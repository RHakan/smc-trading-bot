"""
Wrapper ccxt para Binance USDM Futures.
Mantém duas instâncias — testnet e live — e troca entre elas via set_mode().
"""

import os
import time
import logging
import ccxt
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


_clients: dict[str, ccxt.Exchange] = {}
_active_mode: str = "testnet"

# Endpoint do novo ambiente demo da Binance (antigo testnet.binancefuture.com)
_DEMO_BASE = "https://demo-fapi.binance.com"


def _get_api_keys(mode: str) -> tuple[str, str]:
    """Retorna (api_key, secret) — BD tem prioridade sobre .env."""
    try:
        from api.db import get_credentials, get_auth_username
        # A chave de armazenamento na BD é o username. Até 24/07/2026 vinha do
        # .env (DASHBOARD_USER) — agora a BD é a única fonte da identidade
        # (auth + 2FA migrados para lá). Ler daqui em vez do .env evita que a
        # remoção do DASHBOARD_USER do .env quebre silenciosamente a procura
        # das chaves da Binance (guardadas sob o username real, ex. "meu_utilizador").
        user = get_auth_username() or "admin"
        creds = get_credentials(user, mode)
        if creds:
            return creds["api_key"], creds["secret"]
    except Exception:
        pass

    # Fallback: .env (apenas para compatibilidade — preferir BD)
    if mode == "testnet":
        return (os.getenv("BINANCE_TESTNET_API_KEY", ""),
                os.getenv("BINANCE_TESTNET_SECRET", ""))
    return (os.getenv("BINANCE_LIVE_API_KEY", ""),
            os.getenv("BINANCE_LIVE_SECRET", ""))


def _build_client(mode: str) -> ccxt.Exchange:
    # fetchCurrencies=False evita chamadas desnecessárias ao endpoint Spot
    # (api.binance.com/sapi) que não existe no ambiente demo e não é necessário
    # para operar futuros.
    common_options = {
        "defaultType": "future",
        "fetchCurrencies": False,
        # Sem isto o ccxt LEVANTA EXCEÇÃO em fetch_open_orders() sem símbolo
        # (aviso de rate-limit). A varredura de órfãs precisa da conta inteira e
        # corre no máximo 1x/5min, muito abaixo do limite.
        "warnOnFetchOpenOrdersWithoutSymbol": False,
    }

    api_key, secret = _get_api_keys(mode)

    if mode == "testnet":
        client = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": secret,
            "options": common_options,
        })
        # Redireciona endpoints fapi para o ambiente demo da Binance.
        # set_sandbox_mode(True) aponta para testnet.binancefuture.com (antigo)
        # que não aceita as chaves do novo demo.binance.com.
        client.urls["api"] = {
            key: url.replace("https://fapi.binance.com", _DEMO_BASE)
            for key, url in client.urls["api"].items()
        }
    else:
        client = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": secret,
            "options": common_options,
        })
    return client


def init_clients() -> None:
    """Inicializa os dois clientes ccxt. Chamado uma vez na inicialização do bot."""
    global _clients
    _clients["testnet"] = _build_client("testnet")
    _clients["live"] = _build_client("live")


def get_client() -> ccxt.Exchange:
    """Retorna o cliente ativo (testnet ou live)."""
    if not _clients:
        init_clients()
    return _clients[_active_mode]


def get_mode() -> str:
    return _active_mode


def set_mode(mode: str) -> None:
    """Troca entre 'testnet' e 'live' em runtime (sem reiniciar o bot)."""
    global _active_mode
    if mode not in ("testnet", "live"):
        raise ValueError(f"Modo inválido: {mode}. Use 'testnet' ou 'live'.")
    _active_mode = mode


def reinit_client(mode: str) -> None:
    """Reconstrói o cliente ccxt com as credenciais mais recentes (após guardar na BD)."""
    _clients[mode] = _build_client(mode)


# ---------------------------------------------------------------------------
# Dados de mercado
# ---------------------------------------------------------------------------

_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
}


def fetch_candles(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """
    Busca candles OHLCV da Binance.
    symbol: ex. 'BTC/USDC:USDC'
    timeframe: '15m', '1h', '4h', '1d', etc.
    """
    client = get_client()
    tf = _TIMEFRAME_MAP.get(timeframe, timeframe)
    raw = client.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df


def get_ticker(symbol: str) -> dict:
    """Retorna preço atual e stats 24h."""
    return get_client().fetch_ticker(symbol)


def get_balance() -> dict:
    """Retorna saldo disponível em USDC na conta de futuros."""
    balance = get_client().fetch_balance()
    usdc = balance.get("USDC", {})
    return {
        "total": usdc.get("total", 0),
        "free": usdc.get("free", 0),
        "used": usdc.get("used", 0),
    }


def get_positions() -> list[dict]:
    """Retorna todas as posições abertas."""
    positions = get_client().fetch_positions()
    return [p for p in positions if float(p.get("contracts", 0) or 0) != 0]


def position_exists(symbol: str) -> bool | None:
    """Confirma, com uma consulta DIRIGIDA, se ainda há posição viva neste par.

    Porquê isto existe (06/08/2026): `get_positions()` faz um `fetch_positions()`
    GLOBAL. Se essa resposta vier bem-sucedida mas incompleta (soluço da API, dados
    em cache, o demo a processar o balde ALGO), os símbolos em falta desaparecem da
    lista e o `position_manager` dava-os por FECHADOS — registava a saída ao preço
    do momento e cancelava as ordens. Assinatura no histórico: várias posições
    "fechadas" no mesmo minuto, com lucros pequenos, fora do horário de vela
    (ids 24-27 a 08/07 e 28/29/32 a 12/07 — 7 dos 39 trades, 18% da amostra).

    Uma consulta a UM símbolo é autoritativa e barata, e só corre no caminho em que
    já íamos fechar a posição — custo desprezável.

    Devolve:
        True  → há posição viva (NÃO fechar)
        False → confirmado liso (pode fechar)
        None  → não deu para confirmar; o chamador decide (aqui: não fecha)
    """
    try:
        positions = get_client().fetch_positions([symbol])
    except Exception as e:
        logger.warning(f"[{symbol}] Falha ao confirmar posição individualmente: {e}")
        return None
    for p in positions:
        if p.get("symbol") != symbol:
            continue
        return float(p.get("contracts", 0) or 0) != 0
    return False


# ---------------------------------------------------------------------------
# Camada de compatibilidade: ordens em DOIS baldes
# ---------------------------------------------------------------------------
# A Binance devolve as ordens abertas em dois sítios diferentes e o ccxt só lê um:
#
#   BASIC  → GET /fapi/v1/openOrders     (ccxt fetch_open_orders)  id = orderId
#   ALGO   → GET /fapi/v1/openAlgoOrders (implícito)               id = algoId
#
# O ambiente demo encaminha TODOS os SL/TP (STOP_MARKET/TAKE_PROFIT_MARKET) para o
# balde ALGO. Como o bot só lia o BASIC, era CEGO às próprias ordens de proteção:
# lia 0 → não cancelava nada → recolocava SL/TP a cada boot/trailing → duplicatas
# e órfãs a acumular indefinidamente (ver bot/position_manager.sweep_orphan_orders).
#
# Estas funções leem/cancelam nos DOIS baldes e devolvem um formato único, por isso
# funcionam na demo, na testnet e na conta real — sem depender de qual delas
# encaminha para onde. Formato normalizado:
#   {id, symbol (unificado), kind: 'stop'|'tp'|'other', trigger, reduce_only, is_algo}


def _is_reduce_only(order: dict) -> bool:
    """O 'reduceOnly' unificado do ccxt nem sempre vem preenchido em ordens
    condicionais — cai no 'info' bruto da Binance como fallback."""
    if order.get("reduceOnly") is True:
        return True
    raw = (order.get("info") or {}).get("reduceOnly")
    return raw is True or str(raw).lower() == "true"


def _kind_from_type(*type_strings: str) -> str:
    """Classifica em 'tp' | 'stop' | 'other' a partir de strings de tipo.
    TP é testado primeiro: 'take_profit_market' não contém 'stop', mas a ordem
    inversa daria falso positivo em formatos como 'stop_take_profit'."""
    blob = " ".join(str(t or "").lower() for t in type_strings)
    if "take_profit" in blob or "takeprofit" in blob:
        return "tp"
    if "stop" in blob:
        return "stop"
    return "other"


def _order_kind(order: dict) -> str:
    """Classifica uma ordem do balde BASIC (formato ccxt)."""
    return _kind_from_type(order.get("type"),
                           (order.get("info") or {}).get("type"))


def _algo_open_orders(symbol: str | None = None) -> list[dict]:
    """Ordens abertas do balde ALGO, já normalizadas. Falha em segurança ([])."""
    client = get_client()
    params = {}
    if symbol:
        try:
            params["symbol"] = client.market(symbol)["id"]
        except Exception:
            return []
    try:
        raw = client.fapiPrivateGetOpenAlgoOrders(params)
    except Exception as e:
        logger.warning(f"[EXCHANGE] Falha ao ler ordens ALGO: {e}")
        return []

    out = []
    for o in raw:
        if str(o.get("algoStatus")) != "NEW":      # só as que estão realmente vivas
            continue
        trigger = o.get("triggerPrice")
        try:
            uni = client.safe_market(o.get("symbol"))["symbol"]
        except Exception:
            continue
        out.append({
            "id": str(o.get("algoId")),
            "symbol": uni,
            "kind": _kind_from_type(o.get("orderType"), o.get("algoType")),
            "trigger": float(trigger) if trigger else None,
            "reduce_only": bool(o.get("reduceOnly")),
            "is_algo": True,
        })
    return out


def get_all_open_orders(symbol: str | None = None) -> list[dict]:
    """Ordens abertas dos DOIS baldes (BASIC + ALGO), em formato normalizado.

    Sem `symbol`, devolve a conta inteira — usado na varredura de órfãs.
    Cada balde falha de forma independente: se um não responder, o outro ainda
    devolve o que sabe (nunca levanta exceção).
    """
    client = get_client()
    out: list[dict] = []

    try:
        basic = client.fetch_open_orders(symbol) if symbol else client.fetch_open_orders()
    except Exception as e:
        logger.warning(f"[EXCHANGE] Falha ao ler ordens BASIC: {e}")
        basic = []

    for o in basic:
        trigger = o.get("triggerPrice") or o.get("stopPrice") \
            or (o.get("info") or {}).get("stopPrice")
        out.append({
            "id": str(o.get("id")),
            "symbol": o.get("symbol"),
            "kind": _order_kind(o),
            "trigger": float(trigger) if trigger else None,
            "reduce_only": _is_reduce_only(o),
            "is_algo": False,
        })

    out.extend(_algo_open_orders(symbol))
    return out


def cancel_order_unified(order: dict) -> None:
    """Cancela uma ordem normalizada, no balde a que ela pertence."""
    client = get_client()
    if order.get("is_algo"):
        client.fapiPrivateDeleteAlgoOrder({"algoId": order["id"]})
    else:
        client.cancel_order(order["id"], order["symbol"])


def get_stop_orders(symbol: str) -> dict:
    """
    Lê as ordens condicionais abertas (SL/TP) de um par e devolve os preços.

    Usado na reconciliação ao arranque: depois de um restart o estado em memória
    do position_manager é perdido, mas as ordens SL/TP continuam vivas na Binance.
    Esta função recupera-as para reconstruir a posição.

    CRÍTICO: tem de ler os dois baldes. Enquanto lia só o BASIC devolvia sempre
    {None, None} na demo, e a reconciliação concluía "posição sem stop" e
    RECOLOCAVA SL/TP a cada arranque — criando um par duplicado por restart.

    Retorna {"stop_loss": preço|None, "take_profit": preço|None}.
    """
    sl = tp = None
    for o in get_all_open_orders(symbol):
        if o["trigger"] is None:
            continue
        if o["kind"] == "tp":
            tp = o["trigger"]
        elif o["kind"] == "stop":
            sl = o["trigger"]
    return {"stop_loss": sl, "take_profit": tp}


# ---------------------------------------------------------------------------
# Configuração de posição
# ---------------------------------------------------------------------------

def set_leverage(symbol: str, leverage: int) -> None:
    """Define alavancagem para o par. 1–20x."""
    leverage = max(1, min(20, leverage))
    get_client().set_leverage(leverage, symbol)


def set_margin_isolated(symbol: str) -> None:
    """Garante margem ISOLATED para o par (não contamina o restante do saldo)."""
    try:
        get_client().set_margin_mode("isolated", symbol)
    except ccxt.MarginModeAlreadySet:
        pass


# ---------------------------------------------------------------------------
# Execução de ordens
# ---------------------------------------------------------------------------

def place_market_order(symbol: str, side: str, notional_usdt: float,
                       stop_loss: float, take_profit: float) -> dict:
    """
    Abre ordem a mercado com stop loss e take profit.

    side: 'buy' (long) ou 'sell' (short)
    notional_usdt: valor NOTIONAL da posição em USDT — NÃO a margem.
        A margem é derivada pela Binance (notional / alavancagem), por isso quem
        chama tem de passar o notional. O parâmetro chamava-se `usdt_amount` e
        estava documentado como "margem", mas era usado como notional
        (amount = usdt_amount / price) — e o engine passava mesmo a margem.
        Resultado: as posições abriam `alavancagem`x menores do que o pretendido
        (risco real ≈ 1/3 do configurado). Renomeado para o contrato ser explícito.
    stop_loss: preço do stop loss
    take_profit: preço do take profit
    """
    client = get_client()

    ticker = client.fetch_ticker(symbol)
    price = ticker["last"]

    # Calcula quantidade de contratos baseado no valor USDT e alavancagem implícita
    market = client.market(symbol)
    contract_size = market.get("contractSize", 1)
    amount = notional_usdt / price / contract_size

    # Arredonda para a precisão do par
    amount = client.amount_to_precision(symbol, amount)

    # Ordem principal
    order = client.create_order(symbol, "market", side, amount)

    # Stop loss e take profit como ordens separadas.
    sl_side = "sell" if side == "buy" else "buy"
    tp_side = sl_side

    # ── Stop loss: tenta várias vezes antes de desistir ──────────────────────
    # Uma posição sem SL é inaceitável. Falhas costumam ser transitórias
    # (rede, rate limit, timeout), por isso fazemos retry com backoff curto.
    # Só fechamos a posição se TODAS as tentativas falharem (último recurso).
    SL_MAX_RETRIES = 4
    SL_RETRY_DELAY = 1.5  # segundos entre tentativas

    sl_ok = False
    last_err = None
    for attempt in range(1, SL_MAX_RETRIES + 1):
        try:
            client.create_order(
                symbol, "stop_market", sl_side, amount,
                params={"stopPrice": stop_loss, "reduceOnly": True},
            )
            sl_ok = True
            if attempt > 1:
                logger.info(f"[{symbol}] SL colocado na tentativa {attempt}/{SL_MAX_RETRIES}")
            break
        except Exception as e:
            last_err = e
            logger.warning(
                f"[{symbol}] Falha ao colocar SL (tentativa {attempt}/{SL_MAX_RETRIES}): {e}"
            )
            if attempt < SL_MAX_RETRIES:
                time.sleep(SL_RETRY_DELAY)

    if not sl_ok:
        # Esgotou as tentativas — fecha a posição para não ficar desprotegida
        logger.error(
            f"[{symbol}] SL falhou após {SL_MAX_RETRIES} tentativas — "
            f"a fechar posição para evitar exposição sem stop"
        )
        try:
            client.create_order(symbol, "market", sl_side, amount,
                                params={"reduceOnly": True})
        except Exception as close_err:
            logger.critical(
                f"[{symbol}] CRÍTICO: SL falhou E fecho falhou ({close_err}) — "
                f"INTERVENÇÃO MANUAL URGENTE na Binance!"
            )
        raise RuntimeError(
            f"SL falhou após {SL_MAX_RETRIES} tentativas ({last_err}) — "
            f"posição em {symbol} fechada automaticamente"
        )

    # ── Take profit: igual, mas falha não é fatal (SL já protege) ────────────
    try:
        client.create_order(
            symbol, "take_profit_market", tp_side, amount,
            params={"stopPrice": take_profit, "reduceOnly": True},
        )
    except Exception as e:
        logger.warning(
            f"[{symbol}] TP falhou ({e}) — posição aberta com SL mas sem TP automático. "
            f"O position_manager fecha no target via trailing."
        )

    return order


def cancel_all_orders(symbol: str) -> None:
    """Cancela todas as ordens abertas do par, nos DOIS baldes (BASIC + ALGO).

    O `cancel_all_orders` do ccxt só limpa o BASIC — os SL/TP da demo vivem no
    ALGO e sobreviviam a esta chamada, ficando órfãos e armados.
    """
    try:
        get_client().cancel_all_orders(symbol)
    except Exception as e:
        logger.warning(f"[{symbol}] Falha ao cancelar ordens BASIC: {e}")
    for o in _algo_open_orders(symbol):
        try:
            cancel_order_unified(o)
        except Exception as e:
            logger.warning(f"[{symbol}] Falha ao cancelar ordem ALGO {o['id']}: {e}")


def close_position(symbol: str, position_side: str) -> dict | None:
    """
    Fecha a posição inteira para o par.
    position_side: 'long' ou 'short'
    """
    client = get_client()
    positions = get_positions()
    pos = next((p for p in positions
                if p["symbol"] == symbol and p["side"] == position_side), None)
    if not pos:
        return None

    contracts = abs(float(pos["contracts"]))
    close_side = "sell" if position_side == "long" else "buy"
    amount = client.amount_to_precision(symbol, contracts)
    return client.create_order(
        symbol, "market", close_side, amount,
        params={"reduceOnly": True},
    )


def get_last_trade(symbol: str) -> dict | None:
    """Retorna o último trade de fechamento para o par.
    Usado para obter o preço real de saída quando a exchange fecha a posição via SL/TP.
    """
    try:
        trades = get_client().fetch_my_trades(symbol, limit=10)
        closing = [
            t for t in trades
            if t.get("reduceOnly") or t.get("info", {}).get("reduceOnly") in (True, "true")
        ]
        return closing[-1] if closing else (trades[-1] if trades else None)
    except Exception:
        return None


def close_partial(symbol: str, position_side: str, fraction: float) -> dict | None:
    """
    Fecha uma fração da posição (ex: 0.5 = fecha 50%).
    Usado para take profit parcial no target.
    """
    client = get_client()
    positions = get_positions()
    pos = next((p for p in positions
                if p["symbol"] == symbol and p["side"] == position_side), None)
    if not pos:
        return None

    contracts = abs(float(pos["contracts"])) * fraction
    close_side = "sell" if position_side == "long" else "buy"
    amount = client.amount_to_precision(symbol, contracts)
    return client.create_order(
        symbol, "market", close_side, amount,
        params={"reduceOnly": True},
    )


def place_protective_orders(symbol: str, position_side: str, contracts: float,
                            stop_loss: float, take_profit: float | None = None) -> bool:
    """
    (Re)coloca ordens de proteção reduceOnly para uma posição JÁ ABERTA:
    stop loss (obrigatório) e take profit (opcional). Não abre posição — as ordens
    reduceOnly só podem fechar/reduzir, nunca aumentar exposição.

    Usado na reconciliação ao arranque para reproteger posições que ficaram sem
    SL/TP na exchange (ex: ordens canceladas antes de um restart). Devolve True se
    o stop loss foi colocado com sucesso.
    """
    client = get_client()
    amount = client.amount_to_precision(symbol, abs(float(contracts)))
    close_side = "sell" if position_side == "long" else "buy"

    # Stop loss — obrigatório
    client.create_order(
        symbol, "stop_market", close_side, amount,
        params={"stopPrice": stop_loss, "reduceOnly": True},
    )

    # Take profit — opcional (falha não é fatal, o SL já protege)
    if take_profit:
        try:
            client.create_order(
                symbol, "take_profit_market", close_side, amount,
                params={"stopPrice": take_profit, "reduceOnly": True},
            )
        except Exception as e:
            logger.warning(f"[{symbol}] TP não recolocado ({e}) — posição fica só com SL.")

    return True


def update_stop_loss(symbol: str, position_side: str, new_stop: float) -> None:
    """
    Atualiza o stop loss de uma posição aberta.
    Cancela o SL existente e coloca um novo.
    """
    client = get_client()

    # Cancela os stops existentes nos DOIS baldes antes de colocar o novo. Enquanto
    # só olhava o BASIC, não via nada e ia empilhando: o SOL chegou a ter dois
    # stops idênticos vivos ao mesmo tempo.
    for o in get_all_open_orders(symbol):
        if o["kind"] == "stop" and o["reduce_only"]:
            try:
                cancel_order_unified(o)
            except Exception as e:
                logger.warning(f"[{symbol}] Falha ao cancelar stop antigo {o['id']}: {e}")

    # Busca tamanho atual da posição
    positions = get_positions()
    pos = next((p for p in positions
                if p["symbol"] == symbol and p["side"] == position_side), None)
    if not pos:
        return

    contracts = abs(float(pos["contracts"]))
    sl_side = "sell" if position_side == "long" else "buy"
    amount = client.amount_to_precision(symbol, contracts)
    # Arredonda ao tick size do par (a Binance rejeita preços fora da grade)
    trigger = float(client.price_to_precision(symbol, new_stop))

    client.create_order(
        symbol, "stop_market", sl_side, amount,
        params={"stopPrice": trigger, "reduceOnly": True},
    )


def update_take_profit(symbol: str, position_side: str, new_tp: float) -> None:
    """
    Atualiza o take profit de uma posição aberta.
    Cancela o TP existente e coloca um novo.
    """
    client = get_client()

    # Cancela os TPs existentes nos DOIS baldes (mesmo motivo do update_stop_loss:
    # sem ler o ALGO, os TPs antigos acumulam — o BNB chegou a ter 3 iguais).
    for o in get_all_open_orders(symbol):
        if o["kind"] == "tp" and o["reduce_only"]:
            try:
                cancel_order_unified(o)
            except Exception as e:
                logger.warning(f"[{symbol}] Falha ao cancelar TP antigo {o['id']}: {e}")

    # Busca tamanho atual da posição
    positions = get_positions()
    pos = next((p for p in positions
                if p["symbol"] == symbol and p["side"] == position_side), None)
    if not pos:
        return

    contracts = abs(float(pos["contracts"]))
    tp_side = "sell" if position_side == "long" else "buy"
    amount = client.amount_to_precision(symbol, contracts)
    # Arredonda ao tick size do par (a Binance rejeita preços fora da grade)
    trigger = float(client.price_to_precision(symbol, new_tp))

    client.create_order(
        symbol, "take_profit_market", tp_side, amount,
        params={"stopPrice": trigger, "reduceOnly": True},
    )
