"""
Gerenciador de posições abertas.
Roda a cada 30s e ajusta stop loss automaticamente:
  Fase 1 — Breakeven: quando preço atinge 50% do caminho ao target
  Fase 2 — Trailing: aperta o stop conforme preço avança
  Fase 3 — Parcial:  fecha 50% da posição quando bate o target
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bot import exchange

logger = logging.getLogger(__name__)


RT = 0.0014  # round-trip fees (0.05% + 0.02%) × 2

@dataclass
class ManagedPosition:
    symbol: str
    side: str               # 'long' ou 'short'
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    leverage: int = 1
    margin_usdt: float = 0.0         # capital colocado em margem
    position_size_usdt: float = 0.0  # valor notional da posição
    trade_id: int = 0                # id no SQLite para fechar o registo

    # True = há ordem de stop viva na exchange protegendo a posição.
    # False = posição readotada SEM stop na Binance (ex: SL cancelado/preenchido
    # antes do restart). É apenas MONITORADA/fechável pelo dashboard — a gestão
    # automática (trailing/breakeven) é ignorada para não criar ordens sozinha.
    has_live_stop: bool = True

    # Parametros de gestao (variam por estrategia)
    breakeven_trigger: float = 0.50   # fracao do caminho entry→TP para mover BE
    trailing_atr_mult: float = 1.5    # multiplicador do ATR para trailing apos BE

    # Invalidacao de breakout falho (so lateral_breakout): se o preco fechar de volta
    # pra dentro do range dentro de invalidation_bars velas, sai. Ver check_invalidation.
    invalidation_level: float | None = None   # borda do range rompida
    invalidation_bars: int = 0                # nº de velas em que a regra vale
    bars_since_entry: int = 0                 # velas 1H contadas desde a entrada
    last_inval_ts: object = None              # candle_ts ja contado (dedupe); = vela de entrada

    # Estado interno do trailing
    breakeven_hit: bool = False
    partial_closed: bool = False
    current_stop: float = field(init=False)

    def __post_init__(self):
        self.current_stop = self.stop_loss

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward(self) -> float:
        return abs(self.take_profit - self.entry)

    def progress(self, current_price: float) -> float:
        """Retorna % do caminho percorrido entre entry e take_profit (0 a 1+)."""
        if self.reward == 0:
            return 0.0
        if self.side == "long":
            return (current_price - self.entry) / self.reward
        else:
            return (self.entry - current_price) / self.reward

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Retorna P&L não realizado em % do risco inicial."""
        prog = self.progress(current_price)
        return round(prog * 100, 2)


# Posições gerenciadas em memória (chave: symbol)
_positions: dict[str, ManagedPosition] = {}


def register_position(pos: ManagedPosition) -> None:
    """Registra uma nova posição para acompanhamento."""
    _positions[pos.symbol] = pos
    logger.info(f"[PM] Posição registrada: {pos.symbol} {pos.side.upper()} "
                f"entrada={pos.entry} SL={pos.stop_loss} TP={pos.take_profit}")


def remove_position(symbol: str) -> None:
    _positions.pop(symbol, None)


def get_positions() -> dict[str, ManagedPosition]:
    return dict(_positions)


def check_invalidation(symbol: str, candle_close: float, candle_ts) -> dict | None:
    """Invalidação de breakout falho (só lateral). Chamada pelo engine UMA vez por
    vela 1H fechada. Se, dentro de `invalidation_bars` velas após a entrada, o candle
    FECHAR de volta pra dentro do range (perde a borda rompida), fecha a posição a
    mercado — troca o -1R por ~-0.3R. Baseado em FECHO (candle_ts faz o dedupe).

    Devolve um evento 'position_closed' (para o engine persistir no DB) ou None.
    """
    pos = _positions.get(symbol)
    if pos is None or pos.invalidation_level is None or pos.invalidation_bars <= 0:
        return None

    # Conta cada vela só uma vez. Na entrada, last_inval_ts = vela de entrada, então
    # a 1ª vela nova já conta como bar 1 (janela = entrada+1 .. entrada+N).
    if candle_ts == pos.last_inval_ts:
        return None
    pos.last_inval_ts = candle_ts
    pos.bars_since_entry += 1
    if pos.bars_since_entry > pos.invalidation_bars:
        return None

    inside = (candle_close < pos.invalidation_level) if pos.side == "long" \
        else (candle_close > pos.invalidation_level)
    if not inside:
        return None

    # Rompimento falhou → fecha a mercado (mesmo caminho do fecho manual).
    try:
        order = exchange.close_position(symbol, pos.side)
    except Exception as e:
        logger.warning(f"[PM] Falha ao fechar {symbol} por invalidação: {e}")
        return None
    if order is None:
        # Posição já não existe na exchange (SL/TP disparou no entretanto) — o tick
        # normal trata da limpeza. Não força nada aqui.
        return None
    try:
        exchange.cancel_all_orders(symbol)
    except Exception as e:
        logger.warning(f"[PM] Falha ao cancelar ordens órfãs de {symbol}: {e}")

    # Preço de saída — 3 fallbacks (demo Binance às vezes não preenche 'average').
    exit_price = 0.0
    try:
        exit_price = float(order.get("average") or order.get("price") or 0)
    except (TypeError, ValueError):
        exit_price = 0.0
    if not exit_price:
        try:
            lt = exchange.get_last_trade(symbol)
            exit_price = float(lt["price"]) if lt else 0.0
        except Exception:
            exit_price = 0.0
    if not exit_price:
        exit_price = candle_close

    net_r, pnl_usdt = _calc_close_pnl(pos, exit_price)
    bars = pos.bars_since_entry
    trade_id = pos.trade_id
    remove_position(symbol)
    return {
        "symbol": symbol, "event": "position_closed", "reason": "invalidation",
        "exit_price": exit_price, "pnl_r": net_r, "pnl_usdt": pnl_usdt, "trade_id": trade_id,
        "message": (f"🚫 {symbol}: breakout falso — preço voltou pra dentro do range "
                    f"em {bars} vela(s). Saída @ {exit_price:.4f} "
                    f"({net_r:+.2f}R / {pnl_usdt:+.2f} USDT)"),
    }


def calc_pnl(side: str, entry: float, stop_loss: float,
             notional: float, exit_price: float):
    """
    Calcula R líquido e P&L em USDT para um dado tamanho `notional`.

    Base partilhada pelo fecho total e pelos fechos PARCIAIS: como o P&L escala
    linearmente com o notional, basta passar a fatia (ex: 25% do notional) para
    obter o P&L realizado dessa parcial. O R líquido não depende do tamanho.
    """
    risk = abs(entry - stop_loss)
    # Sem distância de risco (stop == entry) não há como medir R — evita divisão
    # por zero em posições readotadas sem stop conhecido.
    if not risk or not entry:
        return 0.0, 0.0
    if side == "short":
        gross_r = (entry - exit_price) / risk
    else:
        gross_r = (exit_price - entry) / risk
    net_r     = round(gross_r - entry * RT / risk, 4)
    risk_usdt = notional * risk / entry
    pnl_usdt  = round(net_r * risk_usdt, 2)
    return net_r, pnl_usdt


def _calc_close_pnl(pos: ManagedPosition, exit_price: float):
    """Calcula R líquido e P&L em USDT para fechamento total da posição."""
    return calc_pnl(pos.side, pos.entry, pos.stop_loss,
                    pos.position_size_usdt, exit_price)


def reduce_position(symbol: str, fraction: float):
    """
    Reduz o tamanho da posição gerida por uma fração (0<f<1) após um fecho
    parcial na exchange. Retorna (notional_fechado, margem_fechada) — ou (0,0)
    se a posição não estiver em memória. Marca `partial_closed`.
    """
    pos = _positions.get(symbol)
    if not pos or not (0 < fraction < 1):
        return 0.0, 0.0
    closed_notional = round(pos.position_size_usdt * fraction, 2)
    closed_margin   = round(pos.margin_usdt * fraction, 2)
    pos.position_size_usdt = round(pos.position_size_usdt - closed_notional, 2)
    pos.margin_usdt        = round(pos.margin_usdt - closed_margin, 2)
    pos.partial_closed = True
    return closed_notional, closed_margin


# ── Varredura de ordens órfãs ────────────────────────────────────────────────
# Uma ordem reduceOnly num par SEM posição é sobra de um trade já encerrado (a
# ordem irmã da que disparou). Ela fica ARMADA: quando o bot abrir um trade novo
# nesse par, a ordem velha dispara e fecha a posição nova num nível de um trade
# antigo — perda que o backtest nunca modela.
#
# A limpeza dentro do tick() só cobre símbolos presentes em _positions, que é
# MEMÓRIA e morre a cada restart. Posições fechadas enquanto o bot esteve em
# baixo nunca eram detetadas e as órfãs sobreviviam indefinidamente.
_SWEEP_INTERVAL_SEC = 300          # leitura global de ordens é pesada (weight 40)
_last_sweep: datetime | None = None


def sweep_orphan_orders() -> list[str]:
    """Cancela ordens condicionais de pares SEM posição aberta na exchange.

    Pares presentes em _positions são PRESERVADOS de propósito: se o bot acabou
    de abrir uma posição e a exchange ainda não a reporta, não queremos cancelar
    o stop recém-colocado. Esses casos já são tratados dentro do tick().

    Devolve a lista de símbolos limpos.
    """
    try:
        orders = exchange.get_all_open_orders()
        live = exchange.get_positions()
    except Exception as e:
        logger.warning(f"[PM] Varredura de órfãs falhou ao ler a exchange: {e}")
        return []

    with_position = {p.get("symbol") for p in live}
    with_orders = {o.get("symbol") for o in orders if o.get("symbol")}
    orphans = with_orders - with_position - set(_positions.keys())

    cleaned: list[str] = []
    for symbol in sorted(orphans):
        try:
            exchange.cancel_all_orders(symbol)
            cleaned.append(symbol)
            logger.info(f"[PM] Órfãs canceladas em {symbol} (par sem posição aberta).")
        except Exception as e:
            logger.warning(f"[PM] Falha ao cancelar órfãs de {symbol}: {e}")
    return cleaned


def tick(partial_close_pct: float = 0.5) -> list[dict]:
    """
    Executa um ciclo de verificação em todas as posições gerenciadas.
    Retorna lista de eventos gerados neste tick para logging/dashboard.
    Verifica fechamentos na exchange (SL/TP atingidos) e ajusta trailing stop.
    """
    events = []

    # Varredura periódica de órfãs — não depende da memória, logo sobrevive a
    # restarts. Throttled: a leitura global de ordens tem peso alto na API.
    global _last_sweep
    now = datetime.now(timezone.utc)
    if _last_sweep is None or (now - _last_sweep).total_seconds() >= _SWEEP_INTERVAL_SEC:
        _last_sweep = now
        sweep_orphan_orders()

    # Busca posições abertas na exchange para detectar fechamentos por SL/TP
    try:
        live_pos_list = exchange.get_positions()
        live_positions = {p["symbol"]: p for p in live_pos_list}
    except Exception as e:
        logger.warning(f"[PM] Não foi possível buscar posições da exchange: {e}")
        live_positions = None

    for symbol, pos in list(_positions.items()):
        try:
            ticker = exchange.get_ticker(symbol)
            price = ticker["last"]

            # ── Detecta fechamento ─────────────────────────────────────────
            if live_positions is not None:
                # Verifica se a posição ainda existe na Binance
                if symbol not in live_positions:
                    # ── GUARDA CONTRA FECHO-FANTASMA (06/08/2026) ──────────────
                    # Faltar no snapshot GLOBAL não prova que fechou: uma resposta
                    # incompleta da API fazia o bot dar por fechadas TODAS as
                    # posições de uma vez, ao preço do momento (ver
                    # exchange.position_exists). Confirma com consulta dirigida
                    # antes de destruir estado. Na dúvida (None), NÃO fecha —
                    # deixar viver mais um tick é inofensivo; fechar por engano
                    # cancela as ordens de proteção e amputa o trade.
                    still_open = exchange.position_exists(symbol)
                    if still_open is not False:
                        logger.warning(
                            f"[PM] {symbol}: ausente do snapshot global mas a consulta "
                            f"dirigida diz {'VIVA' if still_open else 'INCONCLUSIVO'} — "
                            f"fecho ignorado (protecao contra fecho-fantasma)."
                        )
                        events.append({
                            "symbol": symbol,
                            "event": "phantom_close_avoided",
                            "message": (f"🛡 {symbol.split('/')[0]}: fecho fantasma evitado "
                                        f"(snapshot da exchange veio incompleto)"),
                        })
                        if still_open is None:
                            # Inconclusivo: não fecha, mas também não mexe em ordens
                            # (mover um stop numa posição que afinal fechou criaria
                            # uma ordem órfã). Espera pelo próximo tick.
                            continue
                        # Confirmada VIVA: NÃO fecha e segue para a gestão normal
                        # deste tick, senão o trailing/breakeven ficavam parados.
                    else:
                        # Confirmado liso pelas DUAS leituras: fechou mesmo na
                        # exchange (SL ou TP disparou). A ordem IRMÃ (a que não
                        # disparou) fica órfã — reduceOnly, mas some para nada.
                        try:
                            exchange.cancel_all_orders(symbol)
                        except Exception as e:
                            logger.warning(f"[PM] Falha ao cancelar ordens órfãs de {symbol}: {e}")
                        try:
                            last_trade = exchange.get_last_trade(symbol)
                            exit_price = float(last_trade["price"]) if last_trade else price
                        except Exception:
                            exit_price = price
                        net_r, pnl_usdt = _calc_close_pnl(pos, exit_price)
                        remove_position(symbol)
                        events.append({
                            "symbol":     symbol,
                            "event":      "position_closed",
                            "reason":     "exchange",   # SL/TP disparou na Binance
                            "exit_price": exit_price,
                            "pnl_r":      net_r,
                            "pnl_usdt":   pnl_usdt,
                            "trade_id":   pos.trade_id,
                            "message":    (f"🔴 {symbol}: fechado pela exchange @ {exit_price:.4f} "
                                           f"({net_r:+.2f}R / {pnl_usdt:+.2f} USDT)"),
                        })
                        continue
            else:
                # Fallback quando exchange inacessível: detecta stop por cruzamento de preço
                stop_hit = (pos.side == "short" and price >= pos.current_stop) or \
                           (pos.side == "long"  and price <= pos.current_stop)
                if stop_hit:
                    net_r, pnl_usdt = _calc_close_pnl(pos, pos.current_stop)
                    remove_position(symbol)
                    events.append({
                        "symbol":     symbol,
                        "event":      "position_closed",
                        "reason":     "stop",
                        "exit_price": pos.current_stop,
                        "pnl_r":      net_r,
                        "pnl_usdt":   pnl_usdt,
                        "trade_id":   pos.trade_id,
                        "message":    (f"🔴 {symbol}: stop atingido @ {pos.current_stop:.4f} "
                                       f"({net_r:+.2f}R / {pnl_usdt:+.2f} USDT)"),
                    })
                    continue

            # Posição readotada sem stop vivo na exchange: apenas monitorada.
            # Não movemos SL/breakeven/parcial (não temos ordem para atualizar e
            # não queremos criar uma sozinhos). A deteção de fechamento acima já
            # cobre o caso de ela sumir da exchange.
            if not pos.has_live_stop:
                continue

            prog = pos.progress(price)
            be_trig = pos.breakeven_trigger
            trail_mult = pos.trailing_atr_mult

            # ---------------------------------------------------------------
            # Fase 3 — ALVO ATINGIDO: fecha A MERCADO, sem confiar na ordem
            # ---------------------------------------------------------------
            # PORQUÊ (07/08/2026, trade ETH/USDC #60): a ordem TAKE_PROFIT_MARKET
            # é colocada corretamente (triggerPrice certo, CONTRACT_PRICE, quantidade
            # certa, reduceOnly) mas NÃO EXECUTA neste ambiente — ficou em
            # `algoStatus: NEW` / `triggerTime: 0` depois de o preço passar o alvo
            # DUAS vezes, e a posição continuou aberta 40 horas.
            # No histórico: 24 trades fecharam no stop (−2299 USDT) e apenas 4 no
            # alvo — e nenhum desses 4 é uma execução limpa da ordem de TP. Nenhuma
            # estratégia sobrevive a um ambiente onde só o lado da perda executa.
            #
            # A ordem na exchange fica na mesma, como rede secundária. Esta é a
            # primária. Fecha 100% (é o que o backtest valida e o que a ordem de TP
            # pretendia fazer) — o fecho parcial de 50% que estava aqui era
            # inconsistente com ambos e nunca chegou a correr.
            if prog >= 1.0:
                order = None
                try:
                    order = exchange.close_position(symbol, pos.side)
                except Exception as e:
                    logger.error(f"[PM] {symbol}: ALVO atingido mas o fecho a mercado "
                                 f"falhou: {e} — tenta de novo no próximo tick")
                if order is None:
                    # Falhou, ou a posição já não existe (a ordem da exchange pode ter
                    # disparado entretanto). Em qualquer dos casos não inventa fecho:
                    # o próximo tick reavalia com dados frescos.
                    continue

                try:
                    exchange.cancel_all_orders(symbol)
                except Exception as e:
                    logger.warning(f"[PM] Falha ao cancelar ordens de {symbol}: {e}")

                # Preço real de saída — 3 fallbacks (a demo às vezes não preenche
                # 'average'), o mesmo padrão de check_invalidation.
                exit_price = 0.0
                try:
                    exit_price = float(order.get("average") or order.get("price") or 0)
                except (TypeError, ValueError):
                    exit_price = 0.0
                if not exit_price:
                    try:
                        lt = exchange.get_last_trade(symbol)
                        exit_price = float(lt["price"]) if lt else 0.0
                    except Exception:
                        exit_price = 0.0
                if not exit_price:
                    exit_price = price

                net_r, pnl_usdt = _calc_close_pnl(pos, exit_price)
                trade_id = pos.trade_id
                remove_position(symbol)
                events.append({
                    "symbol":     symbol,
                    "event":      "position_closed",
                    "reason":     "alvo",
                    "exit_price": exit_price,
                    "pnl_r":      net_r,
                    "pnl_usdt":   pnl_usdt,
                    "trade_id":   trade_id,
                    "message":    (f"🎯 {symbol}: ALVO atingido — fechado a mercado @ "
                                   f"{exit_price:.4f} ({net_r:+.2f}R / {pnl_usdt:+.2f} USDT)"),
                })
                continue

            # ---------------------------------------------------------------
            # Fase 2 — Trailing dinâmico (apos breakeven)
            # ---------------------------------------------------------------
            elif prog >= be_trig and pos.breakeven_hit:
                trail_distance = pos.atr * trail_mult
                if pos.side == "long":
                    candidate_stop = price - trail_distance
                    if candidate_stop > pos.current_stop:
                        _update_stop(pos, symbol, candidate_stop)
                        events.append({
                            "symbol": symbol,
                            "event": "trailing",
                            "price": price,
                            "new_stop": pos.current_stop,
                            "message": f"📈 {symbol}: trailing stop → {pos.current_stop:.4f}",
                        })
                else:
                    candidate_stop = price + trail_distance
                    if candidate_stop < pos.current_stop:
                        _update_stop(pos, symbol, candidate_stop)
                        events.append({
                            "symbol": symbol,
                            "event": "trailing",
                            "price": price,
                            "new_stop": pos.current_stop,
                            "message": f"📉 {symbol}: trailing stop → {pos.current_stop:.4f}",
                        })

            # ---------------------------------------------------------------
            # Fase 1 — Breakeven (quando atinge be_trigger% do caminho)
            # ---------------------------------------------------------------
            elif prog >= be_trig and not pos.breakeven_hit:
                _update_stop(pos, symbol, pos.entry)
                pos.breakeven_hit = True
                events.append({
                    "symbol": symbol,
                    "event": "breakeven",
                    "price": price,
                    "new_stop": pos.entry,
                    "message": f"🔒 {symbol}: stop movido para breakeven ({pos.entry}). Trade gratuito!",
                })

        except Exception as e:
            logger.warning(f"[PM] Erro ao processar {symbol}: {e}")

    return events


def _update_stop(pos: ManagedPosition, symbol: str, new_stop: float) -> None:
    pos.current_stop = new_stop
    try:
        exchange.update_stop_loss(symbol, pos.side, new_stop)
    except Exception as e:
        logger.warning(f"[PM] Falha ao atualizar SL na exchange: {e}")
