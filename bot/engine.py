"""
Engine principal do bot.
Gerencia o scheduler APScheduler, executa as estratégias nos pares configurados,
registra trades no SQLite e faz broadcast dos eventos via WebSocket.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

from bot import exchange, position_manager, regime as regime_mod
from bot import strategies
from bot.risk import calc_position_size, calc_required_leverage, validate_trade

load_dotenv()
logger = logging.getLogger(__name__)


def _to_python(obj):
    """Converte recursivamente tipos numpy para Python nativo (bool, int, float).
    Necessário porque pandas/numpy retornam tipos próprios que quebram json e sqlite3.
    """
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar: bool_, int64, float64…
        return obj.item()
    return obj

# Broadcast function injetada pelo api/main.py para enviar eventos ao WebSocket
_broadcast_fn = None

def set_broadcast(fn) -> None:
    global _broadcast_fn
    _broadcast_fn = fn

def _broadcast(event: dict) -> None:
    if not _broadcast_fn:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_broadcast_fn(event))
    except RuntimeError:
        # Chamado fora do event loop (ex: thread do scheduler) — agenda no loop correto
        try:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(_broadcast_fn(event), loop)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Estado do engine
# ---------------------------------------------------------------------------

_running = False
_scheduler: AsyncIOScheduler | None = None
_last_trade_time: dict[str, datetime] = {}     # symbol → UTC datetime do último trade
_last_candle_ts: dict[str, object] = {}        # symbol → timestamp do último candle analisado (deduplicação)

# Cache do regime do BTC (contexto global p/ o filtro BTC da bull_smc). O BTC é o
# líder: só compramos alt em bull quando o próprio BTC está em BULL. Recalcula-se
# no máx. 1x por candle diário (chave = data UTC) — barato, evita 1 fetch por par.
BTC_SYMBOL = "BTC/USDT:USDT"
_btc_regime_cache: dict[str, str] = {"day": "", "regime": "NEUTRAL"}


def _get_btc_regime() -> str:
    """Regime do BTC (EMA20 diária + slope), com cache por dia UTC. Em caso de
    falha na leitura, devolve NEUTRAL (fail-safe: bloqueia o long da bull)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _btc_regime_cache["day"] == today:
        return _btc_regime_cache["regime"]
    try:
        btc_daily = exchange.fetch_candles(BTC_SYMBOL, "1d", limit=60)
        reg = regime_mod.detect_regime(btc_daily)
    except Exception as e:                       # rede/exchange indisponível
        logger.warning(f"[BTC_FILTER] falha ao ler regime do BTC: {e} — assumindo NEUTRAL")
        reg = "NEUTRAL"
    _btc_regime_cache.update(day=today, regime=reg)
    return reg


def is_running() -> bool:
    return _running


def get_config() -> dict:
    return {
        "mode": exchange.get_mode(),
        "strategy": os.getenv("STRATEGY_MODE", "bear_v13"),
        "symbols": _get_symbols(),
        "leverage": int(os.getenv("DEFAULT_LEVERAGE", "3")),
        "risk_pct": float(os.getenv("RISK_PER_TRADE_PCT", "1.0")),
        "max_trades_per_day": int(os.getenv("MAX_TRADES_PER_DAY", "5")),
    }


def _get_symbols() -> list[str]:
    raw = os.getenv("SYMBOLS", "BTC/USDC:USDC")
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Scanner de sinais — executado pelo scheduler
# ---------------------------------------------------------------------------

async def scan_signals(execute_trades: bool = True) -> None:
    """Varre todos os pares configurados e executa trades se houver sinal.

    execute_trades=False: modo diagnóstico — mostra condições sem abrir posições.
    """
    cfg = get_config()
    strategy_mode = cfg["strategy"]
    symbols = cfg["symbols"]
    leverage = cfg["leverage"]
    risk_pct = cfg["risk_pct"]

    now = datetime.now(timezone.utc).strftime("%H:%M")
    label = "🔍 Scan" if execute_trades else "🔎 Scan diagnóstico (sem trades)"
    _broadcast({"type": "log", "level": "info",
                "message": f"{label} — {len(symbols)} pares [{now} UTC]"})

    # Scan manual ignora deduplicação para forçar nova leitura
    if not execute_trades:
        _last_candle_ts.clear()

    for symbol in symbols:
        try:
            await _process_symbol(symbol, strategy_mode,
                                  leverage, risk_pct,
                                  execute_trades=execute_trades)
        except Exception as e:
            logger.error(f"[ENGINE] Erro ao processar {symbol}: {e}")
            _broadcast({"type": "log", "level": "error",
                        "message": f"Erro {symbol}: {e}"})

    if execute_trades:
        # Atualiza posições abertas (trailing stop, breakeven, parciais)
        events = position_manager.tick()
        for event in events:
            logger.info(event["message"])
            _broadcast({"type": "position_update", **event})


async def _process_symbol(symbol: str, strategy: str,
                          leverage: int, risk_pct: float,
                          execute_trades: bool = True) -> None:
    """Busca dados 1H e analisa sinal para um par específico."""

    if strategy not in ("auto", "auto_v2") and not strategies.is_valid(strategy):
        logger.warning(f"[ENGINE] Estratégia desconhecida: {strategy} — ignorando {symbol}")
        return

    df_raw = exchange.fetch_candles(symbol, "1h", limit=200)

    # Deduplicação: analisa apenas o ultimo candle 1H fechado
    candle_ts = df_raw.index[-2]
    if _last_candle_ts.get(symbol) == candle_ts:
        return
    _last_candle_ts[symbol] = candle_ts

    df_entry = df_raw.iloc[:-1]

    # Invalidação de breakout falho (lateral): roda 1x por vela 1H fechada (o dedup
    # acima garante isso). Se o candle fechou de volta pra dentro do range dentro da
    # janela, fecha a posição — corta o -1R para ~-0.3R. Ver position_manager e
    # lateral_breakout.INVAL_BARS. Baseado em FECHO de vela (não pavio).
    inval_close = float(df_entry.iloc[-1]["close"])
    inval_event = position_manager.check_invalidation(symbol, inval_close, candle_ts)
    if inval_event is not None:
        logger.info(inval_event["message"])
        _broadcast({"type": "position_update", **inval_event})
        _persist_close_event(inval_event)
        return

    # Cooldown em barras 1H
    last_dt = _last_trade_time.get(symbol)
    if last_dt is not None:
        elapsed_h = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        bars_elapsed = max(0, int(elapsed_h))
        last_bar = max(0, len(df_entry) - 1 - bars_elapsed)
    else:
        last_bar = -999

    df_daily = exchange.fetch_candles(symbol, "1d", limit=60)

    # Modo "auto" (o Decisor): o roteador de regime escolhe a estratégia do par.
    # BEAR → bear_v13. BULL/NEUTRAL → atualmente sem estratégia aprovada (fica de
    # fora). Quando uma Bull/Neutral passar no backtest, liga-se em regime.py.
    effective_strategy = strategy
    if strategy in ("auto", "auto_v2"):
        market_regime = regime_mod.detect_regime(df_daily)
        # auto    → mapa de produção (BEAR→bear_v13). auto_v2 → sistema v2
        # (NEUTRAL→lateral_breakout, BULL→bull_smc, BEAR off). Ver bot/regime.py.
        if strategy == "auto_v2":
            effective_strategy = regime_mod.strategy_for_regime_v2(market_regime)
        else:
            effective_strategy = regime_mod.strategy_for_regime(market_regime)
        if effective_strategy is None:
            logger.info(f"[{symbol}] Regime {market_regime} — sem estratégia ativa, ignorando")
            _broadcast({"type": "log", "level": "info",
                        "message": f"⏸ {symbol.split('/')[0]}: regime {market_regime} — fora do mercado"})
            return

    strat_module = strategies.get_strategy(effective_strategy)
    if strat_module is None:
        logger.warning(f"[ENGINE] Estratégia '{effective_strategy}' não registada — ignorando {symbol}")
        return
    # Filtro BTC: a bull_smc só compra alt quando o líder (BTC) também está em
    # BULL. Injeta-se o regime do BTC como contexto global (o analyze por par não
    # o tem). Validado no backtest v3: +387 vs +362, corta 46 trades "pump de alt
    # sem líder" (ex.: ETH/AVAX 06/07/2026, -366 ao vivo). Ver backtest_bull_filtro_btc.py.
    if effective_strategy == "bull_smc":
        result = strat_module.analyze(df_entry, df_daily, last_bar,
                                      btc_regime=_get_btc_regime())
    else:
        result = strat_module.analyze(df_entry, df_daily, last_bar)

    # A partir daqui 'strategy' reflete a estratégia efetivamente usada (relevante no modo auto)
    strategy = effective_strategy

    # CRÍTICO: converte numpy.bool_/float64/int64 para Python nativo antes de
    # qualquer broadcast ou persistência (sqlite3 e json.dumps rejeitam tipos numpy)
    result = _to_python(result)

    signal = result["signal"]
    conditions_passed = sum(1 for c in result["conditions"] if c["passed"])
    conditions_total = len(result["conditions"])

    logger.info(
        f"[{symbol}] {strategy.upper()} → {signal} "
        f"({conditions_passed}/{conditions_total} condições) "
        f"R:R={result['rr']:.2f}"
    )

    # Loga resultado no dashboard
    short_sym = symbol.replace("/USDC:USDC", "").replace("/USDC", "")
    rr_txt = f" R:R {result['rr']:.1f}" if result["rr"] else ""
    level = "ok" if signal != "NONE" else "info"
    _broadcast({"type": "log", "level": level,
                "message": f"{short_sym}: {signal} ({conditions_passed}/{conditions_total}){rr_txt}"})

    # Envia snapshot de condições para o dashboard
    _broadcast({
        "type": "signal_update",
        "symbol": symbol,
        "strategy": strategy,
        "signal": signal,
        "conditions": result["conditions"],
        "entry": result["entry"],
        "stop_loss": result["stop_loss"],
        "take_profit": result["take_profit"],
        "rr": result["rr"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    if signal == "NONE":
        return

    if not execute_trades:
        # Modo diagnóstico: mostra o que abriria mas não executa
        _broadcast({"type": "log", "level": "ok",
                    "message": f"💡 {short_sym}: {signal} DETECTADO — sem execução (modo diagnóstico)"})
        return

    # Verifica se já tem posição aberta neste par
    open_positions = position_manager.get_positions()
    if symbol in open_positions:
        logger.info(f"[{symbol}] Já tem posição aberta — ignorando sinal")
        _broadcast({"type": "log", "level": "info",
                    "message": f"⏸ {short_sym}: sinal {signal} — posição já aberta"})
        return

    # Limite de posições simultâneas
    max_pos = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
    if len(open_positions) >= max_pos:
        logger.info(f"[{symbol}] Limite de posições atingido ({max_pos}) — ignorando")
        _broadcast({"type": "log", "level": "info",
                    "message": f"⏸ {short_sym}: sinal {signal} — limite de {max_pos} posições atingido"})
        return

    # Valida direção do stop e R:R antes de executar
    side = "long" if signal == "LONG" else "short"

    # Limite de posições POR LADO (anti-correlação). Sem isto o bot podia abrir
    # 5 posições do mesmo lado que stopam juntas num movimento contrário (incidente
    # 02/07). Confirmado em backtest (5 anos + OOS): cap 2 corta o DD 10.1%->8.1% e
    # o ret/DD 34->42, sem tocar no WR (o cap não muda o sinal, só quantas abrem
    # juntas). MAX_PER_SIDE=2 no .env; default 3 = o do backtest v3.
    max_side = int(os.getenv("MAX_PER_SIDE", "3"))
    n_same_side = sum(1 for p in open_positions.values() if p.side == side)
    if n_same_side >= max_side:
        logger.info(f"[{symbol}] Limite de {max_side} posições em '{side}' atingido — ignorando")
        _broadcast({"type": "log", "level": "info",
                    "message": f"⏸ {short_sym}: sinal {signal} — limite de {max_side} {side.upper()} atingido"})
        return

    entry = result["entry"]
    sl = result["stop_loss"]
    tp = result["take_profit"]
    cfg_rr = result["rr"]

    if sl is None or tp is None:
        _broadcast({"type": "log", "level": "error",
                    "message": f"⚠ {short_sym}: SL ou TP ausente — trade cancelado"})
        return

    # validate_trade com rr_minimo=0 — o R:R já foi verificado pelas condições da estratégia.
    # Aqui só checamos se stop/TP estão na direção correta (sanidade básica).
    valid, reason = validate_trade(entry, sl, tp, rr_minimo=0.0, side=side)
    if not valid:
        logger.warning(f"[{symbol}] Trade rejeitado: {reason}")
        _broadcast({"type": "log", "level": "error",
                    "message": f"⚠ {short_sym}: rejeitado — {reason}"})
        return

    # Executa trade (passa a estratégia efetiva — relevante no modo auto)
    await _execute_trade(symbol, result, leverage, risk_pct, strategy, candle_ts)

    # Regista timestamp do trade para o cooldown
    _last_trade_time[symbol] = datetime.now(timezone.utc)


async def _execute_trade(symbol: str, result: dict,
                         leverage: int, risk_pct: float,
                         strategy: str | None = None, candle_ts=None) -> None:
    """Abre a posição e registra no position_manager e no banco.

    strategy: estratégia efetivamente usada (no modo auto é a que o Decisor
    escolheu, ex. 'bear_v13' — não 'auto'). Cai no STRATEGY_MODE se não for dado.
    """
    from api.db import save_trade

    signal = result["signal"]
    # result já foi processado pelo _to_python em _process_symbol —
    # garantimos Python float/bool aqui como camada extra de segurança
    entry = float(result["entry"])
    sl = float(result["stop_loss"])
    tp = float(result["take_profit"])
    rr = float(result["rr"])
    atr = float(result.get("atr", 0) or 0)

    balance = exchange.get_balance()

    position_size = calc_position_size(balance["free"], risk_pct, entry, sl)
    required_lev = calc_required_leverage(balance["free"], position_size)
    effective_lev = max(leverage, required_lev)

    if position_size <= 0:
        logger.error(f"[{symbol}] position_size inválido ({position_size}) — trade cancelado")
        return

    # calc_position_size devolve o NOTIONAL da posição. A margem é só o capital
    # imobilizado (notional/alavancagem) — serve para registo, não para dimensionar
    # a ordem. Passar a margem aqui abria posições `effective_lev`x menores do que o
    # risco pedido (bug encontrado 17/07: risco real ≈ 1.3% em vez dos 4% configurados).
    margin_used = position_size / effective_lev

    exchange.set_margin_isolated(symbol)
    exchange.set_leverage(symbol, effective_lev)
    order = exchange.place_market_order(
        symbol,
        side="buy" if signal == "LONG" else "sell",
        notional_usdt=position_size,
        stop_loss=sl,
        take_profit=tp,
    )
    order_id = order.get("id", "N/A")

    mode_label_short = "DEMO" if exchange.get_mode() == "testnet" else "LIVE"

    # Salva no banco PRIMEIRO para obter o trade_id antes de registrar no position_manager
    trade_record = {
        "symbol": symbol,
        "side": signal,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "rr": rr,
        "leverage": effective_lev,
        "margin_usdt": margin_used,
        "position_size_usdt": position_size,
        "order_id": order_id,
        "mode": mode_label_short,
        "exchange_mode": exchange.get_mode(),
        "strategy": strategy or os.getenv("STRATEGY_MODE", "bear_v13"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "open",
    }
    trade_id = save_trade(trade_record)

    # Registra no position_manager com o trade_id correto
    pos = position_manager.ManagedPosition(
        symbol=symbol,
        side="long" if signal == "LONG" else "short",
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        atr=atr,
        leverage=effective_lev,
        margin_usdt=round(margin_used, 2),
        position_size_usdt=round(position_size, 2),
        trade_id=trade_id,
        breakeven_trigger=float(result.get("be_trigger_pct", 0.50)),
        trailing_atr_mult=float(result.get("trail_atr", 1.5)),
        invalidation_level=result.get("invalidation_level"),
        invalidation_bars=int(result.get("invalidation_bars", 0) or 0),
        last_inval_ts=candle_ts,   # vela de entrada = baseline da contagem
    )
    position_manager.register_position(pos)

    mode_label = f"{'🧪' if mode_label_short == 'DEMO' else '🔴'} {mode_label_short}"
    logger.info(
        f"{mode_label} [{symbol}] {signal} aberto — "
        f"entrada={entry:.4f} SL={sl:.4f} TP={tp:.4f} R:R={rr:.2f} {effective_lev}x "
        f"pos={position_size:.2f} USDC margem={margin_used:.2f} USDC"
    )

    _broadcast({
        "type": "trade_opened",
        **trade_record,
    })


# ---------------------------------------------------------------------------
# Reconciliação com a exchange
# ---------------------------------------------------------------------------

def _find_open_trade(symbol: str) -> dict | None:
    """Devolve o registo do trade ainda aberto para este par (None se não houver)."""
    try:
        from api.db import get_trades
        for t in get_trades(limit=50, status="open"):
            if t.get("symbol") == symbol:
                return t
    except Exception:
        pass
    return None


def _find_open_trade_id(symbol: str) -> int:
    """Procura no DB o id do trade ainda aberto para este par (0 se não houver)."""
    t = _find_open_trade(symbol)
    return int(t.get("id", 0) or 0) if t else 0


def _reconcile_positions() -> int:
    """
    Reconstrói o estado em memória do position_manager a partir das posições
    REAIS abertas na exchange.

    Porquê: o `_positions` do position_manager vive só em memória e perde-se em
    cada restart do bot. As posições, porém, continuam abertas na Binance. Sem
    reconciliação, o bot fica cego a elas e pode:
      1. Tentar reabrir um par já aberto → erro -4067 (set_leverage com posição viva)
      2. Pior: abrir uma 2ª posição duplicando o risco se não houver SL/TP a bloquear

    Esta função lê as posições reais + as ordens SL/TP e readota cada uma para
    gestão (trailing, breakeven). Corre uma vez no arranque, antes do scheduler.

    Retorna o número de posições readotadas.
    """
    try:
        live = exchange.get_positions()
    except Exception as e:
        logger.warning(f"[RECONCILE] Não foi possível ler posições da exchange: {e}")
        return 0

    adopted = 0
    for p in live:
        symbol = p.get("symbol")
        if not symbol or symbol in position_manager.get_positions():
            continue
        try:
            side     = p.get("side")                          # 'long' | 'short'
            entry    = float(p.get("entryPrice") or 0)
            notional = abs(float(p.get("notional") or 0))
            leverage = int(float(p.get("leverage") or 1))
            if entry <= 0 or side not in ("long", "short"):
                logger.warning(f"[RECONCILE] {symbol}: dados incompletos — ignorado")
                continue

            stops = exchange.get_stop_orders(symbol)
            sl, tp = stops["stop_loss"], stops["take_profit"]

            db_trade = _find_open_trade(symbol)
            has_live_stop = sl is not None

            if sl is None:
                # Sem stop VIVO na exchange (cancelado/preenchido antes do restart).
                # Decisão do Rafa: RECOLOCAR o SL/TP do DB para reproteger a posição.
                db_sl = float(db_trade.get("stop_loss") or 0) if db_trade else 0.0
                db_tp = float(db_trade.get("take_profit") or 0) if db_trade else 0.0
                contracts = abs(float(p.get("contracts") or 0))

                if db_sl and contracts > 0:
                    try:
                        # Limpa ordens órfãs e recoloca proteção (reduceOnly só fecha)
                        exchange.cancel_all_orders(symbol)
                        exchange.place_protective_orders(
                            symbol, side, contracts, db_sl, db_tp or None
                        )
                        sl = db_sl
                        if db_tp:
                            tp = db_tp
                        has_live_stop = True
                        logger.info(
                            f"[RECONCILE] {symbol} {side.upper()} SEM stop na exchange — "
                            f"SL/TP RECOLOCADOS do DB (SL={db_sl:.4f} TP={db_tp or 0:.4f})."
                        )
                        _broadcast({"type": "log", "level": "ok",
                                    "message": f"🛡 {symbol.split('/')[0]}: stop recolocado "
                                               f"na Binance @ {db_sl:.4f}"})
                    except Exception as e:
                        logger.warning(f"[RECONCILE] {symbol}: falha ao recolocar SL/TP: {e}")

                if not has_live_stop:
                    # Não deu para recolocar (sem stop no DB ou erro) → monitora flagueada.
                    if sl is None:
                        sl = db_sl or entry * (1.01 if side == "short" else 0.99)
                    logger.warning(
                        f"[RECONCILE] {symbol} {side.upper()} SEM stop e sem recolocar — "
                        f"readotada só para monitoramento (stop de referência={sl:.4f})."
                    )
                    _broadcast({"type": "log", "level": "warn",
                                "message": f"⚠ {symbol.split('/')[0]}: posição SEM stop na Binance — "
                                           f"monitorada no painel, feche pelo card se quiser."})

            # Referência de R = stop ORIGINAL (do DB). O `sl` lido da exchange pode já
            # ter sido movido pelo trailing ou arrastado à mão — usá-lo como referência
            # encolheria o risco e falsearia o R/P&L a cada restart (no breakeven daria
            # risco 0 → P&L registado como 0). O `sl` vivo entra no current_stop.
            orig_sl = float(db_trade.get("stop_loss") or 0) if db_trade else 0.0
            if orig_sl <= 0:
                orig_sl = sl                    # sem registo no DB → melhor esforço

            # O stop da estratégia é entry ± 1×ATR, logo o risco ≈ ATR original.
            # Boa aproximação para reconstruir o trailing sem o ATR guardado.
            risk    = abs(entry - orig_sl)
            atr_est = risk if risk > 0 else entry * 0.01

            # Se o TP tiver sido atingido/cancelado, reconstrói com RR padrão da v1.3 (2.5)
            if tp is None:
                tp = entry - 2.5 * risk if side == "short" else entry + 2.5 * risk

            margin = notional / leverage if leverage else notional

            # Gestão herdada da estratégia que ABRIU o trade (o DB guarda o nome).
            # Sem isto a posição readotada caía nos defaults genéricos do
            # ManagedPosition (BE 50% + 1.5xATR) e passava a ser gerida DIFERENTE
            # do que a estratégia define — a lateral_breakout, que não tem trailing
            # (TRAIL_ATR=0), começava a arrastar o stop depois de um restart e
            # divergia do backtest.
            be_trig, trail_mult = _strategy_mgmt_params(
                db_trade.get("strategy") if db_trade else None
            )

            pos = position_manager.ManagedPosition(
                symbol=symbol, side=side, entry=entry,
                stop_loss=orig_sl, take_profit=tp, atr=atr_est,
                leverage=leverage, margin_usdt=round(margin, 2),
                position_size_usdt=round(notional, 2),
                trade_id=int(db_trade.get("id", 0) or 0) if db_trade else 0,
                has_live_stop=has_live_stop,
                breakeven_trigger=be_trig,
                trailing_atr_mult=trail_mult,
            )
            # current_stop é field(init=False) e nasce = stop_loss; repõe o nível VIVO
            # que está mesmo na exchange (pode já estar bem à frente do original).
            pos.current_stop = sl
            position_manager.register_position(pos)
            adopted += 1
            logger.info(
                f"[RECONCILE] {symbol} {side.upper()} readotada "
                f"({'gerida' if has_live_stop else 'só monitorada'}) — "
                f"entry={entry:.4f} SL={sl:.4f} TP={tp:.4f} lev={leverage}x"
            )
            _broadcast({"type": "log", "level": "info",
                        "message": f"♻ {symbol.split('/')[0]}: posição readotada "
                                   f"({'gerida' if has_live_stop else 'só monitorada'}, SL={sl:.4f})"})
        except Exception as e:
            logger.warning(f"[RECONCILE] Erro ao readotar {symbol}: {e}")

    if adopted:
        logger.info(f"[RECONCILE] {adopted} posição(ões) readotada(s) da exchange")

    # Fecha no DB qualquer registo 'open' que já não existe na exchange.
    # Acontece quando posições são fechadas manualmente na Binance (fora do bot).
    try:
        from api.db import get_trades, close_trade
        live_symbols = {p.get("symbol") for p in live}
        now_str = datetime.now(timezone.utc).isoformat()
        orphaned = 0
        for t in get_trades(limit=100, status="open"):
            sym = t.get("symbol")
            if sym and sym not in live_symbols:
                # Este trade fechou enquanto o bot não estava a ver (restart, ou
                # fecho na Binance fora do bot).
                #
                # ANTES (bug, corrigido 06/08/2026): escrevia exit_price=entry e
                # pnl=0.0 — um ZERO FABRICADO. O trade teve P&L real, e o histórico
                # ficava a mentir. 10 dos 59 trades do histórico têm este zero falso
                # e por isso não são analisáveis.
                #
                # AGORA: tenta recuperar o preço REAL de saída no histórico de
                # execuções da exchange. Só cai no entry (P&L 0) se não conseguir —
                # e nesse caso o exit_reason diz que o número não é de confiar.
                entry_price = float(t.get("entry_price") or t.get("entry") or 0)
                exit_price = 0.0
                try:
                    lt = exchange.get_last_trade(sym)
                    exit_price = float(lt["price"]) if lt else 0.0
                except Exception as e:
                    logger.warning(f"[RECONCILE] {sym}: falha ao ler última execução: {e}")

                if exit_price > 0:
                    side_str = (t.get("side") or "").lower()
                    side_str = "long" if side_str in ("long", "buy") else "short"
                    net_r, pnl_usdt = position_manager.calc_pnl(
                        side_str, entry_price,
                        float(t.get("stop_loss") or 0),
                        float(t.get("position_size_usdt") or 0),
                        exit_price,
                    )
                    reason = "reconciliacao"
                else:
                    exit_price, net_r, pnl_usdt = entry_price, 0.0, 0.0
                    reason = "reconciliacao_sem_preco"   # P&L = 0 é fabricado

                close_trade(
                    trade_id=int(t["id"]),
                    exit_price=exit_price,
                    pnl_usdt=pnl_usdt,
                    pnl_pct=net_r,
                    closed_at=now_str,
                    exit_reason=reason,
                )
                orphaned += 1
                logger.info(
                    f"[RECONCILE] Trade #{t['id']} ({sym}) fechado no DB @ {exit_price:.6f} "
                    f"({pnl_usdt:+.2f} USDT, motivo={reason}) — posição já não existe na exchange"
                )
        if orphaned:
            logger.info(f"[RECONCILE] {orphaned} trade(s) órfão(s) marcado(s) como fechado(s) no DB")
            _broadcast({"type": "log", "level": "info",
                        "message": f"♻ {orphaned} trade(s) fechado(s) manualmente sincronizado(s) no histórico"})
    except Exception as e:
        logger.warning(f"[RECONCILE] Erro ao limpar trades órfãos: {e}")

    return adopted


# ---------------------------------------------------------------------------
# Controle do scheduler
# ---------------------------------------------------------------------------

def _init_exchange() -> None:
    """Inicializa os clientes ccxt e define o modo conforme TRADING_MODE."""
    exchange.init_clients()
    trading_mode = os.getenv("TRADING_MODE", "paper")
    exchange.set_mode("live" if trading_mode == "live" else "testnet")


def _strategy_mgmt_params(name: str | None) -> tuple[float, float]:
    """Params de gestão (breakeven, trailing) da estratégia, pelo nome guardado no DB.

    Devolve (breakeven_trigger, trailing_atr_mult). Espelha o que o _execute_trade
    lê do `analyze()` (be_trigger_pct / trail_atr), mas a partir das constantes do
    módulo — porque ao readotar não temos o resultado do analyze.

    Estratégia desconhecida/ausente → defaults conservadores do manager.
    """
    mod = strategies.get_strategy(name) if name else None
    if mod is None:
        logger.warning(f"[RECONCILE] Estratégia '{name}' desconhecida — gestão nos defaults.")
        return 0.50, 1.5
    return (float(getattr(mod, "BE_TRIGGER_PCT", 0.50)),
            float(getattr(mod, "TRAIL_ATR", 1.5)))


def reconcile_on_boot() -> int:
    """
    Readota as posições abertas na Binance para a memória LOGO no arranque do
    servidor, SEM ligar o scheduler (não escaneia nem abre trades novos).

    Motivo: o `start.bat` só sobe o uvicorn — o bot fica parado até o utilizador
    clicar "Start". Sem isto, após um restart o dashboard mostra zero posições
    (a memória foi zerada) e não há como fechá-las pelo painel, obrigando a ir à
    Binance na mão. Ao reconciliar no boot, os cards reaparecem, o botão fechar
    funciona com P&L correto e as ordens SL/TP na exchange seguem protegendo.

    Retorna o número de posições readotadas.
    """
    try:
        _init_exchange()
        adopted = _reconcile_positions()
        # Limpa ordens órfãs no arranque. Posições que fecharam enquanto o bot
        # esteve em baixo nunca passam pelo tick(), deixando a ordem irmã viva e
        # armada. Corre DEPOIS da readoção para que os pares já readotados (que
        # estão em _positions) fiquem protegidos da varredura.
        cleaned = position_manager.sweep_orphan_orders()
        if cleaned:
            logger.info(f"[RECONCILE] Órfãs limpas no arranque: {', '.join(cleaned)}")
        return adopted
    except Exception as e:
        logger.warning(f"[ENGINE] Reconciliação no arranque falhou: {e}")
        return 0


async def start(strategy: str = None) -> None:
    global _running, _scheduler

    if _running:
        return

    if strategy:
        os.environ["STRATEGY_MODE"] = strategy

    _init_exchange()

    # Reconcilia posições reais da exchange com o estado em memória.
    # Crítico após restart: readota posições já abertas para gestão e evita
    # tentar reabri-las (que causaria erro -4067 ou pior, duplicação de risco).
    # Idempotente: pula pares já readotados no boot (reconcile_on_boot).
    try:
        _reconcile_positions()
    except Exception as e:
        logger.warning(f"[ENGINE] Reconciliação falhou: {e}")

    _scheduler = AsyncIOScheduler()

    active_strategy = os.getenv("STRATEGY_MODE", "bear_v13")

    # Todas as estrategias usam candles 1H — dispara 1 minuto apos o fecho
    _scheduler.add_job(scan_signals, "cron", minute=1, id="bear_scan")

    # Gestão de posições: trailing stop, breakeven, parcial (a cada 30s)
    _scheduler.add_job(
        _tick_positions,
        "interval",
        seconds=30,
        id="position_manager",
        max_instances=1,
    )

    # Broadcast de preços para o dashboard (a cada 5s — só se houver posições abertas)
    _scheduler.add_job(
        _broadcast_prices,
        "interval",
        seconds=5,
        id="price_broadcast",
        max_instances=1,
    )

    _scheduler.start()
    _running = True
    logger.info(f"[ENGINE] Bot iniciado — estratégia={active_strategy} "
                f"modo={exchange.get_mode()}")

    _broadcast({"type": "bot_started", "strategy": active_strategy,
                "mode": exchange.get_mode()})


async def stop() -> None:
    global _running, _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)

    # Limpa o cache de deduplicação para que ao reiniciar o bot ele
    # processe imediatamente sem esperar pelo próximo candle novo
    _last_candle_ts.clear()

    _running = False
    logger.info("[ENGINE] Bot parado")
    _broadcast({"type": "bot_stopped"})


def _persist_close_event(event: dict) -> None:
    """Fecha o trade no DB e faz broadcast de stats/saldo. Partilhado pelo tick de
    posições (SL/TP/trailing) e pela invalidação de breakout falho."""
    from api.db import close_trade, get_stats
    try:
        # O motivo vem do evento ('exchange' = SL/TP disparou na Binance,
        # 'invalidation' = breakout falso, 'stop' = deteção por cruzamento no
        # fallback). Sem isto o histórico não distingue um alvo de um acidente.
        close_trade(
            trade_id=event["trade_id"],
            exit_price=event["exit_price"],
            pnl_usdt=event["pnl_usdt"],
            pnl_pct=event["pnl_r"],
            closed_at=datetime.now(timezone.utc).isoformat(),
            exit_reason=event.get("reason"),
        )
        stats = get_stats()
        try:
            balance = exchange.get_balance()
        except Exception:
            base = float(os.getenv("PORTFOLIO_VALUE_USD", "1000"))
            new_bal = round(base + (stats.get("total_pnl") or 0), 2)
            balance = {"total": new_bal, "free": new_bal, "used": 0}
        _broadcast({"type": "stats_update", "stats": stats, "balance": balance})
    except Exception as e:
        logger.warning(f"[ENGINE] Erro ao fechar trade no DB: {e}")


async def _tick_positions() -> None:
    """Gestão de posições: trailing stop, breakeven, fechamento parcial (30s)."""
    events = position_manager.tick()
    for event in events:
        logger.info(event["message"])
        if event.get("event") == "position_closed":
            _persist_close_event(event)
        _broadcast({"type": "position_update", **event})


def _build_positions_snapshot() -> list[dict]:
    """Constrói snapshot das posições com preço atual. Retorna lista vazia se não há posições."""
    positions = position_manager.get_positions()
    if not positions:
        return []
    snapshot = []
    for sym, pos in positions.items():
        try:
            price = exchange.get_ticker(sym)["last"]
        except Exception:
            price = pos.entry
        risk_usdt = round(
            pos.position_size_usdt * abs(pos.entry - pos.stop_loss) / pos.entry, 2
        ) if pos.entry else 0
        snapshot.append({
            "symbol": sym,
            "side": pos.side,
            "entry": pos.entry,
            "stop_loss": pos.current_stop,
            "take_profit": pos.take_profit,
            "leverage": pos.leverage,
            "margin_usdt": pos.margin_usdt,
            "position_size_usdt": pos.position_size_usdt,
            "risk_usdt": risk_usdt,
            "breakeven_hit": pos.breakeven_hit,
            "partial_closed": pos.partial_closed,
            "has_live_stop": pos.has_live_stop,
            "progress_pct": round(pos.progress(price) * 100, 1),
            "pnl_pct": pos.unrealized_pnl_pct(price),
        })
    return snapshot


async def _broadcast_prices() -> None:
    """Atualiza P&L no dashboard a cada segundo (só preços, sem lógica de gestão)."""
    snapshot = _build_positions_snapshot()
    if snapshot:
        _broadcast({"type": "positions_tick", "positions": snapshot})
