"""
Estrategia: Lateral Breakout (NEUTRAL)
A peca mais ROBUSTA do sistema v2 — validada em 5 anos (2021-2025): 4/5 anos
positivos, DD maximo 6.6%, PF 1.08-1.46. Funciona em bull, bear e lateral.

Logica (regime NEUTRAL = mercado sem tendencia clara):
  Detecta um RANGE consolidado (ADX baixo) e SEGUE o rompimento (momentum):
    LONG  : close rompe a resistencia do range (+buffer) e a barra anterior
            estava dentro do range.
    SHORT : close rompe o suporte do range.
  Stop = lado OPOSTO do range. Alvo = altura do range projetada (RR ~1, WR alto).
  Sem trailing — a gestao e mecanica (TP fixo na projecao).

Porque funciona onde a reversao falhou: cripto paga MOMENTUM. O range acumula
energia; o rompimento e o movimento que paga. Entrar so apos consolidacao (ADX<20)
filtra o ruido. Backtest: backtests/backtest_breakout_base.py e _sistema_v2.py.
"""
import logging

import numpy as np
import pandas as pd

from bot.risk import calc_rr
from bot import regime as regime_mod

logger = logging.getLogger(__name__)

# ── Parametros (config robusta validada em 5 anos) ────────────────────────────
LOOKBACK    = 30     # barras do range
ADX_PERIOD  = 14
ADX_MAX     = 20     # so opera apos consolidacao (sem tendencia forte)
BUFFER_ATR  = 0.10   # rompimento tem de superar a borda por 0.1xATR (filtra falsos)
ATR_PERIOD  = 14

# Gestao mecanica: TP fixo na projecao, sem trailing.
BE_TRIGGER_PCT = 0.99   # praticamente nao move para BE — deixa o TP/SL fixos resolverem
TRAIL_ATR      = 0.0    # sem trailing

# Invalidacao de breakout falho: se o preco FECHAR de volta pra dentro do range
# dentro de INVAL_BARS velas 1H, sai (o rompimento falhou) — corta o -1R para ~-0.3R.
# Validado 5 anos + OOS 2026 (backtests/backtest_inval_confirma.py): +347 vs +318,
# OOS +60 vs +50, 5/5 anos+, DD 12.1->10.1%. Plato 3-6 velas robusto. So a lateral.
INVAL_BARS = 5


def _calc_atr(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()


def _calc_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def analyze(df_entry: pd.DataFrame, df_daily: pd.DataFrame,
            last_trade_bar: int = -999) -> dict:
    result_none = {
        "signal": "NONE", "conditions": [], "entry": None,
        "stop_loss": None, "take_profit": None, "rr": 0.0, "atr": 0.0,
        "be_trigger_pct": BE_TRIGGER_PCT, "trail_atr": TRAIL_ATR,
    }

    if len(df_entry) < LOOKBACK + ADX_PERIOD + 5:
        return result_none

    df = df_entry.copy()
    df["atr"] = _calc_atr(df)
    df["adx"] = _calc_adx(df)

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    close = float(last["close"])
    atr   = float(last["atr"])
    adx_prev = float(prev["adx"])     # ADX da barra anterior (consolidacao previa)

    if any(pd.isna(v) for v in [close, atr, adx_prev]) or atr <= 0:
        return result_none

    # Range das LOOKBACK barras ANTERIORES ao candle atual
    window = df.iloc[-(LOOKBACK + 1):-1]
    range_low  = float(window["low"].min())
    range_high = float(window["high"].max())
    height = range_high - range_low
    pc = float(prev["close"])

    regime = regime_mod.detect_regime(df_daily)
    buf = BUFFER_ATR * atr

    is_consolidated = adx_prev <= ADX_MAX and range_high > range_low > 0
    breakout_up   = close > range_high + buf and pc <= range_high
    breakout_down = close < range_low  - buf and pc >= range_low

    signal, entry, sl, tp = "NONE", close, None, None
    # Nível de invalidação = borda do range que foi rompida (a que o preço tem de
    # perder de volta para o rompimento ser considerado falho).
    invalidation_level = None
    if regime == "NEUTRAL" and is_consolidated:
        if breakout_up:
            signal, entry, sl, tp = "LONG", close, range_low, close + height
            invalidation_level = range_high
        elif breakout_down:
            signal, entry, sl, tp = "SHORT", close, range_high, close - height
            invalidation_level = range_low

    rr = calc_rr(entry, sl, tp) if (sl is not None and tp is not None) else 0.0

    conditions = [
        {"name": "Regime NEUTRAL (lateral/sem tendencia)",
         "passed": regime == "NEUTRAL", "detail": f"Regime atual: {regime}"},
        {"name": f"Consolidacao previa (ADX < {ADX_MAX})",
         "passed": adx_prev <= ADX_MAX, "detail": f"ADX={adx_prev:.1f}"},
        {"name": "Rompimento de range (cima ou baixo, +buffer)",
         "passed": signal != "NONE",
         "detail": f"close={close:.4f} range=[{range_low:.4f}, {range_high:.4f}]"},
    ]

    result = {
        "signal": signal, "conditions": conditions,
        "entry": entry, "stop_loss": sl, "take_profit": tp,
        "rr": rr, "atr": atr,
        "be_trigger_pct": BE_TRIGGER_PCT, "trail_atr": TRAIL_ATR,
        "invalidation_level": invalidation_level, "invalidation_bars": INVAL_BARS,
    }

    if signal != "NONE":
        logger.info(
            f"[LATERAL] {signal} → entry={entry:.4f} SL={sl:.4f} TP={tp:.4f} "
            f"R:R={rr:.2f} (range height={height:.4f})"
        )
    return result
