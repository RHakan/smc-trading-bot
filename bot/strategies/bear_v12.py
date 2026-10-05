"""
Estrategia: Bear Market v1.2
Backtest: $100 -> $488 (+388%) em 174 dias (Jan-Jun 2026)

Evolucao face a v1.0 (breakout_short):
  BE_TRIGGER : 0.8R -> 1.0R  (da mais espaco antes de mover para breakeven)
  TRAIL_ATR  : 1.5x -> 2.0x  (trailing mais largo = deixa correr mais)
  RR_CAP     : 3.0R -> 4.0R  (alvo maximo mais ambicioso)

Entrada identica a v1.0:
  Regime BEAR (Daily EMA20 declinante) +
  Close < minimo das ultimas 4 velas 1H +
  Candle bearish (close < open) +
  ATR acima da media 48H
  -> SHORT
"""

import logging

import pandas as pd
import numpy as np

from bot.risk import calc_rr

logger = logging.getLogger(__name__)

# ── Parametros de entrada (identicos a v1.0) ─────────────────────────────────
REGIME_EMA_PERIOD   = 20
REGIME_SLOPE_BARS   = 5
REGIME_SLOPE_THRESH = 0.001
LOW_LOOKBACK        = 4
ATR_PERIOD          = 14
ATR_AVG_PERIOD      = 48
ATR_STOP_MULT       = 1.0
COOLDOWN_BARS       = 8

# ── Parametros de saida (optimizados em backtest_exit_sweep.py) ───────────────
BE_TRIGGER = 1.0    # move stop para breakeven ao atingir 1.0R ganho (era 0.8R)
TRAIL_ATR  = 2.0    # trailing = 2.0xATR apos breakeven (era 1.5x)
RR_CAP     = 4.0    # TP maximo em 4R (era 3.0R)


def _calc_atr(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()


def _calc_regime(df_daily: pd.DataFrame) -> str:
    if len(df_daily) < REGIME_EMA_PERIOD + REGIME_SLOPE_BARS + 2:
        return "NEUTRAL"
    ema   = df_daily["close"].ewm(span=REGIME_EMA_PERIOD, adjust=False).mean()
    slope = (ema.iloc[-2] - ema.iloc[-2 - REGIME_SLOPE_BARS]) / ema.iloc[-2 - REGIME_SLOPE_BARS]
    price = df_daily["close"].iloc[-2]
    ema_y = ema.iloc[-2]
    if price < ema_y and slope < -REGIME_SLOPE_THRESH:
        return "BEAR"
    if price > ema_y and slope > REGIME_SLOPE_THRESH:
        return "BULL"
    return "NEUTRAL"


def analyze(df_entry: pd.DataFrame, df_daily: pd.DataFrame,
            last_trade_bar: int = -999) -> dict:
    result_none = {
        "signal": "NONE", "conditions": [], "entry": None,
        "stop_loss": None, "take_profit": None, "rr": 0.0, "atr": 0.0,
    }

    if len(df_entry) < ATR_AVG_PERIOD + LOW_LOOKBACK + 5:
        return result_none

    df = df_entry.copy()
    df["atr"]     = _calc_atr(df)
    df["atr_avg"] = df["atr"].rolling(ATR_AVG_PERIOD).mean()
    df["low_n"]   = df["low"].shift(1).rolling(LOW_LOOKBACK).min()

    last    = df.iloc[-1]
    close   = float(last["close"])
    open_   = float(last["open"])
    atr     = float(last["atr"])
    atr_avg = float(last["atr_avg"])
    low_n   = float(last["low_n"])

    if any(pd.isna(v) for v in [close, atr, atr_avg, low_n]):
        return result_none

    regime = _calc_regime(df_daily)

    sl   = round(close + ATR_STOP_MULT * atr, 8)
    risk = sl - close
    tp   = round(close - RR_CAP * risk, 8) if risk > 0 else None
    rr   = calc_rr(close, sl, tp) if tp else 0.0

    bars_since_last = len(df) - 1 - last_trade_bar
    cooldown_ok     = bars_since_last >= COOLDOWN_BARS if last_trade_bar >= 0 else True

    conditions = [
        {
            "name":   "Regime BEAR (Daily EMA20 declinante)",
            "passed": regime == "BEAR",
            "detail": f"Regime atual: {regime}",
        },
        {
            "name":   f"Rompimento de suporte (close < min. {LOW_LOOKBACK}H anterior)",
            "passed": close < low_n,
            "detail": f"Close {close:.4f} vs Suporte {low_n:.4f}",
        },
        {
            "name":   "Candle bearish (close < open)",
            "passed": close < open_,
            "detail": f"open={open_:.4f}  close={close:.4f}",
        },
        {
            "name":   "Volatilidade elevada (ATR > media 48H)",
            "passed": atr > atr_avg,
            "detail": f"ATR={atr:.4f}  Media48H={atr_avg:.4f}",
        },
        {
            "name":   f"R:R >= {RR_CAP}:1  |  BE={BE_TRIGGER}R  TRAIL={TRAIL_ATR}xATR",
            "passed": rr >= RR_CAP,
            "detail": f"R:R={rr:.2f}  SL={sl:.4f}  TP={tp:.4f}" if tp else "TP nao calculado",
        },
        {
            "name":   f"Cooldown >= {COOLDOWN_BARS} barras (8H)",
            "passed": cooldown_ok,
            "detail": f"{bars_since_last} barras desde ultimo trade",
        },
    ]

    entry = float(last["close"])
    result_base = {
        "signal": "NONE", "conditions": conditions,
        "entry": entry, "stop_loss": sl, "take_profit": tp,
        "rr": rr, "atr": atr,
        "be_trigger": BE_TRIGGER, "trail_atr": TRAIL_ATR,
        "be_trigger_pct": 0.50,  # posicao manager: move BE a 50% do caminho para TP
    }

    if all(c["passed"] for c in conditions):
        logger.info(
            f"[BEAR_V12] SHORT -> entry={entry:.4f} SL={sl:.4f} "
            f"TP={tp:.4f} R:R={rr:.2f} BE={BE_TRIGGER}R TRAIL={TRAIL_ATR}x"
        )
        result_base["signal"] = "SHORT"

    return result_base
