"""
Estrategia: Bear Market v1.3
Base: Bear Market v1.2 com ajustes de timing e gestao de posicao.

Evolucao face a v1.2 (validada em backtest Jan-Jun 2026):
  LOW_LOOKBACK   : 4  -> 3 velas  (entra 1 barra mais cedo em quedas rapidas)
  COOLDOWN       : 8H -> 3H  (permite reentrada em continuacoes — maior frequencia)
  RR_CAP         : 4.0R -> 2.5R  (alvo mais atingivel, drawdown mais controlado)
  BE_TRIGGER_PCT : 50% -> 70% do caminho (da espaco aos repiques normais de queda)

Resultado backtest vs v1.2 (mesmo periodo):
  v1.2: 680 trades, +388.5%
  v1.3: 1014 trades, +502.3%  (+113pp, +334 trades extra todos lucrativos)

Entrada identica a v1.2 (exceto lookback):
  Regime BEAR (Daily EMA20 declinante) +
  Close < minimo das ultimas 3 velas 1H +
  Candle bearish (close < open) +
  ATR > media 48H (volatilidade elevada)
  -> SHORT
"""

import logging

import pandas as pd

from bot.risk import calc_rr
from bot import regime as regime_mod

logger = logging.getLogger(__name__)

# ── Parametros de entrada ─────────────────────────────────────────────────────
# Regime agora vem de bot/regime.py (fonte unica). Estas constantes ficam aqui
# por compatibilidade com scripts de backtest que as importam.
REGIME_EMA_PERIOD    = regime_mod.EMA_PERIOD
REGIME_SLOPE_BARS    = regime_mod.SLOPE_BARS
REGIME_SLOPE_THRESH  = regime_mod.SLOPE_THRESH
LOW_LOOKBACK         = 3      # reduzido de 4 → entra mais cedo em quedas rapidas
ATR_PERIOD           = 14
ATR_AVG_PERIOD       = 48
ATR_STOP_MULT        = 1.0
COOLDOWN_BARS        = 3      # reduzido de 8 → permite reentrada em continuacoes

# ── Parametros de saida ───────────────────────────────────────────────────────
BE_TRIGGER_PCT = 0.70   # move stop para breakeven a 70% do caminho para o TP (era 50%)
TRAIL_ATR      = 2.0    # trailing = 2.0xATR apos breakeven (igual a v1.2)
RR_CAP         = 2.5    # TP maximo em 2.5R — mais atingivel em movimentos de 4-5 barras


def _calc_atr(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()


def analyze(df_entry: pd.DataFrame, df_daily: pd.DataFrame,
            last_trade_bar: int = -999) -> dict:
    result_none = {
        "signal": "NONE", "conditions": [], "entry": None,
        "stop_loss": None, "take_profit": None, "rr": 0.0, "atr": 0.0,
        "be_trigger_pct": BE_TRIGGER_PCT, "trail_atr": TRAIL_ATR,
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

    regime = regime_mod.detect_regime(df_daily)

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
            "name":   f"R:R >= {RR_CAP}:1  |  BE={int(BE_TRIGGER_PCT*100)}%  TRAIL={TRAIL_ATR}xATR",
            "passed": rr >= RR_CAP,
            "detail": f"R:R={rr:.2f}  SL={sl:.4f}  TP={tp:.4f}" if tp else "TP nao calculado",
        },
        {
            "name":   f"Cooldown >= {COOLDOWN_BARS} barras (3H)",
            "passed": cooldown_ok,
            "detail": f"{bars_since_last} barras desde ultimo trade",
        },
    ]

    entry = float(last["close"])
    result_base = {
        "signal": "NONE", "conditions": conditions,
        "entry": entry, "stop_loss": sl, "take_profit": tp,
        "rr": rr, "atr": atr,
        "be_trigger_pct": BE_TRIGGER_PCT,
        "trail_atr": TRAIL_ATR,
    }

    if all(c["passed"] for c in conditions):
        logger.info(
            f"[BEAR_V13] SHORT → entry={entry:.4f} SL={sl:.4f} TP={tp:.4f} "
            f"R:R={rr:.2f} BE={int(BE_TRIGGER_PCT*100)}% TRAIL={TRAIL_ATR}x"
        )
        result_base["signal"] = "SHORT"

    return result_base
