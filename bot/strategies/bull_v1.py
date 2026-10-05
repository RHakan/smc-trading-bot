"""
Estrategia: Bull Market v1.0
Espelho simetrico da Bear Market v1.3 — mesma logica, direcao invertida.

Enquanto a bear shorta rompimentos de SUPORTE em tendencia de baixa, a bull
compra rompimentos de RESISTENCIA em tendencia de alta:

  Regime BULL (Daily EMA20 a subir) +
  Close > maximo das ultimas 3 velas 1H (rompe resistencia) +
  Candle bullish (close > open) +
  ATR > media 48H (volatilidade elevada)
  -> LONG

Gestao de saida identica (espelhada):
  SL = entry - 1.0xATR   (stop abaixo)
  TP = entry + RR_CAP x risco
  Breakeven a 70% do caminho, trailing 2.0xATR, RR_CAP 2.5R.

NOTA: parametros sao ponto de partida (espelho da bear validada). A estrategia
so deve ser ativada apos backtest em mercado bull e aprovacao. Tal como na bear
(v1.0 → v1.2 → v1.3), espera-se iterar estes valores conforme os dados.
"""

import logging

import pandas as pd

from bot.risk import calc_rr
from bot import regime as regime_mod

logger = logging.getLogger(__name__)

# ── Parametros de entrada (espelho da bear_v13) ───────────────────────────────
HIGH_LOOKBACK   = 3      # rompe o maximo das ultimas 3 velas (espelho de LOW_LOOKBACK)
ATR_PERIOD      = 14
ATR_AVG_PERIOD  = 48
ATR_STOP_MULT   = 1.0
COOLDOWN_BARS   = 3

# ── Parametros de saida (identicos a bear_v13) ────────────────────────────────
BE_TRIGGER_PCT = 0.70
TRAIL_ATR      = 2.0
RR_CAP         = 2.5


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

    if len(df_entry) < ATR_AVG_PERIOD + HIGH_LOOKBACK + 5:
        return result_none

    df = df_entry.copy()
    df["atr"]     = _calc_atr(df)
    df["atr_avg"] = df["atr"].rolling(ATR_AVG_PERIOD).mean()
    # Maximo das ultimas N velas ANTERIORES (shift 1) — a resistencia a romper
    df["high_n"]  = df["high"].shift(1).rolling(HIGH_LOOKBACK).max()

    last    = df.iloc[-1]
    close   = float(last["close"])
    open_   = float(last["open"])
    atr     = float(last["atr"])
    atr_avg = float(last["atr_avg"])
    high_n  = float(last["high_n"])

    if any(pd.isna(v) for v in [close, atr, atr_avg, high_n]):
        return result_none

    regime = regime_mod.detect_regime(df_daily)

    # Espelho do stop/TP da bear: stop ABAIXO, alvo ACIMA
    sl   = round(close - ATR_STOP_MULT * atr, 8)
    risk = close - sl
    tp   = round(close + RR_CAP * risk, 8) if risk > 0 else None
    rr   = calc_rr(close, sl, tp) if tp else 0.0

    bars_since_last = len(df) - 1 - last_trade_bar
    cooldown_ok     = bars_since_last >= COOLDOWN_BARS if last_trade_bar >= 0 else True

    conditions = [
        {
            "name":   "Regime BULL (Daily EMA20 ascendente)",
            "passed": regime == "BULL",
            "detail": f"Regime atual: {regime}",
        },
        {
            "name":   f"Rompimento de resistencia (close > max. {HIGH_LOOKBACK}H anterior)",
            "passed": close > high_n,
            "detail": f"Close {close:.4f} vs Resistencia {high_n:.4f}",
        },
        {
            "name":   "Candle bullish (close > open)",
            "passed": close > open_,
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
            f"[BULL_V1] LONG → entry={entry:.4f} SL={sl:.4f} TP={tp:.4f} "
            f"R:R={rr:.2f} BE={int(BE_TRIGGER_PCT*100)}% TRAIL={TRAIL_ATR}x"
        )
        result_base["signal"] = "LONG"

    return result_base
