"""
Estratégia: Breakout SHORT em tendência de baixa
Sinal confirmado no backtest: $100 → $256 (+156%) em 174 dias (Jan-Jun 2026)

Lógica:
  Regime BEAR (Daily EMA20 declinante) +
  Preço fecha abaixo do mínimo das últimas 4 velas 1H (rompimento de suporte) +
  Candle bearish (close < open) +
  ATR acima da média 48H (volatilidade elevada = movimento válido)
  → SHORT

Stop:  close + 1.0 × ATR
TP:    close - 3.0 × risco  (RR 3:1 máximo, trailing stop após breakeven)
"""

import logging

import pandas as pd
import numpy as np

from bot.risk import calc_rr

logger = logging.getLogger(__name__)

# ── Parâmetros da estratégia (validados no backtest) ────────────────────────
REGIME_EMA_PERIOD  = 20      # EMA diária para regime
REGIME_SLOPE_BARS  = 5       # barras para calcular slope da EMA
REGIME_SLOPE_THRESH = 0.001  # slope mínimo de 0.1% para confirmar tendência
LOW_LOOKBACK       = 4       # mínimo das últimas N velas 1H (shift=1)
ATR_PERIOD         = 14      # período do ATR
ATR_AVG_PERIOD     = 48      # janela da média do ATR (filtro de volatilidade)
ATR_STOP_MULT      = 1.0     # stop = entry + N × ATR
RR_CAP             = 3.0     # TP máximo em 3R
COOLDOWN_BARS      = 8       # cooldown mínimo entre trades (barras 1H = 8H)


# ── Helpers de indicadores ──────────────────────────────────────────────────

def _calc_atr(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()


def _calc_regime(df_daily: pd.DataFrame) -> str:
    """Retorna 'BEAR', 'BULL' ou 'NEUTRAL' com base na EMA20 diária."""
    if len(df_daily) < REGIME_EMA_PERIOD + REGIME_SLOPE_BARS + 2:
        return "NEUTRAL"

    ema   = df_daily["close"].ewm(span=REGIME_EMA_PERIOD, adjust=False).mean()
    slope = (ema.iloc[-2] - ema.iloc[-2 - REGIME_SLOPE_BARS]) / ema.iloc[-2 - REGIME_SLOPE_BARS]
    price = df_daily["close"].iloc[-2]   # shift(1): ontem define hoje
    ema_y = ema.iloc[-2]

    if price < ema_y and slope < -REGIME_SLOPE_THRESH:
        return "BEAR"
    if price > ema_y and slope > REGIME_SLOPE_THRESH:
        return "BULL"
    return "NEUTRAL"


# ── Função principal ─────────────────────────────────────────────────────────

def analyze(df_entry: pd.DataFrame, df_daily: pd.DataFrame,
            last_trade_bar: int = -999) -> dict:
    """
    Analisa o sinal de Breakout SHORT.

    df_entry  : OHLCV no timeframe 1H (pelo menos 60 candles fechados)
    df_daily  : OHLCV diário (pelo menos 30 dias)
    last_trade_bar: índice do último trade para controle de cooldown

    Retorna o contrato padrão do engine:
    {
        "signal": "SHORT" | "NONE",
        "conditions": [...],
        "entry": float,
        "stop_loss": float | None,
        "take_profit": float | None,
        "rr": float,
        "atr": float,
    }
    """
    result_none = {
        "signal":      "NONE",
        "conditions":  [],
        "entry":       None,
        "stop_loss":   None,
        "take_profit": None,
        "rr":          0.0,
        "atr":         0.0,
    }

    if len(df_entry) < ATR_AVG_PERIOD + LOW_LOOKBACK + 5:
        return result_none

    # Indicadores na série 1H
    df = df_entry.copy()
    df["atr"]     = _calc_atr(df)
    df["atr_avg"] = df["atr"].rolling(ATR_AVG_PERIOD).mean()
    df["low_n"]   = df["low"].shift(1).rolling(LOW_LOOKBACK).min()   # mínimo das N velas anteriores

    last = df.iloc[-1]

    close   = float(last["close"])
    open_   = float(last["open"])
    atr     = float(last["atr"])
    atr_avg = float(last["atr_avg"])
    low_n   = float(last["low_n"])

    if any(pd.isna(v) for v in [close, atr, atr_avg, low_n]):
        return result_none

    # Regime diário
    regime = _calc_regime(df_daily)

    # Stop e TP
    sl = round(close + ATR_STOP_MULT * atr, 8)
    risk = sl - close
    tp = round(close - RR_CAP * risk, 8) if risk > 0 else None
    rr = calc_rr(close, sl, tp) if tp else 0.0

    # Cooldown
    bars_since_last = len(df) - 1 - last_trade_bar
    cooldown_ok     = bars_since_last >= COOLDOWN_BARS if last_trade_bar >= 0 else True

    # -- Condicoes (lista visivel no dashboard) --
    conditions = [
        {
            "name":   "Regime BEAR (Daily EMA20 declinante)",
            "passed": regime == "BEAR",
            "detail": f"Regime atual: {regime}",
        },
        {
            "name":   f"Rompimento de suporte (close < mín. {LOW_LOOKBACK}H anterior)",
            "passed": close < low_n,
            "detail": f"Close {close:.4f} vs Suporte {low_n:.4f}",
        },
        {
            "name":   "Candle bearish (close < open)",
            "passed": close < open_,
            "detail": f"open={open_:.4f}  close={close:.4f}",
        },
        {
            "name":   "Volatilidade elevada (ATR > média 48H)",
            "passed": atr > atr_avg,
            "detail": f"ATR={atr:.4f}  Media48H={atr_avg:.4f}",
        },
        {
            "name":   f"R:R >= {RR_CAP}:1",
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
        "signal":        "NONE",
        "conditions":    conditions,
        "entry":         entry,
        "stop_loss":     sl,
        "take_profit":   tp,
        "rr":            rr,
        "atr":           atr,
        "be_trigger_pct": 0.50,
        "trail_atr":     1.5,
    }

    if all(c["passed"] for c in conditions):
        logger.info(
            f"[BREAKOUT_SHORT] Sinal SHORT → entry={entry:.4f} "
            f"SL={sl:.4f} TP={tp:.4f} R:R={rr:.2f} ATR={atr:.4f}"
        )
        result_base["signal"] = "SHORT"

    return result_base
