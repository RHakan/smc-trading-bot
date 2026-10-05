"""
Estrategia: Bear Breakout (BEAR) — o bear REABILITADO do sistema v2/v3.
Substitui a bear_v13 (que perdia em 5 anos, ate em 2022) pela versao SELETIVA.

A v1.3 entrava em QUALQUER rompimento de minima de 3 velas — ruido, stop-hunt que
reverte. Isto perdia -13068 em 5 anos. A correcao (validada em 5 anos, 36/36 configs
da vizinhanca positivas): so shortar o rompimento de um RANGE CONSOLIDADO.

Logica (regime BEAR):
  Consolidacao previa : ADX < 20 (sem tendencia forte antes do rompimento).
  Rompimento de suporte: close < minimo das ultimas 30 velas (range, nao 3 velas).
  Candle bearish      : close < open.
  Volatilidade        : ATR > media 48H.
  Stop ESTRUTURAL     : acima da estrutura rompida (max das ultimas 20 velas + 0.1ATR)
                        — aguenta o repique sem stopar (vs 1xATR apertado da v13).
  Alvo: RR 2.5. Gestao: BE 70% + trailing 2xATR.

E o mesmo breakout de range da lateral, aplicado ao regime BEAR (lei do momentum).
Resultado isolado 5 anos: ~+625 medio, 3-4/5 anos+, DD ~16% (vs v13 -13068, 0/5, DD69%).
Backtests: backtest_bear_melhorias.py, _robustez.py, _sistema_v3.py.
"""
import logging

import numpy as np
import pandas as pd

from bot.risk import calc_rr
from bot import regime as regime_mod

logger = logging.getLogger(__name__)

# ── Parametros (config central robusta, validada em 5 anos) ───────────────────
SEL_LOOKBACK   = 30     # range do suporte (rompe a minima de 30 velas, nao 3)
ADX_PERIOD     = 14
ADX_MAX        = 20     # consolidacao previa (sem tendencia forte)
STRUCT_LOOKBACK= 20     # estrutura para o stop (topo das ultimas 20 velas)
STRUCT_BUFFER  = 0.10   # folga acima da estrutura (xATR)
ATR_PERIOD     = 14
ATR_AVG_PERIOD = 48

RR_CAP         = 2.5
BE_TRIGGER_PCT = 0.40   # BE a 40% do caminho ao TP = 1.0R — protege o lucro cedo
                        # (trade em 57% do alvo nunca vira prejuízo). Ver backtest_v3_realizar.py
TRAIL_ATR      = 2.0


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

    need = max(SEL_LOOKBACK, STRUCT_LOOKBACK, ATR_AVG_PERIOD, ADX_PERIOD) + 5
    if len(df_entry) < need:
        return result_none

    df = df_entry.copy()
    df["atr"]     = _calc_atr(df)
    df["atr_avg"] = df["atr"].rolling(ATR_AVG_PERIOD).mean()
    df["adx"]     = _calc_adx(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["close"]); open_ = float(last["open"])
    atr   = float(last["atr"]); atr_avg = float(last["atr_avg"])
    adx_prev = float(prev["adx"])     # consolidacao previa

    if any(pd.isna(v) for v in [close, atr, atr_avg, adx_prev]) or atr <= 0:
        return result_none

    # Suporte = minimo das SEL_LOOKBACK velas ANTERIORES (range, nao 3 velas)
    support = float(df["low"].iloc[-(SEL_LOOKBACK + 1):-1].min())
    # Estrutura para o stop = topo das STRUCT_LOOKBACK velas (incl. a atual)
    struct_high = float(df["high"].iloc[-STRUCT_LOOKBACK:].max())

    regime = regime_mod.detect_regime(df_daily)

    cond_regime = regime == "BEAR"
    cond_break  = close < support
    cond_bear   = close < open_
    cond_vol    = atr > atr_avg
    cond_consol = adx_prev <= ADX_MAX     # range consolidado antes do rompimento

    sl = round(struct_high + STRUCT_BUFFER * atr, 8)
    risk = sl - close
    tp = round(close - RR_CAP * risk, 8) if risk > 0 else None
    rr = calc_rr(close, sl, tp) if tp else 0.0

    conditions = [
        {"name": "Regime BEAR (Daily EMA20 declinante)", "passed": cond_regime,
         "detail": f"Regime atual: {regime}"},
        {"name": f"Consolidacao previa (ADX < {ADX_MAX})", "passed": cond_consol,
         "detail": f"ADX={adx_prev:.1f}"},
        {"name": f"Rompimento de suporte (close < min. {SEL_LOOKBACK} velas)", "passed": cond_break,
         "detail": f"close={close:.4f} suporte={support:.4f}"},
        {"name": "Candle bearish (close < open)", "passed": cond_bear,
         "detail": f"open={open_:.4f} close={close:.4f}"},
        {"name": "Volatilidade elevada (ATR > media 48H)", "passed": cond_vol,
         "detail": f"ATR={atr:.4f} media={atr_avg:.4f}"},
        {"name": f"Stop estrutural | R:R >= {RR_CAP}", "passed": (rr >= RR_CAP),
         "detail": f"SL={sl:.4f} TP={tp:.4f} R:R={rr:.2f}" if tp else "TP nao calculado"},
    ]

    result = {
        "signal": "NONE", "conditions": conditions,
        "entry": close, "stop_loss": sl, "take_profit": tp,
        "rr": rr, "atr": atr,
        "be_trigger_pct": BE_TRIGGER_PCT, "trail_atr": TRAIL_ATR,
    }

    if all(c["passed"] for c in conditions):
        result["signal"] = "SHORT"
        logger.info(
            f"[BEAR_BREAKOUT] SHORT → entry={close:.4f} SL={sl:.4f} TP={tp:.4f} "
            f"R:R={rr:.2f} (suporte={support:.4f} ADX_prev={adx_prev:.1f})"
        )
    return result
