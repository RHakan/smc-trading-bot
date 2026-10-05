"""
Estrategia: Bull SMC (BULL) — complemento do sistema v2.
Sweep de liquidez + CHoCH (Change of Character) com filtros de forca do bull.
Validada nos bulls historicos (2020-21: +51.7%, 2023-25: +36.5% com filtros).

ATENCAO: complemento, NAO o nucleo. Em 5 anos isolada foi +930 mas so 2/5 anos
positivos (depende de bull forte). O nucleo robusto e a lateral_breakout.

Logica (regime BULL):
  Forca do bull: slope da EMA20 diaria >= SLOPE_MIN (bull inclinado, nao so "acima").
  Tendencia 1H intacta: close > EMA100.
  Volatilidade: ATR > media.
  SWEEP+CHoCH: nas ultimas CHOCH_BARS barras houve um sweep (low varreu um swing
    low e fechou acima) cujo ref_high e rompido AGORA pelo close (1a vez) -> entrada.
  Stop = minima do sweep. Alvo = RR_CAP x risco. Gestao: BE 70% + trailing 2xATR.

TODO (melhoria validada, fica p/ depois): filtro BTC — so comprar alt quando o
BTC tambem esta em bull. Requer o regime do BTC (contexto global); o analyze por
par nao o tem. Sem ele a bull e menos robusta. Ver backtests/backtest_bull_dev*.py.
"""
import logging

import numpy as np
import pandas as pd

from bot.risk import calc_rr
from bot import regime as regime_mod

logger = logging.getLogger(__name__)

# ── Parametros (config robusta dos backtests) ─────────────────────────────────
SWING_N     = 10     # janela do swing low (liquidez)
CHOCH_BARS  = 12     # max barras apos sweep para confirmar o CHoCH
CHOCH_REF   = 15     # janela da maxima de referencia
MIN_SWEEP   = 0.05   # % minimo de pierce abaixo do swing low
SLOPE_MIN   = 0.006  # inclinacao minima da EMA20 diaria (forca do bull)
EMA_TREND   = 100    # tendencia 1H: so compra acima desta EMA
ATR_PERIOD  = 14
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


def _daily_slope(df_daily: pd.DataFrame) -> float:
    """Inclinacao da EMA20 diaria (mesma definicao do Decisor), usando a penultima
    vela diaria (a ultima pode estar a formar-se)."""
    if df_daily is None or len(df_daily) < regime_mod.EMA_PERIOD + regime_mod.SLOPE_BARS + 2:
        return 0.0
    ema = df_daily["close"].ewm(span=regime_mod.EMA_PERIOD, adjust=False).mean()
    return float((ema.iloc[-2] - ema.iloc[-2 - regime_mod.SLOPE_BARS]) /
                 ema.iloc[-2 - regime_mod.SLOPE_BARS])


def analyze(df_entry: pd.DataFrame, df_daily: pd.DataFrame,
            last_trade_bar: int = -999, btc_regime: str = "BULL") -> dict:
    """btc_regime: regime do BTC (contexto global, injetado pelo engine). O long
    só dispara se o BTC também está em BULL — evita comprar alt descolado do líder
    ('pump sem BTC', que reverte). Default 'BULL' = neutro p/ chamadas sem o filtro
    (backtests isolados). Validado: backtests/backtest_bull_filtro_btc.py."""
    result_none = {
        "signal": "NONE", "conditions": [], "entry": None,
        "stop_loss": None, "take_profit": None, "rr": 0.0, "atr": 0.0,
        "be_trigger_pct": BE_TRIGGER_PCT, "trail_atr": TRAIL_ATR,
    }

    need = max(SWING_N + CHOCH_REF + CHOCH_BARS, EMA_TREND, ATR_AVG_PERIOD) + 5
    if len(df_entry) < need:
        return result_none

    df = df_entry.copy()
    df["atr"]     = _calc_atr(df)
    df["atr_avg"] = df["atr"].rolling(ATR_AVG_PERIOD).mean()
    df["ema_t"]   = df["close"].ewm(span=EMA_TREND, adjust=False).mean()

    h = df["high"].values; l = df["low"].values; c = df["close"].values
    atr_arr = df["atr"].values; atr_avg = df["atr_avg"].values; ema_t = df["ema_t"].values
    i = len(df) - 1                       # candle atual (ultimo fechado)

    close = float(c[i]); atr = float(atr_arr[i]); av = float(atr_avg[i])
    if any(pd.isna(v) for v in [close, atr, av, ema_t[i]]) or atr <= 0:
        return result_none

    regime = regime_mod.detect_regime(df_daily)
    slope  = _daily_slope(df_daily)

    cond_regime = regime == "BULL"
    cond_btc    = btc_regime == "BULL"          # filtro BTC: alt só sobe com o líder
    cond_slope  = slope >= SLOPE_MIN
    cond_trend  = close > float(ema_t[i])
    cond_vol    = atr > av

    # SWEEP + CHoCH — procura nas ultimas CHOCH_BARS barras
    sweep_choch = False
    entry = close; sl = None; tp = None
    if cond_regime and cond_btc and cond_slope and cond_trend and cond_vol:
        for j in range(i - 1, max(i - CHOCH_BARS - 1, SWING_N + CHOCH_REF) - 1, -1):
            swing_low = np.min(l[j - SWING_N:j])
            if not (l[j] < swing_low and c[j] > swing_low):
                continue
            if MIN_SWEEP > 0 and (swing_low - l[j]) / swing_low * 100 < MIN_SWEEP:
                continue
            ref_high = np.max(h[j - CHOCH_REF:j])
            if np.any(c[j + 1:i] > ref_high):   # ref_high ja rompido antes — nao e CHoCH novo
                continue
            if c[i] > ref_high:
                actual_low = float(np.min(l[j:i + 1]))
                risk = close - actual_low
                if risk > 0 and risk / close <= 0.10:
                    sl = actual_low
                    tp = close + RR_CAP * risk
                    sweep_choch = True
                    break

    rr = calc_rr(entry, sl, tp) if (sl is not None and tp is not None) else 0.0
    signal = "LONG" if sweep_choch else "NONE"

    conditions = [
        {"name": "Regime BULL (Daily EMA20 ascendente)", "passed": cond_regime,
         "detail": f"Regime atual: {regime}"},
        {"name": "Filtro BTC (lider em BULL)", "passed": cond_btc,
         "detail": f"BTC: {btc_regime}"},
        {"name": f"Forca do bull (slope >= {SLOPE_MIN})", "passed": cond_slope,
         "detail": f"slope={slope:.4f}"},
        {"name": f"Tendencia 1H intacta (close > EMA{EMA_TREND})", "passed": cond_trend,
         "detail": f"close={close:.4f} EMA={float(ema_t[i]):.4f}"},
        {"name": "Volatilidade elevada (ATR > media)", "passed": cond_vol,
         "detail": f"ATR={atr:.4f} media={av:.4f}"},
        {"name": "Sweep de liquidez + CHoCH confirmado", "passed": sweep_choch,
         "detail": f"SL={sl:.4f} TP={tp:.4f}" if sweep_choch else "sem setup"},
    ]

    result = {
        "signal": signal, "conditions": conditions,
        "entry": entry, "stop_loss": sl, "take_profit": tp,
        "rr": rr, "atr": atr,
        "be_trigger_pct": BE_TRIGGER_PCT, "trail_atr": TRAIL_ATR,
    }

    if signal != "NONE":
        logger.info(
            f"[BULL_SMC] LONG → entry={entry:.4f} SL={sl:.4f} TP={tp:.4f} R:R={rr:.2f}"
        )
    return result
