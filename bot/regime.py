"""
Detector de regime de mercado — o "roteador" do sistema multi-estrategia.

Decide o estado macro do mercado para cada par com base na EMA20 diaria e na
sua inclinacao (slope). Este modulo e a FONTE UNICA da verdade sobre regime:
tanto as estrategias (bear/bull) como o engine (modo "auto") importam daqui,
para nunca haver duas definicoes de regime divergentes.

Regimes:
  BEAR    → preco abaixo da EMA20 diaria E EMA a cair  → usar estrategia de baixa
  BULL    → preco acima  da EMA20 diaria E EMA a subir → usar estrategia de alta
  NEUTRAL → tudo o resto (lateral / transicao)         → ficar de fora

A zona NEUTRAL e proposital e larga: serve de "buffer" contra whipsaw na
transicao entre tendencias. So operamos quando o regime e claro.
"""

import pandas as pd

# Parametros do detector (iguais aos que a bear_v13 ja usava — comportamento preservado)
EMA_PERIOD   = 20
SLOPE_BARS   = 5
SLOPE_THRESH = 0.001   # 0.1% de inclinacao minima da EMA para confirmar tendencia


def detect_regime(df_daily: pd.DataFrame,
                  ema_period: int = EMA_PERIOD,
                  slope_bars: int = SLOPE_BARS,
                  slope_thresh: float = SLOPE_THRESH) -> str:
    """
    Classifica o regime de mercado a partir dos candles diarios.

    Usa a penultima vela (iloc[-2]) — a ultima do DataFrame pode ainda estar
    a formar-se. Isto evita decidir com base num candle incompleto (lookahead).

    Retorna "BEAR" | "BULL" | "NEUTRAL".
    """
    if df_daily is None or len(df_daily) < ema_period + slope_bars + 2:
        return "NEUTRAL"

    ema   = df_daily["close"].ewm(span=ema_period, adjust=False).mean()
    slope = (ema.iloc[-2] - ema.iloc[-2 - slope_bars]) / ema.iloc[-2 - slope_bars]
    price = df_daily["close"].iloc[-2]
    ema_y = ema.iloc[-2]

    if price < ema_y and slope < -slope_thresh:
        return "BEAR"
    if price > ema_y and slope > slope_thresh:
        return "BULL"
    return "NEUTRAL"


def regime_detail(df_daily: pd.DataFrame,
                  ema_period: int = EMA_PERIOD,
                  slope_bars: int = SLOPE_BARS,
                  slope_thresh: float = SLOPE_THRESH) -> dict:
    """
    Igual a detect_regime, mas devolve os valores intermedios — util para o
    dashboard e para debug ("porque e que esta em NEUTRAL?").

    Retorna {regime, price, ema, slope, slope_thresh}.
    """
    if df_daily is None or len(df_daily) < ema_period + slope_bars + 2:
        return {"regime": "NEUTRAL", "price": None, "ema": None,
                "slope": None, "slope_thresh": slope_thresh}

    ema   = df_daily["close"].ewm(span=ema_period, adjust=False).mean()
    slope = float((ema.iloc[-2] - ema.iloc[-2 - slope_bars]) / ema.iloc[-2 - slope_bars])
    price = float(df_daily["close"].iloc[-2])
    ema_y = float(ema.iloc[-2])

    if price < ema_y and slope < -slope_thresh:
        regime = "BEAR"
    elif price > ema_y and slope > slope_thresh:
        regime = "BULL"
    else:
        regime = "NEUTRAL"

    return {"regime": regime, "price": price, "ema": ema_y,
            "slope": slope, "slope_thresh": slope_thresh}


# ── Mapa regime → estrategia (o "Decisor") ────────────────────────────────────
# Define qual estrategia o modo "auto" do engine usa em cada estado de mercado.
# None = slot reservado: o bot fica de fora (sem trade) nesse regime.
#
# Ligar uma estrategia a um regime e UM passo: troca o None pelo nome registado
# em bot/strategies/__init__.py. O engine passa a usa-la automaticamente.
#
# Estado atual dos slots:
#   BEAR    → bear_v13  (APROVADA em backtest, em teste demo)
#   BULL    → None      (reservado — bull_v1 existe mas foi REPROVADA no backtest
#                        de 2024: -90%. Aguarda Bull v2.0 baseada em pullback.)
#   NEUTRAL → None      (reservado — ainda nao existe estrategia para mercado lateral)
REGIME_STRATEGY = {
    "BEAR":    "bear_v13",
    "BULL":    None,
    "NEUTRAL": None,
}

# Mapa do SISTEMA v2 ("auto_v2") — reconstruido em torno do breakout (jun/2026).
# Validado em 5 anos (2021-2025). O bear_v13 NAO tem edge out-of-sample (perde
# ate em 2022, bear market real) — por isso BEAR fica DE FORA aqui.
#   BEAR    → bear_breakout    (bear REABILITADO — breakout de suporte seletivo,
#                               robusto em 5 anos: 36/36 configs+, brilha em 2022)
#   BULL    → bull_smc         (complemento — sweep+CHoCH)
#   NEUTRAL → lateral_breakout (NUCLEO robusto — 4/5 anos, DD 6.6%)
# As 3 sao breakout de range consolidado (lei do momentum). Sistema v3: +362 em
# 58 meses (500/mes), 4/5 anos+, DD 11.8%. Ver backtests/backtest_sistema_v3.py.
REGIME_STRATEGY_V2 = {
    "BEAR":    "bear_breakout",
    "BULL":    "bull_smc",
    "NEUTRAL": "lateral_breakout",
}


def strategy_for_regime(regime: str) -> str | None:
    """Devolve o nome da estrategia a usar para um dado regime (None = ficar de fora)."""
    return REGIME_STRATEGY.get(regime)


def strategy_for_regime_v2(regime: str) -> str | None:
    """Versao do sistema v2 (modo 'auto_v2' no engine). None = ficar de fora."""
    return REGIME_STRATEGY_V2.get(regime)
