"""
Indicadores técnicos — portados do bot.js (Trend Following v3 + Wyckoff)
Todos os cálculos em pandas para compatibilidade com pandas-ta e performance.
"""

import httpx
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_ema_slope(series: pd.Series, lookback: int) -> str:
    """Retorna 'rising', 'falling' ou 'flat' baseado na variação recente."""
    if len(series) < lookback + 1:
        return "flat"
    diff = series.iloc[-1] - series.iloc[-(lookback + 1)]
    threshold = series.iloc[-1] * 0.001  # 0.1% de variação mínima
    if diff > threshold:
        return "rising"
    elif diff < -threshold:
        return "falling"
    return "flat"


# ---------------------------------------------------------------------------
# RSI com fonte OHLC4 (igual ao Pine Script do Rafa)
# ---------------------------------------------------------------------------

def calc_ohlc4(df: pd.DataFrame) -> pd.Series:
    return (df["open"] + df["high"] + df["low"] + df["close"]) / 4


def calc_rsi(source: pd.Series, period: int = 14) -> pd.Series:
    delta = source.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Bollinger Bands %b (igual ao Pine Script do Rafa)
# ---------------------------------------------------------------------------

def calc_bb(source: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    """
    Retorna DataFrame com colunas: upper, middle, lower, pct_b
    pct_b = (preço - lower) / (upper - lower)
    """
    middle = source.rolling(window=period).mean()
    std = source.rolling(window=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    pct_b = (source - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_pct_b": pct_b,
    })


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ---------------------------------------------------------------------------
# Mayer Multiple (preço / SMA200)
# ---------------------------------------------------------------------------

def calc_mayer_multiple(df: pd.DataFrame, period: int = 200) -> pd.Series:
    sma = df["close"].rolling(window=period).mean()
    return df["close"] / sma.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Volume SMA
# ---------------------------------------------------------------------------

def calc_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(window=period).mean()


# ---------------------------------------------------------------------------
# Swing Highs e Lows (para contexto S/R no Wyckoff)
# ---------------------------------------------------------------------------

def find_swing_highs(df: pd.DataFrame, lookback: int = 10) -> list[dict]:
    """Retorna lista de {index, price} para swing highs nos últimos candles."""
    highs = df["high"].values
    result = []
    for i in range(lookback, len(highs) - lookback):
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            result.append({"index": i, "price": highs[i]})
    return result


def find_swing_lows(df: pd.DataFrame, lookback: int = 10) -> list[dict]:
    """Retorna lista de {index, price} para swing lows nos últimos candles."""
    lows = df["low"].values
    result = []
    for i in range(lookback, len(lows) - lookback):
        if lows[i] == min(lows[i - lookback:i + lookback + 1]):
            result.append({"index": i, "price": lows[i]})
    return result


def nearest_swing_high(df: pd.DataFrame, lookback: int = 10) -> float | None:
    """Retorna o swing high mais recente (preço)."""
    swings = find_swing_highs(df, lookback)
    if not swings:
        return None
    return swings[-1]["price"]


def nearest_swing_low(df: pd.DataFrame, lookback: int = 10) -> float | None:
    """Retorna o swing low mais recente (preço)."""
    swings = find_swing_lows(df, lookback)
    if not swings:
        return None
    return swings[-1]["price"]


def nearest_swing_high_above(df: pd.DataFrame, lookback: int = 10,
                              price_floor: float = 0.0) -> float | None:
    """Retorna o swing high confirmado mais recente que esteja ACIMA de price_floor.
    Usado pela Sniper para encontrar o target de um trade LONG.
    """
    swings = find_swing_highs(df, lookback)
    candidates = [s["price"] for s in swings if s["price"] > price_floor]
    return candidates[-1] if candidates else None


def nearest_swing_low_below(df: pd.DataFrame, lookback: int = 10,
                             price_ceiling: float = float("inf")) -> float | None:
    """Retorna o swing low confirmado mais recente que esteja ABAIXO de price_ceiling.
    Usado pela Sniper para encontrar o target de um trade SHORT.
    """
    swings = find_swing_lows(df, lookback)
    candidates = [s["price"] for s in swings if s["price"] < price_ceiling]
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Fear & Greed Index (alternative.me)
# ---------------------------------------------------------------------------

# Tradução das classificações da API (vêm em inglês)
_FNG_LABELS_PT = {
    "Extreme Fear":  "Medo Extremo",
    "Fear":          "Medo",
    "Neutral":       "Neutro",
    "Greed":         "Ganância",
    "Extreme Greed": "Ganância Extrema",
}

# Cache em memória: o índice só atualiza 1x/dia — não faz sentido bater na API
# externa a cada carregamento do dashboard. TTL de 1h é generoso o suficiente.
_fng_cache: dict | None = None
_fng_cache_at: float = 0.0
_FNG_CACHE_TTL = 3600  # segundos


async def fetch_fear_greed() -> dict | None:
    """
    Busca o Fear & Greed Index atual (valor 0-100 + classificação).
    Retorna {"value": int, "classification": str, "classification_en": str}
    ou None em caso de falha. Usa cache de 1h para poupar a API externa.
    """
    import time
    global _fng_cache, _fng_cache_at

    now = time.monotonic()
    if _fng_cache is not None and (now - _fng_cache_at) < _FNG_CACHE_TTL:
        return _fng_cache

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=1")
            data = resp.json()
        entry = data["data"][0]
        classification_en = entry["value_classification"]
        result = {
            "value": int(entry["value"]),
            "classification": _FNG_LABELS_PT.get(classification_en, classification_en),
            "classification_en": classification_en,
        }
        _fng_cache = result
        _fng_cache_at = now
        return result
    except Exception:
        # Falha na API externa: devolve o último valor em cache (mesmo vencido)
        # em vez de nada — um F&G "velho" é mais útil que um badge vazio.
        return _fng_cache


# ---------------------------------------------------------------------------
# Candlestick patterns (para confirmação Wyckoff)
# ---------------------------------------------------------------------------

def is_bullish_candle(row: pd.Series) -> bool:
    return row["close"] > row["open"]


def is_bearish_candle(row: pd.Series) -> bool:
    return row["close"] < row["open"]


def is_hammer(row: pd.Series) -> bool:
    """Pin bar / Hammer: cauda inferior > 2× corpo, corpo no topo."""
    body = abs(row["close"] - row["open"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    upper_wick = row["high"] - max(row["open"], row["close"])
    if body == 0:
        return False
    return lower_wick > 2 * body and upper_wick < body


def is_shooting_star(row: pd.Series) -> bool:
    """Shooting Star: cauda superior > 2× corpo, corpo na base."""
    body = abs(row["close"] - row["open"])
    upper_wick = row["high"] - max(row["open"], row["close"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    if body == 0:
        return False
    return upper_wick > 2 * body and lower_wick < body


# ---------------------------------------------------------------------------
# Função principal: adiciona todos os indicadores ao DataFrame
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame, ema_bias: int = 200, rsi_period: int = 14,
                   bb_period: int = 20, bb_std: float = 2.0,
                   atr_period: int = 14, volume_sma_period: int = 20) -> pd.DataFrame:
    """
    Recebe um DataFrame com colunas open/high/low/close/volume.
    Retorna o mesmo DataFrame com todos os indicadores como colunas adicionais.
    """
    df = df.copy()

    ohlc4 = calc_ohlc4(df)

    df["ema_bias"] = calc_ema(df["close"], ema_bias)
    df["ema20"] = calc_ema(df["close"], 20)
    df["ema50"] = calc_ema(df["close"], 50)
    df["ema200"] = calc_ema(df["close"], 200)

    df["rsi"] = calc_rsi(ohlc4, rsi_period)
    df["ohlc4"] = ohlc4

    bb = calc_bb(ohlc4, bb_period, bb_std)
    df = pd.concat([df, bb], axis=1)

    df["atr"] = calc_atr(df, atr_period)
    df["mayer"] = calc_mayer_multiple(df)
    df["volume_sma"] = calc_volume_sma(df, volume_sma_period)
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)

    return df
