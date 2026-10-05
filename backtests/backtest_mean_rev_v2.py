"""
backtest_mean_rev_v2.py — RSI+BB Mean Reversion com Filtro de Tendência
Estratégia: identificar extremos de preço e operar NA DIREÇÃO da tendência
  - Extremo de venda (RSI<28, BB%b<0.05) + preço ACIMA da EMA trend → LONG (dip buy)
  - Extremo de venda (RSI<28, BB%b<0.05) + preço ABAIXO da EMA trend → SHORT (continuação)
  - Extremo de compra (RSI>72, BB%b>0.95) + preço ABAIXO da EMA trend → SHORT (exhaustion sell)
  - Extremo de compra (RSI>72, BB%b>0.95) + preço ACIMA da EMA trend → LONG (momentum buy)

Filtragem adicional:
  - Candle de entrada confirma a direção (fechamento do lado certo da abertura)
  - Filtro de momentum: não entrar se os últimos 4 candles mostram mov. contrário forte
"""

import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────── Parâmetros ────────────────
DAYS = 90
TF = "1h"
RSI_P = 14
BB_P = 20
BB_STD = 2.0
ATR_P = 14
MOM_BARS = 4
MOM_ATR = 1.5
STOP_ATR = 1.0
BE_R = 0.8
TRAIL_ATR = 1.5
RR_CAP = 3.0
FEE = 0.0005
SLIP = 0.0002
RT = (FEE + SLIP) * 2
COOLDOWN = 8  # horas entre sinais por par

RSI_LO = 28    # RSI abaixo = oversold extremo
RSI_HI = 72    # RSI acima = overbought extremo
BB_LO = 0.05   # BB%b abaixo = preço próximo/abaixo da banda inferior
BB_HI = 0.95   # BB%b acima = preço próximo/acima da banda superior

PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOGE/USDT:USDT", "DOT/USDT:USDT",
    "XRP/USDT:USDT", "LINK/USDT:USDT",
]

# Configs: (ema_period, slope_lookback, label)
# slope_lookback: se > 0, usa direção da EMA (slope) em vez de preço vs EMA
# Ex: slope=8 compara EMA_now vs EMA_8_bars_atrás
CONFIGS = [
    (200, 0,  "EMA200 nível"),
    (50,  8,  "EMA50 slope8"),
    (50,  16, "EMA50 slope16"),
    (20,  8,  "EMA20 slope8"),
]

# ──────────────── Fetch ────────────────
ex = ccxt.binanceusdm({"enableRateLimit": True})

def fetch(sym):
    limit = DAYS * 24 + 300
    since = int((datetime.now(timezone.utc) - timedelta(days=DAYS + 3)).timestamp() * 1000)
    rows = []
    while len(rows) < limit:
        b = ex.fetch_ohlcv(sym, TF, since=since, limit=1500)
        if not b:
            break
        rows.extend(b)
        since = b[-1][0] + 1
        if len(b) < 1500:
            break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").sort_index().iloc[-limit:]

# ──────────────── Indicadores ────────────────
def add_indicators(df, ema_period, slope_lb):
    df = df.copy()
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_P - 1, adjust=False).mean()

    delta = ohlc4.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=RSI_P - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=RSI_P - 1, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    mid = ohlc4.rolling(BB_P).mean()
    std = ohlc4.rolling(BB_P).std()
    lower = mid - BB_STD * std
    upper = mid + BB_STD * std
    df["bbb"] = (ohlc4 - lower) / ((upper - lower).replace(0, np.nan))

    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    df["ema_trend"] = ema
    if slope_lb > 0:
        # Trend = direção da EMA (slope): True = subindo, False = caindo
        df["trend_up"] = ema > ema.shift(slope_lb)
    else:
        # Trend = preço vs EMA: True = acima, False = abaixo
        df["trend_up"] = df["close"] > ema
    df["mom"] = df["close"].diff(MOM_BARS)
    return df

WARMUP = 200 + max(ATR_P, BB_P) + MOM_BARS + 5

# ──────────────── Detecção de sinal ────────────────
def detect(df, i):
    """Retorna sinal de entrada ou None.

    Lógica trend-aware (3 cenários válidos, 1 ignorado):
    - OV-UP: Oversold (RSI<28, BB%b<0.05) + acima da EMA → LONG (dip buy em uptrend)
    - OV-DN: Oversold (RSI<28, BB%b<0.05) + abaixo da EMA → SHORT (continuação em downtrend)
    - OB-DN: Overbought (RSI>72, BB%b>0.95) + abaixo da EMA → SHORT (exhaustion em downtrend)
    - OB-UP (overbought + acima da EMA) → NÃO OPERAR (comprar no topo = furada)
    """
    r = df.iloc[i]
    rsi = float(r["rsi"]); bbb = float(r["bbb"]); atr = float(r["atr"])
    mom = float(r["mom"]); cl = float(r["close"]); op = float(r["open"])
    ema = float(r["ema_trend"])

    trend_up = bool(r["trend_up"])
    if any(np.isnan(v) for v in [rsi, bbb, atr, mom, ema]):
        return None

    thr = MOM_ATR * atr
    candle_bull = cl > op   # fechou positivo (bounce confirmado)
    candle_bear = cl < op   # fechou negativo (fraqueza confirmada)

    # ── OV-UP: oversold em uptrend → LONG (dip buy) ──
    if rsi < RSI_LO and bbb < BB_LO and trend_up and mom > -thr and candle_bull:
        sl = cl - STOP_ATR * atr
        risk = cl - sl
        if risk > 0:
            return {"side": "long", "entry": cl, "sl": sl, "tp": cl + RR_CAP * risk, "atr": atr, "tag": "OV-UP"}

    # ── OV-DN: oversold em downtrend → SHORT (continuação) ──
    if rsi < RSI_LO and bbb < BB_LO and not trend_up and mom < 0 and candle_bear:
        sl = cl + STOP_ATR * atr
        risk = sl - cl
        if risk > 0:
            return {"side": "short", "entry": cl, "sl": sl, "tp": cl - RR_CAP * risk, "atr": atr, "tag": "OV-DN"}

    # ── OB-DN: overbought em downtrend → SHORT (exhaustion) ──
    if rsi > RSI_HI and bbb > BB_HI and not trend_up and mom < thr and candle_bear:
        sl = cl + STOP_ATR * atr
        risk = sl - cl
        if risk > 0:
            return {"side": "short", "entry": cl, "sl": sl, "tp": cl - RR_CAP * risk, "atr": atr, "tag": "OB-DN"}

    return None

# ──────────────── Simulação de trade ────────────────
def simulate(df, ib, sig):
    e = sig["entry"]; sl = sig["sl"]; side = sig["side"]
    atr0 = sig["atr"]; risk = abs(e - sl)
    fee_r = e * RT / risk
    cur = sl
    be = False
    par = False

    for j in range(ib + 1, len(df)):
        bar = df.iloc[j]
        h = float(bar["high"]); l = float(bar["low"]); c = float(bar["close"])
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else atr0

        hit_stop = (h >= cur) if side == "short" else (l <= cur)
        if hit_stop:
            gr = (e - cur) / risk if side == "short" else (cur - e) / risk
            out = "WIN" if (side == "short" and cur < e) or (side == "long" and cur > e) else "LOSS"
            return {"out": out, "gr": gr, "nr": gr - fee_r, "fee_r": fee_r, "tag": sig["tag"]}

        if side == "short":
            rg = (e - c) / risk
            if not par and l <= sig["tp"]:
                par = True
                locked = e - 2 * risk
                if cur > locked:
                    cur = locked
            if be:
                cand = c + TRAIL_ATR * atr
                if cand < cur:
                    cur = cand
            if not be and rg >= BE_R:
                if e < cur:
                    cur = e
                be = True
        else:
            rg = (c - e) / risk
            if not par and h >= sig["tp"]:
                par = True
                locked = e + 2 * risk
                if cur < locked:
                    cur = locked
            if be:
                cand = c - TRAIL_ATR * atr
                if cand > cur:
                    cur = cand
            if not be and rg >= BE_R:
                if e > cur:
                    cur = e
                be = True

    lc = float(df.iloc[-1]["close"])
    gr = (e - lc) / risk if side == "short" else (lc - e) / risk
    return {"out": "OPEN", "gr": gr, "nr": gr - fee_r, "fee_r": fee_r, "tag": sig["tag"]}

# ──────────────── Relatório ────────────────
def report(all_t, label, days):
    if not all_t:
        print(f"\n[{label}] — 0 trades\n")
        return

    closed = [t for t in all_t if t["out"] != "OPEN"]
    if not closed:
        print(f"\n[{label}] — {len(all_t)} sinais, todos abertos\n")
        return

    nr = [t["nr"] for t in closed]
    gr = [t["gr"] for t in closed]
    fee = [t["fee_r"] for t in all_t]
    wins = [t for t in closed if t["out"] == "WIN"]
    tot = sum(nr); avg = tot / len(nr)
    gw = sum(r for r in nr if r > 0); gl = abs(sum(r for r in nr if r < 0))
    pf = gw / gl if gl > 0 else 99
    ls = sum(1 for t in all_t if t["side"] == "long")
    ss = sum(1 for t in all_t if t["side"] == "short")

    print(f"\n{'='*65}")
    print(f" {label}")
    print(f"{'='*65}")
    print(f"  N={len(all_t)} (L:{ls} S:{ss}) | WR={len(wins)/len(closed)*100:.1f}%")
    print(f"  GrAvg={sum(gr)/len(gr):+.3f}R | FeeR={sum(fee)/len(all_t):.3f} | NetAvg={avg:+.3f}R")
    print(f"  PF={pf:.2f} | T/d={len(all_t)/days:.2f} | Total={tot:+.1f}R")

    # Breakdown por tag
    tags = {}
    for t in closed:
        tag = t.get("tag", "?")
        tags.setdefault(tag, []).append(t["nr"])
    if len(tags) > 1:
        print(f"\n  Por tipo de sinal:")
        for tag, vals in sorted(tags.items()):
            w = sum(1 for v in vals if v > 0)
            print(f"    {tag}: {len(vals)}t {w/len(vals)*100:.0f}%WR avg={sum(vals)/len(vals):+.3f} tot={sum(vals):+.1f}")

    # Breakdown por janela de 30 dias
    ts_list = [pd.Timestamp(t["ts"]) for t in all_t]
    t0 = min(ts_list)
    print(f"\n  Janelas de 30 dias:")
    ws_pos = 0; ws_act = 0
    for w in range(int(days / 30) + 1):
        s = t0 + timedelta(days=w * 30); e = s + timedelta(days=30)
        wt = [t for t, ts_ in zip(all_t, ts_list) if s <= ts_ < e]
        wc = [t for t in wt if t["out"] != "OPEN"]
        if wt:
            if wc:
                ws_act += 1
                wn = [t["nr"] for t in wc]
                wa = sum(wn) / len(wn)
                ww = sum(1 for t in wc if t["out"] == "WIN")
                mark = "+" if wa > 0 else "-"
                ws_pos += (wa > 0)
                print(f"    W{w+1} {s.strftime('%m/%d')}->{e.strftime('%m/%d')}: {len(wt)}t {ww/len(wc)*100:.0f}%WR avg={wa:+.3f} tot={sum(wn):+.1f}R  [{mark}]")
            else:
                print(f"    W{w+1} {s.strftime('%m/%d')}->{e.strftime('%m/%d')}: {len(wt)} abertos")
        else:
            print(f"    W{w+1} {s.strftime('%m/%d')}->{e.strftime('%m/%d')}: 0 trades")
    print(f"  Windows positivos: {ws_pos}/{ws_act}")

    # Breakdown por par
    print(f"\n  Por par:")
    for sym in PAIRS:
        st = [t for t in all_t if t["pair"] == sym]
        sc = [t for t in st if t["out"] != "OPEN"]
        if sc:
            sr = [t["nr"] for t in sc]
            sw = sum(1 for t in sc if t["out"] == "WIN")
            ls2 = sum(1 for t in st if t["side"] == "long")
            ss2 = sum(1 for t in st if t["side"] == "short")
            mark = "+" if sum(sr) > 0 else "-"
            print(f"    {sym.split('/')[0]:<6} {len(st):>3}t L{ls2}S{ss2} {sw/len(sc)*100:.0f}%WR avg={sum(sr)/len(sr):+.3f} tot={sum(sr):+.1f}  [{mark}]")

# ──────────────── Main ────────────────
if __name__ == "__main__":
    print(f"Baixando dados {TF} ({DAYS} dias, {len(PAIRS)} pares)...", end="", flush=True)
    raw = {}
    for sym in PAIRS:
        raw[sym] = fetch(sym)
        print(".", end="", flush=True)
    print(" OK\n")

    for (ema_period, slope_lb, lbl) in CONFIGS:
        # Aplicar indicadores com o EMA period correto
        dfs = {}
        for sym in PAIRS:
            dfs[sym] = add_indicators(raw[sym], ema_period, slope_lb)

        all_trades = []
        for sym, df in dfs.items():
            last_bar = -9999
            for i in range(WARMUP, len(df) - 1):
                if i - last_bar < COOLDOWN:
                    continue
                sig = detect(df, i)
                if not sig:
                    continue
                res = simulate(df, i, sig)
                all_trades.append({
                    "pair": sym,
                    "ts": str(df.index[i]),
                    "side": sig["side"],
                    **res,
                })
                last_bar = i

        report(all_trades, lbl, DAYS)

    print("\nFim.")
