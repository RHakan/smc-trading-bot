"""
backtest_regime_v2.py — Regime DIÁRIO + Entradas 1H
Regime definido pela vela diária: EMA20(daily) + slope
  BULL daily → só LONGs no 1H (RSI<28 + bounce candle = dip em uptrend)
  BEAR daily → só SHORTs no 1H (RSI<28 + bearish candle = continuação)
  NEUTRAL    → nada

A chave: daily EMA não se move quando o 1H tem um spike pontual.
Quando BTC faz um dip horário (RSI 1H < 28), a daily EMA ainda está intacta →
  se a daily é BULL, esse dip é oportunidade de LONG.
  se a daily é BEAR, esse dip é confirmação de SHORT.
"""

import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────── Config ────────────────
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 24, tzinfo=timezone.utc)
DAYS  = (END - START).days  # 174

# Entry (1H)
RSI_P = 14; BB_P = 20; BB_STD = 2.0; ATR_P = 14
MOM_BARS = 4; MOM_ATR = 1.5; STOP_ATR = 1.0
BE_R = 0.8; TRAIL_ATR = 1.5; RR_CAP = 3.0
FEE = 0.0005; SLIP = 0.0002; RT = (FEE + SLIP) * 2
COOLDOWN = 8

# Entry thresholds
RSI_LONG  = 28; BB_LONG  = 0.05   # oversold = LONG em uptrend, SHORT em downtrend
RSI_SHORT = 72; BB_SHORT = 0.95   # overbought = SHORT adicional em downtrend

# Regime (Daily)
REGIME_EMA_D   = 20    # EMA de 20 dias (20 sessões ≈ 1 mês)
REGIME_SLOPE_D = 5     # comparar EMA agora vs 5 dias atrás
SLOPE_THRESH   = 0.0   # qualquer slope positivo/negativo conta

PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOGE/USDT:USDT", "DOT/USDT:USDT",
    "XRP/USDT:USDT", "LINK/USDT:USDT",
]

CAPITAL0 = 100.0
RISK_PCT  = 0.01

# ──────────────── Fetch ────────────────
ex = ccxt.binanceusdm({"enableRateLimit": True})

def fetch(sym, tf, extra_days):
    since_dt = START - timedelta(days=extra_days)
    since = int(since_dt.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    rows = []
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b: break
        rows.extend(b); since = b[-1][0] + 1
        if since >= end_ms or len(b) == 0: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df[df.index < END]

# ──────────────── Indicadores daily ────────────────
def indicators_daily(df):
    df = df.copy()
    df["ema"] = df["close"].ewm(span=REGIME_EMA_D, adjust=False).mean()
    df["slope"] = (df["ema"] - df["ema"].shift(REGIME_SLOPE_D)) / df["ema"].shift(REGIME_SLOPE_D)
    df["regime"] = "NEUTRAL"
    m_bull = (df["close"] > df["ema"]) & (df["slope"] > SLOPE_THRESH)
    m_bear = (df["close"] < df["ema"]) & (df["slope"] < -SLOPE_THRESH)
    df.loc[m_bull, "regime"] = "BULL"
    df.loc[m_bear, "regime"] = "BEAR"
    return df

def get_regime_1h(df_daily, df_1h):
    """Alinha regime diário ao índice 1H (forward-fill pelo dia)."""
    # Usa o fechamento diário para o próximo dia inteiro (sem lookahead)
    reg = df_daily["regime"].shift(1)  # ontem define hoje
    return reg.reindex(df_1h.index, method="ffill")

# ──────────────── Indicadores 1H ────────────────
def indicators_1h(df):
    df = df.copy()
    o4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    prev = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]), (df["high"]-prev).abs(), (df["low"]-prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_P-1, adjust=False).mean()
    d = o4.diff()
    g = d.where(d>0,0.).ewm(com=RSI_P-1, adjust=False).mean()
    l = (-d.where(d<0,0.)).ewm(com=RSI_P-1, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + g / l.replace(0,np.nan)))
    mid = o4.rolling(BB_P).mean(); std = o4.rolling(BB_P).std()
    df["bbb"] = (o4-(mid-BB_STD*std)) / ((2*BB_STD*std).replace(0,np.nan))
    df["mom"] = df["close"].diff(MOM_BARS)
    return df

WARMUP_1H = max(ATR_P, BB_P) + MOM_BARS + 5   # ~29 barras 1H

# ──────────────── Detect ────────────────
def detect(df1h, i, regime):
    r = df1h.iloc[i]
    rsi = float(r["rsi"]); bbb = float(r["bbb"]); atr = float(r["atr"])
    mom = float(r["mom"]); cl = float(r["close"]); op = float(r["open"])
    if any(np.isnan(v) for v in [rsi, bbb, atr, mom]):
        return None
    thr = MOM_ATR * atr

    if regime == "BULL":
        # Dip buy: oversold extremo + candle de reversão (bounce confirmado)
        if rsi < RSI_LONG and bbb < BB_LONG and cl > op and mom > -thr:
            sl = cl - STOP_ATR * atr; risk = cl - sl
            return {"side":"long","entry":cl,"sl":sl,"tp":cl+RR_CAP*risk,"atr":atr,"tag":"OV-L"} if risk>0 else None

    elif regime == "BEAR":
        # Tipo A: OV-S — oversold continuation (continua caindo)
        if rsi < RSI_LONG and bbb < BB_LONG and cl < op and mom < 0:
            sl = cl + STOP_ATR * atr; risk = sl - cl
            return {"side":"short","entry":cl,"sl":sl,"tp":cl-RR_CAP*risk,"atr":atr,"tag":"OV-S"} if risk>0 else None
        # Tipo B: OB-S — dead cat bounce rejection (subiu para overbought, rejeita)
        if rsi > RSI_SHORT and bbb > BB_SHORT and cl < op and mom < thr:
            sl = cl + STOP_ATR * atr; risk = sl - cl
            return {"side":"short","entry":cl,"sl":sl,"tp":cl-RR_CAP*risk,"atr":atr,"tag":"OB-S"} if risk>0 else None

    return None

# ──────────────── Simulate ────────────────
def simulate(df, ib, sig):
    e=sig["entry"]; sl=sig["sl"]; side=sig["side"]
    atr0=sig["atr"]; risk=abs(e-sl); fee_r=e*RT/risk
    cur=sl; be=False; par=False
    for j in range(ib+1, len(df)):
        bar=df.iloc[j]; h=float(bar["high"]); l=float(bar["low"]); c=float(bar["close"])
        atr=float(bar["atr"]) if not np.isnan(bar["atr"]) else atr0
        if (h>=cur) if side=="short" else (l<=cur):
            gr=(e-cur)/risk if side=="short" else (cur-e)/risk
            out="WIN" if (side=="short" and cur<e) or (side=="long" and cur>e) else "LOSS"
            return {"out":out,"gr":gr,"nr":gr-fee_r,"fee_r":fee_r,"close_ts":df.index[j]}
        if side=="short":
            rg=(e-c)/risk
            if not par and l<=sig["tp"]: par=True; locked=e-2*risk; cur=min(cur,locked)
            if be:
                cand=c+TRAIL_ATR*atr
                if cand<cur: cur=cand
            if not be and rg>=BE_R:
                if e<cur: cur=e
                be=True
        else:
            rg=(c-e)/risk
            if not par and h>=sig["tp"]: par=True; locked=e+2*risk; cur=max(cur,locked)
            if be:
                cand=c-TRAIL_ATR*atr
                if cand>cur: cur=cand
            if not be and rg>=BE_R:
                if e>cur: cur=e
                be=True
    lc=float(df.iloc[-1]["close"])
    gr=(e-lc)/risk if side=="short" else (lc-e)/risk
    return {"out":"OPEN","gr":gr,"nr":gr-fee_r,"fee_r":fee_r,"close_ts":END}

# ──────────────── Main ────────────────
if __name__ == "__main__":
    print(f"Baixando Daily + 1H ({DAYS} dias, {len(PAIRS)} pares)...")
    data_d = {}; data_1h = {}
    for sym in PAIRS:
        data_d[sym]  = indicators_daily(fetch(sym, "1d", REGIME_EMA_D + REGIME_SLOPE_D + 5))
        data_1h[sym] = indicators_1h(fetch(sym, "1h", 5))
        print(f"  {sym.split('/')[0]}: daily={len(data_d[sym])}bars  1H={len(data_1h[sym])}bars")

    print("\nRodando backtest...\n")

    all_trades = []
    for sym in PAIRS:
        df_daily = data_d[sym]
        df_1h    = data_1h[sym]
        regime_series = get_regime_1h(df_daily, df_1h)
        last_bar = -9999
        start_i = max(WARMUP_1H, next((i for i,ts in enumerate(df_1h.index) if ts >= START), WARMUP_1H))
        for i in range(start_i, len(df_1h)-1):
            if i - last_bar < COOLDOWN:
                continue
            ts_bar = df_1h.index[i]
            if ts_bar < START:
                continue
            regime = str(regime_series.iloc[i]) if not pd.isna(regime_series.iloc[i]) else "NEUTRAL"
            sig = detect(df_1h, i, regime)
            if not sig:
                continue
            res = simulate(df_1h, i, sig)
            all_trades.append({
                "pair": sym, "open_ts": ts_bar,
                "regime": regime, "side": sig["side"],
                "tag": sig.get("tag","?"), **res,
            })
            last_bar = i

    all_trades.sort(key=lambda t: t["open_ts"])
    closed = [t for t in all_trades if t["out"] != "OPEN"]
    print(f"Total trades: {len(all_trades)} | Fechados: {len(closed)}\n")

    # ──────────────── Simulação composta ────────────────
    capital = CAPITAL0
    months = {}
    for t in closed:
        ts=t["open_ts"]; mk=ts.strftime("%Y-%m")
        risk_usd=capital*RISK_PCT; pnl=t["nr"]*risk_usd; capital+=pnl
        months.setdefault(mk,{"trades":[],"start_cap":None,"end_cap":None})
        if months[mk]["start_cap"] is None: months[mk]["start_cap"]=capital-pnl
        months[mk]["end_cap"]=capital
        months[mk]["trades"].append({**t,"pnl_usd":pnl,"cap_after":capital})

    month_labels = {
        "2026-01":"Janeiro", "2026-02":"Fevereiro", "2026-03":"Março",
        "2026-04":"Abril",   "2026-05":"Maio",       "2026-06":"Junho (parcial)",
    }

    print("="*72)
    print(f"  REGIME DIÁRIO + ENTRADA 1H | $100 USD | 1% risco/trade")
    print(f"  Jan → Jun 2026 | EMA{REGIME_EMA_D}d diária")
    print("="*72)

    tw=0; tl=0
    for mk in sorted(months.keys()):
        m=months[mk]; lbl=month_labels.get(mk,mk)
        trades=[t for t in m["trades"] if t["out"]!="OPEN"]
        if not trades:
            print(f"\n  {lbl}: 0 trades"); continue
        wins=[t for t in trades if t["out"]=="WIN"]
        losses=[t for t in trades if t["out"]=="LOSS"]
        tw+=len(wins); tl+=len(losses)
        pnl_tot=sum(t["pnl_usd"] for t in trades)
        nr_avg=sum(t["nr"] for t in trades)/len(trades)
        pct=(m["end_cap"]/m["start_cap"]-1)*100
        ls=sum(1 for t in trades if t["side"]=="long")
        ss=sum(1 for t in trades if t["side"]=="short")
        # Tags
        tags={}
        for t in trades: tags.setdefault(t.get("tag","?"),[]).append(t["nr"])
        tag_str=" | ".join(f"{k}:{len(v)}t avg{sum(v)/len(v):+.2f}" for k,v in sorted(tags.items()))
        mark="+" if pnl_tot>0 else ""
        print(f"\n  ── {lbl} ──")
        print(f"     Trades: {len(trades)} (L:{ls} S:{ss}) | WR: {len(wins)/len(trades)*100:.0f}% | Avg R: {nr_avg:+.3f}")
        print(f"     ${m['start_cap']:.2f} → ${m['end_cap']:.2f}  ({mark}{pct:.1f}%)   P&L: {mark}${abs(pnl_tot):.2f}")
        print(f"     {tag_str}")

    # Stats gerais
    nr_all=[t["nr"] for t in closed]
    gw=sum(r for r in nr_all if r>0); gl=abs(sum(r for r in nr_all if r<0))
    pf=gw/gl if gl>0 else 99
    bulls=[t for t in closed if t["regime"]=="BULL"]
    bears=[t for t in closed if t["regime"]=="BEAR"]
    bull_wr=sum(1 for t in bulls if t["out"]=="WIN")/len(bulls)*100 if bulls else 0
    bear_wr=sum(1 for t in bears if t["out"]=="WIN")/len(bears)*100 if bears else 0

    print(f"\n{'='*72}")
    print(f"  RESULTADO FINAL")
    print(f"  $100.00 → ${capital:.2f}  ({(capital/100-1)*100:+.1f}%)")
    print(f"  Trades: {len(closed)} | WR: {tw/(tw+tl)*100:.1f}% | Avg R: {sum(nr_all)/len(nr_all):+.3f} | PF: {pf:.2f}")
    print(f"  BULL (LONGs): {len(bulls)} trades | WR: {bull_wr:.1f}%")
    print(f"  BEAR (SHORTs): {len(bears)} trades | WR: {bear_wr:.1f}%")

    # Regime diário por mês (BTC proxy)
    print(f"\n  Regime diário BTC por mês:")
    df_d_btc = data_d["BTC/USDT:USDT"]
    df_1h_btc = data_1h["BTC/USDT:USDT"]
    reg_series = get_regime_1h(df_d_btc, df_1h_btc)
    for mk in ["2026-01","2026-02","2026-03","2026-04","2026-05","2026-06"]:
        y,mo=int(mk[:4]),int(mk[5:])
        sub=reg_series[(reg_series.index.year==y)&(reg_series.index.month==mo)&(reg_series.index>=START)]
        if len(sub)==0: continue
        b=sum(sub=="BULL"); d=sum(sub=="BEAR"); n=sum(sub=="NEUTRAL"); tot=len(sub)
        print(f"    {month_labels.get(mk,mk)[:3]}: BULL {b/tot*100:>4.0f}%  BEAR {d/tot*100:>4.0f}%  NEUTRAL {n/tot*100:>4.0f}%")
    print(f"{'='*72}\n")
