#!/usr/bin/env python3
"""
backtest_mean_rev_v1.py -- RSI + BB Mean Reversion | 1H | 180 dias

Estrategia base (Pine Script do TradingView):
  LONG:  close < EMA200 AND RSI(ohlc4,14) < 30 AND BB%b < 0
  SELL:  RSI > 70 AND BB%b > 1 AND close > EMA200 AND Mayer > 1.3

Problema identificado pelo usuario:
  "quando a queda e forte sao gerados muitos sinais errados"
  -> Sinal de LONG em queda forte = faca caindo (RSI vai a 25 mas preco segue caindo)

Refinamentos testados:
  A) Base: RSI < 33 + BB%b < 0.15 (ligeiramente relaxado para mais sinais, sem/com shorts)
  B) + Filtro momentum: bloqueia entrada se 4-bar change < -1.5x ATR (queda forte em curso)
  C) + Filtro momentum + EMA200 bias (LONGs acima, SHORTs abaixo)

Gestao de posicao (fixada para todos):
  - Stop: 1.0x ATR abaixo/acima da entrada
  - Breakeven: ao ganhar 0.8R
  - Trailing: 1.5x ATR do close apos BE
  - TP cap: 3R (impede alvos ilusorios)
  - Cooldown: 4 barras (4h) entre sinais por par

Parametros:
  RSI_LONG  = 33   (original 30)
  RSI_SHORT = 67   (original 70 invertido)
  BB_LONG   = 0.15 (original 0, ligeiramente relaxado)
  BB_SHORT  = 0.85
  MOM_ATR   = 1.5  (bloqueio de faca caindo: queda/subida > 1.5x ATR em 4 barras)
"""

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- CONFIG ------------------------------------------------------------------

PAIRS = [
    "BTC/USDT:USDT",  "ETH/USDT:USDT",  "SOL/USDT:USDT",  "BNB/USDT:USDT",
    "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT", "DOT/USDT:USDT",
]

DAYS       = 180
TF         = "1h"
COOLDOWN   = 4      # barras entre trades por par

# Indicadores
RSI_P      = 14
BB_P       = 20;  BB_STD = 2.0
ATR_P      = 14
EMA200_P   = 200

# Parametros de sinal
RSI_LONG   = 33    # entrada LONG se RSI < este valor
RSI_SHORT  = 67    # entrada SHORT se RSI > este valor
BB_LONG    = 0.15  # entrada LONG se BB%b < este valor
BB_SHORT   = 0.85  # entrada SHORT se BB%b > este valor
MOM_BARS   = 4     # barras de lookback para filtro de momentum
MOM_ATR    = 1.5   # multiplicador ATR para definir "queda/subida forte"

# Gestao de posicao
STOP_ATR   = 1.0   # stop = 1.0 x ATR
BE_R       = 0.8   # mover para BE ao ganhar 0.8R
TRAIL_ATR  = 1.5   # trailing de 1.5x ATR apos BE
RR_CAP     = 3.0   # teto do alvo em R
RR_MIN     = 1.5   # R:R minimo para aceitar o trade

FEE   = 0.0005;  SLIP = 0.0002
RT    = (FEE + SLIP) * 2   # 0.14%

OUTPUTS_DIR = Path(r"C:\Users\ASUS\OneDrive\Documentos\Claudinho\Outputs")

PASS_AVG = 0.12;  PASS_PF = 1.25;  PASS_RDDR = 2.0;  PASS_N = 100

# Configs: (use_momentum_filter, use_ema200_bias, allow_shorts, label)
CONFIGS = [
    (False, False, False, "Base LONGonly noFilt"),
    (False, False, True,  "Base L+S   noFilt"),
    (True,  False, False, "Mom  LONGonly"),
    (True,  False, True,  "Mom  L+S"),          # alvo principal
    (True,  True,  True,  "Mom+EMA L+S"),        # com bias EMA200
    (True,  True,  False, "Mom+EMA LONGonly"),
]


# --- EXCHANGE ----------------------------------------------------------------

def get_exchange():
    return ccxt.binanceusdm({"enableRateLimit": True})

def fetch_ohlcv(ex, sym, days):
    limit = days * 24 + 300
    since = int((datetime.now(timezone.utc) - timedelta(days=days+3)).timestamp() * 1000)
    rows = []
    while len(rows) < limit:
        batch = ex.fetch_ohlcv(sym, TF, since=since, limit=1500)
        if not batch: break
        rows.extend(batch); since = batch[-1][0] + 1
        if len(batch) < 1500: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").sort_index().iloc[-limit:]


# --- INDICATORS --------------------------------------------------------------

def add_indicators(df):
    df = df.copy()
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    # ATR
    prev = df["close"].shift(1)
    tr   = pd.concat([(df["high"]-df["low"]),
                      (df["high"]-prev).abs(),
                      (df["low"]-prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_P-1, adjust=False).mean()

    # RSI (usando ohlc4, igual ao Pine Script)
    delta = ohlc4.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(com=RSI_P-1, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(com=RSI_P-1, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain/loss.replace(0, np.nan)))

    # BB%b (usando ohlc4)
    mid = ohlc4.rolling(BB_P).mean()
    std = ohlc4.rolling(BB_P).std()
    up  = mid + BB_STD*std
    dn  = mid - BB_STD*std
    df["bbb"]  = (ohlc4 - dn) / (up - dn).replace(0, np.nan)
    df["upper_bb"] = up
    df["lower_bb"] = dn

    # EMA200
    df["ema200"] = df["close"].ewm(span=EMA200_P, adjust=False).mean()

    # Momentum 4-bar
    df["mom4"] = df["close"].diff(MOM_BARS)

    return df


# --- SIGNAL DETECTION --------------------------------------------------------

def detect(df, i, use_mom, use_ema, allow_shorts):
    row = df.iloc[i]
    rsi   = float(row["rsi"])
    bbb   = float(row["bbb"])
    atr   = float(row["atr"])
    ema   = float(row["ema200"])
    mom4  = float(row["mom4"])
    cl    = float(row["close"])

    if any(np.isnan(v) for v in [rsi, bbb, atr, ema, mom4]): return None

    mom_threshold = MOM_ATR * atr

    # ---- LONG ---------------------------------------------------------------
    long_ok = rsi < RSI_LONG and bbb < BB_LONG
    if use_mom:
        # bloqueia se preco caiu forte nas ultimas 4 barras (faca caindo)
        long_ok = long_ok and (mom4 > -mom_threshold)
    if use_ema:
        # prefere LONGs quando preco esta acima da EMA200 (tendencia de alta)
        long_ok = long_ok and (cl > ema)
    if long_ok:
        sl   = cl - STOP_ATR * atr
        risk = cl - sl
        tp   = cl + RR_CAP * risk
        rr   = RR_CAP
        if risk > 0 and rr >= RR_MIN:
            return {"side":"long","entry":cl,"sl":sl,"tp":tp,"rr":rr,"atr":atr}

    # ---- SHORT --------------------------------------------------------------
    if allow_shorts:
        short_ok = rsi > RSI_SHORT and bbb > BB_SHORT
        if use_mom:
            # bloqueia se preco subiu forte nas ultimas 4 barras
            short_ok = short_ok and (mom4 < mom_threshold)
        if use_ema:
            # prefere SHORTs quando preco esta abaixo da EMA200 (tendencia de baixa)
            short_ok = short_ok and (cl < ema)
        if short_ok:
            sl   = cl + STOP_ATR * atr
            risk = sl - cl
            tp   = cl - RR_CAP * risk
            rr   = RR_CAP
            if risk > 0 and rr >= RR_MIN:
                return {"side":"short","entry":cl,"sl":sl,"tp":tp,"rr":rr,"atr":atr}

    return None


# --- TRADE SIMULATION --------------------------------------------------------

def simulate(df, entry_bar, sig):
    entry = sig["entry"]; sl = sig["sl"]; tp = sig["tp"]
    side  = sig["side"];  atr0 = sig["atr"]
    risk  = abs(entry - sl)
    fee_r = entry * RT / risk

    cur = sl; be = False; par = False

    for j in range(entry_bar + 1, len(df)):
        bar = df.iloc[j]
        h = float(bar["high"]); l = float(bar["low"]); c = float(bar["close"])
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else atr0

        hit = (h >= cur) if side == "short" else (l <= cur)
        if hit:
            gr = (entry-cur)/risk if side=="short" else (cur-entry)/risk
            out = "WIN" if (side=="short" and cur<entry) or (side=="long" and cur>entry) else "LOSS"
            return {"out":out,"j":j,"ep":cur,"gr":gr,"nr":gr-fee_r,"be":be,"fee_r":fee_r}

        if side == "short":
            rg = (entry - c) / risk
            if not par and l <= tp:
                par = True; locked = entry - 2.0*risk
                if cur > locked: cur = locked
            if be:
                cand = c + TRAIL_ATR*atr
                if cand < cur: cur = cand
            if not be and rg >= BE_R:
                if entry < cur: cur = entry
                be = True
        else:
            rg = (c - entry) / risk
            if not par and h >= tp:
                par = True; locked = entry + 2.0*risk
                if cur < locked: cur = locked
            if be:
                cand = c - TRAIL_ATR*atr
                if cand > cur: cur = cand
            if not be and rg >= BE_R:
                if entry > cur: cur = entry
                be = True

    lc = float(df.iloc[-1]["close"])
    gr = (entry-lc)/risk if side=="short" else (lc-entry)/risk
    return {"out":"OPEN","j":len(df)-1,"ep":lc,"gr":gr,"nr":gr-fee_r,"be":be,"fee_r":fee_r}


# --- BACKTEST ----------------------------------------------------------------

WARMUP = EMA200_P + max(ATR_P, BB_P) + MOM_BARS + 5   # ~240

def run_pair(data, pair, use_mom, use_ema, allow_shorts):
    df = data["df"]; trades = []; last_bar = -9999
    for i in range(WARMUP, len(df) - 1):
        if i - last_bar < COOLDOWN: continue
        sig = detect(df, i, use_mom, use_ema, allow_shorts)
        if sig is None: continue
        res = simulate(df, i, sig)
        trades.append({
            "pair":pair,"ts":str(df.index[i]),"side":sig["side"],
            "entry":round(sig["entry"],6),"sl":round(sig["sl"],6),
            "tp":round(sig["tp"],6),"rr_t":round(sig["rr"],2),
            "out":res["out"],"ep":round(res["ep"],6),
            "gr":round(res["gr"],3),"nr":round(res["nr"],3),
            "fee_r":round(res["fee_r"],3),"be":res["be"],
        })
        last_bar = i
    return trades


# --- METRICAS ----------------------------------------------------------------

def calc(trades):
    closed = [t for t in trades if t["out"] != "OPEN"]
    wins   = [t for t in closed if t["out"] == "WIN"]
    net_rs = [t["nr"] for t in closed]
    fees   = [t["fee_r"] for t in trades]
    n = len(trades); nc = len(closed)
    if not net_rs:
        return {"n":0,"wr":0,"avg":0,"gr_avg":0,"fee_avg":0,"tot":0,
                "pf":0,"dd":0,"rddr":0,"tpd":0,"no_best":0,"longs":0,"shorts":0}
    tot = sum(net_rs); avg = tot/nc
    grs = [t["gr"] for t in closed]
    gw = sum(r for r in net_rs if r>0); gl = abs(sum(r for r in net_rs if r<0))
    pf = gw/gl if gl>0 else 99.0
    cum = np.cumsum(net_rs); pk = np.maximum.accumulate(cum)
    dd  = float(np.max(pk-cum))
    no_b = sum(sorted(net_rs)[:-1]) if len(net_rs)>1 else tot
    return {"n":n,"wr":round(len(wins)/nc*100,1),
            "avg":round(avg,3),"gr_avg":round(sum(grs)/nc,3),"fee_avg":round(sum(fees)/n,3),
            "tot":round(tot,2),"pf":round(pf,2),"dd":round(dd,2),
            "rddr":round(tot/dd,2) if dd>0 else 0.0,
            "tpd":round(n/DAYS,2),"no_best":round(no_b,2),
            "longs":sum(1 for t in trades if t["side"]=="long"),
            "shorts":sum(1 for t in trades if t["side"]=="short")}

def windows_n(trades, n=6):
    if not trades: return []
    ts_list = [pd.Timestamp(t["ts"]) for t in trades]
    t0 = min(ts_list); out = []
    for w in range(n):
        s = t0 + timedelta(days=w*30); e = s + timedelta(days=30)
        wt = [t for t, ts_ in zip(trades, ts_list) if s <= ts_ < e]
        out.append({"w":f"W{w+1} {s.strftime('%m/%d')}->{e.strftime('%m/%d')}", **calc(wt)})
    return out

def breakdown(trades):
    bd = []
    for sym in sorted(set(t["pair"] for t in trades)):
        pt = [t for t in trades if t["pair"]==sym]
        bd.append({"sym":sym.replace("/USDT:USDT",""), **calc(pt)})
    return sorted(bd, key=lambda x: x["tot"], reverse=True)


# --- DISPLAY -----------------------------------------------------------------

def _tag(m, ws):
    active = [w for w in ws if w["n"] > 0]
    pos_w  = sum(1 for w in active if w["avg"] > 0)
    ok = (m["avg"] >= PASS_AVG and m["pf"] >= PASS_PF and
          m["rddr"] >= PASS_RDDR and m["n"] >= PASS_N and m["no_best"] > 0 and
          pos_w >= len(active) * 0.6)
    return " [OK]" if ok else (" [~]" if m["avg"] >= PASS_AVG and m["pf"] >= PASS_PF else "")

def print_table(results):
    hdr = (f"  {'Config':<24} {'N':>4} {'W%':>5} {'Avg':>7} {'PF':>5} {'R/DD':>5} "
           f"{'NoBest':>7} {'T/d':>4} {'W+/Act'}")
    print(); print("  " + "="*80)
    print("  RSI+BB MEAN REV v1 -- 1H | ATR stops | BE 0.8R | Trail 1.5ATR | RC3")
    print(f"  {DAYS}d | 8 pares | RT={RT*100:.2f}%")
    print("  " + "="*80)
    print(hdr); print("  " + "-"*80)
    for r in sorted(results, key=lambda x: x["m"]["tot"], reverse=True):
        m = r["m"]; ws = r["ws"]; tag = _tag(m, ws)
        active = [w for w in ws if w["n"] > 0]
        pos_w  = sum(1 for w in active if w["avg"] > 0)
        print(f"  {r['lbl']:<24} {m['n']:>4} {m['wr']:>4.0f}% "
              f"{m['avg']:>+7.3f} {m['pf']:>5.2f} {m['rddr']:>5.1f} "
              f"{m['no_best']:>+7.1f} {m['tpd']:>4.2f}  {pos_w}/{len(active)}{tag}")
    print("  " + "-"*80)

def print_deep(trades, ws, label):
    m  = calc(trades)
    bd = breakdown(trades)
    print(f"\n  ===== {label} =====")
    print(f"  {m['n']} trades (L:{m['longs']} S:{m['shorts']}) | "
          f"gross {m['gr_avg']:+.3f}R | fee {m['fee_avg']:.3f}R | net {m['avg']:+.3f}R")
    print(f"  PF {m['pf']:.2f} | DD {m['dd']:.1f}R | R/DD {m['rddr']:.1f} | "
          f"NoBest {m['no_best']:+.2f}R | {m['tpd']:.2f} t/dia")
    print()
    print(f"  {'Janela':<23} {'N':>4} {'W%':>5} {'Avg':>7} {'Tot':>7}")
    for w in ws:
        mk = ("  + " if w["avg"] > 0 else ("  . " if w["n"]==0 else "  - "))
        print(f"  {w['w']:<23} {w['n']:>4} {w['wr']:>4.0f}% {w['avg']:>+7.3f} {w['tot']:>+7.1f}{mk}")
    active = [w for w in ws if w["n"] > 0]
    pos_w  = sum(1 for w in active if w["avg"] > 0)
    if active:
        print(f"\n  Janelas: {pos_w}/{len(active)} positivas")
    n_pos = sum(1 for p in bd if p["tot"] > 0)
    print(f"  Pares:   {n_pos}/{len(bd)} positivos\n")
    print(f"  {'Par':<6} {'N':>4} {'W%':>5} {'L':>4} {'S':>4} {'Avg':>7} {'Tot':>7}")
    for p in bd:
        pt = [t for t in trades if t["pair"]==p["sym"]+"/USDT:USDT"]
        ls = sum(1 for t in pt if t["side"]=="long")
        ss = sum(1 for t in pt if t["side"]=="short")
        print(f"  {p['sym']:<6} {p['n']:>4} {p['wr']:>4.0f}% {ls:>4} {ss:>4} "
              f"{p['avg']:>+7.3f} {p['tot']:>+7.1f}  {'+ ' if p['tot']>0 else '- '}")


# --- MAIN --------------------------------------------------------------------

def main():
    print(f"\n  +--------------------------------------------------+")
    print(f"  |  RSI+BB Mean Reversion v1 -- 1H | 180d | 8p    |")
    print(f"  +--------------------------------------------------+\n")
    print(f"  Parametros: RSI_L={RSI_LONG} RSI_S={RSI_SHORT} BB_L={BB_LONG} BB_S={BB_SHORT}")
    print(f"  Stop={STOP_ATR}xATR  BE@{BE_R}R  Trail={TRAIL_ATR}xATR  RRcap={RR_CAP}  Mom={MOM_BARS}b/{MOM_ATR}xATR\n")

    ex = get_exchange()
    print(f"  [1H] Fetch {DAYS}d x {len(PAIRS)} pares...", end="", flush=True)
    raw_data = {}
    for sym in PAIRS:
        try:
            df = fetch_ohlcv(ex, sym, DAYS)
            df = add_indicators(df)
            raw_data[sym] = {"df": df}
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n  ! {sym}: {e}", file=sys.stderr)
    print(" OK\n")

    print(f"  Rodando {len(CONFIGS)} configs...")
    results = []
    for (use_mom, use_ema, allow_s, lbl) in CONFIGS:
        print(f"    {lbl:<26}", end="", flush=True)
        trades = []
        for sym, d in raw_data.items():
            trades.extend(run_pair(d, sym, use_mom, use_ema, allow_s))
        m  = calc(trades)
        ws = windows_n(trades, 6)
        tag = _tag(m, ws)
        results.append({"lbl":lbl,"m":m,"ws":ws,"trades":trades})
        active = [w for w in ws if w["n"] > 0]
        pos_w  = sum(1 for w in active if w["avg"] > 0)
        print(f"-> {m['n']:>4}t  net={m['avg']:>+.3f}  PF={m['pf']:.2f}  "
              f"{pos_w}/{len(active)} jan+{tag}")

    print_table(results)

    ok_rs = [r for r in results if _tag(r["m"],r["ws"]) in (" [OK]", " [~]")]
    for r in (ok_rs if ok_rs else sorted(results, key=lambda x: x["m"]["tot"], reverse=True)[:3]):
        print_deep(r["trades"], r["ws"], r["lbl"])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    fp = OUTPUTS_DIR / f"{today}_mean_rev_v1.json"
    fp.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": DAYS, "rt_pct": RT*100,
        "params": {"rsi_long":RSI_LONG,"rsi_short":RSI_SHORT,"bb_long":BB_LONG,"bb_short":BB_SHORT,
                   "stop_atr":STOP_ATR,"be_r":BE_R,"trail_atr":TRAIL_ATR,"rr_cap":RR_CAP,
                   "mom_bars":MOM_BARS,"mom_atr":MOM_ATR},
        "results": [{"lbl":r["lbl"],"metrics":r["m"],
                     "windows":[dict(w) for w in r["ws"]],
                     "breakdown":breakdown(r["trades"])} for r in results],
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  Salvo em: {fp}\n")


if __name__ == "__main__":
    main()
