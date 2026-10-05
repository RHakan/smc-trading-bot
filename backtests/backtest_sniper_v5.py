#!/usr/bin/env python3
"""
backtest_sniper_v5.py -- Sniper 1H: validacao em 180 dias

Contexto (v4 resultado):
  BBoff EMAoff 8p (sem LINK e XRP): avg +0.349R, PF 1.90, R/DD 6.6 [OK]
  Problema: W3 = 0 trades em todos configs (junho/2026 calmo)
  Pergunta: esse 0-trade-em-regime-calmo e anomalia ou padrao recorrente?

Objetivo v5:
  - 180 dias em vez de 90 para ver multiplos ciclos de volatilidade
  - 6 janelas de 30 dias para mapear consistencia por periodo
  - Configs testados: BBoff 8p (alvo) + BBoff 10p (comparativo) + BB70 8p (variante)
  - Mesmos parametros fixos: SFatr=1.0, RC3, BE0.5, Trail=ON

Criterio de aprovacao (revisado para 180 dias):
  avg_r >= 0.15, PF >= 1.30, R/DD >= 2.0, N >= 200, NoBest > 0
  Meta de janelas: pelo menos 4 das 6 janelas positivas
"""

import bisect
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

PAIRS_8 = [
    "BTC/USDT:USDT",  "ETH/USDT:USDT",  "SOL/USDT:USDT",  "BNB/USDT:USDT",
    "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT", "DOT/USDT:USDT",
]

PAIRS_10 = PAIRS_8 + ["XRP/USDT:USDT", "LINK/USDT:USDT"]

DAYS       = 180
TF         = "1h"
SWING_LB   = 10
COOLDOWN   = 3
RR_MIN     = 2.0
STOP_BUF   = 0.25
RR_CAP     = 3.0
BE_R       = 0.5
TRAIL_ATR  = 1.5
FLOOR_ATR  = 1.0

ATR_P  = 14;  BB_P = 20;  BB_STD = 2.0

FEE   = 0.0005;  SLIP = 0.0002
RT    = (FEE + SLIP) * 2   # 0.14%

OUTPUTS_DIR = Path(r"C:\Users\ASUS\OneDrive\Documentos\Claudinho\Outputs")

PASS_AVG = 0.15;  PASS_PF = 1.30;  PASS_RDDR = 2.0;  PASS_N = 200

# Configs: (bb_s, bb_l, pairs, label)
CONFIGS = [
    (None, None, PAIRS_8,  "BBoff  8p"),   # alvo principal
    (None, None, PAIRS_10, "BBoff 10p"),   # comparativo com LINK e XRP
    (0.70, 0.30, PAIRS_8,  "BB70   8p"),   # alternativa com filtro BB suave
]


# --- EXCHANGE ----------------------------------------------------------------

def get_exchange():
    return ccxt.binanceusdm({"enableRateLimit": True})

def fetch_ohlcv(ex, sym, days):
    limit = days * 24 + 200
    since = int((datetime.now(timezone.utc) - timedelta(days=days+2)).timestamp() * 1000)
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
    prev  = df["close"].shift(1)
    tr    = pd.concat([(df["high"]-df["low"]),
                       (df["high"]-prev).abs(),
                       (df["low"]-prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_P-1, adjust=False).mean()
    mid = ohlc4.rolling(BB_P).mean()
    std = ohlc4.rolling(BB_P).std()
    df["bb"] = (ohlc4 - (mid - BB_STD*std)) / (2*BB_STD*std).replace(0, np.nan)
    return df


# --- SWING PRECOMPUTATION ----------------------------------------------------

def precompute_swings(df, lb):
    n = len(df); H = df["high"].values; L = df["low"].values
    sh_raw = np.full(n, np.nan); sl_raw = np.full(n, np.nan)
    for i in range(lb, n - lb):
        if H[i] == H[i-lb:i+lb+1].max(): sh_raw[i] = H[i]
        if L[i] == L[i-lb:i+lb+1].min(): sl_raw[i] = L[i]
    last_sh = np.full(n, np.nan); last_sl = np.full(n, np.nan)
    cur_sh = cur_sl = np.nan
    for i in range(n):
        j = i - lb
        if j >= 0:
            if not np.isnan(sh_raw[j]): cur_sh = sh_raw[j]
            if not np.isnan(sl_raw[j]): cur_sl = sl_raw[j]
        last_sh[i] = cur_sh; last_sl[i] = cur_sl
    sh_pts = [(j, float(sh_raw[j])) for j in range(n) if not np.isnan(sh_raw[j])]
    sl_pts = [(j, float(sl_raw[j])) for j in range(n) if not np.isnan(sl_raw[j])]
    return last_sh, last_sl, sh_pts, sl_pts, [p[0] for p in sh_pts], [p[0] for p in sl_pts]

def _sl_below(sl_pts, sl_idx, vis, ceil):
    pos = bisect.bisect_right(sl_idx, vis) - 1
    for k in range(pos, -1, -1):
        if sl_pts[k][1] < ceil: return sl_pts[k][1]
    return None

def _sh_above(sh_pts, sh_idx, vis, floor):
    pos = bisect.bisect_right(sh_idx, vis) - 1
    for k in range(pos, -1, -1):
        if sh_pts[k][1] > floor: return sh_pts[k][1]
    return None


# --- SIGNAL DETECTION --------------------------------------------------------

def detect(data, i, bb_s=None, bb_l=None):
    df = data["df"]; last = df.iloc[i]
    cl  = float(last["close"]); atr = float(last["atr"]); bb = float(last["bb"])
    if np.isnan(atr) or np.isnan(bb): return None

    last_sh = data["last_sh"]; last_sl = data["last_sl"]
    sh_pts  = data["sh_pts"];  sl_pts  = data["sl_pts"]
    sh_idx  = data["sh_idx"];  sl_idx  = data["sl_idx"]
    vis = i - SWING_LB

    sw_h = float(last_sh[i]) if not np.isnan(last_sh[i]) else None
    if sw_h is not None and float(last["high"]) > sw_h and cl < sw_h:
        if (bb > bb_s) if bb_s is not None else True:
            sl = float(last["high"]) + STOP_BUF * atr; risk = sl - cl
            if risk > 0 and risk >= FLOOR_ATR * atr:
                tp = max(_sl_below(sl_pts, sl_idx, vis, cl) or cl-risk*RR_MIN, cl-RR_CAP*risk)
                rr = (cl - tp) / risk
                if rr >= RR_MIN:
                    return {"side":"short","entry":cl,"sl":sl,"tp":tp,"rr":rr,"atr":atr}

    sw_l = float(last_sl[i]) if not np.isnan(last_sl[i]) else None
    if sw_l is not None and float(last["low"]) < sw_l and cl > sw_l:
        if (bb < bb_l) if bb_l is not None else True:
            sl = float(last["low"]) - STOP_BUF * atr; risk = cl - sl
            if risk > 0 and risk >= FLOOR_ATR * atr:
                tp = min(_sh_above(sh_pts, sh_idx, vis, cl) or cl+risk*RR_MIN, cl+RR_CAP*risk)
                rr = (tp - cl) / risk
                if rr >= RR_MIN:
                    return {"side":"long","entry":cl,"sl":sl,"tp":tp,"rr":rr,"atr":atr}
    return None


# --- TRADE SIMULATION --------------------------------------------------------

def simulate(df, entry_bar, sig):
    entry = sig["entry"]; sl = sig["sl"]; side = sig["side"]; atr0 = sig["atr"]
    risk  = abs(entry - sl); fee_r = entry * RT / risk
    cur = sl; be = False; par = False

    for j in range(entry_bar + 1, len(df)):
        bar = df.iloc[j]
        h = float(bar["high"]); l = float(bar["low"]); c = float(bar["close"])
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else atr0

        if (h >= cur) if side == "short" else (l <= cur):
            gr = (entry-cur)/risk if side=="short" else (cur-entry)/risk
            out = "WIN" if (side=="short" and cur<entry) or (side=="long" and cur>entry) else "LOSS"
            return {"out":out,"j":j,"ep":cur,"gr":gr,"nr":gr-fee_r,"be":be,"fee_r":fee_r}

        if side == "short":
            rg = (entry - c) / risk
            if not par and l <= sig["tp"]:
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
            if not par and h >= sig["tp"]:
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


# --- BACKTEST LOOP -----------------------------------------------------------

WARMUP = SWING_LB * 2 + max(ATR_P, BB_P) + 5   # 45 barras

def run_pair(data, pair, bb_s, bb_l):
    df = data["df"]; trades = []; last_bar = -9999
    for i in range(WARMUP, len(df) - 1):
        if i - last_bar < COOLDOWN: continue
        sig = detect(data, i, bb_s, bb_l)
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
    grs    = [t["gr"] for t in closed]
    fees   = [t["fee_r"] for t in trades]
    n = len(trades); nc = len(closed)
    if not net_rs:
        return {"n":0,"wr":0,"avg":0,"gr_avg":0,"fee_avg":0,"tot":0,
                "pf":0,"dd":0,"rddr":0,"tpd":0,"be":0,"best":0,"worst":0,
                "no_best":0,"longs":0,"shorts":0}
    tot = sum(net_rs); avg = tot/nc
    gw = sum(r for r in net_rs if r>0); gl = abs(sum(r for r in net_rs if r<0))
    pf = gw/gl if gl>0 else 99.0
    cum = np.cumsum(net_rs); pk = np.maximum.accumulate(cum)
    dd  = float(np.max(pk-cum))
    no_b = sum(sorted(net_rs)[:-1]) if len(net_rs)>1 else tot
    return {"n":n,"wr":round(len(wins)/nc*100,1),
            "avg":round(avg,3),"gr_avg":round(sum(grs)/nc,3),"fee_avg":round(sum(fees)/n,3),
            "tot":round(tot,2),"pf":round(pf,2),"dd":round(dd,2),
            "rddr":round(tot/dd,2) if dd>0 else 0.0,
            "tpd":round(n/DAYS,2),"be":round(sum(1 for t in trades if t.get("be"))/n*100,1),
            "best":round(max(net_rs),2),"worst":round(min(net_rs),2),"no_best":round(no_b,2),
            "longs":sum(1 for t in trades if t["side"]=="long"),
            "shorts":sum(1 for t in trades if t["side"]=="short")}

def windows_n(trades, n_windows=6):
    """Divide em n janelas de 30 dias a partir do primeiro trade."""
    if not trades: return []
    ts_list = [pd.Timestamp(t["ts"]) for t in trades]
    t0 = min(ts_list); out = []
    for w in range(n_windows):
        s = t0 + timedelta(days=w*30); e = s + timedelta(days=30)
        wt = [t for t, ts_ in zip(trades, ts_list) if s <= ts_ < e]
        m  = calc(wt)
        out.append({"w":f"W{w+1} {s.strftime('%m/%d')}->{e.strftime('%m/%d')}", **m})
    return out

def breakdown(trades):
    bd = []
    for sym in sorted(set(t["pair"] for t in trades)):
        pt = [t for t in trades if t["pair"]==sym]
        m  = calc(pt)
        bd.append({"sym":sym.replace("/USDT:USDT",""), **m})
    return sorted(bd, key=lambda x: x["tot"], reverse=True)


# --- DISPLAY -----------------------------------------------------------------

def _tag(m, ws):
    pos_w = sum(1 for w in ws if w["n"] > 0 and w["avg"] > 0)
    all_ok = (m["avg"] >= PASS_AVG and m["pf"] >= PASS_PF and
              m["rddr"] >= PASS_RDDR and m["n"] >= PASS_N and m["no_best"] > 0)
    window_ok = pos_w >= len([w for w in ws if w["n"] > 0]) * 0.7   # 70% das janelas com trades
    return " [OK]" if all_ok and window_ok else (" [~]" if all_ok else "")

def print_table(results):
    hdr = (f"  {'Config':<14} {'N':>4} {'W%':>5} {'GrAvg':>7} {'FeeR':>5} "
           f"{'Avg':>7} {'PF':>5} {'R/DD':>5} {'NoBest':>7}")
    print(); print("  " + "="*75)
    print("  SNIPER v5 -- 1H | BBoff | SFatr1.0 | RC3 | BE0.5 | Trail=ON | 180d")
    print(f"  {DAYS}d | RT={RT*100:.2f}% | Alvo: avg>={PASS_AVG} PF>={PASS_PF} R/DD>={PASS_RDDR} N>={PASS_N}")
    print("  " + "="*75)
    print(hdr); print("  " + "-"*75)
    for r in sorted(results, key=lambda x: x["m"]["tot"], reverse=True):
        m = r["m"]; tag = _tag(m, r["ws"])
        print(f"  {r['lbl']:<14} {m['n']:>4} {m['wr']:>4.0f}% "
              f"{m['gr_avg']:>+7.3f} {m['fee_avg']:>5.3f} "
              f"{m['avg']:>+7.3f} {m['pf']:>5.2f} {m['rddr']:>5.1f} "
              f"{m['no_best']:>+7.1f}{tag}")
    print("  " + "-"*75)

def print_deep(trades, ws, label):
    m  = calc(trades)
    bd = breakdown(trades)
    print(f"\n  ===== {label} =====")
    print(f"  {m['n']} trades (L:{m['longs']} S:{m['shorts']}) | "
          f"gross {m['gr_avg']:+.3f}R | fee {m['fee_avg']:.3f}R | net {m['avg']:+.3f}R")
    print(f"  PF {m['pf']:.2f} | DD {m['dd']:.1f}R | R/DD {m['rddr']:.1f} | "
          f"Melhor: {m['best']:+.2f}R | Pior: {m['worst']:+.2f}R | NoBest: {m['no_best']:+.2f}R")
    print(f"  {m['tpd']:.2f} trades/dia\n")

    # Janelas mensais com marcacao
    pos_w = 0; tot_w = 0
    print(f"  {'Janela':<23} {'N':>4} {'W%':>5} {'GrAvg':>7} {'Avg':>7} {'Tot':>7} {'Status'}")
    for w in ws:
        if w["n"] == 0:
            status = "  . (sem trades)"
        elif w["avg"] > 0:
            status = "  + POSITIVO"; pos_w += 1; tot_w += 1
        else:
            status = "  - NEGATIVO"; tot_w += 1
        print(f"  {w['w']:<23} {w['n']:>4} {w['wr']:>4.0f}% "
              f"{w['gr_avg']:>+7.3f} {w['avg']:>+7.3f} {w['tot']:>+7.1f}{status}")
    if tot_w > 0:
        print(f"\n  Janelas com trades: {pos_w}/{tot_w} positivas ({pos_w/tot_w*100:.0f}%)")

    n_pos = sum(1 for p in bd if p["tot"] > 0)
    print(f"  Pares positivos: {n_pos}/{len(bd)}\n")
    print(f"  {'Par':<6} {'N':>4} {'W%':>5} {'Avg':>7} {'Tot':>7}")
    for p in bd:
        print(f"  {p['sym']:<6} {p['n']:>4} {p['wr']:>4.0f}% {p['avg']:>+7.3f} {p['tot']:>+7.1f}"
              f"  {'+' if p['tot']>0 else '-'}")


# --- MAIN --------------------------------------------------------------------

def main():
    print(f"\n  +----------------------------------------------+")
    print(f"  |  Sniper v5 -- Validacao 180 dias (1H)       |")
    print(f"  +----------------------------------------------+\n")

    ex = get_exchange()

    all_pairs = list(set(PAIRS_10))
    print(f"  [1H] Fetch {DAYS}d x {len(all_pairs)} pares...", end="", flush=True)
    raw_data = {}
    for sym in all_pairs:
        try:
            df = fetch_ohlcv(ex, sym, DAYS)
            df = add_indicators(df)
            lsh, lsl, shp, slp, shi, sli = precompute_swings(df, SWING_LB)
            raw_data[sym] = {"df":df,"last_sh":lsh,"last_sl":lsl,
                             "sh_pts":shp,"sl_pts":slp,"sh_idx":shi,"sl_idx":sli}
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n  ! {sym}: {e}", file=sys.stderr)
    print(" OK\n")

    print(f"  Rodando {len(CONFIGS)} configs...")
    results = []
    for (bb_s, bb_l, pairs, lbl) in CONFIGS:
        print(f"    {lbl:<14}", end="", flush=True)
        trades = []
        for sym in pairs:
            if sym in raw_data:
                trades.extend(run_pair(raw_data[sym], sym, bb_s, bb_l))
        m  = calc(trades)
        ws = windows_n(trades, 6)
        tag = _tag(m, ws)
        results.append({"lbl":lbl,"m":m,"ws":ws,"trades":trades})
        pos_w = sum(1 for w in ws if w["n"] > 0 and w["avg"] > 0)
        tot_w = sum(1 for w in ws if w["n"] > 0)
        print(f"-> {m['n']:>4}t  net={m['avg']:>+.3f}  PF={m['pf']:.2f}  "
              f"R/DD={m['rddr']:.1f}  janelas={pos_w}/{tot_w} positivas{tag}")

    print_table(results)
    for r in results:
        print_deep(r["trades"], r["ws"], r["lbl"])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    fp = OUTPUTS_DIR / f"{today}_sniper_backtest_v5.json"
    fp.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": DAYS, "rt_pct": RT*100,
        "fixed": {"tf":"1h","rr_cap":RR_CAP,"be_r":BE_R,"trail_atr":TRAIL_ATR,
                  "floor_atr":FLOOR_ATR,"stop_buf":STOP_BUF,"swing_lb":SWING_LB,"cooldown":COOLDOWN},
        "results": [{"lbl":r["lbl"],"metrics":r["m"],
                     "windows":[{k:v for k,v in w.items()} for w in r["ws"]],
                     "breakdown":breakdown(r["trades"])} for r in results],
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  Salvo em: {fp}\n")


if __name__ == "__main__":
    main()
