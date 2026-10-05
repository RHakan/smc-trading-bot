#!/usr/bin/env python3
"""
backtest_sniper_v6.py -- Sniper 1H: visibilidade de swings + EMA200

Diagnóstico v5 (180 dias):
  - Dez/Jan 2025-26 (ATH): estrategia tomou LONGs numa queda continua -> tudo falhou
  - Mar-Jun 2026: last_sh ainda aponta para swings do ATH ($104-110k)
                  preco em abril ($82-88k) nunca chega nesses niveis -> zero trades
                  (por isso v4 com 90 dias capturava abril mas v5 com 180 dias nao)

Dois fixes em v6:
  1. SWING_VIS_LIMIT: so considera swings dos ultimos N barras (nao toda a historia)
     Traders olham estrutura RECENTE, nao swings de 6 meses atras
  2. EMA200 macro trend: LONG so acima da EMA200, SHORT so abaixo
     Alinha o setup com a tendencia macro para evitar pegar quedas (ou ralis) contra-tendencia

Configs em 180 dias:
  A) BBoff + SWING_VIS=200 + sem EMA        -- fix so de visibilidade
  B) BBoff + SWING_VIS=200 + EMA200         -- fix completo (alvo principal)
  C) BBoff + SWING_VIS=500 + EMA200         -- visibilidade um pouco maior
  D) BBoff + SWING_VIS=inf + EMA200         -- EMA200 sem limitar visibilidade (comparativo)
  E) BB70  + SWING_VIS=200 + EMA200         -- com filtro BB suave
  Cada config: 8 pares (sem LINK e XRP)
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

ATR_P  = 14;  BB_P = 20;  BB_STD = 2.0;  EMA_P = 200

FEE   = 0.0005;  SLIP = 0.0002
RT    = (FEE + SLIP) * 2   # 0.14%

OUTPUTS_DIR = Path(r"C:\Users\ASUS\OneDrive\Documentos\Claudinho\Outputs")

PASS_AVG = 0.15;  PASS_PF = 1.30;  PASS_RDDR = 2.0;  PASS_N = 150

# Configs: (bb_s, bb_l, swing_vis, use_ema200, label)
# swing_vis = max bars ago a swing can be (None = sem limite)
CONFIGS = [
    (None, None,  200, False, "VIS200  EMAoff"),
    (None, None,  200, True,  "VIS200  EMA200"),   # alvo principal
    (None, None,  500, True,  "VIS500  EMA200"),
    (None, None, None, True,  "VISinf  EMA200"),
    (0.70, 0.30,  200, True,  "BB70    EMA200"),
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
    prev  = df["close"].shift(1)
    tr    = pd.concat([(df["high"]-df["low"]),
                       (df["high"]-prev).abs(),
                       (df["low"]-prev).abs()], axis=1).max(axis=1)
    df["atr"]   = tr.ewm(com=ATR_P-1, adjust=False).mean()
    mid = ohlc4.rolling(BB_P).mean()
    std = ohlc4.rolling(BB_P).std()
    df["bb"]    = (ohlc4 - (mid - BB_STD*std)) / (2*BB_STD*std).replace(0, np.nan)
    df["ema200"] = df["close"].ewm(span=EMA_P, adjust=False).mean()
    return df


# --- SWING PRECOMPUTATION ----------------------------------------------------

def precompute_swings(df, lb):
    """Detecta swings e retorna arrays forward-filled.
    SWING_VIS_LIMIT e aplicado no momento de usar os swings (em detect()),
    nao aqui — isso permite reusar o mesmo precompute com diferentes vis limits.
    """
    n = len(df); H = df["high"].values; L = df["low"].values
    sh_raw = np.full(n, np.nan); sl_raw = np.full(n, np.nan)
    for i in range(lb, n - lb):
        if H[i] == H[i-lb:i+lb+1].max(): sh_raw[i] = H[i]
        if L[i] == L[i-lb:i+lb+1].min(): sl_raw[i] = L[i]

    # Forward-fill: last_sh[i] = swing high mais recente confirmado ate bar i
    last_sh = np.full(n, np.nan); last_sl = np.full(n, np.nan)
    sh_times = np.full(n, -1, dtype=int);  sl_times = np.full(n, -1, dtype=int)
    cur_sh = cur_sl = np.nan; cur_sh_t = cur_sl_t = -1
    for i in range(n):
        j = i - lb
        if j >= 0:
            if not np.isnan(sh_raw[j]): cur_sh = sh_raw[j]; cur_sh_t = j
            if not np.isnan(sl_raw[j]): cur_sl = sl_raw[j]; cur_sl_t = j
        last_sh[i] = cur_sh;    last_sl[i] = cur_sl
        sh_times[i] = cur_sh_t; sl_times[i] = cur_sl_t

    sh_pts = [(j, float(sh_raw[j])) for j in range(n) if not np.isnan(sh_raw[j])]
    sl_pts = [(j, float(sl_raw[j])) for j in range(n) if not np.isnan(sl_raw[j])]
    return (last_sh, last_sl, sh_times, sl_times,
            sh_pts, sl_pts, [p[0] for p in sh_pts], [p[0] for p in sl_pts])

def _sl_below_vis(sl_pts, sl_idx, vis_idx, ceil, swing_vis):
    """Acha swing low abaixo de ceil, limitado a swings nao mais antigos que swing_vis barras."""
    pos = bisect.bisect_right(sl_idx, vis_idx) - 1
    for k in range(pos, -1, -1):
        if swing_vis is not None and (vis_idx - sl_pts[k][0]) > swing_vis:
            break  # muito antigo
        if sl_pts[k][1] < ceil: return sl_pts[k][1]
    return None

def _sh_above_vis(sh_pts, sh_idx, vis_idx, floor, swing_vis):
    pos = bisect.bisect_right(sh_idx, vis_idx) - 1
    for k in range(pos, -1, -1):
        if swing_vis is not None and (vis_idx - sh_pts[k][0]) > swing_vis:
            break
        if sh_pts[k][1] > floor: return sh_pts[k][1]
    return None


# --- SIGNAL DETECTION --------------------------------------------------------

def detect(data, i, bb_s, bb_l, swing_vis, use_ema200):
    df = data["df"]; last = df.iloc[i]
    cl   = float(last["close"])
    atr  = float(last["atr"])
    bb   = float(last["bb"])
    ema  = float(last["ema200"])
    if np.isnan(atr) or np.isnan(bb) or np.isnan(ema): return None

    last_sh  = data["last_sh"];  last_sl  = data["last_sl"]
    sh_times = data["sh_times"]; sl_times = data["sl_times"]
    sh_pts   = data["sh_pts"];   sl_pts   = data["sl_pts"]
    sh_idx   = data["sh_idx"];   sl_idx   = data["sl_idx"]
    vis = i - SWING_LB   # swings confirmados ate este indice

    # -- SHORT: pavio swepa swing high, corpo fecha abaixo ------------------
    sw_h_t = sh_times[i]   # quando foi confirmado o ultimo swing high
    sw_h   = float(last_sh[i]) if not np.isnan(last_sh[i]) else None

    # Limita visibilidade do swing high
    if sw_h is not None and swing_vis is not None and (i - sw_h_t) > swing_vis:
        sw_h = None   # swing muito antigo, ignorar

    if sw_h is not None and float(last["high"]) > sw_h and cl < sw_h:
        bb_ok    = (bb > bb_s) if bb_s is not None else True
        trend_ok = (cl < ema) if use_ema200 else True   # SHORT abaixo da EMA200
        if bb_ok and trend_ok:
            sl = float(last["high"]) + STOP_BUF * atr; risk = sl - cl
            if risk > 0 and risk >= FLOOR_ATR * atr:
                tp = max(
                    _sl_below_vis(sl_pts, sl_idx, vis, cl, swing_vis) or cl - risk*RR_MIN,
                    cl - RR_CAP * risk
                )
                rr = (cl - tp) / risk
                if rr >= RR_MIN:
                    return {"side":"short","entry":cl,"sl":sl,"tp":tp,"rr":rr,"atr":atr}

    # -- LONG: pavio swepa swing low, corpo fecha acima ---------------------
    sw_l_t = sl_times[i]
    sw_l   = float(last_sl[i]) if not np.isnan(last_sl[i]) else None

    if sw_l is not None and swing_vis is not None and (i - sw_l_t) > swing_vis:
        sw_l = None

    if sw_l is not None and float(last["low"]) < sw_l and cl > sw_l:
        bb_ok    = (bb < bb_l) if bb_l is not None else True
        trend_ok = (cl > ema) if use_ema200 else True   # LONG acima da EMA200
        if bb_ok and trend_ok:
            sl = float(last["low"]) - STOP_BUF * atr; risk = cl - sl
            if risk > 0 and risk >= FLOOR_ATR * atr:
                tp = min(
                    _sh_above_vis(sh_pts, sh_idx, vis, cl, swing_vis) or cl + risk*RR_MIN,
                    cl + RR_CAP * risk
                )
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

WARMUP = EMA_P + SWING_LB * 2 + max(ATR_P, BB_P) + 5   # ~245 barras (EMA200 domina)

def run_pair(data, pair, bb_s, bb_l, swing_vis, use_ema200):
    df = data["df"]; trades = []; last_bar = -9999
    for i in range(WARMUP, len(df) - 1):
        if i - last_bar < COOLDOWN: continue
        sig = detect(data, i, bb_s, bb_l, swing_vis, use_ema200)
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
    if not trades: return []
    ts_list = [pd.Timestamp(t["ts"]) for t in trades]
    t0 = min(ts_list); out = []
    for w in range(n_windows):
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
    active_ws = [w for w in ws if w["n"] > 0]
    pos_w = sum(1 for w in active_ws if w["avg"] > 0)
    ok = (m["avg"] >= PASS_AVG and m["pf"] >= PASS_PF and
          m["rddr"] >= PASS_RDDR and m["n"] >= PASS_N and m["no_best"] > 0 and
          pos_w >= len(active_ws) * 0.6)   # 60%+ das janelas com trades devem ser positivas
    return " [OK]" if ok else (" [~]" if m["avg"] >= PASS_AVG and m["pf"] >= PASS_PF else "")

def print_table(results):
    hdr = (f"  {'Config':<20} {'N':>4} {'W%':>5} {'GrAvg':>7} {'FeeR':>5} "
           f"{'Avg':>7} {'PF':>5} {'R/DD':>5} {'NoBest':>7} {'W+/Wtot'}")
    print(); print("  " + "="*80)
    print("  SNIPER v6 -- 1H | SFatr1.0 | RC3 | BE0.5 | Trail=ON | 180d | 8p")
    print(f"  {DAYS}d | RT={RT*100:.2f}% | Alvo: avg>={PASS_AVG} PF>={PASS_PF} R/DD>={PASS_RDDR} N>={PASS_N}")
    print("  " + "="*80)
    print(hdr); print("  " + "-"*80)
    for r in sorted(results, key=lambda x: x["m"]["tot"], reverse=True):
        m = r["m"]; ws = r["ws"]; tag = _tag(m, ws)
        active = [w for w in ws if w["n"] > 0]
        pos_w  = sum(1 for w in active if w["avg"] > 0)
        print(f"  {r['lbl']:<20} {m['n']:>4} {m['wr']:>4.0f}% "
              f"{m['gr_avg']:>+7.3f} {m['fee_avg']:>5.3f} "
              f"{m['avg']:>+7.3f} {m['pf']:>5.2f} {m['rddr']:>5.1f} "
              f"{m['no_best']:>+7.1f}  {pos_w}/{len(active)}{tag}")
    print("  " + "-"*80)
    print("  W+/Wtot = janelas com trades que foram positivas")

def print_deep(trades, ws, label):
    m  = calc(trades)
    bd = breakdown(trades)
    print(f"\n  ===== {label} =====")
    print(f"  {m['n']} trades (L:{m['longs']} S:{m['shorts']}) | "
          f"gross {m['gr_avg']:+.3f}R | fee {m['fee_avg']:.3f}R | net {m['avg']:+.3f}R")
    print(f"  PF {m['pf']:.2f} | DD {m['dd']:.1f}R | R/DD {m['rddr']:.1f} | "
          f"Melhor: {m['best']:+.2f}R | Pior: {m['worst']:+.2f}R | NoBest: {m['no_best']:+.2f}R")
    print(f"  {m['tpd']:.2f} trades/dia\n")
    print(f"  {'Janela':<23} {'N':>4} {'W%':>5} {'GrAvg':>7} {'Avg':>7} {'Tot':>7} Status")
    for w in ws:
        if w["n"] == 0:
            st = "  . sem trades"
        elif w["avg"] > 0:
            st = "  + POSITIVO"
        else:
            st = "  - NEGATIVO"
        print(f"  {w['w']:<23} {w['n']:>4} {w['wr']:>4.0f}% "
              f"{w['gr_avg']:>+7.3f} {w['avg']:>+7.3f} {w['tot']:>+7.1f}{st}")
    active = [w for w in ws if w["n"] > 0]
    pos_w  = sum(1 for w in active if w["avg"] > 0)
    if active:
        print(f"\n  -> {pos_w}/{len(active)} janelas positivas ({pos_w/len(active)*100:.0f}%)")
    n_pos = sum(1 for p in bd if p["tot"] > 0)
    print(f"  -> {n_pos}/{len(bd)} pares positivos\n")
    print(f"  {'Par':<6} {'N':>4} {'W%':>5} {'Avg':>7} {'Tot':>7}")
    for p in bd:
        print(f"  {p['sym']:<6} {p['n']:>4} {p['wr']:>4.0f}% {p['avg']:>+7.3f} {p['tot']:>+7.1f}  "
              f"{'+ ' if p['tot']>0 else '- '}")


# --- MAIN --------------------------------------------------------------------

def main():
    print(f"\n  +--------------------------------------------------+")
    print(f"  |  Sniper v6 -- SWING_VIS + EMA200 (180d, 8p)    |")
    print(f"  +--------------------------------------------------+\n")
    print(f"  WARMUP = {WARMUP} barras (~{WARMUP/24:.0f} dias) para EMA200 estabilizar\n")

    ex = get_exchange()
    print(f"  [1H] Fetch {DAYS}d x {len(PAIRS_8)} pares...", end="", flush=True)
    raw_data = {}
    for sym in PAIRS_8:
        try:
            df = fetch_ohlcv(ex, sym, DAYS)
            df = add_indicators(df)
            out = precompute_swings(df, SWING_LB)
            raw_data[sym] = {"df":df,"last_sh":out[0],"last_sl":out[1],
                             "sh_times":out[2],"sl_times":out[3],
                             "sh_pts":out[4],"sl_pts":out[5],"sh_idx":out[6],"sl_idx":out[7]}
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n  ! {sym}: {e}", file=sys.stderr)
    print(" OK\n")

    print(f"  Rodando {len(CONFIGS)} configs...")
    results = []
    for (bb_s, bb_l, swing_vis, use_ema200, lbl) in CONFIGS:
        print(f"    {lbl:<22}", end="", flush=True)
        trades = []
        for sym, d in raw_data.items():
            trades.extend(run_pair(d, sym, bb_s, bb_l, swing_vis, use_ema200))
        m  = calc(trades)
        ws = windows_n(trades, 6)
        tag = _tag(m, ws)
        results.append({"lbl":lbl,"m":m,"ws":ws,"trades":trades})
        active = [w for w in ws if w["n"] > 0]
        pos_w  = sum(1 for w in active if w["avg"] > 0)
        print(f"-> {m['n']:>4}t  net={m['avg']:>+.3f}  PF={m['pf']:.2f}  "
              f"R/DD={m['rddr']:.1f}  {pos_w}/{len(active)} jan+{tag}")

    print_table(results)
    ok_rs = [r for r in results if _tag(r["m"],r["ws"]) in (" [OK]", " [~]")]
    for r in (ok_rs if ok_rs else sorted(results, key=lambda x: x["m"]["tot"], reverse=True)[:3]):
        print_deep(r["trades"], r["ws"], r["lbl"])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    fp = OUTPUTS_DIR / f"{today}_sniper_backtest_v6.json"
    fp.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": DAYS, "rt_pct": RT*100, "warmup_bars": WARMUP,
        "fixed": {"tf":"1h","rr_cap":RR_CAP,"be_r":BE_R,"trail_atr":TRAIL_ATR,
                  "floor_atr":FLOOR_ATR,"stop_buf":STOP_BUF,"swing_lb":SWING_LB,
                  "cooldown":COOLDOWN,"ema_p":EMA_P},
        "results": [{"lbl":r["lbl"],"metrics":r["m"],
                     "windows":[dict(w) for w in r["ws"]],
                     "breakdown":breakdown(r["trades"])} for r in results],
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  Salvo em: {fp}\n")


if __name__ == "__main__":
    main()
