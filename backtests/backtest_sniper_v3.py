#!/usr/bin/env python3
"""
backtest_sniper_v3.py -- Sniper 1H: reducao de fee_r + regimes

Contexto (v2 resultado chave):
  1H SFatr RC3: gross_r = +0.249R, fee_r = -0.169R, net = +0.080R (68% comido por taxa)
  - fee_r = entry * RT / risk  ->  para reduzir fee, precisa de risco maior
  - SFatr=1.0 ATR => stop medio ~0.83% do preco => fee_r alto
  - BB 0.85/0.15 bloqueia maio-junho (mercado fora dos extremos de BB)

Objetivo v3:
  1. Reduzir fee_r aumentando FLOOR_ATR de 1.0 para 1.5/2.0
  2. Testar BB relaxado (0.70/0.30) para capturar mai/jun
  3. Testar BB=OFF para quantificar contribuicao do filtro
  4. Manter: SFatr, RC3, BE0.5, Trail=ON, 1H only

Expectativa:
  FLOOR_ATR=2.0 -> fee_r ~0.085R -> net ~0.164R (se gross se mantiver)
  BB relaxado   -> mais trades em mai/jun, win rate possivelmente menor

Parametros fixos: rr_cap=3.0, be_r=0.5, cooldown=3 bars, use_trail=True
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

PAIRS = [
    "BTC/USDT:USDT",  "ETH/USDT:USDT",  "SOL/USDT:USDT",  "BNB/USDT:USDT",
    "XRP/USDT:USDT",  "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT",
    "DOT/USDT:USDT",  "LINK/USDT:USDT",
]

DAYS       = 90
TF         = "1h"
SWING_LB   = 10
COOLDOWN   = 3        # 3 barras 1H = 3h entre trades por par
RR_MIN     = 2.0
STOP_BUF   = 0.25    # buffer ATR alem do extremo do pavio
RR_CAP     = 3.0     # teto do alvo em R (fixo neste teste)
BE_R       = 0.5     # trigger de breakeven em R ganhos (fixo)
TRAIL_ATR  = 1.5

ATR_P  = 14;  BB_P = 20;  BB_STD = 2.0

FEE   = 0.0005;  SLIP = 0.0002
RT    = (FEE + SLIP) * 2   # 0.14%

OUTPUTS_DIR = Path(r"C:\Users\ASUS\OneDrive\Documentos\Claudinho\Outputs")

# Criterios de aprovacao
PASS_AVG = 0.15;  PASS_PF = 1.30;  PASS_RDDR = 2.0;  PASS_N = 50

# Configs a testar: (bb_short, bb_long, floor_atr, label)
CONFIGS = [
    # Baseline v2
    (0.85, 0.15, 1.0, "BB85 ATR1.0  RC3"),
    # Aumentar floor para reduzir fee_r
    (0.85, 0.15, 1.5, "BB85 ATR1.5  RC3"),
    (0.85, 0.15, 2.0, "BB85 ATR2.0  RC3"),
    (0.85, 0.15, 2.5, "BB85 ATR2.5  RC3"),
    # BB relaxado (captura mais regimes)
    (0.70, 0.30, 1.0, "BB70 ATR1.0  RC3"),
    (0.70, 0.30, 1.5, "BB70 ATR1.5  RC3"),
    (0.70, 0.30, 2.0, "BB70 ATR2.0  RC3"),
    # BB desligado (linha de base sem filtro direcional)
    (None, None, 1.0, "BBoff ATR1.0 RC3"),
    (None, None, 1.5, "BBoff ATR1.5 RC3"),
    (None, None, 2.0, "BBoff ATR2.0 RC3"),
]


# --- EXCHANGE ----------------------------------------------------------------

def get_exchange():
    return ccxt.binanceusdm({"enableRateLimit": True})

def fetch_ohlcv(ex, sym: str, days: int) -> pd.DataFrame:
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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

def precompute_swings(df: pd.DataFrame, lb: int):
    n = len(df)
    H = df["high"].values;  L = df["low"].values
    sh_raw = np.full(n, np.nan);  sl_raw = np.full(n, np.nan)
    for i in range(lb, n - lb):
        wh = H[i-lb: i+lb+1]; wl = L[i-lb: i+lb+1]
        if H[i] == wh.max(): sh_raw[i] = H[i]
        if L[i] == wl.min(): sl_raw[i] = L[i]

    last_sh = np.full(n, np.nan);  last_sl = np.full(n, np.nan)
    cur_sh = cur_sl = np.nan
    for i in range(n):
        j = i - lb
        if j >= 0:
            if not np.isnan(sh_raw[j]): cur_sh = sh_raw[j]
            if not np.isnan(sl_raw[j]): cur_sl = sl_raw[j]
        last_sh[i] = cur_sh;  last_sl[i] = cur_sl

    sh_pts = [(j, float(sh_raw[j])) for j in range(n) if not np.isnan(sh_raw[j])]
    sl_pts = [(j, float(sl_raw[j])) for j in range(n) if not np.isnan(sl_raw[j])]
    sh_idx = [p[0] for p in sh_pts]
    sl_idx = [p[0] for p in sl_pts]
    return last_sh, last_sl, sh_pts, sl_pts, sh_idx, sl_idx


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

def detect(data, i, bb_s, bb_l, floor_atr) -> dict | None:
    df    = data["df"];  last = df.iloc[i]
    close = float(last["close"]);  atr = float(last["atr"]);  bb = float(last["bb"])
    if np.isnan(atr) or np.isnan(bb): return None

    last_sh = data["last_sh"];  last_sl = data["last_sl"]
    sh_pts  = data["sh_pts"];   sl_pts  = data["sl_pts"]
    sh_idx  = data["sh_idx"];   sl_idx  = data["sl_idx"]
    vis     = i - SWING_LB

    # -- SHORT: pavio swepa swing high, corpo fecha abaixo ----------------
    sw_h = float(last_sh[i]) if not np.isnan(last_sh[i]) else None
    if sw_h is not None:
        if float(last["high"]) > sw_h and close < sw_h:
            bb_ok = (bb > bb_s) if bb_s is not None else True
            if bb_ok:
                sl   = float(last["high"]) + STOP_BUF * atr
                risk = sl - close
                if risk > 0 and risk >= floor_atr * atr:
                    tp_s = _sl_below(sl_pts, sl_idx, vis, close)
                    tp_r = close - risk * RR_MIN   # fallback
                    tp_u = tp_s if tp_s else tp_r
                    tp   = max(tp_u, close - RR_CAP * risk)   # cap
                    rr   = (close - tp) / risk
                    if rr >= RR_MIN:
                        return {"side":"short","entry":close,"sl":sl,"tp":tp,"rr":rr,"atr":atr}

    # -- LONG: pavio swepa swing low, corpo fecha acima -------------------
    sw_l = float(last_sl[i]) if not np.isnan(last_sl[i]) else None
    if sw_l is not None:
        if float(last["low"]) < sw_l and close > sw_l:
            bb_ok = (bb < bb_l) if bb_l is not None else True
            if bb_ok:
                sl   = float(last["low"]) - STOP_BUF * atr
                risk = close - sl
                if risk > 0 and risk >= floor_atr * atr:
                    tp_s = _sh_above(sh_pts, sh_idx, vis, close)
                    tp_r = close + risk * RR_MIN
                    tp_u = tp_s if tp_s else tp_r
                    tp   = min(tp_u, close + RR_CAP * risk)
                    rr   = (tp - close) / risk
                    if rr >= RR_MIN:
                        return {"side":"long","entry":close,"sl":sl,"tp":tp,"rr":rr,"atr":atr}
    return None


# --- TRADE SIMULATION --------------------------------------------------------

def simulate(df, entry_bar, sig):
    entry = sig["entry"];  sl = sig["sl"];  tp = sig["tp"]
    side  = sig["side"];   atr0 = sig["atr"]
    risk  = abs(entry - sl)
    fee_r = entry * RT / risk

    cur = sl;  be = False;  par = False

    for j in range(entry_bar + 1, len(df)):
        bar  = df.iloc[j]
        h = float(bar["high"]);  l = float(bar["low"]);  c = float(bar["close"])
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else atr0

        hit = (h >= cur) if side == "short" else (l <= cur)
        if hit:
            gr = (entry-cur)/risk if side=="short" else (cur-entry)/risk
            return {"out":"WIN" if (side=="short" and cur<entry) or (side=="long" and cur>entry)
                    else "LOSS", "j":j, "ep":cur, "gr":gr, "nr":gr-fee_r, "be":be, "fee_r":fee_r}

        if side == "short":
            rg = (entry - c) / risk
            if not par and l <= tp:
                par = True
                locked = entry - 2.0 * risk
                if cur > locked: cur = locked
            if be:
                cand = c + TRAIL_ATR * atr
                if cand < cur: cur = cand
            if not be and rg >= BE_R:
                if entry < cur: cur = entry
                be = True
        else:
            rg = (c - entry) / risk
            if not par and h >= tp:
                par = True
                locked = entry + 2.0 * risk
                if cur < locked: cur = locked
            if be:
                cand = c - TRAIL_ATR * atr
                if cand > cur: cur = cand
            if not be and rg >= BE_R:
                if entry > cur: cur = entry
                be = True

    lc = float(df.iloc[-1]["close"])
    gr = (entry-lc)/risk if side=="short" else (lc-entry)/risk
    return {"out":"OPEN","j":len(df)-1,"ep":lc,"gr":gr,"nr":gr-fee_r,"be":be,"fee_r":fee_r}


# --- BACKTEST LOOP -----------------------------------------------------------

WARMUP = SWING_LB * 2 + max(ATR_P, BB_P) + 5

def run_pair(data, pair, bb_s, bb_l, floor_atr):
    df = data["df"];  trades = [];  last_bar = -9999
    for i in range(WARMUP, len(df) - 1):
        if i - last_bar < COOLDOWN: continue
        sig = detect(data, i, bb_s, bb_l, floor_atr)
        if sig is None: continue
        res = simulate(df, i, sig)
        trades.append({
            "pair": pair, "ts": str(df.index[i]),
            "side": sig["side"],
            "entry": round(sig["entry"],6), "sl": round(sig["sl"],6),
            "tp": round(sig["tp"],6), "rr_t": round(sig["rr"],2),
            "out": res["out"], "ep": round(res["ep"],6),
            "gr": round(res["gr"],3), "nr": round(res["nr"],3),
            "fee_r": round(res["fee_r"],3), "be": res["be"],
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

    tot  = sum(net_rs)
    avg  = tot / nc
    gr_avg  = sum(grs) / nc
    fee_avg = sum(fees) / n

    gw = sum(r for r in net_rs if r > 0)
    gl = abs(sum(r for r in net_rs if r < 0))
    pf = gw / gl if gl > 0 else 99.0

    cum = np.cumsum(net_rs); pk = np.maximum.accumulate(cum)
    dd  = float(np.max(pk - cum))
    rddr = round(tot/dd, 2) if dd > 0 else 0.0

    be_pct  = sum(1 for t in trades if t.get("be")) / n * 100
    no_best = sum(sorted(net_rs)[:-1]) if len(net_rs) > 1 else tot

    return {
        "n": n, "wr": round(len(wins)/nc*100,1),
        "avg": round(avg,3), "gr_avg": round(gr_avg,3), "fee_avg": round(fee_avg,3),
        "tot": round(tot,2), "pf": round(pf,2), "dd": round(dd,2), "rddr": rddr,
        "tpd": round(n/DAYS,2), "be": round(be_pct,1),
        "best": round(max(net_rs),2), "worst": round(min(net_rs),2),
        "no_best": round(no_best,2),
        "longs":  sum(1 for t in trades if t["side"]=="long"),
        "shorts": sum(1 for t in trades if t["side"]=="short"),
    }

def windows(trades):
    if not trades: return []
    ts = [pd.Timestamp(t["ts"]) for t in trades]
    t0 = min(ts);  out = []
    for w in range(3):
        s = t0 + timedelta(days=w*30); e = s + timedelta(days=30)
        wt = [t for t, ts_ in zip(trades, ts) if s <= ts_ < e]
        m  = calc(wt)
        out.append({"w": f"W{w+1} {s.date()}->{e.date()}", **m})
    return out

def breakdown(trades):
    bd = []
    for sym in sorted(set(t["pair"] for t in trades)):
        pt = [t for t in trades if t["pair"]==sym]
        m  = calc(pt)
        bd.append({"sym": sym.replace("/USDT:USDT",""), **m})
    return sorted(bd, key=lambda x: x["tot"], reverse=True)


# --- DISPLAY -----------------------------------------------------------------

def _tag(m):
    ok = (m["avg"] >= PASS_AVG and m["pf"] >= PASS_PF and
          m["rddr"] >= PASS_RDDR and m["n"] >= PASS_N and m["no_best"] > 0)
    return " [OK]" if ok else ""

def print_table(results):
    hdr = (f"  {'Config':<22} {'N':>4} {'W%':>5} {'GrAvg':>7} {'FeeR':>6} "
           f"{'Avg':>7} {'Tot':>7} {'PF':>5} {'DD':>5} {'R/DD':>5} {'NoBest':>7}")
    sep = "  " + "-" * (len(hdr)-2)
    print(); print("  " + "="*(len(hdr)-2))
    print(f"  SNIPER v3 -- 1H | SFatr | RC3 | BE0.5 | Trail=ON")
    print(f"  {DAYS}d | {len(PAIRS)} pares | RT={RT*100:.2f}% | Alvo: avg>={PASS_AVG} PF>={PASS_PF} R/DD>={PASS_RDDR}")
    print("  " + "="*(len(hdr)-2))
    print(hdr)
    print(sep)
    for r in sorted(results, key=lambda x: x["m"]["tot"], reverse=True):
        m = r["m"]; t = _tag(m)
        print(f"  {r['lbl']:<22} {m['n']:>4} {m['wr']:>4.0f}% "
              f"{m['gr_avg']:>+7.3f} {m['fee_avg']:>6.3f} "
              f"{m['avg']:>+7.3f} {m['tot']:>+7.1f} "
              f"{m['pf']:>5.2f} {m['dd']:>5.1f} {m['rddr']:>5.1f} "
              f"{m['no_best']:>+7.1f}{t}")
    print(sep)
    print("  Colunas: GrAvg=gross_r_medio  FeeR=taxa_media_por_trade  NoBest=tot_sem_melhor_trade")

def print_deep(trades, label):
    m  = calc(trades)
    ws = windows(trades)
    bd = breakdown(trades)
    print(f"\n  -- {label} --")
    print(f"  {m['n']} trades (L:{m['longs']} S:{m['shorts']}) | "
          f"gross {m['gr_avg']:+.3f}R | fee {m['fee_avg']:.3f}R | net {m['avg']:+.3f}R | "
          f"PF {m['pf']:.2f} | DD {m['dd']:.1f}R | R/DD {m['rddr']:.1f}")
    print(f"  Melhor: {m['best']:+.2f}R | Pior: {m['worst']:+.2f}R | NoBest: {m['no_best']:+.2f}R")
    if ws:
        print(f"  {'Janela':<27} {'N':>4} {'W%':>5} {'GrAvg':>7} {'Avg':>7} {'Tot':>7}")
        for w in ws:
            print(f"  {w['w']:<27} {w['n']:>4} {w['wr']:>4.0f}% "
                  f"{w['gr_avg']:>+7.3f} {w['avg']:>+7.3f} {w['tot']:>+7.1f}")
    n_pos = sum(1 for p in bd if p["tot"] > 0)
    print(f"  Pares positivos: {n_pos}/{len(bd)}")
    print(f"  {'Par':<6} {'N':>4} {'W%':>5} {'Avg':>7} {'Tot':>7}")
    for p in bd:
        mk = " +" if p["tot"] > 0 else " -"
        print(f"  {p['sym']:<6} {p['n']:>4} {p['wr']:>4.0f}% {p['avg']:>+7.3f} {p['tot']:>+7.1f}{mk}")


# --- MAIN --------------------------------------------------------------------

def main():
    print("\n  +--------------------------------------------------+")
    print("  |  Sniper v3 -- 1H | Reducao fee_r + regimes     |")
    print("  +--------------------------------------------------+\n")

    ex = get_exchange()

    # Fetch e prepara dados UMA VEZ
    print(f"  [1H] Fetch {DAYS}d x {len(PAIRS)} pares...", end="", flush=True)
    data = {}
    for sym in PAIRS:
        try:
            df = fetch_ohlcv(ex, sym, DAYS)
            df = add_indicators(df)
            lsh, lsl, shp, slp, shi, sli = precompute_swings(df, SWING_LB)
            data[sym] = {"df":df,"last_sh":lsh,"last_sl":lsl,
                         "sh_pts":shp,"sl_pts":slp,"sh_idx":shi,"sl_idx":sli}
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n  ! {sym}: {e}", file=sys.stderr)
    print(" OK\n")

    # Roda todas as configs
    print(f"  Rodando {len(CONFIGS)} configs x {len(data)} pares...")
    results = []
    for (bb_s, bb_l, floor_atr, lbl) in CONFIGS:
        print(f"    {lbl:<24}", end="", flush=True)
        trades = []
        for sym, d in data.items():
            trades.extend(run_pair(d, sym, bb_s, bb_l, floor_atr))
        m = calc(trades)
        results.append({"lbl":lbl,"bb_s":bb_s,"bb_l":bb_l,"fl":floor_atr,"m":m,"trades":trades})
        tag = _tag(m)
        print(f"-> {m['n']:>4}t  gross={m['gr_avg']:>+.3f}  fee={m['fee_avg']:.3f}  "
              f"net={m['avg']:>+.3f}  PF={m['pf']:.2f}  R/DD={m['rddr']:.1f}{tag}")

    print_table(results)

    # Deep analysis dos top 3 por tot
    top3 = sorted(results, key=lambda x: x["m"]["tot"], reverse=True)[:3]
    for r in top3:
        print_deep(r["trades"], f"{r['lbl']}")

    # Save
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    fp    = OUTPUTS_DIR / f"{today}_sniper_backtest_v3.json"
    fp.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": DAYS, "pairs": PAIRS, "rt_pct": RT*100,
        "fixed": {"tf":"1h","rr_cap":RR_CAP,"be_r":BE_R,"trail_atr":TRAIL_ATR,
                  "stop_buf":STOP_BUF,"swing_lb":SWING_LB,"cooldown":COOLDOWN},
        "results": [{"lbl":r["lbl"],"bb_s":r["bb_s"],"bb_l":r["bb_l"],
                     "fl":r["fl"],"metrics":r["m"]} for r in results],
        "top_trades": {r["lbl"]: r["trades"] for r in top3},
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  Salvo em: {fp}\n")


if __name__ == "__main__":
    main()
