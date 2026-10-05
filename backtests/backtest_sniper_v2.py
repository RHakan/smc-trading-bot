#!/usr/bin/env python3
"""
backtest_sniper_v2.py -- Sniper Backtest com filtros corrigidos

Correções vs v1:
  1. Piso de stop: rejeita trades com risco < K x taxa_round_trip
  2. Filtro de volume: removido (ablação v1 provou que piora)
  3. Teto no alvo: TP limitado a NxR para evitar alvos ilusórios
  4. BE trigger em R (não % da distância até TP)
  5. 1H em paralelo (stop naturalmente maior, menos ruído de taxa)

Parâmetros FIXOS: use_bb=True, use_vol=False, use_trail=True

Ablação 15m: stop_floor x rr_cap x be_trigger = 3x3x2 = 18 configs
Ablação 1H:  stop_floor x rr_cap (best be_r de 15m)  = 3x3 =  9 configs

Critério de aprovação:
  avg_r >= +0.15  |  PF >= 1.30  |  R/DD >= 2.0  |  N >= 100  |  TotalR_sem_melhor > 0
"""

import bisect
import json
import sys
import time
from datetime import datetime, timezone, timedelta

# Força UTF-8 no stdout para evitar erros cp1252 no terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

# --- CONFIG ------------------------------------------------------------------

PAIRS = [
    "BTC/USDT:USDT",  "ETH/USDT:USDT",  "SOL/USDT:USDT",  "BNB/USDT:USDT",
    "XRP/USDT:USDT",  "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT",
    "DOT/USDT:USDT",  "LINK/USDT:USDT",
]

DAYS         = 90
TF_15M       = "15m"
TF_1H        = "1h"

# Parâmetros fixos
SWING_LB     = 10       # barras de cada lado para confirmar swing
COOLDOWN_15  = 12       # barras entre trades em 15m (12 x 15m = 3h)
COOLDOWN_1H  = 3        # barras entre trades em 1h  ( 3 x  1h = 3h)
RR_MIN       = 2.0
STOP_BUF     = 0.25     # buffer ATR além do extremo do pavio
BB_SHORT     = 0.85
BB_LONG      = 0.15
TRAIL_ATR    = 1.5

# Indicadores
ATR_P  = 14;  BB_P = 20;  BB_STD = 2.0

# Custos round-trip
FEE   = 0.0005
SLIP  = 0.0002
RT    = (FEE + SLIP) * 2   # 0.14%

# Piso de stop
FLOOR_K   = 10    # fee_k -> stop >= 10 x RT x entry ~= 1.4%
FLOOR_ATR = 1.0   # atr -> stop >= 1.0 x ATR

OUTPUTS_DIR = Path(r"C:\Users\ASUS\OneDrive\Documentos\Claudinho\Outputs")

# Critérios de aprovação para print
PASS_AVG_R = 0.15;  PASS_PF = 1.30;  PASS_RDDR = 2.0;  PASS_N = 100


# --- EXCHANGE ----------------------------------------------------------------

def get_exchange():
    return ccxt.binanceusdm({"enableRateLimit": True})

def fetch_ohlcv(ex, sym: str, tf: str, days: int) -> pd.DataFrame:
    bars_day = 96 if tf == "15m" else 24
    limit    = days * bars_day + 300
    since    = int((datetime.now(timezone.utc) - timedelta(days=days + 2)).timestamp() * 1000)
    rows = []
    while len(rows) < limit:
        batch = ex.fetch_ohlcv(sym, tf, since=since, limit=1500)
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
    ohlc4   = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    prev    = df["close"].shift(1)
    tr      = pd.concat([(df["high"]-df["low"]),
                         (df["high"]-prev).abs(),
                         (df["low"]-prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_P-1, adjust=False).mean()
    mid = ohlc4.rolling(BB_P).mean()
    std = ohlc4.rolling(BB_P).std()
    df["bb"]  = (ohlc4 - (mid - BB_STD*std)) / (2*BB_STD*std).replace(0, np.nan)
    return df


# --- SWING PRECOMPUTATION ----------------------------------------------------

def precompute_swings(df: pd.DataFrame, lb: int):
    """
    Pré-calcula tudo sobre swings uma vez por par.
    Retorna estruturas otimizadas para lookup O(log n) no loop principal.
    """
    n     = len(df)
    H     = df["high"].values
    L     = df["low"].values

    sh_raw = np.full(n, np.nan)   # swing high confirmado naquele bar
    sl_raw = np.full(n, np.nan)

    for i in range(lb, n - lb):
        wh = H[i-lb: i+lb+1]; wl = L[i-lb: i+lb+1]
        if H[i] == wh.max(): sh_raw[i] = H[i]
        if L[i] == wl.min(): sl_raw[i] = L[i]

    # Forward-fill: último swing visível em cada bar
    # Swing em j torna-se visível em bar i quando j + lb <= i (-> j = i - lb)
    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    cur_sh = cur_sl = np.nan
    for i in range(n):
        j = i - lb
        if j >= 0:
            if not np.isnan(sh_raw[j]): cur_sh = sh_raw[j]
            if not np.isnan(sl_raw[j]): cur_sl = sl_raw[j]
        last_sh[i] = cur_sh
        last_sl[i] = cur_sl

    # Listas ordenadas para busca retroativa por preço (TP detection)
    sh_pts   = [(j, float(sh_raw[j])) for j in range(n) if not np.isnan(sh_raw[j])]
    sl_pts   = [(j, float(sl_raw[j])) for j in range(n) if not np.isnan(sl_raw[j])]
    sh_idx   = [p[0] for p in sh_pts]   # índices (ordenados) -- para bisect
    sl_idx   = [p[0] for p in sl_pts]

    return last_sh, last_sl, sh_pts, sl_pts, sh_idx, sl_idx


def _find_sl_below(sl_pts, sl_idx, vis_limit: int, ceiling: float):
    """Swing low mais recente abaixo de ceiling, confirmado até vis_limit."""
    pos = bisect.bisect_right(sl_idx, vis_limit) - 1
    for k in range(pos, -1, -1):
        if sl_pts[k][1] < ceiling:
            return sl_pts[k][1]
    return None

def _find_sh_above(sh_pts, sh_idx, vis_limit: int, floor: float):
    """Swing high mais recente acima de floor, confirmado até vis_limit."""
    pos = bisect.bisect_right(sh_idx, vis_limit) - 1
    for k in range(pos, -1, -1):
        if sh_pts[k][1] > floor:
            return sh_pts[k][1]
    return None


# --- FILTROS -----------------------------------------------------------------

def passes_floor(entry: float, risk: float, atr: float, mode: str) -> bool:
    if mode == "none":    return True
    if mode == "fee_k":   return risk >= FLOOR_K * RT * entry
    if mode == "atr":     return risk >= FLOOR_ATR * atr
    return True

def apply_rr_cap(entry: float, risk: float, tp: float, side: str, cap) -> float:
    if cap is None: return tp
    if side == "short":
        return max(tp, entry - cap * risk)   # mais próximo = menos otimista
    else:
        return min(tp, entry + cap * risk)


# --- SIGNAL DETECTION --------------------------------------------------------

def detect_signal(data: dict, i: int, floor: str, rr_cap) -> dict | None:
    df     = data["df"]
    last   = df.iloc[i]
    close  = float(last["close"])
    atr    = float(last["atr"])
    bb     = float(last["bb"])
    if np.isnan(atr) or np.isnan(bb): return None

    last_sh = data["last_sh"];  last_sl = data["last_sl"]
    sh_pts  = data["sh_pts"];   sl_pts  = data["sl_pts"]
    sh_idx  = data["sh_idx"];   sl_idx  = data["sl_idx"]
    vis     = i - SWING_LB     # último índice de swing visível aqui

    # -- SHORT ------------------------------------------------------------
    sw_h = float(last_sh[i]) if not np.isnan(last_sh[i]) else None
    if sw_h is not None:
        wick_swept     = float(last["high"]) > sw_h
        body_reclaimed = close < sw_h
        if wick_swept and body_reclaimed and bb > BB_SHORT:
            sl   = float(last["high"]) + STOP_BUF * atr
            risk = sl - close
            if risk > 0 and passes_floor(close, risk, atr, floor):
                tp_s = _find_sl_below(sl_pts, sl_idx, vis, close)
                tp   = apply_rr_cap(close, risk, tp_s if tp_s else close - risk * RR_MIN,
                                    "short", rr_cap)
                rr   = (close - tp) / risk
                if rr >= RR_MIN:
                    return {"side":"short","entry":close,"sl":sl,"tp":tp,"rr":rr,"atr":atr}

    # -- LONG -------------------------------------------------------------
    sw_l = float(last_sl[i]) if not np.isnan(last_sl[i]) else None
    if sw_l is not None:
        wick_swept     = float(last["low"]) < sw_l
        body_reclaimed = close > sw_l
        if wick_swept and body_reclaimed and bb < BB_LONG:
            sl   = float(last["low"]) - STOP_BUF * atr
            risk = close - sl
            if risk > 0 and passes_floor(close, risk, atr, floor):
                tp_s = _find_sh_above(sh_pts, sh_idx, vis, close)
                tp   = apply_rr_cap(close, risk, tp_s if tp_s else close + risk * RR_MIN,
                                    "long", rr_cap)
                rr   = (tp - close) / risk
                if rr >= RR_MIN:
                    return {"side":"long","entry":close,"sl":sl,"tp":tp,"rr":rr,"atr":atr}
    return None


# --- TRADE SIMULATION --------------------------------------------------------

def simulate_trade(df: pd.DataFrame, entry_bar: int, sig: dict, be_r: float) -> dict:
    """
    Trailing: breakeven quando o trade acumula be_r x risk de ganho
    (ex: be_r=0.5 -> BE ao ganhar 0.5R; be_r=1.0 -> BE ao ganhar 1R).
    """
    entry = sig["entry"]; sl = sig["sl"]; tp = sig["tp"]
    side  = sig["side"];  atr0 = sig["atr"]
    risk  = abs(entry - sl)
    fee_r = entry * RT / risk

    cur_stop = sl;  be_hit = False;  partial = False

    for j in range(entry_bar + 1, len(df)):
        bar  = df.iloc[j]
        high = float(bar["high"]); low = float(bar["low"]); cl = float(bar["close"])
        atr  = float(bar["atr"]) if not np.isnan(bar["atr"]) else atr0

        # Stop check
        hit = (high >= cur_stop) if side == "short" else (low <= cur_stop)
        if hit:
            gr = (entry - cur_stop) / risk if side == "short" else (cur_stop - entry) / risk
            return {"outcome": "WIN" if (side=="short" and cur_stop < entry) or
                                        (side=="long"  and cur_stop > entry) else "LOSS",
                    "exit_bar": j, "exit_price": cur_stop,
                    "gross_r": gr, "net_r": gr - fee_r, "hit_be": be_hit}

        if side == "short":
            r_gain = (entry - cl) / risk

            if not partial and low <= tp:             # Fase 3: partial close
                partial = True
                locked  = entry - 2.0 * risk
                if cur_stop > locked: cur_stop = locked

            if be_hit:                                # Fase 2: trailing
                cand = cl + TRAIL_ATR * atr
                if cand < cur_stop: cur_stop = cand

            if not be_hit and r_gain >= be_r:         # Fase 1: breakeven
                if entry < cur_stop: cur_stop = entry
                be_hit = True
        else:
            r_gain = (cl - entry) / risk

            if not partial and high >= tp:
                partial = True
                locked  = entry + 2.0 * risk
                if cur_stop < locked: cur_stop = locked

            if be_hit:
                cand = cl - TRAIL_ATR * atr
                if cand > cur_stop: cur_stop = cand

            if not be_hit and r_gain >= be_r:
                if entry > cur_stop: cur_stop = entry
                be_hit = True

    # Trade aberto no fim dos dados
    lc = float(df.iloc[-1]["close"])
    gr = (entry - lc) / risk if side == "short" else (lc - entry) / risk
    return {"outcome":"OPEN","exit_bar":len(df)-1,"exit_price":lc,
            "gross_r":gr,"net_r":gr - fee_r,"hit_be":be_hit}


# --- BACKTEST LOOP -----------------------------------------------------------

WARMUP = SWING_LB * 2 + max(ATR_P, BB_P) + 5

def run_pair(data: dict, pair: str, cooldown: int,
              floor: str, rr_cap, be_r: float) -> list[dict]:
    df       = data["df"]
    trades   = []
    last_bar = -9999

    for i in range(WARMUP, len(df) - 1):
        if i - last_bar < cooldown: continue
        sig = detect_signal(data, i, floor, rr_cap)
        if sig is None: continue
        res = simulate_trade(df, i, sig, be_r)
        trades.append({
            "pair": pair, "ts": str(df.index[i]),
            "side": sig["side"],
            "entry": round(sig["entry"], 6),
            "sl":    round(sig["sl"], 6),
            "tp":    round(sig["tp"], 6),
            "rr_t":  round(sig["rr"], 2),
            "outcome":    res["outcome"],
            "exit_price": round(res["exit_price"], 6),
            "gross_r":    round(res["gross_r"], 3),
            "net_r":      round(res["net_r"],   3),
            "hit_be":     res["hit_be"],
        })
        last_bar = i
    return trades


# --- MÉTRICAS ----------------------------------------------------------------

def calc(trades: list[dict]) -> dict:
    closed  = [t for t in trades if t["outcome"] != "OPEN"]
    wins    = [t for t in closed if t["outcome"] == "WIN"]
    net_rs  = [t["net_r"] for t in closed]
    total_r = sum(net_rs)
    n       = len(trades)
    nc      = len(closed)

    gw  = sum(r for r in net_rs if r > 0)
    gl  = abs(sum(r for r in net_rs if r < 0))
    pf  = gw / gl if gl > 0 else 99.0

    cum  = np.cumsum(net_rs) if net_rs else np.array([0.0])
    pk   = np.maximum.accumulate(cum)
    dd   = float(np.max(pk - cum)) if len(cum) else 0.0
    rddr = round(total_r / dd, 2) if dd > 0 else 0.0

    be_pct = sum(1 for t in trades if t.get("hit_be")) / n * 100 if n else 0.0

    # TotalR sem o melhor trade (robustez)
    no_best = sum(sorted(net_rs)[:-1]) if len(net_rs) > 1 else total_r

    return {
        "n": n, "wins": len(wins), "losses": nc-len(wins),
        "open": n - nc,
        "wr":  round(len(wins)/nc*100, 1) if nc else 0.0,
        "avg": round(total_r/nc, 3)        if nc else 0.0,
        "tot": round(total_r, 2),
        "pf":  round(pf, 2),
        "dd":  round(dd, 2),
        "rddr": rddr,
        "tpd": round(n / DAYS, 2),
        "be":  round(be_pct, 1),
        "best": round(max(net_rs), 2) if net_rs else 0.0,
        "worst":round(min(net_rs), 2) if net_rs else 0.0,
        "no_best": round(no_best, 2),
        "longs":  sum(1 for t in trades if t["side"]=="long"),
        "shorts": sum(1 for t in trades if t["side"]=="short"),
    }


def windows(trades: list[dict]) -> list[dict]:
    if not trades: return []
    ts   = [pd.Timestamp(t["ts"]) for t in trades]
    t0   = min(ts)
    out  = []
    for w in range(3):
        s = t0 + timedelta(days=w*30); e = s + timedelta(days=30)
        wt = [t for t, ts_ in zip(trades, ts) if s <= ts_ < e]
        m  = calc(wt)
        out.append({"w": f"W{w+1} {s.date()}->{e.date()}", **m})
    return out

def breakdown(trades: list[dict]) -> list[dict]:
    bd = []
    for sym in sorted(set(t["pair"] for t in trades)):
        pt = [t for t in trades if t["pair"]==sym]
        m  = calc(pt)
        bd.append({"sym": sym.replace("/USDT:USDT",""), **m})
    return sorted(bd, key=lambda x: x["tot"], reverse=True)


# --- DISPLAY -----------------------------------------------------------------

def _tag(m: dict) -> str:
    ok = (m["avg"] >= PASS_AVG_R and m["pf"] >= PASS_PF and
          m["rddr"] >= PASS_RDDR  and m["n"]  >= PASS_N  and m["no_best"] > 0)
    return " [OK]" if ok else ""

def _lbl(sf, rc, be):
    s = {"none":"SFoff","fee_k":"SF14p","atr":"SFatr"}[sf]
    r = f"RC{int(rc)}" if rc else "RCinf"
    b = f"BE{be}"
    return f"{s} {r} {b}"

def print_ablation(results: list[dict], title: str):
    W  = 100
    hd = (f"  {'Config':<18} {'N':>4} {'W%':>5} {'Avg':>6} {'Tot':>7} "
          f"{'PF':>5} {'DD':>5} {'R/DD':>5} {'T/d':>4} {'NoBest':>7} {'BE%':>5}")
    sep = "  " + "-" * (W-2)
    print(); print("  " + "=" * (W-2)); print(f"  {title}")
    print(f"  BB=ON Vol=OFF Trail=ON | {DAYS}d | {len(PAIRS)} pares | {RT*100:.2f}% RT")
    print("  " + "="*(W-2)); print(hd); print(sep)
    for r in sorted(results, key=lambda x: x["m"]["tot"], reverse=True):
        m = r["m"]; tag = _tag(m)
        print(f"  {r['lbl']:<18} {m['n']:>4} {m['wr']:>4.0f}% "
              f"{m['avg']:>+6.3f} {m['tot']:>+7.1f} {m['pf']:>5.2f} "
              f"{m['dd']:>5.1f} {m['rddr']:>5.1f} {m['tpd']:>4.2f} "
              f"{m['no_best']:>+7.1f} {m['be']:>4.0f}%{tag}")
    print(sep)
    print(f"  [OK] avg>={PASS_AVG_R} PF>={PASS_PF} R/DD>={PASS_RDDR} N>={PASS_N} NoBest>0")

def print_deep(trades: list[dict], label: str):
    m  = calc(trades)
    ws = windows(trades)
    bd = breakdown(trades)
    print(f"\n  -- {label} --")
    print(f"  {m['n']} trades (L:{m['longs']} S:{m['shorts']}) | "
          f"avg_r {m['avg']:+.3f} | PF {m['pf']:.2f} | "
          f"MaxDD {m['dd']:.1f}R | R/DD {m['rddr']:.1f}")
    print(f"  Melhor: {m['best']:+.2f}R | Pior: {m['worst']:+.2f}R | "
          f"Sem melhor: {m['no_best']:+.2f}R")
    if ws:
        print(f"  {'Janela':<27} {'N':>4} {'W%':>5} {'Avg':>6} {'Tot':>7}")
        for w in ws:
            print(f"  {w['w']:<27} {w['n']:>4} {w['wr']:>4.0f}% "
                  f"{w['avg']:>+6.3f} {w['tot']:>+7.1f}")
    n_pos = sum(1 for p in bd if p["tot"] > 0)
    print(f"  Pares positivos: {n_pos}/{len(bd)}")
    print(f"  {'Par':<7} {'N':>4} {'W%':>5} {'Avg':>6} {'Tot':>7}")
    for p in bd:
        marker = " +" if p["tot"] > 0 else " -"
        print(f"  {p['sym']:<7} {p['n']:>4} {p['wr']:>4.0f}% "
              f"{p['avg']:>+6.3f} {p['tot']:>+7.1f}{marker}")


# --- MAIN --------------------------------------------------------------------

def main():
    print("\n  +-------------------------------------------+")
    print("  |  Sniper Backtest v2 - filtros corrigidos  |")
    print("  +-------------------------------------------+\n")

    ex = get_exchange()

    # -- 1. Fetch + preparação de dados -----------------------------------
    def fetch_prepare(tf, cooldown_label):
        dfs = {}
        print(f"  [{tf}] Fetch {DAYS}d x {len(PAIRS)} pares...", end="", flush=True)
        for sym in PAIRS:
            try:
                df = fetch_ohlcv(ex, sym, tf, DAYS)
                df = add_indicators(df)
                lsh, lsl, shp, slp, shi, sli = precompute_swings(df, SWING_LB)
                dfs[sym] = {"df":df,"last_sh":lsh,"last_sl":lsl,
                            "sh_pts":shp,"sl_pts":slp,"sh_idx":shi,"sl_idx":sli}
                print(".", end="", flush=True)
            except Exception as e:
                print(f"\n  ⚠ {sym}: {e}", file=sys.stderr)
        print(" OK")
        return dfs

    data_15 = fetch_prepare(TF_15M, "15m")
    data_1h = fetch_prepare(TF_1H,  "1h")
    print()

    # -- 2. Ablação 15m ---------------------------------------------------
    floors   = ["none", "fee_k", "atr"]
    rr_caps  = [None, 3.0, 4.0]
    be_rs    = [0.5, 1.0]
    combos15 = [(f,r,b) for f in floors for r in rr_caps for b in be_rs]

    print(f"  [15m] {len(combos15)} configs x {len(data_15)} pares...")
    res_15 = []
    for sf, rc, be in combos15:
        lbl = _lbl(sf, rc, be)
        print(f"    {lbl:<22}", end="", flush=True)
        trades = []
        for sym, d in data_15.items():
            trades.extend(run_pair(d, sym, COOLDOWN_15, sf, rc, be))
        m = calc(trades)
        res_15.append({"lbl":lbl,"sf":sf,"rc":rc,"be":be,"m":m,"trades":trades})
        print(f"-> {m['n']:>4}t  avg={m['avg']:>+.3f}  PF={m['pf']:.2f}  R/DD={m['rddr']:.1f}")

    # -- 3. Ablação 1H (best be_r de 15m) ---------------------------------
    be_avg = {be: sum(r["m"]["tot"] for r in res_15 if r["be"]==be)/
                  max(sum(1 for r in res_15 if r["be"]==be),1) for be in be_rs}
    best_be = max(be_avg, key=be_avg.get)
    print(f"\n  [1H] {len(floors)*len(rr_caps)} configs x {len(data_1h)} pares (be_r={best_be})...")
    res_1h = []
    for sf, rc in [(f,r) for f in floors for r in rr_caps]:
        sf_s = {"none":"SFoff","fee_k":"SF14p","atr":"SFatr"}[sf]
        rc_s = f"RC{int(rc)}" if rc else "RCinf"
        lbl  = f"{sf_s} {rc_s}"
        print(f"    {lbl:<16}", end="", flush=True)
        trades = []
        for sym, d in data_1h.items():
            trades.extend(run_pair(d, sym, COOLDOWN_1H, sf, rc, best_be))
        m = calc(trades)
        res_1h.append({"lbl":lbl,"sf":sf,"rc":rc,"be":best_be,"m":m,"trades":trades})
        print(f"-> {m['n']:>4}t  avg={m['avg']:>+.3f}  PF={m['pf']:.2f}  R/DD={m['rddr']:.1f}")

    # -- 4. Output ---------------------------------------------------------
    print_ablation(res_15, "ABLAÇÃO 15m -- Sniper v2")
    print_ablation(res_1h, "ABLAÇÃO 1H  -- Sniper v2")

    # Deep analysis da melhor config 15m (score: avg_r + PF)
    def score(r):
        m = r["m"]
        if m["n"] < 50: return -99
        return m["avg"] + min(m["pf"], 4.0) * 0.3 + (0.5 if m["no_best"] > 0 else 0)

    best15 = max(res_15, key=score)
    best1h = max(res_1h, key=lambda r: r["m"]["tot"] if r["m"]["n"] >= 20 else -99)

    print_deep(best15["trades"], f"MELHOR 15m: {best15['lbl']}")
    print_deep(best1h["trades"], f"MELHOR 1H:  {best1h['lbl']}")

    # -- 5. Salvar JSON ----------------------------------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    fp    = OUTPUTS_DIR / f"{today}_sniper_backtest_v2.json"
    fp.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": DAYS, "pairs": PAIRS, "rt_pct": RT*100,
        "fixed": {"bb":True,"vol":False,"trail":True,"rr_min":RR_MIN,
                  "stop_buf":STOP_BUF,"trail_atr":TRAIL_ATR},
        "best_be_r": best_be,
        "results_15m": [{"lbl":r["lbl"],"sf":r["sf"],"rc":r["rc"],
                          "be":r["be"],"metrics":r["m"]} for r in res_15],
        "results_1h":  [{"lbl":r["lbl"],"sf":r["sf"],"rc":r["rc"],
                          "be":r["be"],"metrics":r["m"]} for r in res_1h],
        "best_15m": {"lbl":best15["lbl"],"metrics":best15["m"],
                     "windows":windows(best15["trades"]),
                     "breakdown":breakdown(best15["trades"]),
                     "trades":best15["trades"]},
        "best_1h":  {"lbl":best1h["lbl"],"metrics":best1h["m"],
                     "windows":windows(best1h["trades"]),
                     "breakdown":breakdown(best1h["trades"]),
                     "trades":best1h["trades"]},
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  [OK] Salvo em: {fp}\n")


if __name__ == "__main__":
    main()
