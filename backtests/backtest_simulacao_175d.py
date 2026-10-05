"""
backtest_simulacao_175d.py
Simulação: $100 USD em 01/Jan/2026, risco 1% por trade, composto mês a mês.
Estratégia: RSI28 BB005 Rev MOM (sem filtro EMA, sem filtro de tendência).
"""

import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────── Parâmetros ────────────────
START_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2026, 6, 24, tzinfo=timezone.utc)
DAYS = (END_DATE - START_DATE).days  # ~174 dias

TF = "1h"
RSI_P = 14; BB_P = 20; BB_STD = 2.0; ATR_P = 14
MOM_BARS = 4; MOM_ATR = 1.5; STOP_ATR = 1.0
BE_R = 0.8; TRAIL_ATR = 1.5; RR_CAP = 3.0
FEE = 0.0005; SLIP = 0.0002; RT = (FEE + SLIP) * 2
COOLDOWN = 8

RSI_LO = 28; BB_LO = 0.05
RSI_HI = 100 - RSI_LO; BB_HI = 1 - BB_LO

PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOGE/USDT:USDT", "DOT/USDT:USDT",
    "XRP/USDT:USDT", "LINK/USDT:USDT",
]

STARTING_CAPITAL = 100.0  # USD
RISK_PCT = 0.01            # 1% por trade

# ──────────────── Fetch ────────────────
ex = ccxt.binanceusdm({"enableRateLimit": True})
WARMUP = 200 + max(ATR_P, BB_P) + MOM_BARS + 5  # 229 bars

def fetch(sym):
    # Baixa desde (START_DATE - WARMUP horas) para ter warmup suficiente
    since_dt = START_DATE - timedelta(hours=WARMUP + 10)
    since = int(since_dt.timestamp() * 1000)
    end_ms = int(END_DATE.timestamp() * 1000)
    rows = []
    while True:
        b = ex.fetch_ohlcv(sym, TF, since=since, limit=1000)
        if not b:
            break
        rows.extend(b)
        since = b[-1][0] + 1
        if since >= end_ms or len(b) == 0:
            break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[df.index < END_DATE]
    return df

# ──────────────── Indicadores ────────────────
def add_indicators(df):
    df = df.copy()
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    prev = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]), (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_P - 1, adjust=False).mean()
    delta = ohlc4.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=RSI_P - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=RSI_P - 1, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    mid = ohlc4.rolling(BB_P).mean()
    std = ohlc4.rolling(BB_P).std()
    df["bbb"] = (ohlc4 - (mid - BB_STD * std)) / ((2 * BB_STD * std).replace(0, np.nan))
    df["mom"] = df["close"].diff(MOM_BARS)
    return df

# ──────────────── Detect ────────────────
def detect(df, i):
    r = df.iloc[i]
    rsi = float(r["rsi"]); bbb = float(r["bbb"]); atr = float(r["atr"])
    mom = float(r["mom"]); cl = float(r["close"]); op = float(r["open"])
    if any(np.isnan(v) for v in [rsi, bbb, atr, mom]):
        return None
    thr = MOM_ATR * atr
    if rsi < RSI_LO and bbb < BB_LO and mom > -thr and cl > op:
        sl = cl - STOP_ATR * atr; risk = cl - sl
        return {"side": "long", "entry": cl, "sl": sl, "tp": cl + RR_CAP * risk, "atr": atr} if risk > 0 else None
    if rsi > RSI_HI and bbb > BB_HI and mom < thr and cl < op:
        sl = cl + STOP_ATR * atr; risk = sl - cl
        return {"side": "short", "entry": cl, "sl": sl, "tp": cl - RR_CAP * risk, "atr": atr} if risk > 0 else None
    return None

# ──────────────── Simulate trade ────────────────
def simulate(df, ib, sig):
    e = sig["entry"]; sl = sig["sl"]; side = sig["side"]
    atr0 = sig["atr"]; risk = abs(e - sl)
    fee_r = e * RT / risk
    cur = sl; be = False; par = False
    for j in range(ib + 1, len(df)):
        bar = df.iloc[j]; h = float(bar["high"]); l = float(bar["low"]); c = float(bar["close"])
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else atr0
        ts_close = df.index[j]
        if (h >= cur) if side == "short" else (l <= cur):
            gr = (e - cur) / risk if side == "short" else (cur - e) / risk
            out = "WIN" if (side == "short" and cur < e) or (side == "long" and cur > e) else "LOSS"
            return {"out": out, "gr": gr, "nr": gr - fee_r, "fee_r": fee_r, "close_ts": ts_close}
        if side == "short":
            rg = (e - c) / risk
            if not par and l <= sig["tp"]:
                par = True
                locked = e - 2 * risk
                if cur > locked: cur = locked
            if be:
                cand = c + TRAIL_ATR * atr
                if cand < cur: cur = cand
            if not be and rg >= BE_R:
                if e < cur: cur = e
                be = True
        else:
            rg = (c - e) / risk
            if not par and h >= sig["tp"]:
                par = True
                locked = e + 2 * risk
                if cur < locked: cur = locked
            if be:
                cand = c - TRAIL_ATR * atr
                if cand > cur: cur = cand
            if not be and rg >= BE_R:
                if e > cur: cur = e
                be = True
    lc = float(df.iloc[-1]["close"])
    gr = (e - lc) / risk if side == "short" else (lc - e) / risk
    return {"out": "OPEN", "gr": gr, "nr": gr - fee_r, "fee_r": fee_r, "close_ts": END_DATE}

# ──────────────── Main ────────────────
if __name__ == "__main__":
    print(f"Baixando dados 1H ({DAYS} dias, {len(PAIRS)} pares)...", end="", flush=True)
    raw = {}
    for sym in PAIRS:
        raw[sym] = add_indicators(fetch(sym))
        print(".", end="", flush=True)
    print(" OK\n")

    # Encontra o índice do START_DATE em cada df
    all_trades = []
    for sym, df in raw.items():
        last_bar = -9999
        # Primeiro índice >= START_DATE, com warmup de 229 barras
        start_i = max(WARMUP, next((i for i, ts in enumerate(df.index) if ts >= START_DATE), WARMUP))
        for i in range(start_i, len(df) - 1):
            if i - last_bar < COOLDOWN:
                continue
            sig = detect(df, i)
            if not sig:
                continue
            res = simulate(df, i, sig)
            all_trades.append({
                "pair": sym,
                "open_ts": df.index[i],
                "side": sig["side"],
                **res,
            })
            last_bar = i

    # Ordena por data de abertura
    all_trades.sort(key=lambda t: t["open_ts"])

    print(f"Total de trades: {len(all_trades)}\n")

    # ──────────────── Simulação composta ────────────────
    capital = STARTING_CAPITAL
    months = {}

    for t in all_trades:
        if t["out"] == "OPEN":
            continue
        ts = t["open_ts"]
        month_key = ts.strftime("%Y-%m")
        risk_usd = capital * RISK_PCT
        pnl_usd = t["nr"] * risk_usd
        capital += pnl_usd
        months.setdefault(month_key, {"trades": [], "start_cap": None, "end_cap": None})
        if months[month_key]["start_cap"] is None:
            months[month_key]["start_cap"] = capital - pnl_usd
        months[month_key]["end_cap"] = capital
        months[month_key]["trades"].append({**t, "pnl_usd": pnl_usd, "cap_after": capital})

    # ──────────────── Relatório mês a mês ────────────────
    print("=" * 72)
    print(f"  SIMULAÇÃO: $100 USD | 1% risco/trade | {len(PAIRS)} pares | RSI28 BB005")
    print(f"  Período: 01/Jan/2026 → 24/Jun/2026")
    print("=" * 72)

    month_labels = {
        "2026-01": "Janeiro", "2026-02": "Fevereiro", "2026-03": "Março",
        "2026-04": "Abril",   "2026-05": "Maio",       "2026-06": "Junho (parcial)",
    }

    total_wins = 0; total_losses = 0
    for mk in sorted(months.keys()):
        m = months[mk]
        lbl = month_labels.get(mk, mk)
        trades = m["trades"]
        closed = [t for t in trades if t["out"] != "OPEN"]
        if not closed:
            print(f"\n  {lbl}: 0 trades fechados")
            continue
        wins = [t for t in closed if t["out"] == "WIN"]
        losses = [t for t in closed if t["out"] == "LOSS"]
        total_wins += len(wins); total_losses += len(losses)
        pnl_list = [t["pnl_usd"] for t in closed]
        pnl_total = sum(pnl_list)
        start_cap = m["start_cap"]
        end_cap = m["end_cap"]
        pct = (end_cap / start_cap - 1) * 100 if start_cap else 0
        nr_avg = sum(t["nr"] for t in closed) / len(closed)
        mark = "+" if pnl_total > 0 else ""

        print(f"\n  ── {lbl} ──")
        print(f"     Trades: {len(closed)} | Wins: {len(wins)} | Losses: {len(losses)} | WR: {len(wins)/len(closed)*100:.0f}%")
        print(f"     Avg R net: {nr_avg:+.3f}R")
        print(f"     Saldo inicio: ${start_cap:>8.2f}")
        print(f"     Saldo fim:    ${end_cap:>8.2f}  ({mark}{pct:.1f}%)")
        print(f"     P&L mês:      ${mark}{pnl_total:.2f}")

        # Detalhe dos trades
        for t in closed[:]:
            mark2 = "W" if t["out"] == "WIN" else "L"
            pair_short = t["pair"].split("/")[0]
            print(f"       [{mark2}] {str(t['open_ts'])[:16]} {pair_short:<5} {t['side']:<5} "
                  f"nr={t['nr']:+.3f}R  ${t['pnl_usd']:+.2f}  cap=${t['cap_after']:.2f}")

    # ──────────────── Resumo final ────────────────
    final_cap = capital
    total_return = (final_cap / STARTING_CAPITAL - 1) * 100
    wr = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) > 0 else 0
    closed_all = [t for t in all_trades if t["out"] != "OPEN"]
    nr_all = [t["nr"] for t in closed_all]
    gr_all = [t["gr"] for t in closed_all]
    gw = sum(r for r in nr_all if r > 0); gl = abs(sum(r for r in nr_all if r < 0))
    pf = gw / gl if gl > 0 else 99

    print(f"\n{'=' * 72}")
    print(f"  RESULTADO FINAL")
    print(f"  Capital inicial: $100.00")
    print(f"  Capital final:   ${final_cap:.2f}")
    print(f"  Retorno total:   {total_return:+.1f}%")
    print(f"  Trades:          {len(closed_all)} (W:{total_wins} L:{total_losses} WR:{wr:.1f}%)")
    print(f"  Avg R net:       {sum(nr_all)/len(nr_all):+.3f}R")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"{'=' * 72}\n")
