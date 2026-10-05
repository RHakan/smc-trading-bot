"""
backtest_atr_sweep.py
Testa diferentes janelas de media do ATR: 24H, 30H, 36H, 42H, 48H (referencia)
Tudo o mais identico ao breakout_short.py
"""
import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 24, tzinfo=timezone.utc)

PAIRS = [
    'BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','BNB/USDT:USDT',
    'ADA/USDT:USDT','AVAX/USDT:USDT','DOGE/USDT:USDT','DOT/USDT:USDT',
    'XRP/USDT:USDT','LINK/USDT:USDT',
]

FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
LOW_LOOKBACK=4; ATR_STOP=1.0; RR_CAP=3.0; COOLDOWN=8
REGIME_EMA=20; SLOPE_BARS=5; SLOPE_THRESH=0.001
BE_TRIGGER=0.8; TRAIL_ATR=1.5

ATR_WINDOWS = [24, 30, 36, 42, 48]   # janelas a testar

mlabels = {
    '2026-01':'Jan','2026-02':'Fev','2026-03':'Mar',
    '2026-04':'Abr','2026-05':'Mai','2026-06':'Jun',
}

ex = ccxt.binanceusdm({'enableRateLimit': True})

def fetch(sym, tf, extra):
    since  = int((START - timedelta(days=extra)).timestamp()*1000)
    end_ms = int(END.timestamp()*1000)
    rows   = []
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b: break
        rows.extend(b); since = b[-1][0]+1
        if since >= end_ms or len(b)==0: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=['ts','o','h','l','c','v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()[lambda d: d.index < END]

print('Carregando dados...')
raw = {}
for sym in PAIRS:
    dfd = fetch(sym, '1d', 30).copy()
    dfd['ema']    = dfd['c'].ewm(span=REGIME_EMA, adjust=False).mean()
    dfd['slope']  = (dfd['ema'] - dfd['ema'].shift(SLOPE_BARS)) / dfd['ema'].shift(SLOPE_BARS)
    dfd['regime'] = 'NEUTRAL'
    dfd.loc[(dfd['c'] < dfd['ema']) & (dfd['slope'] < -SLOPE_THRESH), 'regime'] = 'BEAR'
    dfd.loc[(dfd['c'] > dfd['ema']) & (dfd['slope'] >  SLOPE_THRESH), 'regime'] = 'BULL'

    df1 = fetch(sym, '1h', 5).copy()
    tr  = pd.concat([
        (df1['h']-df1['l']),
        (df1['h']-df1['c'].shift(1)).abs(),
        (df1['l']-df1['c'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df1['atr']   = tr.ewm(com=13, adjust=False).mean()
    df1['low_n'] = df1['l'].shift(1).rolling(LOW_LOOKBACK).min()
    df1['regime']= dfd['regime'].shift(1).reindex(df1.index, method='ffill')

    # Pre-calcula todas as medias de ATR de uma vez
    for w in ATR_WINDOWS:
        df1[f'atr_avg{w}'] = df1['atr'].rolling(w).mean()

    raw[sym] = df1
    time.sleep(0.1)
print(f'OK\n')

def sim_short(df, ib, entry, sl_price):
    risk  = sl_price - entry
    if risk <= 0: return 0.0
    tp    = entry - RR_CAP * risk
    fee_r = entry * RT / risk
    cur   = sl_price; be = False
    for j in range(ib+1, min(ib+300, len(df))):
        bar = df.iloc[j]
        lo=float(bar['l']); hi=float(bar['h']); c=float(bar['c'])
        atr=float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if hi >= cur: return (entry-cur)/risk - fee_r
        if lo <= tp:  return RR_CAP - fee_r
        if not be and (entry-c)/risk >= BE_TRIGGER:
            cur = entry; be = True
        if be:
            cand = c + TRAIL_ATR*atr
            if cand < cur: cur = cand
    return (entry - float(df.iloc[min(ib+299,len(df)-1)]['c']))/risk - fee_r

def run(atr_window):
    col = f'atr_avg{atr_window}'
    trades = []
    for sym, df1 in raw.items():
        last_bar = -9999
        start_i  = max(atr_window+10,
                       next((i for i,ts in enumerate(df1.index) if ts>=START), atr_window+10))
        for i in range(start_i, len(df1)-1):
            if i - last_bar < COOLDOWN: continue
            ts_bar = df1.index[i]
            if ts_bar < START: continue
            r = df1.iloc[i]
            if str(r['regime']) != 'BEAR': continue
            cl=float(r['c']); op=float(r['o'])
            atr=float(r['atr']); atr_avg=float(r[col])
            low_n=float(r['low_n'])
            if any(np.isnan(v) for v in [cl,atr,atr_avg,low_n]): continue
            if cl < low_n and cl < op and atr > atr_avg:
                sl_p = cl + ATR_STOP*atr
                if sl_p > cl:
                    nr = sim_short(df1, i, cl, sl_p)
                    trades.append({'ts': ts_bar, 'nr': nr})
                    last_bar = i
    return trades

# Executa e reporta
print(f"{'Janela':>8} | {'Trades':>6} | {'WR':>6} | {'Avg R':>7} | {'PF':>5} | {'Jan':>7} | {'Fev':>7} | {'Mar':>7} | {'Abr':>7} | {'Mai':>7} | {'Jun':>7} | {'TOTAL':>8}")
print('-'*105)

for w in ATR_WINDOWS:
    trades = run(w)
    if not trades:
        print(f"{w:>5}H ATR | sem trades"); continue

    trades.sort(key=lambda t: t['ts'])
    capital = 100.0; months = {}
    for t in trades:
        mk  = t['ts'].strftime('%Y-%m')
        pnl = t['nr'] * capital * 0.01
        capital += pnl
        if mk not in months:
            months[mk] = {'cnt':0,'wins':0,'nr_sum':0.0,'start':capital-pnl}
        months[mk]['end_cap'] = capital
        months[mk]['cnt']    += 1
        if t['nr']>0: months[mk]['wins'] += 1
        months[mk]['nr_sum'] += t['nr']

    nr_all = [t['nr'] for t in trades]
    tw = sum(1 for r in nr_all if r>0)
    gw = sum(r for r in nr_all if r>0)
    gl = abs(sum(r for r in nr_all if r<0))
    pf = gw/gl if gl>0 else 99.0
    wr = tw/len(nr_all)*100
    avg_r = sum(nr_all)/len(nr_all)

    month_pcts = []
    for mk in ['2026-01','2026-02','2026-03','2026-04','2026-05','2026-06']:
        if mk in months:
            m = months[mk]
            pct = (m['end_cap']/m['start']-1)*100
            month_pcts.append(f"{pct:+.1f}%")
        else:
            month_pcts.append("  n/a ")

    label = f"{w}H {'*' if w==48 else ' '}"
    total_pct = (capital/100-1)*100
    print(f"{label:>8} | {len(nr_all):>6} | {wr:>5.1f}% | {avg_r:>+7.3f}R | {pf:>5.2f} | "
          + " | ".join(f"{p:>7}" for p in month_pcts)
          + f" | {total_pct:>+7.1f}%")

print('\n* = referencia atual (48H)')
