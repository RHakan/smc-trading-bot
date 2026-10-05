"""
backtest_exit_sweep.py
Sweep das variaveis de saida mantendo entrada fixa (ATR48H, configuracao actual).

Variaveis testadas:
  BE_TRIGGER : quando move stop para breakeven (em R ganho)  [0.5, 0.6, 0.8, 1.0, 1.2]
  TRAIL_ATR  : multiplicador do trailing stop apos BE        [1.0, 1.5, 2.0, 2.5]
  RR_CAP     : take profit maximo                            [2.0, 2.5, 3.0, 4.0, 5.0]

Estrategia de teste: um sweep de cada variavel isolada, mantendo as outras fixas.
Depois combinacao das melhores de cada grupo.
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
LOW_LOOKBACK=4; ATR_STOP=1.0; ATR_AVG=48; COOLDOWN=8
REGIME_EMA=20; SLOPE_BARS=5; SLOPE_THRESH=0.001

# Valores base (configuracao actual)
BASE_BE      = 0.8
BASE_TRAIL   = 1.5
BASE_RR_CAP  = 3.0

mlabels = {'2026-01':'Jan','2026-02':'Fev','2026-03':'Mar',
           '2026-04':'Abr','2026-05':'Mai','2026-06':'Jun'}

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
data = {}
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
    df1['atr']     = tr.ewm(com=13, adjust=False).mean()
    df1['atr_avg'] = df1['atr'].rolling(ATR_AVG).mean()
    df1['low_n']   = df1['l'].shift(1).rolling(LOW_LOOKBACK).min()
    df1['regime']  = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
    data[sym] = df1
    time.sleep(0.1)
print(f'OK\n')

# ── Simulacao com parametros de saida variaveis ───────────────────────────────
def sim_short(df, ib, entry, sl_price, rr_cap, be_trigger, trail_atr):
    risk  = sl_price - entry
    if risk <= 0: return 0.0
    tp    = entry - rr_cap * risk
    fee_r = entry * RT / risk
    cur   = sl_price; be = False
    for j in range(ib+1, min(ib+400, len(df))):
        bar = df.iloc[j]
        lo=float(bar['l']); hi=float(bar['h']); c=float(bar['c'])
        atr=float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if hi >= cur: return (entry-cur)/risk - fee_r
        if lo <= tp:  return rr_cap - fee_r
        if not be and (entry-c)/risk >= be_trigger:
            cur = entry; be = True
        if be:
            cand = c + trail_atr*atr
            if cand < cur: cur = cand
    return (entry - float(df.iloc[min(ib+399,len(df)-1)]['c']))/risk - fee_r

def run(be_trigger, trail_atr, rr_cap):
    trades = []
    for sym, df1 in data.items():
        last_bar = -9999
        start_i  = max(ATR_AVG+10,
                       next((i for i,ts in enumerate(df1.index) if ts>=START), ATR_AVG+10))
        for i in range(start_i, len(df1)-1):
            if i - last_bar < COOLDOWN: continue
            ts_bar = df1.index[i]
            if ts_bar < START: continue
            r = df1.iloc[i]
            if str(r['regime']) != 'BEAR': continue
            cl=float(r['c']); op=float(r['o'])
            atr=float(r['atr']); atr_avg=float(r['atr_avg'])
            low_n=float(r['low_n'])
            if any(np.isnan(v) for v in [cl,atr,atr_avg,low_n]): continue
            if not (cl < low_n and cl < op and atr > atr_avg): continue
            sl_p = cl + ATR_STOP*atr
            if sl_p <= cl: continue
            nr = sim_short(df1, i, cl, sl_p, rr_cap, be_trigger, trail_atr)
            trades.append({'ts': ts_bar, 'nr': nr})
            last_bar = i
    return trades

def summarize(trades):
    if not trades: return None
    trades.sort(key=lambda t: t['ts'])
    capital=100.0; months={}
    for t in trades:
        mk=t['ts'].strftime('%Y-%m'); pnl=t['nr']*capital*0.01; capital+=pnl
        if mk not in months:
            months[mk]={'cnt':0,'wins':0,'nr_sum':0.0,'start':capital-pnl}
        months[mk]['end_cap']=capital; months[mk]['cnt']+=1
        if t['nr']>0: months[mk]['wins']+=1
        months[mk]['nr_sum']+=t['nr']
    nr_all=[t['nr'] for t in trades]
    tw=sum(1 for r in nr_all if r>0)
    gw=sum(r for r in nr_all if r>0); gl=abs(sum(r for r in nr_all if r<0))
    return {
        't':len(nr_all), 'wr':tw/len(nr_all)*100,
        'avgr':sum(nr_all)/len(nr_all), 'pf':gw/gl if gl>0 else 99,
        'total':(capital/100-1)*100, 'cap':capital, 'months':months,
        'tw':tw, 'tl':len(nr_all)-tw,
    }

def month_pct(s, mk):
    if mk not in s['months']: return '  n/a '
    m=s['months'][mk]; pct=(m['end_cap']/m['start']-1)*100
    return f"{pct:+.1f}%"

def print_table(rows, title):
    print(f'\n{"="*100}')
    print(f'  {title}')
    print('='*100)
    print(f"{'Config':<22} | {'T':>5} | {'WR':>6} | {'AvgR':>7} | {'PF':>5} | {'Jan':>7} | {'Fev':>7} | {'Mar':>7} | {'Abr':>7} | {'Mai':>7} | {'Jun':>7} | {'TOTAL':>7}")
    print('-'*100)
    for label, s, is_base in rows:
        if s is None: continue
        mks=['2026-01','2026-02','2026-03','2026-04','2026-05','2026-06']
        mpcts=' | '.join(f"{month_pct(s,mk):>7}" for mk in mks)
        mark='*' if is_base else ' '
        print(f"  {label:<20}{mark}| {s['t']:>5} | {s['wr']:>5.1f}% | {s['avgr']:>+7.3f}R | {s['pf']:>5.2f} | {mpcts} | {s['total']:>+6.1f}%")
    print('* = configuracao actual')

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 1: BE_TRIGGER (quando move stop para breakeven)
# ─────────────────────────────────────────────────────────────────────────────
print('Sweep 1/3: BE_TRIGGER...')
be_values = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
rows_be = []
for be in be_values:
    t = run(be_trigger=be, trail_atr=BASE_TRAIL, rr_cap=BASE_RR_CAP)
    s = summarize(t)
    is_base = (be == BASE_BE)
    rows_be.append((f"BE={be}R", s, is_base))
    mark='*' if is_base else ' '
    print(f"  BE={be}R{mark}: {s['t']} t | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 2: TRAIL_ATR (trailing apos breakeven)
# ─────────────────────────────────────────────────────────────────────────────
print('\nSweep 2/3: TRAIL_ATR...')
trail_values = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
rows_trail = []
for trail in trail_values:
    t = run(be_trigger=BASE_BE, trail_atr=trail, rr_cap=BASE_RR_CAP)
    s = summarize(t)
    is_base = (trail == BASE_TRAIL)
    rows_trail.append((f"TRAIL={trail}xATR", s, is_base))
    mark='*' if is_base else ' '
    print(f"  TRAIL={trail}xATR{mark}: {s['t']} t | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 3: RR_CAP (take profit maximo)
# ─────────────────────────────────────────────────────────────────────────────
print('\nSweep 3/3: RR_CAP...')
rr_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
rows_rr = []
for rr in rr_values:
    t = run(be_trigger=BASE_BE, trail_atr=BASE_TRAIL, rr_cap=rr)
    s = summarize(t)
    is_base = (rr == BASE_RR_CAP)
    rows_rr.append((f"RR_CAP={rr}R", s, is_base))
    mark='*' if is_base else ' '
    print(f"  RR_CAP={rr}R{mark}: {s['t']} t | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# COMBINACOES: melhor de cada sweep
# ─────────────────────────────────────────────────────────────────────────────
print('\nCombinacoes das melhores configuracoes...')
combos = [
    # (label,              be,   trail, rr)
    ('BASE (actual)',      0.8,  1.5,   3.0),
    ('BE0.6 T1.0 RR3',    0.6,  1.0,   3.0),
    ('BE0.6 T1.5 RR4',    0.6,  1.5,   4.0),
    ('BE0.6 T1.5 RR5',    0.6,  1.5,   5.0),
    ('BE1.0 T1.5 RR3',    1.0,  1.5,   3.0),
    ('BE1.0 T2.0 RR3',    1.0,  2.0,   3.0),
    ('BE1.0 T2.0 RR4',    1.0,  2.0,   4.0),
    ('BE0.8 T1.0 RR3',    0.8,  1.0,   3.0),
    ('BE0.8 T2.0 RR3',    0.8,  2.0,   3.0),
    ('BE0.8 T2.0 RR4',    0.8,  2.0,   4.0),
]
rows_combo = []
for label, be, trail, rr in combos:
    t = run(be_trigger=be, trail_atr=trail, rr_cap=rr)
    s = summarize(t)
    is_base = (label == 'BASE (actual)')
    rows_combo.append((label, s, is_base))
    mark='*' if is_base else ' '
    print(f"  {label:<22}{mark}: {s['t']} t | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# TABELAS DETALHADAS
# ─────────────────────────────────────────────────────────────────────────────
print_table(rows_be,    'SWEEP BE_TRIGGER — quando move stop para breakeven')
print_table(rows_trail, 'SWEEP TRAIL_ATR  — trailing stop apos breakeven')
print_table(rows_rr,    'SWEEP RR_CAP     — take profit maximo')
print_table(rows_combo, 'COMBINACOES MELHORES')
