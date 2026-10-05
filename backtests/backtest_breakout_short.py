"""
backtest_breakout_short.py
Replica exata da estrategia em bot/strategies/breakout_short.py
Parametros importados diretamente do modulo para garantir consistencia.
"""
import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from bot.strategies.breakout_short import (
    REGIME_EMA_PERIOD, REGIME_SLOPE_BARS, REGIME_SLOPE_THRESH,
    LOW_LOOKBACK, ATR_PERIOD, ATR_AVG_PERIOD,
    ATR_STOP_MULT, RR_CAP, COOLDOWN_BARS,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 24, tzinfo=timezone.utc)

PAIRS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
    'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOGE/USDT:USDT', 'DOT/USDT:USDT',
    'XRP/USDT:USDT', 'LINK/USDT:USDT',
]

# Fees USDT (conta atual) — RT = (maker 0.02% + taker 0.05%) x2 + slippage
FEE  = 0.0005
SLIP = 0.0002
RT   = (FEE + SLIP) * 2   # 0.0014 round-trip

BE_TRIGGER = 0.8
TRAIL_ATR  = 1.5

mlabels = {
    '2026-01': 'Jan', '2026-02': 'Fev', '2026-03': 'Mar',
    '2026-04': 'Abr', '2026-05': 'Mai', '2026-06': 'Jun',
}

ex = ccxt.binanceusdm({'enableRateLimit': True})

def fetch(sym, tf, extra_days):
    since  = int((START - timedelta(days=extra_days)).timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    rows   = []
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b: break
        rows.extend(b)
        since = b[-1][0] + 1
        if since >= end_ms or len(b) == 0: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=['ts','o','h','l','c','v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()[lambda d: d.index < END]

print('Carregando dados (1H + diario)...')
data = {}
for sym in PAIRS:
    # Regime diario (mesma logica de _calc_regime no breakout_short.py)
    dfd = fetch(sym, '1d', 30).copy()
    dfd['ema']    = dfd['c'].ewm(span=REGIME_EMA_PERIOD, adjust=False).mean()
    dfd['slope']  = (dfd['ema'] - dfd['ema'].shift(REGIME_SLOPE_BARS)) / dfd['ema'].shift(REGIME_SLOPE_BARS)
    dfd['regime'] = 'NEUTRAL'
    dfd.loc[(dfd['c'] > dfd['ema']) & (dfd['slope'] >  REGIME_SLOPE_THRESH), 'regime'] = 'BULL'
    dfd.loc[(dfd['c'] < dfd['ema']) & (dfd['slope'] < -REGIME_SLOPE_THRESH), 'regime'] = 'BEAR'

    # 1H — indicadores identicos ao breakout_short.py
    df1 = fetch(sym, '1h', 5).copy()
    tr  = pd.concat([
        (df1['h'] - df1['l']),
        (df1['h'] - df1['c'].shift(1)).abs(),
        (df1['l'] - df1['c'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df1['atr']     = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
    df1['atr_avg'] = df1['atr'].rolling(ATR_AVG_PERIOD).mean()
    df1['low_n']   = df1['l'].shift(1).rolling(LOW_LOOKBACK).min()   # mínimo das N velas anteriores

    # shift(1) no regime: ontem define hoje (sem lookahead)
    df1['regime'] = dfd['regime'].shift(1).reindex(df1.index, method='ffill')

    data[sym] = df1
    time.sleep(0.1)

print(f'OK - {len(data)} pares\n')

def sim_short(df, ib, entry, sl_price):
    risk  = sl_price - entry
    if risk <= 0: return 0.0
    tp    = entry - RR_CAP * risk
    fee_r = entry * RT / risk
    cur   = sl_price
    be    = False
    for j in range(ib + 1, min(ib + 300, len(df))):
        bar = df.iloc[j]
        lo  = float(bar['l']); hi = float(bar['h']); c = float(bar['c'])
        atr = float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if hi >= cur: return (entry - cur) / risk - fee_r
        if lo <= tp:  return RR_CAP - fee_r
        if not be and (entry - c) / risk >= BE_TRIGGER:
            cur = entry; be = True
        if be:
            cand = c + TRAIL_ATR * atr
            if cand < cur: cur = cand
    return (entry - float(df.iloc[min(ib + 299, len(df) - 1)]['c'])) / risk - fee_r

# Coleta de trades
all_trades = []
for sym, df1 in data.items():
    last_bar = -9999
    start_i  = max(ATR_AVG_PERIOD + 10,
                   next((i for i, ts in enumerate(df1.index) if ts >= START), ATR_AVG_PERIOD + 10))

    for i in range(start_i, len(df1) - 1):
        if i - last_bar < COOLDOWN_BARS: continue
        ts_bar = df1.index[i]
        if ts_bar < START: continue

        r      = df1.iloc[i]
        regime = str(r['regime'])
        if regime != 'BEAR': continue

        cl     = float(r['c']); op  = float(r['o'])
        atr    = float(r['atr']); atr_avg = float(r['atr_avg'])
        low_n  = float(r['low_n'])

        if any(np.isnan(v) for v in [cl, atr, atr_avg, low_n]): continue

        # As 4 condicoes da estrategia (identicas ao breakout_short.py)
        if cl < low_n and cl < op and atr > atr_avg:
            sl_price = cl + ATR_STOP_MULT * atr
            if sl_price <= cl: continue
            nr = sim_short(df1, i, cl, sl_price)
            all_trades.append({'ts': ts_bar, 'sym': sym, 'nr': nr})
            last_bar = i

print(f'Total trades: {len(all_trades)}\n')

# Relatorio
all_trades.sort(key=lambda t: t['ts'])
capital = 100.0
months  = {}

for t in all_trades:
    mk  = t['ts'].strftime('%Y-%m')
    pnl = t['nr'] * capital * 0.01
    capital += pnl
    if mk not in months:
        months[mk] = {'cnt': 0, 'wins': 0, 'nr_sum': 0.0, 'start': capital - pnl}
    months[mk]['end_cap']  = capital
    months[mk]['cnt']     += 1
    if t['nr'] > 0: months[mk]['wins'] += 1
    months[mk]['nr_sum']  += t['nr']

print('Estrategia: Breakout SHORT 1H | Regime BEAR diario | ATR > media 48H')
print('Parametros: LOW_LOOKBACK=%d | ATR_STOP=%.1fx | RR_CAP=%.1f | COOLDOWN=%dH' % (
    LOW_LOOKBACK, ATR_STOP_MULT, RR_CAP, COOLDOWN_BARS))
print('Fees: RT=%.4f%% (USDT maker+taker+slip)' % (RT * 100))
print('=' * 60)

tw = tl = 0
for mk in sorted(months.keys()):
    m    = months[mk]
    lbl  = mlabels.get(mk, mk)
    pct  = (m['end_cap'] / m['start'] - 1) * 100
    wr   = m['wins'] / m['cnt'] * 100
    avgr = m['nr_sum'] / m['cnt']
    tw  += m['wins']
    tl  += m['cnt'] - m['wins']
    mark = '+' if pct >= 0 else ''
    print(f"{lbl}: {m['cnt']:4d} t | WR {wr:4.1f}% | avg {avgr:+.3f}R | {mark}{pct:.1f}% | ${m['end_cap']:.2f}")

nr_all = [t['nr'] for t in all_trades]
gw     = sum(r for r in nr_all if r > 0)
gl     = abs(sum(r for r in nr_all if r < 0))
pf     = gw / gl if gl > 0 else 99.0

print('=' * 60)
print(f"TOTAL : {len(nr_all)} trades | WR {tw/(tw+tl)*100:.1f}% | avg {sum(nr_all)/len(nr_all):+.3f}R | PF {pf:.2f}")
print(f"$100  -> ${capital:.2f}  ({(capital/100-1)*100:+.1f}%)")

# Breakdown por par
print('\nResultado por par:')
by_sym = {}
for t in all_trades:
    s = t['sym'].replace('/USDT:USDT','')
    if s not in by_sym: by_sym[s] = []
    by_sym[s].append(t['nr'])

for s, trades in sorted(by_sym.items()):
    wr  = sum(1 for r in trades if r > 0) / len(trades) * 100
    avg = sum(trades) / len(trades)
    print(f"  {s:<6}: {len(trades):3d} t | WR {wr:4.1f}% | avg {avg:+.3f}R")
