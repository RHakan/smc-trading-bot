"""
backtest_diagnostico.py
Analisa em detalhe o que aconteceu nos meses negativos (Marco e Maio).
Objectivo: perceber o padrao dos perdedores antes de tentar filtros.

Metricas investigadas:
  - Distribuicao de resultados (ganhos vs perdas por tamanho)
  - Slope da EMA diaria no momento do sinal (forca da tendencia)
  - Candle diario no dia do sinal (o dia estava a subir ou a cair?)
  - Hora do sinal (existe padrao horario?)
  - Dias consecutivos em BEAR antes do sinal
  - ATR relativo (quao acima da media estava o ATR?)
  - Par que gerou o sinal
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
LOW_LOOKBACK=4; ATR_STOP=1.0; RR_CAP=4.0; COOLDOWN=8
REGIME_EMA=20; SLOPE_BARS=5; SLOPE_THRESH=0.001
BE_TRIGGER=1.0; TRAIL_ATR=2.0; ATR_AVG=48

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
data    = {}
data_1d = {}
for sym in PAIRS:
    dfd = fetch(sym, '1d', 30).copy()
    dfd['ema']    = dfd['c'].ewm(span=REGIME_EMA, adjust=False).mean()
    dfd['slope']  = (dfd['ema'] - dfd['ema'].shift(SLOPE_BARS)) / dfd['ema'].shift(SLOPE_BARS)
    dfd['regime'] = 'NEUTRAL'
    dfd.loc[(dfd['c'] < dfd['ema']) & (dfd['slope'] < -SLOPE_THRESH), 'regime'] = 'BEAR'
    dfd.loc[(dfd['c'] > dfd['ema']) & (dfd['slope'] >  SLOPE_THRESH), 'regime'] = 'BULL'
    # Dias consecutivos em BEAR
    bear_streak = []
    streak = 0
    for reg in dfd['regime']:
        streak = streak + 1 if reg == 'BEAR' else 0
        bear_streak.append(streak)
    dfd['bear_streak'] = bear_streak
    # Candle diario: bullish ou bearish
    dfd['daily_bull'] = dfd['c'] > dfd['o']
    data_1d[sym] = dfd

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
    df1['slope']   = dfd['slope'].shift(1).reindex(df1.index, method='ffill')
    df1['bear_streak'] = dfd['bear_streak'].shift(1).reindex(df1.index, method='ffill')
    df1['daily_bull']  = dfd['daily_bull'].shift(1).reindex(df1.index, method='ffill')
    data[sym] = df1
    time.sleep(0.1)
print('OK\n')

def sim_short_detail(df, ib, entry, sl_price):
    risk  = sl_price - entry
    if risk <= 0: return 0.0, 'invalid'
    tp    = entry - RR_CAP * risk
    fee_r = entry * RT / risk
    cur   = sl_price; be = False
    for j in range(ib+1, min(ib+400, len(df))):
        bar = df.iloc[j]
        lo=float(bar['l']); hi=float(bar['h']); c=float(bar['c'])
        atr=float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if hi >= cur:
            nr = (entry-cur)/risk - fee_r
            return nr, 'stop' if not be else 'trail'
        if lo <= tp:
            return RR_CAP - fee_r, 'tp'
        if not be and (entry-c)/risk >= BE_TRIGGER:
            cur = entry; be = True
        if be:
            cand = c + TRAIL_ATR*atr
            if cand < cur: cur = cand
    nr = (entry - float(df.iloc[min(ib+399,len(df)-1)]['c']))/risk - fee_r
    return nr, 'timeout'

# Coleta trades com metadados
all_trades = []
for sym, df1 in data.items():
    dfd = data_1d[sym]
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
        low_n=float(r['low_n']); slope=float(r['slope'])
        bear_streak=float(r['bear_streak']); daily_bull=bool(r['daily_bull'])
        if any(np.isnan(v) for v in [cl,atr,atr_avg,low_n,slope]): continue
        if not (cl < low_n and cl < op and atr > atr_avg): continue
        sl_p = cl + ATR_STOP*atr
        if sl_p <= cl: continue
        nr, exit_type = sim_short_detail(df1, i, cl, sl_p)
        all_trades.append({
            'ts': ts_bar,
            'sym': sym.replace('/USDT:USDT',''),
            'nr': nr,
            'slope': slope,
            'slope_pct': slope * 100,
            'bear_streak': int(bear_streak),
            'daily_bull': daily_bull,
            'atr_ratio': atr / atr_avg,   # quao acima da media esta o ATR
            'hour': ts_bar.hour,
            'exit': exit_type,
            'month': ts_bar.strftime('%Y-%m'),
        })
        last_bar = i

all_trades.sort(key=lambda t: t['ts'])
print(f'Total trades: {len(all_trades)}\n')

# ── Funcao auxiliar ───────────────────────────────────────────────────────────
def stats(trades):
    if not trades: return
    nr = [t['nr'] for t in trades]
    wins = [r for r in nr if r > 0]
    losses = [r for r in nr if r <= 0]
    wr = len(wins)/len(nr)*100
    avg = sum(nr)/len(nr)
    gw = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0
    pf = gw/gl if gl > 0 else 99
    return {'t':len(nr),'wr':wr,'avg':avg,'pf':pf,'wins':wins,'losses':losses}

def print_stats(s, indent='  '):
    if not s: return
    print(f"{indent}{s['t']} trades | WR {s['wr']:.1f}% | avg {s['avg']:+.3f}R | PF {s['pf']:.2f}")

# ── ANALISE POR MES ───────────────────────────────────────────────────────────
print('='*65)
print('RESUMO GERAL')
print('='*65)
for mk, label in [('2026-01','Jan'),('2026-02','Fev'),('2026-03','Mar'),
                  ('2026-04','Abr'),('2026-05','Mai'),('2026-06','Jun')]:
    mt = [t for t in all_trades if t['month'] == mk]
    if not mt: continue
    s = stats(mt)
    print(f"\n{label}: ", end='')
    print_stats(s, indent='')

# ── FOCO: MARCO E MAIO ────────────────────────────────────────────────────────
for mk, label in [('2026-03','MARCO'), ('2026-05','MAIO')]:
    mt = [t for t in all_trades if t['month'] == mk]
    if not mt: continue
    wins   = [t for t in mt if t['nr'] > 0]
    losses = [t for t in mt if t['nr'] <= 0]

    print(f'\n{"="*65}')
    print(f'ANALISE DETALHADA — {label}')
    print('='*65)
    s = stats(mt)
    print_stats(s)

    # Distribuicao de resultados
    print(f'\n  Distribuicao dos trades:')
    buckets = [(-99,-2,'< -2R'),(-2,-1,'-2R a -1R'),(-1,-0.5,'-1R a -0.5R'),
               (-0.5,0,'-0.5R a 0'),(0,1,'0 a +1R'),(1,2,'+1R a +2R'),
               (2,3,'+2R a +3R'),(3,99,'> +3R')]
    for lo,hi,lbl in buckets:
        count = sum(1 for t in mt if lo <= t['nr'] < hi)
        bar = '#' * count
        print(f"    {lbl:>14}: {count:3d}  {bar}")

    # Tipo de saida
    print(f'\n  Tipo de saida:')
    for exit_t in ['stop','trail','tp','timeout']:
        et = [t for t in mt if t['exit'] == exit_t]
        if not et: continue
        s2 = stats(et)
        print(f"    {exit_t:<8}: {s2['t']:3d} t | WR {s2['wr']:4.1f}% | avg {s2['avg']:+.3f}R")

    # Slope da EMA diaria (forca da tendencia)
    print(f'\n  Slope EMA diaria (forca do BEAR):')
    slope_buckets = [(-99,-0.01,'forte  (< -1.0%)'),(-0.01,-0.005,'medio  (-1.0% a -0.5%)'),
                     (-0.005,-0.002,'fraco  (-0.5% a -0.2%)'),(-0.002,0,'minimo (-0.2% a 0%)')]
    for lo,hi,lbl in slope_buckets:
        bt = [t for t in mt if lo <= t['slope'] < hi]
        if not bt: continue
        s2 = stats(bt)
        print(f"    {lbl}: {s2['t']:3d} t | WR {s2['wr']:4.1f}% | avg {s2['avg']:+.3f}R")

    # Bear streak (dias consecutivos em BEAR)
    print(f'\n  Dias consecutivos em regime BEAR:')
    streak_buckets = [(0,5,'1-5 dias'),(5,10,'6-10 dias'),(10,20,'11-20 dias'),(20,999,'> 20 dias')]
    for lo,hi,lbl in streak_buckets:
        bt = [t for t in mt if lo < t['bear_streak'] <= hi]
        if not bt: continue
        s2 = stats(bt)
        print(f"    {lbl:>12}: {s2['t']:3d} t | WR {s2['wr']:4.1f}% | avg {s2['avg']:+.3f}R")

    # Candle diario
    print(f'\n  Candle diario no dia do sinal:')
    for bull, lbl in [(True,'Dia de alta (diario bullish)'),(False,'Dia de baixa (diario bearish)')]:
        bt = [t for t in mt if t['daily_bull'] == bull]
        if not bt: continue
        s2 = stats(bt)
        print(f"    {lbl}: {s2['t']:3d} t | WR {s2['wr']:4.1f}% | avg {s2['avg']:+.3f}R")

    # ATR ratio
    print(f'\n  Intensidade do ATR (ATR / media 48H):')
    atr_buckets = [(1.0,1.2,'1.0x-1.2x'),(1.2,1.5,'1.2x-1.5x'),
                   (1.5,2.0,'1.5x-2.0x'),(2.0,99,'> 2.0x')]
    for lo,hi,lbl in atr_buckets:
        bt = [t for t in mt if lo <= t['atr_ratio'] < hi]
        if not bt: continue
        s2 = stats(bt)
        print(f"    {lbl:>12}: {s2['t']:3d} t | WR {s2['wr']:4.1f}% | avg {s2['avg']:+.3f}R")

    # Hora do sinal
    print(f'\n  Hora do sinal (UTC):')
    hour_groups = [(0,6,'00-06H'),(6,12,'06-12H'),(12,18,'12-18H'),(18,24,'18-24H')]
    for lo,hi,lbl in hour_groups:
        bt = [t for t in mt if lo <= t['hour'] < hi]
        if not bt: continue
        s2 = stats(bt)
        print(f"    {lbl}: {s2['t']:3d} t | WR {s2['wr']:4.1f}% | avg {s2['avg']:+.3f}R")

    # Por par
    print(f'\n  Por par:')
    syms = sorted(set(t['sym'] for t in mt))
    for sym in syms:
        bt = [t for t in mt if t['sym'] == sym]
        s2 = stats(bt)
        bar_w = int(s2['wr']/5)
        print(f"    {sym:<6}: {s2['t']:3d} t | WR {s2['wr']:4.1f}% | avg {s2['avg']:+.3f}R")

# ── COMPARACAO JAN+FEV vs MAR+MAI ────────────────────────────────────────────
print(f'\n{"="*65}')
print('COMPARACAO: meses bons (Jan+Fev) vs meses maus (Mar+Mai)')
print('='*65)
good = [t for t in all_trades if t['month'] in ('2026-01','2026-02')]
bad  = [t for t in all_trades if t['month'] in ('2026-03','2026-05')]

for group, label in [(good,'Jan+Fev (bons)'),(bad,'Mar+Mai (maus)')]:
    print(f'\n  {label}:')
    s = stats(group)
    print_stats(s)
    if not group: continue
    slopes = [t['slope_pct'] for t in group]
    streaks = [t['bear_streak'] for t in group]
    atr_ratios = [t['atr_ratio'] for t in group]
    print(f"    Slope medio:      {sum(slopes)/len(slopes):.3f}%")
    print(f"    Slope min/max:    {min(slopes):.3f}% / {max(slopes):.3f}%")
    print(f"    Bear streak medio:{sum(streaks)/len(streaks):.1f} dias")
    print(f"    ATR ratio medio:  {sum(atr_ratios)/len(atr_ratios):.2f}x")
    daily_bull_pct = sum(1 for t in group if t['daily_bull'])/len(group)*100
    print(f"    Dias bullish:     {daily_bull_pct:.1f}% dos sinais foram em dias de alta")
