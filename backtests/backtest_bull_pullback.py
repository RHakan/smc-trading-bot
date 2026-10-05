"""
backtest_bull_pullback.py
Estrategia: Bull Market — Pullback para EMA20 (1H)

Logica:
  Em vez de entrar em breakouts (que falham), entrar quando o preco
  recua ate a EMA20 em tendencia de alta e ressalta de volta.

  Regime BULL (Daily EMA20 crescente, price > EMA20 diaria) +
  Low do candle toca a zona da EMA20 1H (dentro de PULLBACK_BUFFER%) +
  Close fecha ACIMA da EMA20 1H (bounce confirmado) +
  Candle bullish (close > open) +
  ATR acima da media 48H (volatilidade suficiente)
  -> LONG

SL: abaixo da EMA20 — 1xATR (se a EMA falhou como suporte, sai)
TP: close + RR_CAP x risco

Periodo: 2024-01-01 a 2026-06-24
  (cobre bull run 2024 + periodos bull 2025/2026)

Sweep:
  PULLBACK_BUFFER : quao perto do EMA o low tem de chegar  [0.003, 0.005, 0.01, 0.02]
  BE_TRIGGER      : quando move para breakeven              [0.5, 0.8, 1.0, 1.2, 1.5]
  TRAIL_ATR       : trailing apos breakeven                 [1.0, 1.5, 2.0, 2.5]
  RR_CAP          : take profit maximo                      [2.0, 3.0, 4.0, 5.0]
"""
import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 24, tzinfo=timezone.utc)

PAIRS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
    'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOGE/USDT:USDT', 'XRP/USDT:USDT',
    'LINK/USDT:USDT',
]

FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
EMA_PERIOD = 20
ATR_AVG    = 48
ATR_STOP   = 1.0
COOLDOWN   = 8
REGIME_EMA = 20; SLOPE_BARS = 5; SLOPE_THRESH = 0.001

# Valores base para o sweep
BASE_BUFFER = 0.005   # low tem de tocar EMA20 +/- 0.5%
BASE_BE     = 1.0
BASE_TRAIL  = 2.0
BASE_RR_CAP = 3.0

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

print('Carregando dados (2024-2026)...')
data = {}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)

    dfd = fetch(sym, '1d', 30).copy()
    dfd['ema']    = dfd['c'].ewm(span=REGIME_EMA, adjust=False).mean()
    dfd['slope']  = (dfd['ema'] - dfd['ema'].shift(SLOPE_BARS)) / dfd['ema'].shift(SLOPE_BARS)
    dfd['regime'] = 'NEUTRAL'
    dfd.loc[(dfd['c'] > dfd['ema']) & (dfd['slope'] >  SLOPE_THRESH), 'regime'] = 'BULL'
    dfd.loc[(dfd['c'] < dfd['ema']) & (dfd['slope'] < -SLOPE_THRESH), 'regime'] = 'BEAR'

    df1 = fetch(sym, '1h', 5).copy()
    tr  = pd.concat([
        (df1['h']-df1['l']),
        (df1['h']-df1['c'].shift(1)).abs(),
        (df1['l']-df1['c'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df1['atr']     = tr.ewm(com=13, adjust=False).mean()
    df1['atr_avg'] = df1['atr'].rolling(ATR_AVG).mean()
    df1['ema20']   = df1['c'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df1['regime']  = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
    data[sym] = df1
    time.sleep(0.1)
    print('OK')
print()


def sim_long(df, ib, entry, sl_price, rr_cap, be_trigger, trail_atr):
    risk  = entry - sl_price
    if risk <= 0: return 0.0
    tp    = entry + rr_cap * risk
    fee_r = entry * RT / risk
    cur   = sl_price; be = False
    for j in range(ib+1, min(ib+400, len(df))):
        bar = df.iloc[j]
        lo=float(bar['l']); hi=float(bar['h']); c=float(bar['c'])
        atr=float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if lo <= cur: return (cur - entry)/risk - fee_r
        if hi >= tp:  return rr_cap - fee_r
        if not be and (c - entry)/risk >= be_trigger:
            cur = entry; be = True
        if be:
            cand = c - trail_atr * atr
            if cand > cur: cur = cand
    return (float(df.iloc[min(ib+399, len(df)-1)]['c']) - entry)/risk - fee_r


def run(pullback_buffer, be_trigger, trail_atr, rr_cap):
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
            if str(r['regime']) != 'BULL': continue

            cl    = float(r['c']); op = float(r['o'])
            lo    = float(r['l'])
            atr   = float(r['atr']); atr_avg = float(r['atr_avg'])
            ema20 = float(r['ema20'])

            if any(np.isnan(v) for v in [cl, op, lo, atr, atr_avg, ema20]): continue

            # Condicoes de entrada
            pullback_ok = lo <= ema20 * (1 + pullback_buffer)  # low tocou zona da EMA
            bounce_ok   = cl > ema20                            # fechou acima da EMA
            bullish     = cl > op                               # candle bullish
            vol_ok      = atr > atr_avg                        # volatilidade

            if not (pullback_ok and bounce_ok and bullish and vol_ok): continue

            # SL abaixo da EMA — se a EMA falhou como suporte, saimos
            sl_p = ema20 - ATR_STOP * atr
            if sl_p >= cl: continue

            nr = sim_long(df1, i, cl, sl_p, rr_cap, be_trigger, trail_atr)
            trades.append({'ts': ts_bar, 'sym': sym, 'nr': nr})
            last_bar = i
    return trades


def summarize(trades):
    if not trades: return None
    trades.sort(key=lambda t: t['ts'])
    capital=100.0; months={}; years={}
    for t in trades:
        mk=t['ts'].strftime('%Y-%m'); yk=t['ts'].strftime('%Y')
        pnl=t['nr']*capital*0.01; capital+=pnl
        if mk not in months:
            months[mk]={'cnt':0,'wins':0,'nr_sum':0.0,'start':capital-pnl}
        months[mk]['end_cap']=capital; months[mk]['cnt']+=1
        if t['nr']>0: months[mk]['wins']+=1
        months[mk]['nr_sum']+=t['nr']
        if yk not in years: years[yk]={'start':capital-pnl}
        years[yk]['end_cap']=capital
    nr_all=[t['nr'] for t in trades]
    tw=sum(1 for r in nr_all if r>0)
    gw=sum(r for r in nr_all if r>0); gl=abs(sum(r for r in nr_all if r<0))
    return {
        't':len(nr_all),'wr':tw/len(nr_all)*100,
        'avgr':sum(nr_all)/len(nr_all),'pf':gw/gl if gl>0 else 99,
        'total':(capital/100-1)*100,'cap':capital,
        'months':months,'years':years,'tw':tw,'tl':len(nr_all)-tw,
    }


def year_pct(s, yk):
    if yk not in s.get('years',{}): return '  n/a '
    y=s['years'][yk]; pct=(y['end_cap']/y['start']-1)*100
    return f"{pct:+.1f}%"


def print_table(rows, title):
    print(f'\n{"="*92}')
    print(f'  {title}')
    print('='*92)
    print(f"{'Config':<24} | {'T':>5} | {'WR':>6} | {'AvgR':>7} | {'PF':>5} | {'2024':>7} | {'2025':>7} | {'2026':>7} | {'TOTAL':>8}")
    print('-'*92)
    for label, s, is_base in rows:
        if s is None:
            mark='*' if is_base else ' '
            print(f"  {label:<22}{mark}| sem trades")
            continue
        yp=' | '.join(year_pct(s,str(y)) for y in [2024,2025,2026])
        mark='*' if is_base else ' '
        print(f"  {label:<22}{mark}| {s['t']:>5} | {s['wr']:>5.1f}% | {s['avgr']:>+7.3f}R | {s['pf']:>5.2f} | {yp} | {s['total']:>+7.1f}%")
    if any(is_base for _,_,is_base in rows):
        print('  * = configuracao base')


ALL_MONTHS = sorted(set(
    f"{y}-{m:02d}" for y in range(2024,2027) for m in range(1,13)
    if (y < 2026) or (y==2026 and m<=6)
))
mlabels = {
    '2024-01':'Jan24','2024-02':'Fev24','2024-03':'Mar24','2024-04':'Abr24',
    '2024-05':'Mai24','2024-06':'Jun24','2024-07':'Jul24','2024-08':'Ago24',
    '2024-09':'Set24','2024-10':'Out24','2024-11':'Nov24','2024-12':'Dez24',
    '2025-01':'Jan25','2025-02':'Fev25','2025-03':'Mar25','2025-04':'Abr25',
    '2025-05':'Mai25','2025-06':'Jun25','2025-07':'Jul25','2025-08':'Ago25',
    '2025-09':'Set25','2025-10':'Out25','2025-11':'Nov25','2025-12':'Dez25',
    '2026-01':'Jan26','2026-02':'Fev26','2026-03':'Mar26','2026-04':'Abr26',
    '2026-05':'Mai26','2026-06':'Jun26',
}

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 0: PULLBACK_BUFFER (quao perto do EMA o low tem de chegar)
# ─────────────────────────────────────────────────────────────────────────────
print('Sweep 0/4: PULLBACK_BUFFER...')
buf_values = [0.002, 0.005, 0.01, 0.015, 0.02]
rows_buf = []
for buf in buf_values:
    t = run(pullback_buffer=buf, be_trigger=BASE_BE, trail_atr=BASE_TRAIL, rr_cap=BASE_RR_CAP)
    s = summarize(t)
    is_base = (buf == BASE_BUFFER)
    label = f"BUF={buf*100:.1f}%"
    rows_buf.append((label, s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  {label}{mark}: {s['t']} trades | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  {label}: sem trades")

# Escolhe o melhor buffer para os sweeps seguintes
best_buf_row = max((r for r in rows_buf if r[1] is not None), key=lambda x: x[1]['total'])
best_buf_val = buf_values[rows_buf.index(best_buf_row)]
print(f'  -> Melhor buffer: {best_buf_val*100:.1f}% ({best_buf_row[1]["total"]:+.1f}%)')

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 1: BE_TRIGGER
# ─────────────────────────────────────────────────────────────────────────────
print('\nSweep 1/4: BE_TRIGGER...')
be_values = [0.5, 0.8, 1.0, 1.2, 1.5]
rows_be = []
for be in be_values:
    t = run(pullback_buffer=best_buf_val, be_trigger=be, trail_atr=BASE_TRAIL, rr_cap=BASE_RR_CAP)
    s = summarize(t)
    is_base = (be == BASE_BE)
    rows_be.append((f"BE={be}R", s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  BE={be}R{mark}: {s['t']} trades | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  BE={be}R: sem trades")

best_be = max((r for r in rows_be if r[1] is not None), key=lambda x: x[1]['total'])
best_be_val = be_values[rows_be.index(best_be)]

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 2: TRAIL_ATR
# ─────────────────────────────────────────────────────────────────────────────
print('\nSweep 2/4: TRAIL_ATR...')
trail_values = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
rows_trail = []
for trail in trail_values:
    t = run(pullback_buffer=best_buf_val, be_trigger=best_be_val, trail_atr=trail, rr_cap=BASE_RR_CAP)
    s = summarize(t)
    is_base = (trail == BASE_TRAIL)
    rows_trail.append((f"TRAIL={trail}xATR", s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  TRAIL={trail}xATR{mark}: {s['t']} trades | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  TRAIL={trail}xATR: sem trades")

best_trail = max((r for r in rows_trail if r[1] is not None), key=lambda x: x[1]['total'])
best_trail_val = trail_values[rows_trail.index(best_trail)]

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 3: RR_CAP
# ─────────────────────────────────────────────────────────────────────────────
print('\nSweep 3/4: RR_CAP...')
rr_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
rows_rr = []
for rr in rr_values:
    t = run(pullback_buffer=best_buf_val, be_trigger=best_be_val, trail_atr=best_trail_val, rr_cap=rr)
    s = summarize(t)
    is_base = (rr == BASE_RR_CAP)
    rows_rr.append((f"RR_CAP={rr}R", s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  RR_CAP={rr}R{mark}: {s['t']} trades | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  RR_CAP={rr}R: sem trades")

best_rr = max((r for r in rows_rr if r[1] is not None), key=lambda x: x[1]['total'])
best_rr_val = rr_values[rows_rr.index(best_rr)]

# ─────────────────────────────────────────────────────────────────────────────
# COMBINACOES com os melhores valores de cada sweep
# ─────────────────────────────────────────────────────────────────────────────
print('\nCombinacoes...')
combos = [
    ('BASE',                    BASE_BUFFER, BASE_BE, BASE_TRAIL, BASE_RR_CAP),
    ('Melhor BE',               best_buf_val, best_be_val, BASE_TRAIL, BASE_RR_CAP),
    ('Melhor TRAIL',            best_buf_val, best_be_val, best_trail_val, BASE_RR_CAP),
    ('Melhor RR',               best_buf_val, best_be_val, best_trail_val, best_rr_val),
    # Variantes adicionais
    ('BE0.5 T1.0 RR2',          best_buf_val, 0.5, 1.0, 2.0),
    ('BE0.5 T1.5 RR3',          best_buf_val, 0.5, 1.5, 3.0),
    ('BE0.8 T1.5 RR3',          best_buf_val, 0.8, 1.5, 3.0),
    ('BE1.0 T2.0 RR3',          best_buf_val, 1.0, 2.0, 3.0),
    ('BE1.0 T2.0 RR4',          best_buf_val, 1.0, 2.0, 4.0),
    ('BE1.2 T2.5 RR4',          best_buf_val, 1.2, 2.5, 4.0),
]
rows_combo = []
for label, buf, be, trail, rr in combos:
    t = run(pullback_buffer=buf, be_trigger=be, trail_atr=trail, rr_cap=rr)
    s = summarize(t)
    is_base = (label == 'BASE')
    rows_combo.append((label, s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  {label:<22}{mark}: {s['t']} trades | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  {label}: sem trades")

# ─────────────────────────────────────────────────────────────────────────────
# TABELAS
# ─────────────────────────────────────────────────────────────────────────────
print_table(rows_buf,   f'SWEEP PULLBACK_BUFFER — distancia ao EMA20 (BE={BASE_BE}R TRAIL={BASE_TRAIL}x RR={BASE_RR_CAP}R)')
print_table(rows_be,    f'SWEEP BE_TRIGGER — quando move para breakeven (BUF={best_buf_val*100:.1f}%)')
print_table(rows_trail, f'SWEEP TRAIL_ATR  — trailing apos breakeven (BUF={best_buf_val*100:.1f}% BE={best_be_val}R)')
print_table(rows_rr,    f'SWEEP RR_CAP     — take profit maximo (BUF={best_buf_val*100:.1f}% BE={best_be_val}R TRAIL={best_trail_val}x)')
print_table(rows_combo, 'COMBINACOES MELHORES')

# ─────────────────────────────────────────────────────────────────────────────
# DETALHE MENSAL DA MELHOR COMBINACAO
# ─────────────────────────────────────────────────────────────────────────────
valid_combos = [(l,s,b) for l,s,b in rows_combo if s is not None]
if valid_combos:
    best_label, best_s, _ = max(valid_combos, key=lambda x: x[1]['total'])
    print(f'\n{"="*92}')
    print(f'  DETALHE MENSAL — {best_label}')
    print('='*92)
    print(f"{'Mes':<8} | {'Trades':>6} | {'Wins':>5} | {'WR':>6} | {'Soma R':>7} | {'Capital':>10}")
    print('-'*92)
    for mk in ALL_MONTHS:
        if mk not in best_s['months']: continue
        m=best_s['months'][mk]
        wr=m['wins']/m['cnt']*100 if m['cnt']>0 else 0
        pct=(m['end_cap']/m['start']-1)*100
        print(f"  {mlabels.get(mk,mk):<6} | {m['cnt']:>6} | {m['wins']:>5} | {wr:>5.1f}% | {m['nr_sum']:>+6.2f}R | ${m['end_cap']:>9.2f}  ({pct:+.1f}%)")
    print(f"\n  Capital final: ${best_s['cap']:.2f}  |  Total: {best_s['total']:+.1f}%")
    print(f"  Trades: {best_s['t']}  |  WR: {best_s['wr']:.1f}%  |  AvgR: {best_s['avgr']:+.3f}R  |  PF: {best_s['pf']:.2f}")

    # Resumo dos melhores parametros encontrados
    best_combo = next((c for c in combos if c[0]==best_label), None)
    if best_combo:
        _,buf,be,trail,rr = best_combo
        print(f'\n  Parametros optimos encontrados:')
        print(f'    PULLBACK_BUFFER = {buf*100:.1f}%')
        print(f'    BE_TRIGGER      = {be}R')
        print(f'    TRAIL_ATR       = {trail}x')
        print(f'    RR_CAP          = {rr}R')
