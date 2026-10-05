"""
backtest_bear_v13.py
Sweep das variaveis de saida para Bear Market v1.3.

Diferencas face ao backtest_exit_sweep.py (v1.2):
  LOW_LOOKBACK   : 4 → 3 velas  (entra mais cedo em quedas rapidas)
  ATR_MOMENTUM   : ATR > avg → ATR > 1.5x avg  (so momentum real)
  COOLDOWN       : 8 → 3 barras  (permite reentrada em continuacoes)
  BASE_RR_CAP    : 4.0R → 2.5R
  BASE_BE        : 1.75R  (equivale a 70% do caminho para TP=2.5R)
  BASE_TRAIL     : 2.0x (igual a v1.2)

Periodo: Jan-Jun 2026 (mesmo do backtest v1.2 para comparacao directa)

No final, mostra comparacao v1.2 vs v1.3 com os parametros base de cada versao.
"""
import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 24, tzinfo=timezone.utc)

PAIRS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
    'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOGE/USDT:USDT', 'DOT/USDT:USDT',
    'XRP/USDT:USDT', 'LINK/USDT:USDT',
]

FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2

# ── Parametros de ENTRADA v1.3 ────────────────────────────────────────────────
LOW_LOOKBACK       = 3      # era 4
ATR_MOMENTUM_MULT  = 1.5    # novo: ATR > 1.5x media (era ATR > media)
ATR_AVG            = 48
ATR_STOP           = 1.0
COOLDOWN           = 3      # era 8
REGIME_EMA         = 20
SLOPE_BARS         = 5
SLOPE_THRESH       = 0.001

# ── Valores BASE para o sweep (parametros v1.3) ───────────────────────────────
# BE=1.75R equivale a 70% do caminho quando RR_CAP=2.5  (0.70 * 2.5 = 1.75)
BASE_BE     = 1.75
BASE_TRAIL  = 2.0
BASE_RR_CAP = 2.5

# ── Parametros v1.2 (para comparacao final) ───────────────────────────────────
V12_LOW_LOOKBACK      = 4
V12_ATR_MOMENTUM_MULT = 1.0   # era so ATR > media
V12_COOLDOWN          = 8
V12_BE                = 1.0
V12_TRAIL             = 2.0
V12_RR_CAP            = 4.0

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


print('Carregando dados (Jan-Jun 2026)...')
raw_data = {}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
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
    df1['regime']  = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
    raw_data[sym] = (df1, dfd)
    time.sleep(0.1)
    print('OK')
print()


# ── Prepara dataset com os lookback calculados para v1.3 e v1.2 ───────────────
def build_dataset(lookback):
    data = {}
    for sym, (df1_raw, _) in raw_data.items():
        df1 = df1_raw.copy()
        df1['low_n'] = df1['l'].shift(1).rolling(lookback).min()
        data[sym] = df1
    return data

data_v13 = build_dataset(LOW_LOOKBACK)        # lookback=3
data_v12  = build_dataset(V12_LOW_LOOKBACK)   # lookback=4


# ── Simulacao SHORT ───────────────────────────────────────────────────────────
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


def run(be_trigger, trail_atr, rr_cap, dataset, cooldown, atr_mult):
    trades = []
    for sym, df1 in dataset.items():
        last_bar = -9999
        start_i  = max(ATR_AVG+10,
                       next((i for i,ts in enumerate(df1.index) if ts>=START), ATR_AVG+10))
        for i in range(start_i, len(df1)-1):
            if i - last_bar < cooldown: continue
            ts_bar = df1.index[i]
            if ts_bar < START: continue
            r = df1.iloc[i]
            if str(r['regime']) != 'BEAR': continue
            cl=float(r['c']); op=float(r['o'])
            atr=float(r['atr']); atr_avg=float(r['atr_avg'])
            low_n=float(r['low_n'])
            if any(np.isnan(v) for v in [cl,atr,atr_avg,low_n]): continue
            # Condicao de momentum: v1.3 exige ATR > mult*avg; v1.2 exige ATR > avg
            if not (cl < low_n and cl < op and atr > atr_mult * atr_avg): continue
            sl_p = cl + ATR_STOP*atr
            if sl_p <= cl: continue
            nr = sim_short(df1, i, cl, sl_p, rr_cap, be_trigger, trail_atr)
            trades.append({'ts': ts_bar, 'sym': sym, 'nr': nr})
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
        't':len(nr_all),'wr':tw/len(nr_all)*100,
        'avgr':sum(nr_all)/len(nr_all),'pf':gw/gl if gl>0 else 99,
        'total':(capital/100-1)*100,'cap':capital,'months':months,
        'tw':tw,'tl':len(nr_all)-tw,
    }


def month_pct(s, mk):
    if mk not in s['months']: return '  n/a '
    m=s['months'][mk]; pct=(m['end_cap']/m['start']-1)*100
    return f"{pct:+.1f}%"


MKS = ['2026-01','2026-02','2026-03','2026-04','2026-05','2026-06']

def print_table(rows, title):
    print(f'\n{"="*108}')
    print(f'  {title}')
    print('='*108)
    print(f"{'Config':<22} | {'T':>5} | {'WR':>6} | {'AvgR':>7} | {'PF':>5} | "
          f"{'Jan':>7} | {'Fev':>7} | {'Mar':>7} | {'Abr':>7} | {'Mai':>7} | {'Jun':>7} | {'TOTAL':>7}")
    print('-'*108)
    for label, s, is_base in rows:
        if s is None:
            print(f"  {label:<20} | sem trades")
            continue
        mpcts=' | '.join(f"{month_pct(s,mk):>7}" for mk in MKS)
        mark='*' if is_base else ' '
        print(f"  {label:<20}{mark}| {s['t']:>5} | {s['wr']:>5.1f}% | "
              f"{s['avgr']:>+7.3f}R | {s['pf']:>5.2f} | {mpcts} | {s['total']:>+6.1f}%")
    if any(b for _,_,b in rows):
        print('  * = configuracao base v1.3')


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 0: ATR_MOMENTUM_MULT — qual e o limiar optimo?
# ─────────────────────────────────────────────────────────────────────────────
print('Sweep 0/4: ATR_MOMENTUM_MULT (limiar de momentum)...')
atr_mult_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
rows_atr = []
for mult in atr_mult_values:
    # Para o sweep de mult, usamos lookback=3 e cooldown=3 mas dataset com low_n=3
    t = run(BASE_BE, BASE_TRAIL, BASE_RR_CAP, data_v13, COOLDOWN, mult)
    s = summarize(t)
    is_base = (mult == ATR_MOMENTUM_MULT)
    label = f"ATR>{mult:.1f}x"
    rows_atr.append((label, s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  {label}{mark}: {s['t']} t | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  {label}: sem trades")

# Melhor multiplicador para usar nos sweeps seguintes
best_mult_row = max((r for r in rows_atr if r[1] is not None), key=lambda x: x[1]['total'])
best_mult_val = atr_mult_values[rows_atr.index(best_mult_row)]
print(f'  -> Melhor ATR mult: {best_mult_val}x (total {best_mult_row[1]["total"]:+.1f}%)')
print()

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 1: BE_TRIGGER
# ─────────────────────────────────────────────────────────────────────────────
print('Sweep 1/4: BE_TRIGGER...')
# BE=1.75 equivale a 70% do TP=2.5; sweep cobre abaixo e acima
be_values = [0.6, 0.8, 1.0, 1.25, 1.5, 1.75]
rows_be = []
for be in be_values:
    t = run(be, BASE_TRAIL, BASE_RR_CAP, data_v13, COOLDOWN, ATR_MOMENTUM_MULT)
    s = summarize(t)
    is_base = (be == BASE_BE)
    rows_be.append((f"BE={be}R", s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  BE={be}R{mark}: {s['t']} t | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  BE={be}R: sem trades")

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 2: TRAIL_ATR
# ─────────────────────────────────────────────────────────────────────────────
print('\nSweep 2/4: TRAIL_ATR...')
trail_values = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
rows_trail = []
for trail in trail_values:
    t = run(BASE_BE, trail, BASE_RR_CAP, data_v13, COOLDOWN, best_mult_val)
    s = summarize(t)
    is_base = (trail == BASE_TRAIL)
    rows_trail.append((f"TRAIL={trail}xATR", s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  TRAIL={trail}xATR{mark}: {s['t']} t | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  TRAIL={trail}xATR: sem trades")

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 3: RR_CAP
# ─────────────────────────────────────────────────────────────────────────────
print('\nSweep 3/4: RR_CAP...')
rr_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
rows_rr = []
for rr in rr_values:
    t = run(BASE_BE, BASE_TRAIL, rr, data_v13, COOLDOWN, best_mult_val)
    s = summarize(t)
    is_base = (rr == BASE_RR_CAP)
    rows_rr.append((f"RR_CAP={rr}R", s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  RR_CAP={rr}R{mark}: {s['t']} t | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  RR_CAP={rr}R: sem trades")

# ─────────────────────────────────────────────────────────────────────────────
# COMBINACOES
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nSweep 4/4: Combinacoes (usando ATR>{best_mult_val:.1f}x)...')
combos = [
    ('BASE v1.3',          1.75, 2.0,  2.5),
    ('BE1.0 T1.5 RR2.5',   1.0,  1.5,  2.5),
    ('BE1.0 T2.0 RR2.5',   1.0,  2.0,  2.5),
    ('BE1.0 T2.0 RR3.0',   1.0,  2.0,  3.0),
    ('BE1.25 T2.0 RR2.5',  1.25, 2.0,  2.5),
    ('BE1.25 T2.0 RR3.0',  1.25, 2.0,  3.0),
    ('BE1.5 T2.0 RR2.5',   1.5,  2.0,  2.5),
    ('BE1.5 T2.5 RR3.0',   1.5,  2.5,  3.0),
    ('BE1.75 T2.0 RR3.0',  1.75, 2.0,  3.0),
    ('BE1.75 T2.5 RR2.5',  1.75, 2.5,  2.5),
]
rows_combo = []
for label, be, trail, rr in combos:
    t = run(be, trail, rr, data_v13, COOLDOWN, best_mult_val)
    s = summarize(t)
    is_base = (label == 'BASE v1.3')
    rows_combo.append((label, s, is_base))
    mark='*' if is_base else ' '
    if s: print(f"  {label:<22}{mark}: {s['t']} t | WR {s['wr']:.1f}% | avg {s['avgr']:+.3f}R | {s['total']:+.1f}%")
    else: print(f"  {label}: sem trades")

# ─────────────────────────────────────────────────────────────────────────────
# TABELAS
# ─────────────────────────────────────────────────────────────────────────────
print_table(rows_atr,   f'SWEEP ATR_MOMENTUM_MULT — limiar de momentum (lookback={LOW_LOOKBACK}, cooldown={COOLDOWN}H)')
print_table(rows_be,    f'SWEEP BE_TRIGGER — (ATR>{best_mult_val:.1f}x)')
print_table(rows_trail, f'SWEEP TRAIL_ATR  — (ATR>{best_mult_val:.1f}x)')
print_table(rows_rr,    f'SWEEP RR_CAP     — (ATR>{best_mult_val:.1f}x)')
print_table(rows_combo, f'COMBINACOES MELHORES v1.3 (ATR>{best_mult_val:.1f}x)')

# ─────────────────────────────────────────────────────────────────────────────
# DETALHE MENSAL DA MELHOR COMBINACAO
# ─────────────────────────────────────────────────────────────────────────────
valid_combos = [(l,s,b) for l,s,b in rows_combo if s is not None]
if valid_combos:
    best_label, best_s, _ = max(valid_combos, key=lambda x: x[1]['total'])
    print(f'\n{"="*108}')
    print(f'  DETALHE MENSAL — Melhor: {best_label}')
    print('='*108)
    print(f"{'Mes':<6} | {'Trades':>6} | {'Wins':>5} | {'WR':>6} | {'Soma R':>7} | {'Capital':>9}")
    print('-'*108)
    cap = 100.0
    for mk in MKS:
        if mk not in best_s['months']: continue
        m = best_s['months'][mk]
        wr  = m['wins']/m['cnt']*100 if m['cnt']>0 else 0
        pct = (m['end_cap']/m['start']-1)*100
        print(f"  {mlabels[mk]:<4} | {m['cnt']:>6} | {m['wins']:>5} | {wr:>5.1f}% | "
              f"{m['nr_sum']:>+6.2f}R | ${m['end_cap']:>8.2f}  ({pct:+.1f}%)")
    print(f"\n  Capital final: ${best_s['cap']:.2f}  |  Total: {best_s['total']:+.1f}%")
    print(f"  Trades: {best_s['t']}  |  WR: {best_s['wr']:.1f}%  |  "
          f"AvgR: {best_s['avgr']:+.3f}R  |  PF: {best_s['pf']:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# COMPARACAO DIRECTA v1.2 vs v1.3
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*108}')
print('  COMPARACAO DIRECTA — v1.2 vs v1.3 (mesmo periodo Jan-Jun 2026)')
print('='*108)

t_v12 = run(V12_BE, V12_TRAIL, V12_RR_CAP, data_v12, V12_COOLDOWN, V12_ATR_MOMENTUM_MULT)
s_v12 = summarize(t_v12)

# Melhor config v1.3 encontrada
best_combo_params = next((c for c in combos if c[0]==best_label), None)
if best_combo_params:
    _, best_be, best_trail, best_rr = best_combo_params
    t_v13_best = run(best_be, best_trail, best_rr, data_v13, COOLDOWN, ATR_MOMENTUM_MULT)
    s_v13_best = summarize(t_v13_best)
else:
    t_v13_best = run(BASE_BE, BASE_TRAIL, BASE_RR_CAP, data_v13, COOLDOWN, ATR_MOMENTUM_MULT)
    s_v13_best = summarize(t_v13_best)
    best_label = 'BASE v1.3'

rows_vs = [
    (f'v1.2 (BE={V12_BE}R T={V12_TRAIL}x RR={V12_RR_CAP}R CD={V12_COOLDOWN}H)', s_v12, False),
    (f'v1.3 {best_label}', s_v13_best, True),
]
print_table(rows_vs, 'v1.2 vs v1.3 — parametros base de cada versao')

if s_v12 and s_v13_best:
    delta_t     = s_v13_best['t']   - s_v12['t']
    delta_wr    = s_v13_best['wr']  - s_v12['wr']
    delta_avgr  = s_v13_best['avgr']- s_v12['avgr']
    delta_total = s_v13_best['total']- s_v12['total']
    print(f"\n  Delta v1.3 - v1.2:")
    print(f"    Trades : {delta_t:+d}  ({'+mais' if delta_t>0 else 'menos'} sinais)")
    print(f"    WR     : {delta_wr:+.1f}%")
    print(f"    AvgR   : {delta_avgr:+.3f}R")
    print(f"    Total  : {delta_total:+.1f}%")
    verdict = "MELHOR" if delta_total > 0 else "PIOR" if delta_total < 0 else "IGUAL"
    print(f"\n  Veredicto: v1.3 e {verdict} que v1.2 neste periodo.")
