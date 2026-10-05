"""
backtest_wyckoff.py
Bull via WYCKOFF SPRING — abordagem de acumulacao/reversao, oposta ao pullback.

Conceito (entrada LONG):
  Range de acumulacao = min/max das ultimas N velas 1H.
  SPRING = a vela quebra ABAIXO do minimo do range (varre stops) mas FECHA de
           volta DENTRO do range (close > range_low) — falsa quebra, a "mola".
  Confirmacao = candle bullish (close > open) + [opcional] volume elevado.
  Entrada no fecho do spring. Stop abaixo do low do spring (risco pequeno).
  TP por RR fixo (ou topo do range).

Porque pode funcionar onde o pullback falhou: o Spring tem timing especifico —
captura a reversao APOS a limpeza de stops, nao aposta num "dip" qualquer.

Testa tambem EM QUE REGIME o spring funciona (BULL puro vs incluir NEUTRAL/
transicao, que e onde a acumulacao Wyckoff classica acontece).

Periodo: 2024. Uso: python backtest_wyckoff.py [ANO]
"""
import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

from bot import regime as regime_mod

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

YEAR  = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
START = datetime(YEAR, 1, 1,  tzinfo=timezone.utc)
END   = datetime(YEAR, 12, 31, tzinfo=timezone.utc)

RISK_PCT = 1.0
PAIRS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
    'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOGE/USDT:USDT',
    'LINK/USDT:USDT', 'DOT/USDT:USDT',
]
FEE = 0.0005; SLIP = 0.0002; RT = (FEE + SLIP) * 2

ATR_PERIOD = 14
COOLDOWN   = 3
BE_PCT     = 0.70
TRAIL_ATR  = 2.0

RANGE_LOOKBACKS = (20, 30, 50)   # pre-calculados

# ── Base do sweep ─────────────────────────────────────────────────────────────
BASE = dict(range_n=30, vol_mult=1.0, regime='BULL', rr_cap=2.5,
            stop_atr=0.3, require_bullish=True)

mlabels = {f'{YEAR}-{m:02d}': nome for m, nome in enumerate(
    ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'], start=1)}
MKS = [f'{YEAR}-{m:02d}' for m in range(1, 13)]

ex = ccxt.binanceusdm({'enableRateLimit': True})


def fetch(sym, tf, extra_days):
    since  = int((START - timedelta(days=extra_days)).timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    rows   = []
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b: break
        rows.extend(b); since = b[-1][0] + 1
        if since >= end_ms: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=['ts','o','h','l','c','v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()[lambda d: d.index <= END]


print(f'Backtest Wyckoff Spring — ano {YEAR}')
print('A carregar dados...')
raw = {}
ok_pairs = []
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd = fetch(sym, '1d', 40).copy()
        if len(dfd) < 60:
            print('SEM DADOS'); continue
        ema_d = dfd['c'].ewm(span=regime_mod.EMA_PERIOD, adjust=False).mean()
        slope = (ema_d - ema_d.shift(regime_mod.SLOPE_BARS)) / ema_d.shift(regime_mod.SLOPE_BARS)
        dfd['regime'] = 'NEUTRAL'
        dfd.loc[(dfd['c'] < ema_d) & (slope < -regime_mod.SLOPE_THRESH), 'regime'] = 'BEAR'
        dfd.loc[(dfd['c'] > ema_d) & (slope >  regime_mod.SLOPE_THRESH), 'regime'] = 'BULL'

        df1 = fetch(sym, '1h', 6).copy()
        tr  = pd.concat([
            (df1['h'] - df1['l']),
            (df1['h'] - df1['c'].shift(1)).abs(),
            (df1['l'] - df1['c'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df1['atr'] = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
        df1['vol_avg'] = df1['v'].rolling(48).mean()
        # range_low/high das N velas ANTERIORES (shift 1 — nao inclui a vela atual)
        for N in RANGE_LOOKBACKS:
            df1[f'rl{N}'] = df1['l'].shift(1).rolling(N).min()
            df1[f'rh{N}'] = df1['h'].shift(1).rolling(N).max()
        df1['regime'] = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
        raw[sym] = df1
        ok_pairs.append(sym)
        time.sleep(0.1)
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def sim_long(arrays, ib, entry, sl_price, rr_cap):
    l_arr, h_arr, c_arr, atr_arr, n = arrays
    risk = entry - sl_price
    if risk <= 0: return 0.0
    tp = entry + rr_cap * risk
    fee_r = entry * RT / risk
    cur = sl_price; be = False
    for j in range(ib + 1, min(ib + 400, n)):
        lo = l_arr[j]; hi = h_arr[j]; c = c_arr[j]
        atr = atr_arr[j] if not np.isnan(atr_arr[j]) else risk
        if lo <= cur:
            return (cur - entry) / risk - fee_r
        if hi >= tp:
            return rr_cap - fee_r
        if not be and (c - entry) / risk >= BE_PCT * rr_cap:
            cur = entry; be = True
        if be:
            cand = c - TRAIL_ATR * atr
            if cand > cur: cur = cand
    last_c = c_arr[min(ib + 399, n - 1)]
    return (last_c - entry) / risk - fee_r


def regime_ok(reg, mode):
    if mode == 'BULL':     return reg == 'BULL'
    if mode == 'BULL_NEU': return reg in ('BULL', 'NEUTRAL')
    return True  # ALL


def run(range_n, vol_mult, regime, rr_cap, stop_atr, require_bullish):
    trades = []
    rl_name = f'rl{range_n}'; rh_name = f'rh{range_n}'
    for sym in ok_pairs:
        df = raw[sym]
        rl = df[rl_name].values; rh = df[rh_name].values
        atr_arr = df['atr'].values; vol = df['v'].values; vavg = df['vol_avg'].values
        c_arr = df['c'].values; l_arr = df['l'].values
        h_arr = df['h'].values; o_arr = df['o'].values
        reg_arr = df['regime'].values
        idx = df.index; n = len(df)
        arrays = (l_arr, h_arr, c_arr, atr_arr, n)
        last_bar = -9999
        for i in range(max(RANGE_LOOKBACKS) + 50, n - 1):
            ts = idx[i]
            if ts < START or ts > END: continue
            if i - last_bar < COOLDOWN: continue
            if not regime_ok(str(reg_arr[i]), regime): continue
            range_low = rl[i]; atr = atr_arr[i]
            cl = c_arr[i]; lo = l_arr[i]; op = o_arr[i]
            if np.isnan(range_low) or np.isnan(atr): continue
            # SPRING: quebrou abaixo do minimo do range mas fechou de volta para dentro
            if not (lo < range_low and cl > range_low): continue
            # Confirmacao bullish
            if require_bullish and not (cl > op): continue
            # Volume elevado na vela do spring (absorcao)
            if vol_mult > 1.0:
                if np.isnan(vavg[i]) or vol[i] < vol_mult * vavg[i]: continue
            entry = cl
            sl_p = lo - stop_atr * atr   # stop abaixo do low do spring
            if sl_p >= entry: continue
            nr = sim_long(arrays, i, entry, sl_p, rr_cap)
            trades.append({'ts': ts, 'nr': nr})
            last_bar = i
    return trades


def summarize(ts):
    if not ts: return None
    nr = [t['nr'] for t in ts]
    wins = [r for r in nr if r > 0]
    gw = sum(wins); gl = abs(sum(r for r in nr if r <= 0))
    cap = 100.0; months = {}
    for t in sorted(ts, key=lambda x: x['ts']):
        mk = t['ts'].strftime('%Y-%m'); pnl = t['nr'] * cap * (RISK_PCT/100); cap += pnl
        if mk not in months: months[mk] = {'cnt':0,'wins':0,'start':cap-pnl}
        months[mk]['end_cap'] = cap; months[mk]['cnt'] += 1
        if t['nr'] > 0: months[mk]['wins'] += 1
    return {'t':len(nr),'wr':len(wins)/len(nr)*100,'avgr':sum(nr)/len(nr),
            'pf':gw/gl if gl>0 else 99,'total':cap-100,'cap':cap,'months':months}


def show(label, s, base=False):
    mark = '*' if base else ' '
    if s is None:
        print(f"  {label:<26}{mark}: sem trades"); return
    print(f"  {label:<26}{mark}: {s['t']:>4} t | WR {s['wr']:>4.1f}% | "
          f"avg {s['avgr']:>+6.3f}R | PF {s['pf']:>4.2f} | {s['total']:>+9.1f}%")


def run_base(**over):
    return summarize(run(**{**BASE, **over}))


# ── SWEEP 0: regime onde o spring opera ───────────────────────────────────────
print('Sweep 0/4: regime (onde o Spring funciona)...')
for rg in ['BULL', 'BULL_NEU', 'ALL']:
    show(f"regime={rg}", run_base(regime=rg), rg == BASE['regime'])

# ── SWEEP 1: tamanho do range ─────────────────────────────────────────────────
print('\nSweep 1/4: range_n (barras do range de acumulacao)...')
for rn in RANGE_LOOKBACKS:
    show(f"range_n={rn}", run_base(range_n=rn), rn == BASE['range_n'])

# ── SWEEP 2: filtro de volume ─────────────────────────────────────────────────
print('\nSweep 2/4: vol_mult (volume na vela do spring)...')
for vm in [1.0, 1.3, 1.5, 2.0]:
    show(f"vol_mult={vm}", run_base(vol_mult=vm), vm == BASE['vol_mult'])

# ── SWEEP 3: RR_CAP ───────────────────────────────────────────────────────────
print('\nSweep 3/4: RR_CAP...')
for rr in [1.5, 2.0, 2.5, 3.0]:
    show(f"RR_CAP={rr}", run_base(rr_cap=rr), rr == BASE['rr_cap'])

# ── COMBINACOES ───────────────────────────────────────────────────────────────
print('\nCombinacoes:')
combos = [
    dict(range_n=30, vol_mult=1.5, regime='BULL',     rr_cap=2.5, stop_atr=0.3, require_bullish=True),
    dict(range_n=30, vol_mult=1.5, regime='BULL_NEU', rr_cap=2.5, stop_atr=0.3, require_bullish=True),
    dict(range_n=50, vol_mult=1.5, regime='BULL',     rr_cap=3.0, stop_atr=0.3, require_bullish=True),
    dict(range_n=50, vol_mult=2.0, regime='BULL_NEU', rr_cap=3.0, stop_atr=0.5, require_bullish=True),
    dict(range_n=20, vol_mult=1.3, regime='BULL',     rr_cap=2.0, stop_atr=0.3, require_bullish=True),
    dict(range_n=30, vol_mult=2.0, regime='ALL',      rr_cap=2.5, stop_atr=0.3, require_bullish=True),
    dict(range_n=50, vol_mult=1.3, regime='BULL',     rr_cap=2.5, stop_atr=0.5, require_bullish=True),
]
results = []
for p in combos:
    s = summarize(run(**p))
    results.append((p, s))
    show(f"r{p['range_n']} v{p['vol_mult']} {p['regime']} RR{p['rr_cap']}", s)

valid = [(p, s) for p, s in results if s is not None]
if valid:
    best_p, best_s = max(valid, key=lambda x: x[1]['total'])
    print(f'\n{"="*78}')
    print(f"  MELHOR: {best_p}")
    print('='*78)
    print(f"  Trades {best_s['t']} | WR {best_s['wr']:.1f}% | AvgR {best_s['avgr']:+.3f}R | "
          f"PF {best_s['pf']:.2f} | Total {best_s['total']:+.1f}%")
    print(f"\n  {'Mes':<5} | {'Trades':>6} | {'WR':>6} | {'Capital':>20}")
    print('  ' + '-'*52)
    for mk in MKS:
        if mk not in best_s['months']: continue
        m = best_s['months'][mk]
        wr = m['wins']/m['cnt']*100 if m['cnt']>0 else 0
        pct = (m['end_cap']/m['start']-1)*100
        print(f"  {mlabels[mk]:<5} | {m['cnt']:>6} | {wr:>5.1f}% | ${m['end_cap']:>8.2f} ({pct:+.1f}%)")

print('\nReferencia: espelho -90.6% | pullback v2.0 -38.4% | pullback v2.1 -28.6% | bear v1.3 +60%')
print('Custos: round-trip 0.14%.')
