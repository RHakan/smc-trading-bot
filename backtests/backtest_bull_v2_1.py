"""
backtest_bull_v2_1.py
Bull v2.1 — PULLBACK com SELETIVIDADE + CONFIRMACAO.

Os testes anteriores falharam por falta de seletividade (3186 trades de ruido,
qualquer rocar na EMA disparava) e por entrar sem confirmacao (pegava faca caindo).
A v2.1 adiciona os dois ingredientes que faltaram:

  1. CORRECAO REAL: o preco esteve acima da EMA de suporte por N barras antes de
     recuar (houve uma perna de alta antes do pullback — nao e so ruido na media).
  2. CONFIRMACAO: candle bullish (close > open) no toque, e fechou acima da EMA
     (suporte segurou). Nao compra a faca a cair.
  3. STOP LARGO: 1.5-2.0xATR abaixo do suporte (o que os sweeps mostraram funcionar).

Logica (1H, regime BULL):
  Regime BULL + close > EMA_trend +
  [ultimas PULLBACK_BARS velas estiveram acima da EMA_support] +
  vela atual tocou (low <= EMA_support) e fechou acima (close > EMA_support) +
  [opcional] candle bullish (close > open)
  -> LONG ; stop = EMA_support - stop_mult x ATR

Periodo: 2024. Uso: python backtest_bull_v2_1.py [ANO]
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

# ── Base do sweep (parte do melhor que o v2.0 encontrou) ──────────────────────
BASE = dict(ema_support=50, ema_trend=200, stop_mult=1.5, rr_cap=2.5,
            pullback_bars=5, require_bullish=True)

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


print(f'Backtest Bull v2.1 (PULLBACK + seletividade) — ano {YEAR}')
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

        df1 = fetch(sym, '1h', 18).copy()
        tr  = pd.concat([
            (df1['h'] - df1['l']),
            (df1['h'] - df1['c'].shift(1)).abs(),
            (df1['l'] - df1['c'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df1['atr'] = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
        for p in (20, 50, 100, 200):
            df1[f'ema{p}'] = df1['c'].ewm(span=p, adjust=False).mean()
        df1['regime'] = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
        raw[sym] = df1
        ok_pairs.append(sym)
        time.sleep(0.1)
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def sim_long(df_arrays, ib, entry, sl_price, rr_cap):
    l_arr, h_arr, c_arr, atr_arr, n = df_arrays
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


def run(ema_support, ema_trend, stop_mult, rr_cap, pullback_bars, require_bullish):
    trades = []
    sup_name = f'ema{ema_support}'
    trend_name = f'ema{ema_trend}' if ema_trend else None
    for sym in ok_pairs:
        df = raw[sym]
        sup_arr = df[sup_name].values
        trend_arr = df[trend_name].values if trend_name else None
        atr_arr = df['atr'].values
        c_arr = df['c'].values; l_arr = df['l'].values
        h_arr = df['h'].values; o_arr = df['o'].values
        reg_arr = df['regime'].values
        idx = df.index
        n = len(df)
        arrays = (l_arr, h_arr, c_arr, atr_arr, n)
        last_bar = -9999
        for i in range(210, n - 1):
            ts = idx[i]
            if ts < START or ts > END: continue
            if i - last_bar < COOLDOWN: continue
            if str(reg_arr[i]) != 'BULL': continue
            sup = sup_arr[i]; atr = atr_arr[i]
            cl = c_arr[i]; lo = l_arr[i]; op = o_arr[i]
            if np.isnan(sup) or np.isnan(atr): continue
            # Filtro de tendencia
            if trend_arr is not None:
                tr = trend_arr[i]
                if np.isnan(tr) or cl <= tr: continue
            # Seletividade: as ultimas N velas estiveram acima do suporte (perna de alta)
            if pullback_bars > 0:
                prev_c   = c_arr[i - pullback_bars:i]
                prev_sup = sup_arr[i - pullback_bars:i]
                if np.isnan(prev_sup).any() or not np.all(prev_c > prev_sup):
                    continue
            # Toque + suporte segurou
            if not (lo <= sup and cl > sup): continue
            # Confirmacao: candle bullish
            if require_bullish and not (cl > op): continue
            entry = cl
            sl_p  = sup - stop_mult * atr
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
          f"avg {s['avgr']:>+6.3f}R | PF {s['pf']:>4.2f} | {s['total']:>+8.1f}%")


def run_base(**over):
    p = {**BASE, **over}
    return summarize(run(**p))


# ── SWEEP 0: seletividade (barras acima da EMA antes do toque) ─────────────────
print('Sweep 0/3: PULLBACK_BARS (seletividade — perna de alta antes do recuo)...')
for pb in [0, 3, 5, 8, 12]:
    show(f"pullback_bars={pb}", run_base(pullback_bars=pb), pb == BASE['pullback_bars'])

# ── SWEEP 1: confirmacao bullish ──────────────────────────────────────────────
print('\nSweep 1/3: confirmacao (candle bullish no toque)...')
for rb in [False, True]:
    show(f"require_bullish={rb}", run_base(require_bullish=rb), rb == BASE['require_bullish'])

# ── SWEEP 2: stop ─────────────────────────────────────────────────────────────
print('\nSweep 2/3: stop_mult...')
for sm in [1.0, 1.5, 2.0, 2.5]:
    show(f"stop={sm}xATR", run_base(stop_mult=sm), sm == BASE['stop_mult'])

# ── SWEEP 3: RR_CAP ───────────────────────────────────────────────────────────
print('\nSweep 3/3: RR_CAP...')
for rr in [1.5, 2.0, 2.5, 3.0]:
    show(f"RR_CAP={rr}", run_base(rr_cap=rr), rr == BASE['rr_cap'])

# ── COMBINACOES ───────────────────────────────────────────────────────────────
print('\nCombinacoes (support, trend, stop, RR, pullback_bars, bullish):')
combos = [
    dict(ema_support=50, ema_trend=200, stop_mult=1.5, rr_cap=2.5, pullback_bars=5,  require_bullish=True),
    dict(ema_support=50, ema_trend=200, stop_mult=2.0, rr_cap=3.0, pullback_bars=5,  require_bullish=True),
    dict(ema_support=50, ema_trend=200, stop_mult=2.0, rr_cap=2.5, pullback_bars=8,  require_bullish=True),
    dict(ema_support=20, ema_trend=100, stop_mult=1.5, rr_cap=2.5, pullback_bars=5,  require_bullish=True),
    dict(ema_support=20, ema_trend=200, stop_mult=2.0, rr_cap=3.0, pullback_bars=8,  require_bullish=True),
    dict(ema_support=50, ema_trend=100, stop_mult=2.0, rr_cap=3.0, pullback_bars=8,  require_bullish=True),
    dict(ema_support=50, ema_trend=200, stop_mult=2.5, rr_cap=3.0, pullback_bars=12, require_bullish=True),
    dict(ema_support=20, ema_trend=200, stop_mult=2.0, rr_cap=2.5, pullback_bars=12, require_bullish=True),
]
results = []
for p in combos:
    s = summarize(run(**p))
    results.append((p, s))
    lbl = f"S{p['ema_support']} T{p['ema_trend']} St{p['stop_mult']} RR{p['rr_cap']} pb{p['pullback_bars']}"
    show(lbl, s)

# ── MELHOR ────────────────────────────────────────────────────────────────────
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

print('\nReferencia: espelho -90.6% | pullback-no-toque (v2.0) melhor -38.4% | bear v1.3 +60%')
print('Custos: round-trip 0.14%.')
