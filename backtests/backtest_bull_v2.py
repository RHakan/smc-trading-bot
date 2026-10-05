"""
backtest_bull_v2.py
Sweep da estrategia Bull Market v2.0 — PULLBACK (comprar a correcao no suporte).

Logica base (1H, regime BULL):
  Regime BULL (Daily EMA20 ascendente) +
  Tendencia intacta no 1H (close > EMA_trend) +
  Correcao tocou o suporte (low <= EMA_support) +
  Suporte segurou (close > EMA_support)
  -> LONG no toque (sem esperar confirmacao)

  Stop = EMA_support - stop_mult x ATR   (abaixo do suporte, espaco para respirar)
  TP   = entry + RR_CAP x risco
  Breakeven a 70% + trailing 2.0xATR (igual as outras)

Diferenca face ao espelho (bull_v1, que fez -90%): aqui compramos o DESCONTO
num suporte, com stop estrutural largo — nao o rompimento com stop apertado.

Sweep (varia um parametro de cada vez a partir da base):
  0: EMA_support  — qual EMA e o suporte (20 vs 50)
  1: EMA_trend    — filtro de tendencia (sem / 50 / 100 / 200)
  2: stop_mult    — quao fundo fica o stop abaixo da EMA (0.5 / 1.0 / 1.5 / 2.0)
  3: RR_CAP       — alvo (1.5 / 2.0 / 2.5 / 3.0)

Periodo: 2024 (mesmo ano onde o espelho fez -90%, para comparacao directa).
Uso: python backtest_bull_v2.py [ANO]
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

# ── Base do sweep ─────────────────────────────────────────────────────────────
BASE_EMA_SUPPORT = 20
BASE_EMA_TREND   = 50      # 0 = sem filtro de tendencia
BASE_STOP_MULT   = 1.0
BASE_RR_CAP      = 2.0

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


print(f'Backtest Bull v2.0 (PULLBACK) — sweep — ano {YEAR}')
print('A carregar dados da Binance...')
raw = {}
ok_pairs = []
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd = fetch(sym, '1d', 40).copy()
        if len(dfd) < 60:
            print('SEM DADOS — ignorado'); continue
        ema_d = dfd['c'].ewm(span=regime_mod.EMA_PERIOD, adjust=False).mean()
        slope = (ema_d - ema_d.shift(regime_mod.SLOPE_BARS)) / ema_d.shift(regime_mod.SLOPE_BARS)
        dfd['regime'] = 'NEUTRAL'
        dfd.loc[(dfd['c'] < ema_d) & (slope < -regime_mod.SLOPE_THRESH), 'regime'] = 'BEAR'
        dfd.loc[(dfd['c'] > ema_d) & (slope >  regime_mod.SLOPE_THRESH), 'regime'] = 'BULL'

        df1 = fetch(sym, '1h', 15).copy()
        tr  = pd.concat([
            (df1['h'] - df1['l']),
            (df1['h'] - df1['c'].shift(1)).abs(),
            (df1['l'] - df1['c'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df1['atr'] = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
        # Pre-calcula todas as EMAs que o sweep pode usar
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


def sim_long(df, ib, entry, sl_price, rr_cap):
    risk = entry - sl_price
    if risk <= 0:
        return 0.0
    tp = entry + rr_cap * risk
    fee_r = entry * RT / risk
    cur = sl_price; be = False
    for j in range(ib + 1, min(ib + 400, len(df))):
        bar = df.iloc[j]
        lo = float(bar['l']); hi = float(bar['h']); c = float(bar['c'])
        atr = float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if lo <= cur:
            return (cur - entry) / risk - fee_r
        if hi >= tp:
            return rr_cap - fee_r
        if not be and (c - entry) / risk >= BE_PCT * rr_cap:
            cur = entry; be = True
        if be:
            cand = c - TRAIL_ATR * atr
            if cand > cur: cur = cand
    last_c = float(df.iloc[min(ib + 399, len(df) - 1)]['c'])
    return (last_c - entry) / risk - fee_r


def run(ema_support, ema_trend, stop_mult, rr_cap):
    """Executa a estrategia com um conjunto de parametros sobre todos os pares."""
    trades = []
    sup_col = f'ema{ema_support}'
    trend_col = f'ema{ema_trend}' if ema_trend else None
    for sym in ok_pairs:
        df = raw[sym]
        sup_arr   = df[sup_col].values
        trend_arr = df[trend_col].values if trend_col else None
        atr_arr = df['atr'].values
        c_arr = df['c'].values; l_arr = df['l'].values
        reg_arr = df['regime'].values
        idx = df.index
        last_bar = -9999
        n = len(df)
        start_i = 210  # garante EMA200 e ATR estaveis
        for i in range(start_i, n - 1):
            ts = idx[i]
            if ts < START or ts > END: continue
            if i - last_bar < COOLDOWN: continue
            if str(reg_arr[i]) != 'BULL': continue
            sup = sup_arr[i]; atr = atr_arr[i]
            cl = c_arr[i]; lo = l_arr[i]
            if np.isnan(sup) or np.isnan(atr): continue
            # Filtro de tendencia: preco acima da EMA de tendencia
            if trend_arr is not None:
                tr = trend_arr[i]
                if np.isnan(tr) or cl <= tr: continue
            # Toque no suporte: a vela penetrou a EMA mas fechou acima (suporte segurou)
            if not (lo <= sup and cl > sup): continue
            entry = cl
            sl_p  = sup - stop_mult * atr   # stop abaixo do suporte
            if sl_p >= entry: continue
            nr = sim_long(df, i, entry, sl_p, rr_cap)
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
            'pf':gw/gl if gl>0 else 99,'total':cap-100,'cap':cap,'months':months,
            'soma_r':sum(nr)}


def show(label, s, base=False):
    mark = '*' if base else ' '
    if s is None:
        print(f"  {label:<22}{mark}: sem trades")
        return
    print(f"  {label:<22}{mark}: {s['t']:>4} t | WR {s['wr']:>4.1f}% | "
          f"avg {s['avgr']:>+6.3f}R | PF {s['pf']:>4.2f} | {s['total']:>+7.1f}%")


# ── SWEEP 0: EMA de suporte ───────────────────────────────────────────────────
print('Sweep 0/3: EMA de suporte...')
for ema_s in [20, 50]:
    s = summarize(run(ema_s, BASE_EMA_TREND, BASE_STOP_MULT, BASE_RR_CAP))
    show(f"EMA_support={ema_s}", s, ema_s == BASE_EMA_SUPPORT)

# ── SWEEP 1: filtro de tendencia ──────────────────────────────────────────────
print('\nSweep 1/3: filtro de tendencia (EMA_trend)...')
for ema_t in [0, 50, 100, 200]:
    s = summarize(run(BASE_EMA_SUPPORT, ema_t, BASE_STOP_MULT, BASE_RR_CAP))
    lbl = "sem filtro" if ema_t == 0 else f"close>EMA{ema_t}"
    show(lbl, s, ema_t == BASE_EMA_TREND)

# ── SWEEP 2: profundidade do stop ─────────────────────────────────────────────
print('\nSweep 2/3: stop_mult (ATR abaixo do suporte)...')
for sm in [0.5, 1.0, 1.5, 2.0]:
    s = summarize(run(BASE_EMA_SUPPORT, BASE_EMA_TREND, sm, BASE_RR_CAP))
    show(f"stop={sm}xATR", s, sm == BASE_STOP_MULT)

# ── SWEEP 3: RR_CAP ───────────────────────────────────────────────────────────
print('\nSweep 3/3: RR_CAP...')
for rr in [1.5, 2.0, 2.5, 3.0]:
    s = summarize(run(BASE_EMA_SUPPORT, BASE_EMA_TREND, BASE_STOP_MULT, rr))
    show(f"RR_CAP={rr}", s, rr == BASE_RR_CAP)

# ── COMBINACOES PROMISSORAS ───────────────────────────────────────────────────
print('\nCombinacoes (EMA_support, EMA_trend, stop_mult, RR_CAP):')
combos = [
    (20, 50, 1.0, 2.0), (20, 50, 1.0, 2.5), (20, 50, 1.5, 2.5),
    (20, 100, 1.0, 2.0), (20, 200, 1.0, 2.0),
    (50, 100, 1.0, 2.0), (50, 200, 1.0, 2.5), (50, 200, 1.5, 3.0),
    (20, 50, 0.5, 1.5),  (20, 100, 1.5, 3.0),
]
results = []
for es, et, sm, rr in combos:
    s = summarize(run(es, et, sm, rr))
    results.append(((es, et, sm, rr), s))
    show(f"S{es} T{et} St{sm} RR{rr}", s)

# ── DETALHE DA MELHOR COMBINACAO ──────────────────────────────────────────────
valid = [(p, s) for p, s in results if s is not None]
if valid:
    best_p, best_s = max(valid, key=lambda x: x[1]['total'])
    es, et, sm, rr = best_p
    print(f'\n{"="*78}')
    print(f'  MELHOR: EMA_support={es}  EMA_trend={et}  stop={sm}xATR  RR_CAP={rr}')
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

print('\nReferencia: o espelho (bull_v1) fez -90.6% em 2024. Bear v1.3 (junho) fez +60%.')
print('Custos: taxa 0.05% + slippage 0.02% por lado (round-trip 0.14%).')
