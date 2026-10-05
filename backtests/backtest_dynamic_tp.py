"""
backtest_dynamic_tp.py
Compara TP fixo (3xATR) vs TP dinamico baseado no proximo suporte (swing low).

TP dinamico:
  - Encontra o swing low mais proximo abaixo do entry
  - Usa esse nivel como TP
  - Filtra trades onde o R:R resultante < RR_MIN
  - Cap em RR_CAP (nao deixa correr mais que 3R)

Swing low: barra cujo low e menor que os N vizinhos de cada lado.
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
BE_TRIGGER=0.8; TRAIL_ATR=1.5; ATR_AVG=48

SWING_N = 5        # N barras de cada lado para identificar swing low
RR_MIN  = 1.5      # R:R minimo para aceitar trade com TP dinamico

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

def find_nearest_support(df, bar_idx, entry, sl):
    """
    Procura o swing low mais proximo ABAIXO do entry, olhando para tras
    a partir de bar_idx (sem lookahead).
    Swing low: low[i] < low[i-N..i+N] para N=SWING_N barras.
    Retorna o nivel do suporte ou None se nao encontrar.
    """
    risk = sl - entry
    # Janela de busca: ultimas 100 barras antes do sinal
    start_search = max(0, bar_idx - 100)
    sub = df.iloc[start_search:bar_idx - SWING_N]   # exclui as N barras finais (incompletas como swing)

    swing_lows = []
    for i in range(SWING_N, len(sub) - SWING_N):
        low_i = float(sub.iloc[i]['l'])
        window = sub.iloc[i - SWING_N: i + SWING_N + 1]['l']
        if low_i == window.min() and low_i < entry:
            swing_lows.append(low_i)

    if not swing_lows:
        return None

    # Suporte mais proximo abaixo do entry (o mais alto dos swing lows abaixo do entry)
    candidates = [s for s in swing_lows if s < entry - risk * 0.5]  # pelo menos 0.5R de distancia
    if not candidates:
        return None
    return max(candidates)   # o mais proximo (o mais alto abaixo do entry)

print('Carregando dados (1H + diario)...')
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

# ── Simulacao generica (aceita TP variavel) ──────────────────────────────────
def sim_short(df, ib, entry, sl_price, tp_price):
    risk  = sl_price - entry
    if risk <= 0: return 0.0
    fee_r = entry * RT / risk
    cur   = sl_price; be = False
    for j in range(ib+1, min(ib+300, len(df))):
        bar = df.iloc[j]
        lo=float(bar['l']); hi=float(bar['h']); c=float(bar['c'])
        atr=float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if hi >= cur: return (entry-cur)/risk - fee_r
        if lo <= tp_price: return (entry-tp_price)/risk - fee_r
        if not be and (entry-c)/risk >= BE_TRIGGER:
            cur = entry; be = True
        if be:
            cand = c + TRAIL_ATR*atr
            if cand < cur: cur = cand
    return (entry - float(df.iloc[min(ib+299,len(df)-1)]['c']))/risk - fee_r

# ── Coleta de trades ──────────────────────────────────────────────────────────
def run(dynamic_tp=False, rr_min=RR_MIN):
    trades = []
    skipped = 0
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

            sl_price = cl + ATR_STOP * atr

            if dynamic_tp:
                support = find_nearest_support(df1, i, cl, sl_price)
                if support is None:
                    skipped += 1
                    continue
                risk = sl_price - cl
                rr   = (cl - support) / risk
                if rr < rr_min:
                    skipped += 1
                    continue
                # Cap em RR_CAP
                tp_price = max(support, cl - RR_CAP * risk)
            else:
                risk     = sl_price - cl
                tp_price = cl - RR_CAP * risk

            nr = sim_short(df1, i, cl, sl_price, tp_price)
            trades.append({'ts': ts_bar, 'nr': nr})
            last_bar = i

    return trades, skipped

# ── Relatorio ────────────────────────────────────────────────────────────────
def report(trades, label, skipped=0):
    print(f'\n{"="*62}')
    print(f'  {label}')
    if skipped: print(f'  (trades descartados por R:R insuficiente: {skipped})')
    print('='*62)

    if not trades:
        print('  Nenhum trade.'); return

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
        if t['nr'] > 0: months[mk]['wins'] += 1
        months[mk]['nr_sum'] += t['nr']

    tw = tl = 0
    for mk in sorted(months.keys()):
        m    = months[mk]; lbl = mlabels.get(mk, mk)
        pct  = (m['end_cap']/m['start']-1)*100
        wr   = m['wins']/m['cnt']*100
        avgr = m['nr_sum']/m['cnt']
        tw  += m['wins']; tl += m['cnt']-m['wins']
        mark = '+' if pct >= 0 else ''
        print(f"  {lbl}: {m['cnt']:4d} t | WR {wr:4.1f}% | avg {avgr:+.3f}R | {mark}{pct:.1f}%  ${m['end_cap']:.2f}")

    nr_all = [t['nr'] for t in trades]
    gw = sum(r for r in nr_all if r>0)
    gl = abs(sum(r for r in nr_all if r<0))
    pf = gw/gl if gl>0 else 99.0
    print(f'\n  TOTAL: {len(nr_all)} t | WR {tw/(tw+tl)*100:.1f}% | avg {sum(nr_all)/len(nr_all):+.3f}R | PF {pf:.2f}')
    print(f'  $100 -> ${capital:.2f}  ({(capital/100-1)*100:+.1f}%)')

print('Rodando comparacao TP fixo vs TP dinamico...')

t_fixo, _ = run(dynamic_tp=False)
report(t_fixo, 'TP FIXO — 3xATR (referencia atual)')

t_din_15, sk15 = run(dynamic_tp=True, rr_min=1.5)
report(t_din_15, 'TP DINAMICO — proximo suporte | R:R minimo 1.5', sk15)

t_din_20, sk20 = run(dynamic_tp=True, rr_min=2.0)
report(t_din_20, 'TP DINAMICO — proximo suporte | R:R minimo 2.0', sk20)

t_din_25, sk25 = run(dynamic_tp=True, rr_min=2.5)
report(t_din_25, 'TP DINAMICO — proximo suporte | R:R minimo 2.5', sk25)
