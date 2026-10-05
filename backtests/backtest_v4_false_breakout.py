"""
backtest_v4_false_breakout.py
Estratégia: False Breakout (Bull Trap / Bear Trap) em 15M
Fees: USDC Futures — Maker 0.0000% / Taker 0.0360%
Regime: EMA20 diária (BULL/BEAR)
Sinais:
  BULL TRAP → SHORT: high > resistance, fecha abaixo (pin bear), volume spike
  BEAR TRAP → LONG:  low < support,    fecha acima (pin bull), volume spike
Stop: natural do trap (extremo da vela) OU 1xATR (testamos os dois)
"""
import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 24, tzinfo=timezone.utc)

ex = ccxt.binanceusdm({'enableRateLimit': True})

PAIRS = [
    'BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','BNB/USDT:USDT',
    'ADA/USDT:USDT','AVAX/USDT:USDT','DOGE/USDT:USDT','DOT/USDT:USDT',
    'XRP/USDT:USDT','LINK/USDT:USDT'
]

# --- Fees USDC ---
# Entrada market (taker) = 0.036%, TP/Stop limit (maker) = 0.000%
# Slippage conservador: 0.01% por side
TAKER = 0.00036
MAKER = 0.00000
SLIP  = 0.0001
RT    = TAKER + MAKER + SLIP * 2   # 0.00056 round-trip total

# --- Gestao de risco ---
RR_CAP      = 3.0
BE_TRIGGER  = 0.8
TRAIL_ATR   = 1.5
COOLDOWN    = 4        # barras 15M (= 1 hora)
MAX_RISK_ATR = 4.0     # rejeita sinal se stop > 4xATR

# --- False Breakout ---
FB_WINDOW    = 20      # 20 x 15M = 5 horas de S/R
TRAP_BUFFER  = 0.15    # buffer alem do extremo (em x ATR)
VOL_MULT     = 1.3     # spike de volume minimo

# --- Regime diario ---
REGIME_EMA   = 20
SLOPE_BARS   = 5
SLOPE_THRESH = 0.001

mlabels = {
    '2026-01': 'Jan', '2026-02': 'Fev', '2026-03': 'Mar',
    '2026-04': 'Abr', '2026-05': 'Mai', '2026-06': 'Jun'
}

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

def calc_regime_daily(sym):
    dfd = fetch(sym, '1d', 30).copy()
    dfd['ema']    = dfd['c'].ewm(span=REGIME_EMA, adjust=False).mean()
    dfd['slope']  = (dfd['ema'] - dfd['ema'].shift(SLOPE_BARS)) / dfd['ema'].shift(SLOPE_BARS)
    dfd['regime'] = 'NEUTRAL'
    dfd.loc[(dfd['c'] > dfd['ema']) & (dfd['slope'] >  SLOPE_THRESH), 'regime'] = 'BULL'
    dfd.loc[(dfd['c'] < dfd['ema']) & (dfd['slope'] < -SLOPE_THRESH), 'regime'] = 'BEAR'
    return dfd

print('Carregando dados (15M + diario)...')
data    = {}
data_1h = {}

for sym in PAIRS:
    dfd = calc_regime_daily(sym)

    # 15M
    df15 = fetch(sym, '15m', 5).copy()
    tr = pd.concat([
        (df15['h'] - df15['l']),
        (df15['h'] - df15['c'].shift(1)).abs(),
        (df15['l'] - df15['c'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df15['atr']     = tr.ewm(com=13, adjust=False).mean()
    o4 = (df15['o'] + df15['h'] + df15['l'] + df15['c']) / 4
    d  = o4.diff()
    g  = d.where(d > 0, 0.).ewm(com=13, adjust=False).mean()
    lv = (-d.where(d < 0, 0.)).ewm(com=13, adjust=False).mean()
    df15['rsi']        = 100 - (100 / (1 + g / lv.replace(0, np.nan)))
    df15['vol_avg']    = df15['v'].rolling(20).mean()
    df15['resistance'] = df15['h'].shift(1).rolling(FB_WINDOW).max()
    df15['support']    = df15['l'].shift(1).rolling(FB_WINDOW).min()
    df15['regime']     = dfd['regime'].shift(1).reindex(df15.index, method='ffill')
    data[sym] = df15

    # 1H (para breakout SHORT comparativo)
    df1 = fetch(sym, '1h', 5).copy()
    tr1 = pd.concat([
        (df1['h'] - df1['l']),
        (df1['h'] - df1['c'].shift(1)).abs(),
        (df1['l'] - df1['c'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df1['atr']       = tr1.ewm(com=13, adjust=False).mean()
    df1['atr_avg48'] = df1['atr'].rolling(48).mean()
    df1['low4']      = df1['l'].shift(1).rolling(4).min()
    df1['regime']    = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
    data_1h[sym] = df1

    time.sleep(0.15)

print(f'OK - {len(data)} pares\n')

# ─────────────────────────────────────────────
# Simulacao
# ─────────────────────────────────────────────
def sim_long(df, ib, entry, sl_price, rt=RT):
    risk = entry - sl_price
    if risk <= 0: return 0.0
    tp    = entry + RR_CAP * risk
    fee_r = entry * rt / risk
    cur   = sl_price; be = False
    for j in range(ib + 1, min(ib + 400, len(df))):
        bar = df.iloc[j]
        lo  = float(bar['l']); hi = float(bar['h']); c = float(bar['c'])
        atr = float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if lo <= cur: return (cur - entry) / risk - fee_r
        if hi >= tp:  return RR_CAP - fee_r
        if not be and (c - entry) / risk >= BE_TRIGGER:
            cur = entry; be = True
        if be:
            cand = c - TRAIL_ATR * atr
            if cand > cur: cur = cand
    return (float(df.iloc[min(ib + 399, len(df) - 1)]['c']) - entry) / risk - fee_r

def sim_short(df, ib, entry, sl_price, rt=RT):
    risk = sl_price - entry
    if risk <= 0: return 0.0
    tp    = entry - RR_CAP * risk
    fee_r = entry * rt / risk
    cur   = sl_price; be = False
    for j in range(ib + 1, min(ib + 400, len(df))):
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
    return (entry - float(df.iloc[min(ib + 399, len(df) - 1)]['c'])) / risk - fee_r

# ─────────────────────────────────────────────
# Backtest false breakout 15M
# ─────────────────────────────────────────────
def run_fb(stop_mode='trap'):
    all_trades = []
    for sym, df15 in data.items():
        last_bar = -9999
        start_i  = max(FB_WINDOW + 30,
                       next((i for i, ts in enumerate(df15.index) if ts >= START),
                            FB_WINDOW + 30))
        for i in range(start_i, len(df15) - 1):
            if i - last_bar < COOLDOWN: continue
            ts_bar = df15.index[i]
            if ts_bar < START: continue

            r      = df15.iloc[i]
            regime = str(r['regime'])
            if regime == 'NEUTRAL': continue

            cl  = float(r['c']); op  = float(r['o'])
            hi  = float(r['h']); lo  = float(r['l'])
            atr = float(r['atr']); vol = float(r['v'])
            vol_avg = float(r['vol_avg'])
            res = float(r['resistance']); sup = float(r['support'])

            if any(np.isnan(v) for v in [cl, atr, vol_avg, res, sup]): continue
            if vol_avg <= 0 or atr <= 0: continue

            vol_spike = vol >= vol_avg * VOL_MULT
            sig = None

            # BULL TRAP -> SHORT
            if regime == 'BEAR' and hi > res and cl < res and cl < op and vol_spike:
                if stop_mode == 'trap':
                    sl_p = hi + TRAP_BUFFER * atr
                else:
                    sl_p = cl + 1.0 * atr
                risk = sl_p - cl
                if 0 < risk <= MAX_RISK_ATR * atr:
                    sig = ('short', cl, sl_p)

            # BEAR TRAP -> LONG
            elif regime == 'BULL' and lo < sup and cl > sup and cl > op and vol_spike:
                if stop_mode == 'trap':
                    sl_p = lo - TRAP_BUFFER * atr
                else:
                    sl_p = cl - 1.0 * atr
                risk = cl - sl_p
                if 0 < risk <= MAX_RISK_ATR * atr:
                    sig = ('long', cl, sl_p)

            if not sig: continue
            side, entry, sl_p = sig
            nr = (sim_short(df15, i, entry, sl_p) if side == 'short'
                  else sim_long(df15, i, entry, sl_p))
            all_trades.append({'ts': ts_bar, 'sym': sym, 'side': side, 'nr': nr})
            last_bar = i

    return all_trades

# ─────────────────────────────────────────────
# Backtest breakout SHORT 1H (melhor resultado anterior)
# ─────────────────────────────────────────────
def run_breakout_short_1h():
    RT_USDT = 0.0014   # fees USDT originais para comparacao honesta
    all_trades = []
    for sym, df1 in data_1h.items():
        last_bar = -9999
        start_i  = max(55, next((i for i, ts in enumerate(df1.index) if ts >= START), 55))
        for i in range(start_i, len(df1) - 1):
            if i - last_bar < 8: continue
            ts_bar = df1.index[i]
            if ts_bar < START: continue
            r = df1.iloc[i]
            if str(r['regime']) != 'BEAR': continue
            cl  = float(r['c']); op = float(r['o'])
            atr = float(r['atr']); atr_avg = float(r['atr_avg48'])
            l4  = float(r['low4'])
            if any(np.isnan(v) for v in [cl, atr, atr_avg, l4]): continue
            if atr < atr_avg: continue
            if cl < l4 and cl < op:
                sl_p = cl + 1.0 * atr
                if sl_p > cl:
                    nr = sim_short(df1, i, cl, sl_p, rt=RT_USDT)
                    all_trades.append({'ts': ts_bar, 'sym': sym, 'side': 'short', 'nr': nr})
                    last_bar = i
    return all_trades

# ─────────────────────────────────────────────
# Relatorio
# ─────────────────────────────────────────────
def report(all_trades, label):
    print(f'\n{"="*70}')
    print(f'  {label}')
    print('='*70)
    if not all_trades:
        print('  Nenhum trade.'); return

    all_trades = sorted(all_trades, key=lambda t: t['ts'])
    capital = 100.0; months = {}
    for t in all_trades:
        mk  = t['ts'].strftime('%Y-%m')
        pnl = t['nr'] * capital * 0.01
        capital += pnl
        if mk not in months:
            months[mk] = {'cnt':0,'wins':0,'nr_sum':0.0,
                          'longs':0,'shorts':0,'start':capital - pnl}
        months[mk]['end_cap'] = capital
        months[mk]['cnt']    += 1
        if t['nr'] > 0: months[mk]['wins'] += 1
        months[mk]['nr_sum'] += t['nr']
        if t['side'] == 'long': months[mk]['longs'] += 1
        else:                   months[mk]['shorts'] += 1

    tw = tl = 0
    for mk in sorted(months.keys()):
        m    = months[mk]; lbl = mlabels.get(mk, mk)
        pct  = (m['end_cap'] / m['start'] - 1) * 100
        wr   = m['wins'] / m['cnt'] * 100
        avgr = m['nr_sum'] / m['cnt']
        tw  += m['wins']; tl += m['cnt'] - m['wins']
        mark = '+' if pct >= 0 else ''
        print(f"  {lbl}: {m['cnt']:4d} t (L:{m['longs']:3d} S:{m['shorts']:3d})"
              f"  WR {wr:4.1f}%  avg {avgr:+.3f}R  {mark}{pct:.1f}%  ${m['end_cap']:.2f}")

    nr_all   = [t['nr'] for t in all_trades]
    gw       = sum(r for r in nr_all if r > 0)
    gl       = abs(sum(r for r in nr_all if r < 0))
    pf       = gw / gl if gl > 0 else 99.0
    total_wr = tw / (tw + tl) * 100
    avg_r    = sum(nr_all) / len(nr_all)

    print(f'\n  TOTAL : {len(nr_all)} t | WR {total_wr:.1f}% | avg {avg_r:+.3f}R | PF {pf:.2f}')
    print(f'  $100  -> ${capital:.2f}  ({(capital/100-1)*100:+.1f}%)')

    longs  = [t['nr'] for t in all_trades if t['side'] == 'long']
    shorts = [t['nr'] for t in all_trades if t['side'] == 'short']
    if longs:
        lwr = sum(1 for r in longs if r > 0) / len(longs) * 100
        print(f'    LONGs : {len(longs)} t | WR {lwr:.1f}% | avg {sum(longs)/len(longs):+.3f}R')
    if shorts:
        swr = sum(1 for r in shorts if r > 0) / len(shorts) * 100
        print(f'    SHORTs: {len(shorts)} t | WR {swr:.1f}% | avg {sum(shorts)/len(shorts):+.3f}R')

# ─────────────────────────────────────────────
# Executa
# ─────────────────────────────────────────────
print('Rodando backtests...')

t_trap = run_fb(stop_mode='trap')
t_atr  = run_fb(stop_mode='atr')
t_bo1h = run_breakout_short_1h()

report(t_trap, 'FALSE BREAKOUT 15M — Stop no extremo do trap  (fees USDC)')
report(t_atr,  'FALSE BREAKOUT 15M — Stop 1xATR fixo          (fees USDC)')
report(t_bo1h, 'REFERENCIA: Breakout SHORT 1H ATR>avg          (fees USDT)')

# Combinado: FB SHORT 15M + Breakout SHORT 1H
fb_shorts = [t for t in t_trap if t['side'] == 'short']
combined  = sorted(fb_shorts + t_bo1h, key=lambda t: t['ts'])
report(combined, 'COMBINADO: FB-SHORT 15M + Breakout-SHORT 1H')

# Combinado completo: FB ambos + Breakout SHORT
combined_full = sorted(t_trap + t_bo1h, key=lambda t: t['ts'])
report(combined_full, 'COMBINADO COMPLETO: FB LONG+SHORT 15M + Breakout-SHORT 1H')
