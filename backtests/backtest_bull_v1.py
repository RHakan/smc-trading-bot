"""
backtest_bull_v1.py
Backtest da estrategia Bull Market v1.0 (espelho da Bear v1.3) num periodo de
mercado de ALTA, para validar antes de aprovar/ativar.

Parametros vem direto de bot/strategies/bull_v1.py (fonte unica — se a
estrategia mudar, o backtest segue). Regime vem de bot/regime.py.

Uso:
    python backtest_bull_v1.py [ANO]
    python backtest_bull_v1.py 2024        # default
    python backtest_bull_v1.py 2023

Mostra:
  - Resumo: trades, win rate, soma R, resultado % com risco 1%/trade
  - Distribuicao mensal
  - Regime diario por par (confirma que o periodo foi mesmo BULL)
"""
import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

from bot.strategies import bull_v1 as S
from bot import regime as regime_mod

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Periodo (ano passado como argumento, default 2024) ────────────────────────
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

# Parametros vindos da estrategia bull (espelho da bear)
ATR_AVG   = S.ATR_AVG_PERIOD
ATR_STOP  = S.ATR_STOP_MULT
COOLDOWN  = S.COOLDOWN_BARS
LOOKBACK  = S.HIGH_LOOKBACK
RR_CAP    = S.RR_CAP
BE_PCT    = S.BE_TRIGGER_PCT
TRAIL_ATR = S.TRAIL_ATR

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


print(f'Backtest Bull v1.0 — ano {YEAR} ({START:%d/%m} a {END:%d/%m})')
print(f'Parametros (de bull_v1.py): lookback={LOOKBACK}  cooldown={COOLDOWN}H  '
      f'RR_CAP={RR_CAP}R  BE={int(BE_PCT*100)}%  TRAIL={TRAIL_ATR}xATR  '
      f'ATR_STOP={ATR_STOP}xATR  momentum=ATR>media{ATR_AVG}H\n')

print('A carregar dados da Binance...')
raw = {}
ok_pairs = []
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd = fetch(sym, '1d', 40).copy()
        if len(dfd) < 60:
            print('SEM DADOS suficientes — ignorado'); continue
        # Regime diario via modulo central (calculado vela a vela)
        ema = dfd['c'].ewm(span=regime_mod.EMA_PERIOD, adjust=False).mean()
        slope = (ema - ema.shift(regime_mod.SLOPE_BARS)) / ema.shift(regime_mod.SLOPE_BARS)
        dfd['regime'] = 'NEUTRAL'
        dfd.loc[(dfd['c'] < ema) & (slope < -regime_mod.SLOPE_THRESH), 'regime'] = 'BEAR'
        dfd.loc[(dfd['c'] > ema) & (slope >  regime_mod.SLOPE_THRESH), 'regime'] = 'BULL'

        df1 = fetch(sym, '1h', 6).copy()
        tr  = pd.concat([
            (df1['h'] - df1['l']),
            (df1['h'] - df1['c'].shift(1)).abs(),
            (df1['l'] - df1['c'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df1['atr']     = tr.ewm(com=S.ATR_PERIOD - 1, adjust=False).mean()
        df1['atr_avg'] = df1['atr'].rolling(ATR_AVG).mean()
        df1['regime']  = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
        df1['high_n']  = df1['h'].shift(1).rolling(LOOKBACK).max()
        raw[sym] = (df1, dfd)
        ok_pairs.append(sym)
        time.sleep(0.1)
        print('OK')
    except Exception as e:
        print(f'ERRO ({e}) — ignorado')
print()


def sim_long(df, ib, entry, sl_price):
    """Espelho de sim_short: SL abaixo, breakeven, trailing, TP acima."""
    risk = entry - sl_price
    if risk <= 0:
        return 0.0, 'invalido', entry
    tp    = entry + RR_CAP * risk
    fee_r = entry * RT / risk
    cur   = sl_price; be = False
    for j in range(ib + 1, min(ib + 400, len(df))):
        bar = df.iloc[j]
        lo = float(bar['l']); hi = float(bar['h']); c = float(bar['c'])
        atr = float(bar['atr']) if not np.isnan(bar['atr']) else risk
        # Stop / breakeven / trailing — atingido se a vela tocou o stop atual (em baixo)
        if lo <= cur:
            motivo = 'breakeven' if (be and cur == entry) else ('trailing' if be else 'stop_loss')
            return (cur - entry) / risk - fee_r, motivo, cur
        # Take profit (em cima)
        if hi >= tp:
            return RR_CAP - fee_r, 'take_profit', tp
        # Move para breakeven
        if not be and (c - entry) / risk >= BE_PCT * RR_CAP:
            cur = entry; be = True
        # Trailing depois do breakeven (stop sobe atras do preco)
        if be:
            cand = c - TRAIL_ATR * atr
            if cand > cur: cur = cand
    last_c = float(df.iloc[min(ib + 399, len(df) - 1)]['c'])
    return (last_c - entry) / risk - fee_r, 'timeout', last_c


# ── Simulacao ─────────────────────────────────────────────────────────────────
trades = []
regime_days = {}

for sym in ok_pairs:
    df1, dfd = raw[sym]
    dmask = (dfd.index >= START) & (dfd.index <= END)
    regime_days[sym] = dfd.loc[dmask, 'regime'].value_counts().to_dict()

    last_bar = -9999
    idx = df1.index
    for i in range(ATR_AVG + LOOKBACK + 2, len(df1) - 1):
        ts_bar = idx[i]
        if ts_bar < START or ts_bar > END:
            continue
        if i - last_bar < COOLDOWN:
            continue
        r = df1.iloc[i]
        if str(r['regime']) != 'BULL':
            continue
        cl = float(r['c']); op = float(r['o'])
        atr = float(r['atr']); atr_avg = float(r['atr_avg']); high_n = float(r['high_n'])
        if any(np.isnan(v) for v in [cl, atr, atr_avg, high_n]):
            continue
        # Condicoes de entrada — identicas a bull_v1.analyze (LONG)
        if not (cl > high_n and cl > op and atr > atr_avg):
            continue
        sl_p = cl - ATR_STOP * atr
        if sl_p >= cl:
            continue
        nr, motivo, exit_p = sim_long(df1, i, cl, sl_p)
        trades.append({'ts': ts_bar, 'sym': sym.replace('/USDT:USDT', ''),
                       'nr': nr, 'motivo': motivo})
        last_bar = i

trades.sort(key=lambda t: t['ts'])


def summarize(ts):
    if not ts: return None
    nr = [t['nr'] for t in ts]
    wins = [r for r in nr if r > 0]
    gw = sum(wins); gl = abs(sum(r for r in nr if r <= 0))
    cap = 100.0; months = {}
    for t in ts:
        mk = t['ts'].strftime('%Y-%m'); pnl = t['nr'] * cap * (RISK_PCT/100); cap += pnl
        if mk not in months: months[mk] = {'cnt':0,'wins':0,'start':cap-pnl}
        months[mk]['end_cap'] = cap; months[mk]['cnt'] += 1
        if t['nr'] > 0: months[mk]['wins'] += 1
    return {'t':len(nr),'wr':len(wins)/len(nr)*100,'avgr':sum(nr)/len(nr),
            'pf':gw/gl if gl>0 else 99,'total':cap-100,'cap':cap,'months':months,
            'soma_r':sum(nr),'wins':len(wins),'losses':len(nr)-len(wins)}


s = summarize(trades)

# ── Resumo ────────────────────────────────────────────────────────────────────
print('=' * 90)
print(f'  RESUMO — Bull v1.0 em {YEAR}')
print('=' * 90)
if s:
    motivos = {}
    for t in trades: motivos[t['motivo']] = motivos.get(t['motivo'], 0) + 1
    print(f"  Total de trades : {s['t']}")
    print(f"  Vitorias        : {s['wins']}  ({s['wr']:.1f}%)")
    print(f"  Derrotas        : {s['losses']}  ({100-s['wr']:.1f}%)")
    print(f"  Soma em R       : {s['soma_r']:+.2f}R")
    print(f"  Profit Factor   : {s['pf']:.2f}")
    print(f"  Resultado capital: {s['total']:+.2f}%  (risco {RISK_PCT}%/trade, composto)")
    print(f"\n  Motivos de saida:")
    for m, c in sorted(motivos.items(), key=lambda x: -x[1]):
        print(f"    {m:<12}: {c}")
else:
    print("  Sem trades — nao houve setups BULL validos no periodo.")

# ── Distribuicao mensal ───────────────────────────────────────────────────────
if s:
    print('\n' + '=' * 90)
    print('  DISTRIBUICAO MENSAL')
    print('=' * 90)
    print(f"{'Mes':<5} | {'Trades':>6} | {'Wins':>5} | {'WR':>6} | {'Capital':>20}")
    print('-' * 90)
    for mk in MKS:
        if mk not in s['months']: continue
        m = s['months'][mk]
        wr = m['wins']/m['cnt']*100 if m['cnt']>0 else 0
        pct = (m['end_cap']/m['start']-1)*100
        print(f"  {mlabels[mk]:<3} | {m['cnt']:>6} | {m['wins']:>5} | {wr:>5.1f}% | "
              f"${m['end_cap']:>8.2f}  ({pct:+.1f}%)")
    print(f"\n  Capital final: ${s['cap']:.2f}  |  Total: {s['total']:+.1f}%")

# ── Regime diario por par ─────────────────────────────────────────────────────
print('\n' + '=' * 90)
print('  REGIME DIARIO NO PERIODO (a bull so opera em dias BULL)')
print('=' * 90)
print(f"{'Par':<6} | {'Dias BULL':>9} | {'Dias BEAR':>9} | {'Dias NEUTRAL':>12}")
print('-' * 90)
tot_bull = tot_bear = tot_neu = 0
for sym in ok_pairs:
    s2 = sym.replace('/USDT:USDT', '')
    rd = regime_days.get(sym, {})
    b, be, n = rd.get('BULL',0), rd.get('BEAR',0), rd.get('NEUTRAL',0)
    tot_bull += b; tot_bear += be; tot_neu += n
    print(f"  {s2:<4} | {b:>9} | {be:>9} | {n:>12}")
print('-' * 90)
print(f"  {'TOTAL':<4} | {tot_bull:>9} | {tot_bear:>9} | {tot_neu:>12}")

dias_total = tot_bull + tot_bear + tot_neu
if dias_total:
    print(f"\n  {YEAR} foi {tot_bull/dias_total*100:.0f}% BULL, "
          f"{tot_bear/dias_total*100:.0f}% BEAR, {tot_neu/dias_total*100:.0f}% NEUTRAL")
print('\nNota: dados USDT. Custos: taxa 0.05% + slippage 0.02% por lado (round-trip 0.14%).')
