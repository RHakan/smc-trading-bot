"""
backtest_bull_dev.py
Banco de desenvolvimento da estrategia de ALTA (bull SMC).

Objetivo: tornar a bull robusta no periodo DURO (jun/2025 -> jun/2026, que inclui
topo + transicao + bear), onde ela contribuiu -461 USDC no teste de portfolio.
O sinal (sweep + CHoCH) tem edge em bull real; o problema e QUANDO ela opera —
o Decisor aciona-a em "bull falso" (repique dentro de transicao/bear).

Testa 3 filtros de FORCA do bull (isola a bull no motor de portfolio realista —
saldo unico, correlacao, posicoes simultaneas — mas so com sinais de alta):

  A) min_slope_d  — exigir a EMA20 diaria mais inclinada (bull com forca, nao so "acima")
  B) btc_filter   — so comprar ALT quando o proprio BTC esta em bull (lider de mercado;
                    na transicao o BTC vira primeiro e arrasta as alts)
  C) ema_trend_1h — so comprar com a tendencia 1H intacta (close > EMA_N de 1H)

Meta da ALTA: contribuicao POSITIVA em USDC (medida em $, nao em nº de trades —
a frequencia alta vem da lateral, nao daqui). Sem destruir o lucro em bull real.

⚠️  Bull SMC continua HIPOTETICA. Aprovacao so apos validar tambem nos bulls
    historicos (2020-21, 2023-25) como out-of-sample reverso.
Uso: python backtests/backtest_bull_dev.py
"""
import sys, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ccxt, numpy as np, pandas as pd
from bot.strategies import bear_v13 as BEAR
from bot import regime as regime_mod

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

START = datetime(2025, 6,  1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 29, tzinfo=timezone.utc)
INITIAL_BALANCE = 5000.0
RISK_PCT        = 1.0
PERIOD_DAYS     = (END - START).days

PAIRS = [
    'BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
    'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
    'AVAX/USDT:USDT','DOT/USDT:USDT',
]
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD; B_ATR_AVG=BEAR.ATR_AVG_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH

# Bull SMC — config robusta (ponto de partida)
U_SWING_N=10; U_CHOCH_BARS=12; U_CHOCH_REF=15; U_MIN_SWEEP=0.05
U_RR=2.5; U_BE_PCT=0.70; U_TRAIL=2.0; U_COOLDOWN=3

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
ex = ccxt.binanceusdm({'enableRateLimit': True})


def fetch(sym, tf, extra_days):
    safe  = sym.replace('/', '_').replace(':', '_')
    cache = CACHE_DIR / f'{safe}_{tf}_jun25_jun26.csv'
    if cache.exists():
        df = pd.read_csv(cache, index_col='ts', parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    since  = int((START - timedelta(days=extra_days)).timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    rows=[]
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=['ts','o','h','l','c','v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('ts').sort_index()[lambda d: d.index <= END]
    df.to_csv(cache)
    return df


print('Banco de desenvolvimento da ALTA (bull SMC)')
print(f'  Periodo: {START:%d/%m/%Y} -> {END:%d/%m/%Y} ({PERIOD_DAYS} dias)  |  saldo {INITIAL_BALANCE:.0f} USDC')
print('  (periodo DURO: topo + transicao + bear — onde a bull sangrava)\n')
print('A carregar dados (cache)...')

pair_data = {}
btc_bull = None
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd = fetch(sym, '1d', 40).copy()
        df1 = fetch(sym, '1h', 25).copy()
        if len(df1) < 300: print('SEM DADOS'); continue

        ema_d = dfd['c'].ewm(span=R_EMA, adjust=False).mean()
        slope = (ema_d - ema_d.shift(R_SLOPE)) / ema_d.shift(R_SLOPE)
        dfd['regime'] = 'NEUTRAL'
        dfd.loc[(dfd['c'] < ema_d) & (slope < -R_THRESH), 'regime'] = 'BEAR'
        dfd.loc[(dfd['c'] > ema_d) & (slope >  R_THRESH), 'regime'] = 'BULL'
        dfd['slope_d'] = slope

        tr = pd.concat([(df1['h']-df1['l']),
                        (df1['h']-df1['c'].shift(1)).abs(),
                        (df1['l']-df1['c'].shift(1)).abs()], axis=1).max(axis=1)
        df1['atr']     = tr.ewm(com=ATR_PERIOD-1, adjust=False).mean()
        df1['atr_avg'] = df1['atr'].rolling(B_ATR_AVG).mean()
        df1['regime']  = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
        df1['slope_d'] = dfd['slope_d'].shift(1).reindex(df1.index, method='ffill')
        for p in (50, 100, 200):
            df1[f'ema{p}'] = df1['c'].ewm(span=p, adjust=False).mean()

        pair_data[sym] = df1
        if sym.startswith('BTC'):
            # serie booleana: BTC em bull (regime do dia anterior, ja no 1H)
            btc_bull = (df1['regime'] == 'BULL')
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()

# Indice mestre + alinhamento do filtro BTC
master_index=None
for sym, df in pair_data.items():
    master_index = df.index if master_index is None else master_index.union(df.index)
master_index = master_index[(master_index>=START)&(master_index<=END)]
btc_bull_arr = btc_bull.reindex(master_index).fillna(False).values


def gen_bull_signals(df, btc_arr_local, min_slope, btc_filter, ema_trend):
    """Sinais de alta (sweep+CHoCH) com filtros de forca do bull."""
    n=len(df)
    h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values
    reg=df['regime'].values; slope=df['slope_d'].values
    ema_t = df[f'ema{ema_trend}'].values if ema_trend else None

    side=np.array([None]*n, dtype=object)
    entry=np.full(n, np.nan); sl=np.full(n, np.nan)
    start_i=max(U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, 210)

    for i in range(start_i, n):
        if reg[i] != 'BULL': continue
        # Filtro A: forca do bull (slope diario)
        if min_slope > 0 and (np.isnan(slope[i]) or slope[i] < min_slope): continue
        # Filtro B: BTC tambem em bull
        if btc_filter and not btc_arr_local[i]: continue
        # Filtro C: tendencia 1H intacta
        if ema_t is not None and (np.isnan(ema_t[i]) or c[i] <= ema_t[i]): continue
        a=atr[i]; av=atr_avg[i]
        if np.isnan(a) or (not np.isnan(av) and a < av): continue
        best=None
        for j in range(i-1, max(i-U_CHOCH_BARS-1, U_SWING_N+U_CHOCH_REF)-1, -1):
            swing_low=np.min(l[j-U_SWING_N:j])
            if not (l[j]<swing_low and c[j]>swing_low): continue
            if U_MIN_SWEEP>0 and (swing_low-l[j])/swing_low*100 < U_MIN_SWEEP: continue
            ref_high=np.max(h[j-U_CHOCH_REF:j])
            if np.any(c[j+1:i] > ref_high): continue
            if c[i]>ref_high:
                actual_low=np.min(l[j:i+1]); risk=c[i]-actual_low
                if risk>0 and risk/c[i]<=0.10:
                    best=(c[i], actual_low); break
        if best:
            side[i]='LONG'; entry[i]=best[0]; sl[i]=best[1]
    return side, entry, sl


def build_arrays(min_slope, btc_filter, ema_trend):
    A={}
    for sym, df in pair_data.items():
        # Filtro BTC alinhado ao indice deste par (regime do BTC por barra)
        btc_local = btc_bull.reindex(df.index).fillna(False).values
        side, entry, sl = gen_bull_signals(df, btc_local, min_slope, btc_filter, ema_trend)
        d = df.assign(_s=side,_e=entry,_sl=sl).reindex(master_index)
        A[sym]={'h':d['h'].values,'l':d['l'].values,'c':d['c'].values,
                'atr':d['atr'].values,'side':d['_s'].values,
                'entry':d['_e'].values,'sl':d['_sl'].values}
    return A


def update_position(pos, hi, lo, c, atr):
    entry=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
    a=atr if not np.isnan(atr) else risk
    if lo<=pos['cur']:
        m='breakeven' if (pos['be'] and pos['cur']==entry) else ('trailing' if pos['be'] else 'stop')
        return True, (pos['cur']-entry)/risk-fee_r, m
    if hi>=pos['tp']: return True, U_RR-fee_r, 'take_profit'
    if not pos['be'] and (c-entry)/risk>=U_BE_PCT*U_RR: pos['cur']=entry; pos['be']=True
    if pos['be']:
        cand=c-U_TRAIL*a
        if cand>pos['cur']: pos['cur']=cand
    return False, 0.0, None


def run_portfolio(A, max_concurrent=None):
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]; max_conc=0
    for k in range(len(master_index)):
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]
            closed,nr,m=update_position(pos, d['h'][k], d['l'][k], c, d['atr'][k])
            if closed:
                balance+=nr*pos['risk_usd']
                trades.append({'ts_out':master_index[k],'nr':nr})
                del positions[sym]; cooldown_until[sym]=k+U_COOLDOWN
        for sym in A:
            if sym in positions: continue
            if max_concurrent is not None and len(positions)>=max_concurrent: break
            if k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            entry=d['entry'][k]; sl=d['sl'][k]
            if np.isnan(entry) or np.isnan(sl) or balance<=0: continue
            risk_usd=balance*(RISK_PCT/100.0); risk_px=entry-sl
            if risk_px<=0: continue
            positions[sym]={'entry':entry,'cur':sl,'tp':entry+U_RR*risk_px,'be':False,
                            'risk_px':risk_px,'risk_usd':risk_usd,'fee_r':entry*RT/risk_px}
        max_conc=max(max_conc,len(positions))
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return {'balance':balance,'trades':trades,'max_dd':max_dd,'max_conc':max_conc}


def show(label, res, base=False):
    t=res['trades']; bal=res['balance']
    mark='*' if base else ' '
    if not t:
        print(f'  {label:<30}{mark}: sem trades'); return res
    w=[x for x in t if x['nr']>0]
    gw=sum(x['nr'] for x in w); gl=abs(sum(x['nr'] for x in t if x['nr']<=0))
    ret=(bal/INITIAL_BALANCE-1)*100; pnl=bal-INITIAL_BALANCE
    print(f'  {label:<30}{mark}: {pnl:>+8.0f} USDC ({ret:>+6.1f}%) | {len(t):>4}t '
          f'({len(t)/PERIOD_DAYS:>4.2f}/dia) | WR {len(w)/len(t)*100:>4.1f}% | '
          f'PF {gw/gl if gl>0 else 99:>4.2f} | DD {res["max_dd"]:>4.1f}%')
    return res


print('='*120)
print('  DESENVOLVIMENTO DA ALTA — bull isolada no motor de portfolio (so sinais de alta)')
print('='*120)
print('  Base = bull SMC atual, sem filtros novos (referencia: -461 USDC no portfolio completo)\n')

show('BASE (sem filtros)', run_portfolio(build_arrays(0.0, False, 0)), base=True)

print('\nFiltro A — forca do bull (slope minimo da EMA20 diaria):')
for ms in [0.003, 0.006, 0.010, 0.015]:
    show(f'min_slope={ms}', run_portfolio(build_arrays(ms, False, 0)))

print('\nFiltro B — BTC tambem em bull (lider de mercado):')
show('btc_filter=ON', run_portfolio(build_arrays(0.0, True, 0)))

print('\nFiltro C — tendencia 1H intacta (close > EMA):')
for et in [50, 100, 200]:
    show(f'ema_trend_1h={et}', run_portfolio(build_arrays(0.0, False, et)))

print('\nCombinacoes dos filtros:')
combos = [
    (0.006, True,  0),
    (0.006, False, 100),
    (0.0,   True,  100),
    (0.006, True,  100),
    (0.010, True,  200),
    (0.006, True,  50),
]
results=[]
for ms,bf,et in combos:
    lbl=f'slope{ms} btc{"ON" if bf else "--"} ema{et or "--"}'
    r=show(lbl, run_portfolio(build_arrays(ms, bf, et)))
    results.append(((ms,bf,et), r))

best_p,best_r = max(results, key=lambda x:x[1]['balance'])
print('\n' + '='*120)
print(f'  MELHOR FILTRO: slope={best_p[0]}  btc={"ON" if best_p[1] else "OFF"}  '
      f'ema_trend={best_p[2] or "OFF"}  ->  {best_r["balance"]:,.0f} USDC '
      f'({(best_r["balance"]/INITIAL_BALANCE-1)*100:+.1f}%)  DD {best_r["max_dd"]:.1f}%')
print('='*120)
print('  Objetivo: sair do negativo (-461) para POSITIVO no periodo duro, sem matar a frequencia.')
print('  Proximo passo se algum filtro funcionar: validar nos bulls historicos (2020-21, 2023-25).')
print('  Custos round-trip 0.14% incluidos. Bull SMC ainda HIPOTETICA.')
