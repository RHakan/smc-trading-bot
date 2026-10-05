"""
backtest_bull_dev_hist.py
Validacao out-of-sample dos FILTROS da Alta nos bulls reais.

O backtest_bull_dev.py achou que os filtros (slope>=0.006 + BTC bull + close>EMA100)
reduzem o sangramento da bull no periodo DURO (jun25-jun26): -990 -> -218 USDC.
Mas esses filtros foram escolhidos NESSE periodo. A pergunta decisiva:

  Os filtros mantem o LUCRO nos bulls reais, ou cortaram tanto que mataram tudo?

Roda a bull no MESMO motor de portfolio (saldo unico, correlacao), nos dois bulls
historicos, comparando SEM filtros vs COM filtros:
  Bull 2023-2025 : 18/06/2023 -> 06/10/2025
  Bull 2020-2021 : 13/03/2020 -> 11/11/2021

Se COM filtros ainda der lucro forte -> Alta resolvida (paga em bull, neutra fora).
Se COM filtros zerar o lucro -> filtros agressivos demais, repensar.

Usa o cache _1h_2020_2025.csv (do backtest_bull_smc). 1d e baixado se faltar.
Uso: python backtests/backtest_bull_dev_hist.py
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

PERIODS = [
    {'name':'Bull 2023-2025','start':datetime(2023,6,18,tzinfo=timezone.utc),'end':datetime(2025,10,6,tzinfo=timezone.utc)},
    {'name':'Bull 2020-2021','start':datetime(2020,3,13,tzinfo=timezone.utc),'end':datetime(2021,11,11,tzinfo=timezone.utc)},
]
FETCH_START=datetime(2020,1,1,tzinfo=timezone.utc)
FETCH_END  =datetime(2025,10,6,tzinfo=timezone.utc)

INITIAL_BALANCE=5000.0; RISK_PCT=1.0
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD; B_ATR_AVG=BEAR.ATR_AVG_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
U_SWING_N=10; U_CHOCH_BARS=12; U_CHOCH_REF=15; U_MIN_SWEEP=0.05
U_RR=2.5; U_BE_PCT=0.70; U_TRAIL=2.0; U_COOLDOWN=3

CACHE_DIR=Path(__file__).parent/'cache'; CACHE_DIR.mkdir(exist_ok=True)
ex=ccxt.binanceusdm({'enableRateLimit':True})


def fetch(sym, tf):
    safe=sym.replace('/','_').replace(':','_')
    cache=CACHE_DIR/f'{safe}_{tf}_2020_2025.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True); return df
    since=int((FETCH_START-timedelta(days=5)).timestamp()*1000)
    end_ms=int(FETCH_END.timestamp()*1000); rows=[]
    while True:
        b=ex.fetch_ohlcv(sym,tf,since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=df.set_index('ts').sort_index()[lambda d:d.index<=FETCH_END]
    df.to_csv(cache); return df


print('Validacao dos filtros da Alta nos bulls historicos')
print('A carregar dados (1h em cache; 1d baixa se faltar)...')
pair_data={}; btc_bull=None
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        df1=fetch(sym,'1h').copy()
        dfd=fetch(sym,'1d').copy()
        if len(df1)<300: print('SEM DADOS'); continue
        ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
        slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
        dfd['regime']='NEUTRAL'
        dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
        dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
        dfd['slope_d']=slope
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-df1['c'].shift(1)).abs(),
                      (df1['l']-df1['c'].shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['atr_avg']=df1['atr'].rolling(B_ATR_AVG).mean()
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['slope_d']=dfd['slope_d'].shift(1).reindex(df1.index,method='ffill')
        for p in (50,100,200): df1[f'ema{p}']=df1['c'].ewm(span=p,adjust=False).mean()
        pair_data[sym]=df1
        if sym.startswith('BTC'): btc_bull=(df1['regime']=='BULL')
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen_bull_signals(df, btc_local, min_slope, btc_filter, ema_trend):
    n=len(df)
    h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values
    reg=df['regime'].values; slope=df['slope_d'].values
    ema_t=df[f'ema{ema_trend}'].values if ema_trend else None
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan)
    start_i=max(U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5,210)
    for i in range(start_i,n):
        if reg[i]!='BULL': continue
        if min_slope>0 and (np.isnan(slope[i]) or slope[i]<min_slope): continue
        if btc_filter and not btc_local[i]: continue
        if ema_t is not None and (np.isnan(ema_t[i]) or c[i]<=ema_t[i]): continue
        a=atr[i]; av=atr_avg[i]
        if np.isnan(a) or (not np.isnan(av) and a<av): continue
        best=None
        for j in range(i-1,max(i-U_CHOCH_BARS-1,U_SWING_N+U_CHOCH_REF)-1,-1):
            swing_low=np.min(l[j-U_SWING_N:j])
            if not (l[j]<swing_low and c[j]>swing_low): continue
            if U_MIN_SWEEP>0 and (swing_low-l[j])/swing_low*100<U_MIN_SWEEP: continue
            ref_high=np.max(h[j-U_CHOCH_REF:j])
            if np.any(c[j+1:i]>ref_high): continue
            if c[i]>ref_high:
                actual_low=np.min(l[j:i+1]); risk=c[i]-actual_low
                if risk>0 and risk/c[i]<=0.10: best=(c[i],actual_low); break
        if best: side[i]='LONG'; entry[i]=best[0]; sl[i]=best[1]
    return side,entry,sl


def update_position(pos,hi,lo,c,atr):
    entry=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
    a=atr if not np.isnan(atr) else risk
    if lo<=pos['cur']:
        return True,(pos['cur']-entry)/risk-fee_r
    if hi>=pos['tp']: return True,U_RR-fee_r
    if not pos['be'] and (c-entry)/risk>=U_BE_PCT*U_RR: pos['cur']=entry; pos['be']=True
    if pos['be']:
        cand=c-U_TRAIL*a
        if cand>pos['cur']: pos['cur']=cand
    return False,0.0


def run_period(period, min_slope, btc_filter, ema_trend):
    # indice mestre do periodo
    midx=None
    arrs={}
    for sym,df in pair_data.items():
        btc_local=btc_bull.reindex(df.index).fillna(False).values
        side,entry,sl=gen_bull_signals(df,btc_local,min_slope,btc_filter,ema_trend)
        d=df.assign(_s=side,_e=entry,_sl=sl)
        midx=d.index if midx is None else midx.union(d.index)
        arrs[sym]=d
    midx=midx[(midx>=period['start'])&(midx<=period['end'])]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,
                'atr':dd['atr'].values,'side':dd['_s'].values,
                'entry':dd['_e'].values,'sl':dd['_sl'].values}
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]
    for k in range(len(midx)):
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]; closed,nr=update_position(pos,d['h'][k],d['l'][k],c,d['atr'][k])
            if closed:
                balance+=nr*pos['risk_usd']; trades.append(nr)
                del positions[sym]; cooldown_until[sym]=k+U_COOLDOWN
        for sym in A:
            if sym in positions: continue
            if k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            entry=d['entry'][k]; sl=d['sl'][k]
            if np.isnan(entry) or np.isnan(sl) or balance<=0: continue
            risk_usd=balance*(RISK_PCT/100.0); risk_px=entry-sl
            if risk_px<=0: continue
            positions[sym]={'entry':entry,'cur':sl,'tp':entry+U_RR*risk_px,'be':False,
                            'risk_px':risk_px,'risk_usd':risk_usd,'fee_r':entry*RT/risk_px}
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return {'balance':balance,'trades':trades,'max_dd':max_dd,'days':(period['end']-period['start']).days}


def show(label, res):
    t=res['trades']; bal=res['balance']
    if not t: print(f'    {label:<22}: sem trades'); return
    w=[x for x in t if x>0]; gw=sum(w); gl=abs(sum(x for x in t if x<=0))
    ret=(bal/INITIAL_BALANCE-1)*100; pnl=bal-INITIAL_BALANCE
    print(f'    {label:<22}: {pnl:>+9.0f} USDC ({ret:>+7.1f}%) | {len(t):>4}t '
          f'({len(t)/res["days"]:>4.2f}/dia) | WR {len(w)/len(t)*100:>4.1f}% | '
          f'PF {gw/gl if gl>0 else 99:>4.2f} | DD {res["max_dd"]:>4.1f}%')


CONFIGS = [
    ('SEM filtros',          0.0,   False, 0),
    ('slope0.006',           0.006, False, 0),
    ('+BTC',                 0.006, True,  0),
    ('+BTC+EMA100 (escolha)',0.006, True,  100),
    ('slope0.015 (mais duro)',0.015,False, 0),
]

print('='*112)
print('  FILTROS NOS BULLS REAIS  (queremos: COM filtro MANTER o lucro)')
print('='*112)
for p in PERIODS:
    print(f'\n  ── {p["name"]}  ({p["start"].date()} -> {p["end"].date()}) ──')
    for lbl,ms,bf,et in CONFIGS:
        show(lbl, run_period(p, ms, bf, et))

print('\n' + '='*112)
print('  VEREDICTO')
print('='*112)
print('  Compara cada filtro vs "SEM filtros" em CADA bull.')
print('  Periodo DURO (jun25-jun26) ja mostrou: filtros levam -990 -> -218 USDC.')
print('  Se aqui o lucro AGUENTAR -> Alta resolvida (paga em bull, ~neutra fora).')
print('  Custos 0.14% round-trip. Saldo 5000 USDC, risco 1%/trade. Bull SMC HIPOTETICA.')
