"""
backtest_sistema_v2.py
Sistema v2 — base LATERAL (robusta) + BULL (complemento), validado em 5 anos.

A lateral (breakout NEUTRAL) e o alicerce: 4/5 anos+, DD max 6.6%. A bull sozinha
foi +930 mas so 2/5 anos. Pergunta: somar a bull MELHORA a consistencia ou
atrapalha? E: com DD tao baixo, ate onde da p/ escalar o risco com seguranca?

Testa, em 2021-2025:
  1. LATERAL so          (a base)
  2. LATERAL + BULL       (a bull ajuda?)
  3. LATERAL + BULL com risco 1/2/3/4% (escala segura rumo a magnitude)

Decisor: NEUTRAL->lateral breakout, BULL->bull SMC+filtros. SEM bear (sem edge).
max_per_side=3. Saldo 5000/ano. Usa cache _2020_2025.
Uso: python backtests/backtest_sistema_v2.py
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

YEARS=[2021,2022,2023,2024,2025]
YEAR_END={2025:datetime(2025,10,6,tzinfo=timezone.utc)}
YEAR_CTX={2021:'bull',2022:'BEAR',2023:'recup',2024:'bull',2025:'misto'}
FETCH_END=datetime(2025,10,6,tzinfo=timezone.utc)
INITIAL_BALANCE=5000.0; MAX_PER_SIDE=3
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
# Bull SMC + filtros
U_SWING_N=10; U_CHOCH_BARS=12; U_CHOCH_REF=15; U_MIN_SWEEP=0.05; U_RR=2.5; U_BE=0.70; U_TRAIL=2.0
U_COOLDOWN=3; U_SLOPE_MIN=0.006; U_EMA_TREND=100
# Lateral breakout
L_LOOKBACK=30; L_ADX_MAX=20; L_BUFFER=0.1; L_COOLDOWN=4; L_TIMEOUT=48

CACHE_DIR=Path(__file__).parent/'cache'; CACHE_DIR.mkdir(exist_ok=True)
ex=ccxt.binanceusdm({'enableRateLimit':True})


def fetch(sym, tf):
    safe=sym.replace('/','_').replace(':','_'); cache=CACHE_DIR/f'{safe}_{tf}_2020_2025.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True); return df
    since=int((datetime(2020,1,1,tzinfo=timezone.utc)-timedelta(days=5)).timestamp()*1000)
    end_ms=int(FETCH_END.timestamp()*1000); rows=[]
    while True:
        b=ex.fetch_ohlcv(sym,tf,since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=df.set_index('ts').sort_index()[lambda d:d.index<=FETCH_END]; df.to_csv(cache); return df


def adx(df, period=14):
    h=df['h']; l=df['l']; c=df['c']; up=h.diff(); dn=-l.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/period,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    mdi=100*pd.Series(mdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period,adjust=False).mean()


print('SISTEMA v2 — lateral (base) + bull (complemento), validado em 5 anos')
print('  2021 bull | 2022 BEAR | 2023 recup | 2024 bull | 2025 misto\n')
print('A carregar dados (cache 2020-2025)...')
pair_data={}; btc_bull_full=None
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd=fetch(sym,'1d').copy(); df1=fetch(sym,'1h').copy()
        if len(df1)<500: print('SEM DADOS'); continue
        ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
        slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
        dfd['regime']='NEUTRAL'
        dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
        dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
        dfd['slope_d']=slope
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['adx']=adx(df1,14)
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['slope_d']=dfd['slope_d'].shift(1).reindex(df1.index,method='ffill')
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        df1['btc_bull']=(df1['regime']=='BULL')
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen(df, use_bull):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; adx_a=df['adx'].values; reg=df['regime'].values
    slope=df['slope_d'].values; ema_t=df['ema_trend'].values; btc_local=df['btc_bull'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object)
    start_i=max(U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, L_LOOKBACK+5, 210)
    for i in range(start_i,n):
        r=reg[i]; a=atr[i]
        if np.isnan(a) or a<=0: continue
        if r=='NEUTRAL':
            if np.isnan(adx_a[i-1]) or adx_a[i-1]>L_ADX_MAX: continue
            rl=np.min(l[i-L_LOOKBACK:i]); rh=np.max(h[i-L_LOOKBACK:i])
            if rl<=0 or rh<=rl: continue
            buf=L_BUFFER*a; cl=c[i]; pc=c[i-1]; height=rh-rl
            if cl>rh+buf and pc<=rh:
                stop=rl; risk=cl-stop
                if risk>0: side[i]='LONG'; entry[i]=cl; sl[i]=stop; tp[i]=cl+height; strat[i]='lat'
            elif cl<rl-buf and pc>=rl:
                stop=rh; risk=stop-cl
                if risk>0: side[i]='SHORT'; entry[i]=cl; sl[i]=stop; tp[i]=cl-height; strat[i]='lat'
        elif use_bull and r=='BULL':
            if np.isnan(slope[i]) or slope[i]<U_SLOPE_MIN: continue
            if not btc_local[i]: continue
            if np.isnan(ema_t[i]) or c[i]<=ema_t[i]: continue
            best=None
            for j in range(i-1,max(i-U_CHOCH_BARS-1,U_SWING_N+U_CHOCH_REF)-1,-1):
                swing_low=np.min(l[j-U_SWING_N:j])
                if not (l[j]<swing_low and c[j]>swing_low): continue
                if U_MIN_SWEEP>0 and (swing_low-l[j])/swing_low*100<U_MIN_SWEEP: continue
                ref_high=np.max(h[j-U_CHOCH_REF:j])
                if np.any(c[j+1:i]>ref_high): continue
                if c[i]>ref_high:
                    al=np.min(l[j:i+1]); risk=c[i]-al
                    if risk>0 and risk/c[i]<=0.10: best=(c[i],al); break
            if best:
                e,stop=best; risk=e-stop
                side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=e+U_RR*risk; strat[i]='bull'
    return side,entry,sl,tp,strat


GEST={'bull':dict(be=U_BE,trail=U_TRAIL,manage='trail',cd=U_COOLDOWN),
      'lat': dict(be=0,trail=0,manage='fixed',cd=L_COOLDOWN)}


def run(use_bull, start, end, risk_pct):
    midx=None; arrs={}
    for sym,df in pair_data.items():
        s,e,sl,tp,st=gen(df, use_bull)
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st)
        midx=d.index if midx is None else midx.union(d.index); arrs[sym]=d
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,'tp':dd['_tp'].values,'strat':dd['_st'].values}
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]
    for k in range(len(midx)):
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]; atr=d['atr'][k]
            risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']; a=atr if not np.isnan(atr) else risk
            closed=False; nr=0.0
            if pos['side']=='LONG':
                if lo<=pos['cur']: nr=(pos['cur']-e)/risk-fee_r; closed=True
                elif hi>=pos['tp']: nr=(pos['tp']-e)/risk-fee_r; closed=True
                elif pos['manage']=='trail':
                    if not pos['be_done'] and (c-e)/risk>=pos['be']*((pos['tp']-e)/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done']:
                        cand=c-pos['trail']*a
                        if cand>pos['cur']: pos['cur']=cand
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
            if not closed and pos['manage']=='fixed':
                pos['age']+=1
                if pos['age']>=L_TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']; trades.append({'nr':nr,'strat':pos['strat']})
                del positions[sym]; cooldown_until[sym]=k+pos['cd']
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sym in A:
            if sym in positions: continue
            if k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            if side=='LONG' and n_long>=MAX_PER_SIDE: continue
            if side=='SHORT' and n_short>=MAX_PER_SIDE: continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]; strat=d['strat'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            g=GEST[strat]
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'be_done':False,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(risk_pct/100.0),'fee_r':e*RT/risk_px,
                            'be':g['be'],'trail':g['trail'],'manage':g['manage'],'cd':g['cd']}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    nr=[t['nr'] for t in trades]; w=[x for x in nr if x>0]
    pf=sum(w)/abs(sum(x for x in nr if x<=0)) if any(x<=0 for x in nr) else 99
    return dict(pnl=balance-INITIAL_BALANCE,t=len(nr),pf=pf,dd=max_dd,trades=trades)


def matriz(label, use_bull, risk_pct):
    pnls={}
    for yr in YEARS:
        start=datetime(yr,1,1,tzinfo=timezone.utc); end=YEAR_END.get(yr,datetime(yr,12,31,tzinfo=timezone.utc))
        pnls[yr]=run(use_bull, start, end, risk_pct)
    tot=sum(p['pnl'] for p in pnls.values()); pos=sum(1 for p in pnls.values() if p['pnl']>0)
    maxdd=max(p['dd'] for p in pnls.values())
    row='  '+f"{label:<24}"+''.join(f"{pnls[y]['pnl']:>+9.0f}" for y in YEARS)+f"{tot:>+9.0f}{pos:>5}/5{maxdd:>7.1f}%"
    print(row)
    return tot,pos,pnls


print('='*104)
print('  LATERAL so vs LATERAL+BULL (risco 1%) — anos+ = consistencia, DDmax = pior ano')
print('='*104)
print('  '+f"{'Config':<24}"+''.join(f"{str(y)[2:]+YEAR_CTX[y][:1]:>9}" for y in YEARS)+f"{'TOTAL':>9}{'anos+':>6}{'DDmax':>8}")
print('  '+'-'*102)
matriz('LATERAL so', False, 1.0)
matriz('LATERAL + BULL', True, 1.0)

print('\n'+'='*104)
print('  ESCALA DE RISCO (lateral+bull) — o DD baixo permite subir o risco/trade')
print('='*104)
print('  '+f"{'Risco':<24}"+''.join(f"{str(y)[2:]+YEAR_CTX[y][:1]:>9}" for y in YEARS)+f"{'TOTAL':>9}{'anos+':>6}{'DDmax':>8}")
print('  '+'-'*102)
for rp in [1.0,2.0,3.0,4.0]:
    matriz(f'risco {rp}%', True, rp)

print('\n  Meta: manter >=4/5 anos+ com DDmax < 25% e o maior total. Robustez antes de magnitude.')
print('  Custos 0.14% round-trip. Bull e Lateral HIPOTETICAS. Bear APOSENTADO (sem edge).')
