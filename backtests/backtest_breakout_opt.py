"""
backtest_breakout_opt.py
Refinamento da LATERAL = BREAKOUT do range (a tese que venceu o showdown:
robusta, DD baixo, alinhada com a lei "cripto paga momentum").

No showdown as melhorias foram testadas ISOLADAS (adx20, stop=opposite, fixed,
rr2.0). Aqui COMBINAMOS e procuramos a melhor config robusta — e validamos com
um split temporal (1a metade vs 2a metade do periodo) para distinguir edge real
de sorte (overfit aparece como "boa numa metade, pessima na outra").

Motor de portfolio (saldo 5000, correlacao). Periodo jun/2025->jun/2026.
Uso: python backtests/backtest_breakout_opt.py
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

START=datetime(2025,6,1,tzinfo=timezone.utc); END=datetime(2026,6,29,tzinfo=timezone.utc)
MID_DATE=datetime(2025,12,15,tzinfo=timezone.utc)   # divisor do split temporal
INITIAL_BALANCE=5000.0; RISK_PCT=1.0; PERIOD_DAYS=(END-START).days
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
COOLDOWN=4; TIMEOUT=48

CACHE_DIR=Path(__file__).parent/'cache'; CACHE_DIR.mkdir(exist_ok=True)
ex=ccxt.binanceusdm({'enableRateLimit':True})


def fetch(sym, tf, extra_days):
    safe=sym.replace('/','_').replace(':','_'); cache=CACHE_DIR/f'{safe}_{tf}_jun25_jun26.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True); return df
    since=int((START-timedelta(days=extra_days)).timestamp()*1000); end_ms=int(END.timestamp()*1000); rows=[]
    while True:
        b=ex.fetch_ohlcv(sym,tf,since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=df.set_index('ts').sort_index()[lambda d:d.index<=END]; df.to_csv(cache); return df


def adx(df, period=14):
    h=df['h']; l=df['l']; c=df['c']; up=h.diff(); dn=-l.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/period,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    mdi=100*pd.Series(mdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period,adjust=False).mean()


print('BREAKOUT do range — refinamento + validacao temporal')
print(f'  Periodo: {START:%d/%m/%Y} -> {END:%d/%m/%Y} ({PERIOD_DAYS} dias) | saldo {INITIAL_BALANCE:.0f} | 1H')
print(f'  Split: 1a metade ate {MID_DATE:%d/%m/%Y} | 2a metade depois\n')
print('A carregar dados (cache)...')
pair_data={}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd=fetch(sym,'1d',40).copy(); df1=fetch(sym,'1h',25).copy()
        if len(df1)<300: print('SEM DADOS'); continue
        ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
        slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
        dfd['regime']='NEUTRAL'
        dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
        dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['adx']=adx(df1,14)
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()

master_index=None
for sym,df in pair_data.items():
    master_index=df.index if master_index is None else master_index.union(df.index)
master_index=master_index[(master_index>=START)&(master_index<=END)]


def gen_breakout(df, lookback, adx_max, stop_mode, rr_cap, buffer_atr, width_max, regime_filter='NEUTRAL'):
    n=len(df); h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; adx_a=df['adx'].values; reg=df['regime'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    for i in range(lookback+5,n):
        if regime_filter!='ALL' and reg[i]!=regime_filter: continue
        a=atr[i]
        if np.isnan(a): continue
        rl=np.min(l[i-lookback:i]); rh=np.max(h[i-lookback:i])
        if rl<=0 or rh<=rl: continue
        width=(rh-rl)/rl*100
        if width_max>0 and width>width_max: continue
        if adx_max>0 and (np.isnan(adx_a[i-1]) or adx_a[i-1]>adx_max): continue
        buf=buffer_atr*a; cl=c[i]; pc=c[i-1]; height=rh-rl
        if cl>rh+buf and pc<=rh:
            e=cl
            stop=(rl+rh)/2 if stop_mode=='mid' else (rl if stop_mode=='opposite' else rh-1.0*a)
            risk=e-stop
            if risk>0:
                target=e+rr_cap*risk if rr_cap>0 else e+height
                if target>e: side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=target; continue
        if cl<rl-buf and pc>=rl:
            e=cl
            stop=(rl+rh)/2 if stop_mode=='mid' else (rh if stop_mode=='opposite' else rl+1.0*a)
            risk=stop-e
            if risk>0:
                target=e-rr_cap*risk if rr_cap>0 else e-height
                if target<e: side[i]='SHORT'; entry[i]=e; sl[i]=stop; tp[i]=target
    return side,entry,sl,tp


def build_arrays(**kw):
    A={}
    for sym,df in pair_data.items():
        s,e,sl,tp=gen_breakout(df, **kw)
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp).reindex(master_index)
        A[sym]={'h':d['h'].values,'l':d['l'].values,'c':d['c'].values,'atr':d['atr'].values,
                'side':d['_s'].values,'entry':d['_e'].values,'sl':d['_sl'].values,'tp':d['_tp'].values}
    return A


def run_portfolio(A):
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]
    for k in range(len(master_index)):
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]
            risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']; closed=False; nr=0.0
            if pos['side']=='LONG':
                if lo<=pos['cur']: nr=(pos['cur']-e)/risk-fee_r; closed=True
                elif hi>=pos['tp']: nr=(pos['tp']-e)/risk-fee_r; closed=True
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
            if not closed:
                pos['age']+=1
                if pos['age']>=TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']
                trades.append({'ts':master_index[k],'nr':nr})
                del positions[sym]; cooldown_until[sym]=k+COOLDOWN
        for sym in A:
            if sym in positions: continue
            if k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px}
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return {'balance':balance,'trades':trades,'max_dd':max_dd}


def metrics(trades):
    if not trades: return None
    nr=[t['nr'] for t in trades]; w=[r for r in nr if r>0]
    gw=sum(w); gl=abs(sum(r for r in nr if r<=0))
    return dict(t=len(nr), wr=len(w)/len(nr)*100, pf=gw/gl if gl>0 else 99, sumr=sum(nr))


def split_halves(trades):
    h1=[t for t in trades if t['ts']<MID_DATE]; h2=[t for t in trades if t['ts']>=MID_DATE]
    return metrics(h1), metrics(h2)


def show(label, res, base=False):
    t=res['trades']; bal=res['balance']; mark='*' if base else ' '
    if not t: print(f'  {label:<40}{mark}: sem trades'); return res
    m=metrics(t); pnl=bal-INITIAL_BALANCE
    print(f'  {label:<40}{mark}: {pnl:>+7.0f} USDC | {pnl/PERIOD_DAYS:>+5.1f}/dia | {m["t"]:>3}t | '
          f'WR {m["wr"]:>4.1f}% | PF {m["pf"]:>4.2f} | DD {res["max_dd"]:>4.1f}%')
    return res


BASE=dict(lookback=30, adx_max=20, stop_mode='opposite', rr_cap=0, buffer_atr=0.1, width_max=0)

print('='*120)
print('  GRID DE COMBINACOES (a partir das melhorias que venceram isoladas)')
print('='*120)
grid=[]
for stop_mode in ['opposite','mid']:
    for adx_max in [20,25]:
        for rr_cap in [0,2.0]:
            for lookback in [30,50]:
                p=dict(lookback=lookback, adx_max=adx_max, stop_mode=stop_mode,
                       rr_cap=rr_cap, buffer_atr=0.1, width_max=0)
                r=run_portfolio(build_arrays(**p))
                lbl=f"{stop_mode[:3]} adx{adx_max} rr{rr_cap or 'H'} lb{lookback}"
                grid.append((p, r, lbl))

grid.sort(key=lambda x:-x[1]['balance'])
for p,r,lbl in grid: show(lbl, r)

print('\n'+'='*120)
print('  VALIDACAO TEMPORAL das TOP 3 (edge real = positiva nas DUAS metades)')
print('='*120)
for p,r,lbl in grid[:3]:
    m1,m2=split_halves(r['trades'])
    print(f'\n  {lbl}  (total {r["balance"]-INITIAL_BALANCE:+.0f} USDC, DD {r["max_dd"]:.1f}%)')
    for nome,m in [('1a metade',m1),('2a metade',m2)]:
        if m: print(f'    {nome}: {m["t"]:>3}t | WR {m["wr"]:>4.1f}% | PF {m["pf"]:>4.2f} | somaR {m["sumr"]:>+6.1f}')
        else: print(f'    {nome}: sem trades')

best_p,best_r,best_lbl=grid[0]
print('\n'+'='*120)
print(f'  MELHOR COMBO: {best_lbl}')
print(f'  -> {best_r["balance"]-INITIAL_BALANCE:+.0f} USDC | {(best_r["balance"]-INITIAL_BALANCE)/PERIOD_DAYS:+.1f}/dia | '
      f'PF {metrics(best_r["trades"])["pf"]:.2f} | DD {best_r["max_dd"]:.1f}%')
print(f'  Params: {best_p}')
print('='*120)
print('  Se positiva nas DUAS metades = edge robusto -> validar nos bulls historicos e ir ao motor conjunto.')
print('  Custos 0.14% round-trip. Risco 1%/trade. Lateral HIPOTETICA.')
