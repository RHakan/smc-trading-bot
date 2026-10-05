"""
backtest_bear_robustez.py
Confirma se o BEAR MELHORADO (seletividade + stop estrutural) e robusto ou sorte.

Vencedora em backtest_bear_melhorias: sel_lb=30, adx_max=20, stop=struct lb20.
  -> +1027, 3/5 anos+, DD 15.4%.

Aqui variamos os parametros EM TORNO dela. Se a vizinhanca toda for positiva /
3+ anos, e edge real (regiao robusta). Se so o ponto exato funciona, e overfit.

Bear isolada, regime BEAR, 5000/ano, risco 1%, max_per_side 3. Stop sempre
estrutural (lb variavel). Sem RSI dinamico (nao ajudou isolado).
Uso: python backtests/backtest_bear_robustez.py
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
INITIAL_BALANCE=5000.0; RISK_PCT=1.0; MAX_PER_SIDE=3
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD; B_ATR_AVG=BEAR.ATR_AVG_PERIOD
B_COOLDOWN=BEAR.COOLDOWN_BARS; B_RR=BEAR.RR_CAP; B_BE=BEAR.BE_TRIGGER_PCT; B_TRAIL=BEAR.TRAIL_ATR
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH

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


print('ROBUSTEZ do BEAR melhorado (seletividade + stop estrutural)')
print('  Vencedora: sel30 adx20 struct20 -> +1027, 3/5 anos, DD 15.4%\n')
print('A carregar dados (cache 2020-2025)...')
pair_data={}
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
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['atr_avg']=df1['atr'].rolling(B_ATR_AVG).mean()
        df1['adx']=adx(df1,14)
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen(df, sel_lb, adx_max, struct_lb):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values; reg=df['regime'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    start_i=max(B_ATR_AVG+sel_lb+5, struct_lb+5, 210)
    for i in range(start_i,n):
        if reg[i]!='BEAR': continue
        a=atr[i]; cl=c[i]; op=o[i]; av=atr_avg[i]
        if np.isnan(a) or a<=0 or np.isnan(av): continue
        low_n=np.min(l[i-sel_lb:i])
        if not (cl<low_n and cl<op and a>av): continue
        if np.isnan(adx_a[i-1]) or adx_a[i-1]>adx_max: continue
        stop=np.max(h[i-struct_lb:i+1])+0.1*a
        risk=stop-cl
        if risk<=0: continue
        side[i]='SHORT'; entry[i]=cl; sl[i]=stop; tp[i]=cl-B_RR*risk
    return side,entry,sl,tp


def run(cfg, start, end):
    midx=None; arrs={}
    for sym,df in pair_data.items():
        s,e,sl,tp=gen(df, **cfg)
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp)
        midx=d.index if midx is None else midx.union(d.index); arrs[sym]=d
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,'tp':dd['_tp'].values}
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]
    for k in range(len(midx)):
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]; atr=d['atr'][k]
            risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']; a=atr if not np.isnan(atr) else risk
            closed=False; nr=0.0
            if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
            elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
            else:
                if not pos['be'] and (e-c)/risk>=B_BE*((e-pos['tp'])/risk): pos['cur']=e; pos['be']=True
                if pos['be']:
                    cand=c+B_TRAIL*a
                    if cand<pos['cur']: pos['cur']=cand
            if closed:
                balance+=nr*pos['risk_usd']; trades.append(nr)
                del positions[sym]; cooldown_until[sym]=k+B_COOLDOWN
        n_short=len(positions)
        for sym in A:
            if sym in positions: continue
            if k<=cooldown_until[sym]: continue
            if n_short>=MAX_PER_SIDE: break
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(stop-e)
            if risk_px<=0: continue
            positions[sym]={'entry':e,'cur':stop,'tp':tp,'be':False,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px}
            n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return balance-INITIAL_BALANCE, max_dd


def evaluate(cfg):
    tot=0; pos=0; ddmax=0
    for yr in YEARS:
        start=datetime(yr,1,1,tzinfo=timezone.utc); end=YEAR_END.get(yr,datetime(yr,12,31,tzinfo=timezone.utc))
        pnl,dd=run(cfg,start,end)
        tot+=pnl; pos+=(1 if pnl>0 else 0); ddmax=max(ddmax,dd)
    return tot,pos,ddmax


print('='*92)
print('  VIZINHANCA da vencedora (sel30 adx20 struct20). Robusto = vizinhos tambem +/3+ anos')
print('='*92)
print(f"  {'sel_lb':>6} {'adx_max':>8} {'struct':>7} | {'TOTAL':>9} | {'anos+':>6} | {'DDmax':>7}")
print('  '+'-'*60)
# grid em torno da vencedora
best=None
for sel_lb in [25,30,35,40]:
    for adx_max in [18,20,25]:
        for struct_lb in [15,20,25]:
            cfg=dict(sel_lb=sel_lb, adx_max=adx_max, struct_lb=struct_lb)
            tot,pos,ddmax=evaluate(cfg)
            mark=' <' if (sel_lb==30 and adx_max==20 and struct_lb==20) else ''
            print(f"  {sel_lb:>6} {adx_max:>8} {struct_lb:>7} | {tot:>+9.0f} | {pos:>4}/5 | {ddmax:>6.1f}%{mark}")
            if best is None or (pos,tot)>(best[1],best[2]): best=(cfg,pos,tot,ddmax)

# resumo de robustez
print('\n'+'='*92)
print('  RESUMO DE ROBUSTEZ')
print('='*92)
allcfgs=[]
for sel_lb in [25,30,35,40]:
    for adx_max in [18,20,25]:
        for struct_lb in [15,20,25]:
            tot,pos,ddmax=evaluate(dict(sel_lb=sel_lb,adx_max=adx_max,struct_lb=struct_lb))
            allcfgs.append((tot,pos,ddmax))
n=len(allcfgs); pos_tot=sum(1 for t,p,d in allcfgs if t>0); cons=sum(1 for t,p,d in allcfgs if p>=3)
print(f"  {n} configs na vizinhanca | {pos_tot} com total positivo | {cons} com >=3/5 anos+")
print(f"  Total medio: {np.mean([t for t,p,d in allcfgs]):+.0f} | DD medio: {np.mean([d for t,p,d in allcfgs]):.1f}%")
print(f"\n  Melhor: {best[0]} -> {best[2]:+.0f}, {best[1]}/5 anos, DD {best[3]:.1f}%")
if pos_tot >= n*0.7:
    print('  >>> REGIAO ROBUSTA: a maioria dos vizinhos e positiva = edge real, nao overfit.')
else:
    print('  >>> CUIDADO: poucos vizinhos positivos = pode ser sensivel/overfit.')
print('\n  Custos 0.14% round-trip. Bear isolada, regime BEAR, 5000/ano, risco 1%.')
