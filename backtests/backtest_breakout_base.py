"""
backtest_breakout_base.py
NOVO NUCLEO — breakout de range, validado em 5 ANOS desde o inicio.

Lecoes aplicadas: (1) simples, poucos parametros; (2) so o breakout (unica logica
robusta em 5 anos); (3) escolher a config pela CONSISTENCIA ano-a-ano, nao por
maximizar um periodo (= anti-overfit por design).

Logica: range das ultimas N barras (consolidacao = ADX baixo) -> SEGUE o rompimento.
  LONG  : close rompe a resistencia do range (+buffer), barra anterior dentro.
  SHORT : close rompe o suporte do range.
  stop = lado oposto do range ; alvo = altura do range projetada. Opera os 2 lados.

Variante-chave testada: precisa do Decisor de regime (so NEUTRAL) ou auto-detecta
(qualquer regime, so ADX baixo)? Se auto-detecta = -1 camada de overfit.

Metrica: P&L por ano (2021-2025) + nº de anos POSITIVOS (consistencia). Saldo 5000
reinicia por ano, risco 1%, max_per_side=3. Usa cache _2020_2025.
Uso: python backtests/backtest_breakout_base.py
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
ATR_PERIOD=BEAR.ATR_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
COOLDOWN=4; TIMEOUT=48

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


print('NOVO NUCLEO — breakout de range, validado em 5 anos (consistencia = robustez)')
print('  2021 bull | 2022 BEAR | 2023 recup | 2024 bull | 2025 misto\n')
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
        df1['adx']=adx(df1,14)
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen(df, use_regime, lookback, adx_max, stop_mode, buffer_atr):
    n=len(df); h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; adx_a=df['adx'].values; reg=df['regime'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    for i in range(lookback+5,n):
        if use_regime and reg[i]!='NEUTRAL': continue
        a=atr[i]
        if np.isnan(a): continue
        if adx_max>0 and (np.isnan(adx_a[i-1]) or adx_a[i-1]>adx_max): continue
        rl=np.min(l[i-lookback:i]); rh=np.max(h[i-lookback:i])
        if rl<=0 or rh<=rl: continue
        buf=buffer_atr*a; cl=c[i]; pc=c[i-1]; height=rh-rl
        if cl>rh+buf and pc<=rh:
            stop=(rl+rh)/2 if stop_mode=='mid' else rl
            risk=cl-stop
            if risk>0: side[i]='LONG'; entry[i]=cl; sl[i]=stop; tp[i]=cl+height
        elif cl<rl-buf and pc>=rl:
            stop=(rl+rh)/2 if stop_mode=='mid' else rh
            risk=stop-cl
            if risk>0: side[i]='SHORT'; entry[i]=cl; sl[i]=stop; tp[i]=cl-height
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
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]
            risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']; closed=False; nr=0.0
            if pos['side']=='LONG':
                if lo<=pos['sl']: nr=(pos['sl']-e)/risk-fee_r; closed=True
                elif hi>=pos['tp']: nr=(pos['tp']-e)/risk-fee_r; closed=True
            else:
                if hi>=pos['sl']: nr=(e-pos['sl'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
            if not closed:
                pos['age']+=1
                if pos['age']>=TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']; trades.append(nr)
                del positions[sym]; cooldown_until[sym]=k+COOLDOWN
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sym in A:
            if sym in positions: continue
            if k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            if side=='LONG' and n_long>=MAX_PER_SIDE: continue
            if side=='SHORT' and n_short>=MAX_PER_SIDE: continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            positions[sym]={'side':side,'entry':e,'sl':stop,'tp':tp,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    nr=trades; w=[x for x in nr if x>0]
    pf=sum(w)/abs(sum(x for x in nr if x<=0)) if any(x<=0 for x in nr) else 99
    return dict(pnl=balance-INITIAL_BALANCE,t=len(nr),pf=pf,dd=max_dd)


CONFIGS=[
    ('NEUTRAL lb30 adx20', dict(use_regime=True,  lookback=30, adx_max=20, stop_mode='opposite', buffer_atr=0.1)),
    ('AUTO    lb30 adx20', dict(use_regime=False, lookback=30, adx_max=20, stop_mode='opposite', buffer_atr=0.1)),
    ('AUTO    lb30 adx25', dict(use_regime=False, lookback=30, adx_max=25, stop_mode='opposite', buffer_atr=0.1)),
    ('AUTO    lb50 adx20', dict(use_regime=False, lookback=50, adx_max=20, stop_mode='opposite', buffer_atr=0.1)),
    ('AUTO    lb20 adx20', dict(use_regime=False, lookback=20, adx_max=20, stop_mode='opposite', buffer_atr=0.1)),
    ('AUTO    lb30 adx15', dict(use_regime=False, lookback=30, adx_max=15, stop_mode='opposite', buffer_atr=0.1)),
    ('AUTO    lb30 adx20 mid', dict(use_regime=False, lookback=30, adx_max=20, stop_mode='mid', buffer_atr=0.1)),
]

print('='*100)
print('  P&L por ano e config (anos+ = nº de anos positivos de 5 = CONSISTENCIA)')
print('='*100)
hdr='  '+f"{'Config':<22}"+''.join(f"{str(y)[2:]+YEAR_CTX[y][:1]:>10}" for y in YEARS)+f"{'TOTAL':>9}{'anos+':>6}"
print(hdr); print('  '+'-'*98)
results=[]
for name,cfg in CONFIGS:
    pnls={}
    for yr in YEARS:
        start=datetime(yr,1,1,tzinfo=timezone.utc); end=YEAR_END.get(yr,datetime(yr,12,31,tzinfo=timezone.utc))
        pnls[yr]=run(cfg, start, end)
    tot=sum(p['pnl'] for p in pnls.values()); pos=sum(1 for p in pnls.values() if p['pnl']>0)
    row='  '+f"{name:<22}"+''.join(f"{pnls[y]['pnl']:>+10.0f}" for y in YEARS)+f"{tot:>+9.0f}{pos:>5}/5"
    print(row)
    results.append((name,cfg,tot,pos,pnls))

# melhor = mais anos positivos, desempate por total
results.sort(key=lambda x:(x[3],x[2]), reverse=True)
name,cfg,tot,pos,pnls=results[0]
print('\n'+'='*100)
print(f'  MAIS ROBUSTO: {name}  ->  {pos}/5 anos positivos | total {tot:+.0f} USDC')
print('='*100)
print(f"  {'Ano':<14} | {'P&L':>9} | {'Trades':>6} | {'PF':>5} | {'DD':>6}")
print('  '+'-'*48)
for yr in YEARS:
    p=pnls[yr]
    print(f"  {yr} ({YEAR_CTX[yr]:<5}) | {p['pnl']:>+9.0f} | {p['t']:>6} | {p['pf']:>5.2f} | {p['dd']:>5.1f}%")
print(f"\n  Config: {cfg}")
print('\n  Se >=4/5 anos positivos = base robusta para construir (e escalar depois).')
print('  Custos 0.14% round-trip. Risco 1%/trade. Validado em 5 anos desde o inicio.')
