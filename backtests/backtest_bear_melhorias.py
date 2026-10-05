"""
backtest_bear_melhorias.py
Melhorar a estrategia de BAIXA — testa 4 angulos sobre a bear v1.3, em 5 anos.

Baseline (v1.3): regime BEAR + close<min(3 velas) + bearish + ATR>media.
  Resultado isolado 5 anos: -13068 USDC, 0/5 anos positivos (perde ate em 2022).
  Diagnostico: a condicao 2 (close<min 3 velas) entra em QUALQUER rompimento de
  minima — a maioria e stop-hunt que reverte. Falta SELETIVIDADE.

4 angulos (cada um isolado + combinacoes):
  1. SELETIVIDADE  : exige consolidacao previa (ADX baixo) + rompe range maior
                     (lookback N em vez de 3). = trazer ao bear o que faz o breakout
                     funcionar.
  2. CONFIRMACAO   : so entra se o rompimento for DECISIVO ((low_n-close)/ATR >= X)
                     — filtra os falsos rompimentos fracos.
  3. STOP ESTRUTURAL: stop acima da estrutura rompida (max das ultimas K barras)
                     em vez de close+1xATR (aguenta o repique sem stopar).
  4. FILTRO DO FUNDO: nao shortar em sobrevenda (RSI < media(RSI,200)-k*desvio =
                     ja esticado, repique provavel).

So opera em regime BEAR (fiel a v13). Isolado, 5000/ano, risco 1%, max_per_side 3.
Metrica: P&L/ano + anos positivos (consistencia). Levantamento MENSAL da melhor.
Uso: python backtests/backtest_bear_melhorias.py
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
RSI_W=200

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


def rsi_calc(s, period=14):
    d=s.diff(); up=d.clip(lower=0).ewm(com=period-1,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(com=period-1,adjust=False).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))


print('MELHORIAS DA BEAR — 4 angulos sobre a v1.3, em 5 anos')
print('  Baseline v1.3 isolada: -13068 USDC, 0/5 anos+\n')
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
        rsi=rsi_calc(c,14); df1['rsi']=rsi
        df1['rsi_ma']=rsi.rolling(RSI_W).mean(); df1['rsi_sd']=rsi.rolling(RSI_W).std()
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen(df, sel_lb, adx_max, conf_min, stop_mode, struct_lb, rsi_dyn, rsi_k):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    reg=df['regime'].values; rsi=df['rsi'].values; rsi_ma=df['rsi_ma'].values; rsi_sd=df['rsi_sd'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    start_i=max(B_ATR_AVG+sel_lb+5, struct_lb+5, RSI_W+5, 210)
    for i in range(start_i,n):
        if reg[i]!='BEAR': continue
        a=atr[i]; cl=c[i]; op=o[i]; av=atr_avg[i]
        if np.isnan(a) or a<=0 or np.isnan(av): continue
        low_n=np.min(l[i-sel_lb:i])    # minimo das ultimas sel_lb barras (suporte)
        # Condicoes base (v13)
        if not (cl<low_n and cl<op and a>av): continue
        # Angulo 1: consolidacao previa (ADX baixo antes do rompimento)
        if adx_max>0 and (np.isnan(adx_a[i-1]) or adx_a[i-1]>adx_max): continue
        # Angulo 2: rompimento decisivo (convicao em ATR)
        if conf_min>0 and (low_n-cl)/a < conf_min: continue
        # Angulo 4: nao shortar em sobrevenda (RSI dinamico)
        if rsi_dyn and not np.isnan(rsi_ma[i]) and not np.isnan(rsi_sd[i]) and not np.isnan(rsi[i]):
            if rsi[i] < rsi_ma[i]-rsi_k*rsi_sd[i]: continue
        # Angulo 3: stop
        if stop_mode=='struct':
            stop=np.max(h[i-struct_lb:i+1])+0.1*a   # acima da estrutura rompida
        else:
            stop=cl+1.0*a                            # v13: close + 1xATR
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
            # SHORT
            if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
            elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
            else:
                if not pos['be'] and (e-c)/risk>=B_BE*((e-pos['tp'])/risk): pos['cur']=e; pos['be']=True
                if pos['be']:
                    cand=c+B_TRAIL*a
                    if cand<pos['cur']: pos['cur']=cand
            if closed:
                balance+=nr*pos['risk_usd']; trades.append({'ts':midx[k],'nr':nr,'pnl':nr*pos['risk_usd']})
                del positions[sym]; cooldown_until[sym]=k+B_COOLDOWN
        n_short=sum(1 for p in positions.values())
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
            positions[sym]={'side':'SHORT','entry':e,'cur':stop,'tp':tp,'be':False,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px}
            n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    nr=[t['nr'] for t in trades]; w=[x for x in nr if x>0]
    pf=sum(w)/abs(sum(x for x in nr if x<=0)) if any(x<=0 for x in nr) else 99
    return dict(pnl=balance-INITIAL_BALANCE,t=len(nr),pf=pf,dd=max_dd,trades=trades)


def evaluate(cfg):
    pnls={}; allt=[]
    for yr in YEARS:
        start=datetime(yr,1,1,tzinfo=timezone.utc); end=YEAR_END.get(yr,datetime(yr,12,31,tzinfo=timezone.utc))
        r=run(cfg,start,end); pnls[yr]=r; allt+=r['trades']
    tot=sum(p['pnl'] for p in pnls.values()); pos=sum(1 for p in pnls.values() if p['pnl']>0)
    ddmax=max(p['dd'] for p in pnls.values())
    return pnls,tot,pos,ddmax,allt


BASE=dict(sel_lb=3, adx_max=0, conf_min=0.0, stop_mode='atr', struct_lb=10, rsi_dyn=False, rsi_k=2.0)

CONFIGS=[
    ('BASELINE v1.3',            BASE),
    # Angulo 1 — seletividade
    ('1) sel20 adx25',           {**BASE,'sel_lb':20,'adx_max':25}),
    ('1) sel30 adx20',           {**BASE,'sel_lb':30,'adx_max':20}),
    ('1) sel30 adx25',           {**BASE,'sel_lb':30,'adx_max':25}),
    # Angulo 2 — confirmacao
    ('2) conf 0.3',              {**BASE,'conf_min':0.3}),
    ('2) conf 0.5',              {**BASE,'conf_min':0.5}),
    # Angulo 3 — stop estrutural
    ('3) stop struct lb10',      {**BASE,'stop_mode':'struct','struct_lb':10}),
    ('3) stop struct lb20',      {**BASE,'stop_mode':'struct','struct_lb':20}),
    # Angulo 4 — filtro do fundo (RSI dinamico)
    ('4) rsi_dyn k2.0',          {**BASE,'rsi_dyn':True,'rsi_k':2.0}),
    ('4) rsi_dyn k1.5',          {**BASE,'rsi_dyn':True,'rsi_k':1.5}),
    # Combinacoes dos que costumam ajudar
    ('1+4 sel30adx20 +rsi',      {**BASE,'sel_lb':30,'adx_max':20,'rsi_dyn':True,'rsi_k':2.0}),
    ('1+3 sel30adx20 +struct',   {**BASE,'sel_lb':30,'adx_max':20,'stop_mode':'struct','struct_lb':20}),
    ('1+2+4 sel30 conf0.3 rsi',  {**BASE,'sel_lb':30,'adx_max':20,'conf_min':0.3,'rsi_dyn':True}),
    ('TUDO sel30 adx20 conf0.3 struct rsi', {'sel_lb':30,'adx_max':20,'conf_min':0.3,'stop_mode':'struct','struct_lb':20,'rsi_dyn':True,'rsi_k':2.0}),
]

print('='*104)
print('  MELHORIAS DA BEAR — P&L por ano (anos+ = consistencia | meta: virar os 0/5 da v1.3)')
print('='*104)
print('  '+f"{'Config':<38}"+''.join(f"{str(y)[2:]+YEAR_CTX[y][:1]:>9}" for y in YEARS)+f"{'TOTAL':>9}{'+':>4}{'DD':>7}")
print('  '+'-'*102)
results=[]
for name,cfg in CONFIGS:
    pnls,tot,pos,ddmax,allt=evaluate(cfg)
    row='  '+f"{name:<38}"+''.join(f"{pnls[y]['pnl']:>+9.0f}" for y in YEARS)+f"{tot:>+9.0f}{pos:>3}/5{ddmax:>6.1f}%"
    print(row)
    results.append((name,cfg,tot,pos,ddmax,allt))

results.sort(key=lambda x:(x[3],x[2]), reverse=True)
name,cfg,tot,pos,ddmax,allt=results[0]
print('\n'+'='*104)
print(f'  MELHOR: {name}  ->  {pos}/5 anos+ | total {tot:+.0f} | DDmax {ddmax:.1f}%')
print('='*104)
ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
months={}
for t in allt: months.setdefault(t['ts'].strftime('%Y-%m'),0.0); months[t['ts'].strftime('%Y-%m')]+=t['pnl']
yrs=sorted(set(int(mk[:4]) for mk in months))
print(f"  {'Ano':<6} | "+' '.join(f'{ML[m]:>5}' for m in range(12))+f" | {'Soma':>7}")
print('  '+'-'*90)
for yr in yrs:
    vals=[months.get(f'{yr}-{m+1:02d}') for m in range(12)]
    cells=' '.join((f'{v:>+5.0f}' if v is not None else '    .') for v in vals)
    s=sum(v for v in vals if v is not None)
    print(f"  {yr:<6} | {cells} | {s:>+7.0f}")
print(f"\n  Config vencedora: {cfg}")
print('  Custos 0.14% round-trip. Bear isolada, regime BEAR, 5000/ano, risco 1%.')
