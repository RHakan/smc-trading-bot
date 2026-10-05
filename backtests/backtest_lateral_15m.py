"""
backtest_lateral_15m.py
Testa a LATERAL (o motor mais robusto do v3) no 15m vs 1h — tese de FREQUENCIA do Rafa
(mais trades menores > esperar 1 trade grande). A logica e IDENTICA (regime diario NEUTRAL,
rompimento de range consolidado, stop no lado oposto, TP=altura, invalidacao rh-5). So muda
o timeframe de ENTRADA.

Duas formas de 15m, pra separar "frequencia ajuda" de "janela mais curta":
  - 15m 'mesmas barras': LOOKBACK=30, inval=5, timeout=48 em barras de 15m
    (= janelas 4x MAIS CURTAS que no 1h -> muito mais trades, mais ruido).
  - 15m 'tempo-equiv':   parametros x4 (LB=120, inval=20, timeout=192, ATR/ADX x4)
    (= MESMA janela real do 1h, so com granularidade fina de entrada/saida).
Baseline = lateral no 1h (identica a do bot).

Regime sempre do DIARIO. 10 pares. 5 anos (2021..2025-10) + OOS 2026 (jan..jun-29,
fim do cache 15m). 500/mes reset. MAX 3/lado. Custos 0.14% RT (fee pesa mais no 15m!).
Levantamento mensal. Bot rodando NAO tocado.
Uso: python backtests/backtest_lateral_15m.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from bot import regime as regime_mod

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MONTHLY_BASE=500.0; RISK_PCT=1.0; MAX_PER_SIDE=3
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT','ADA/USDT:USDT',
       'DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT','AVAX/USDT:USDT','SUI/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH

CACHE=Path(__file__).parent/'cache'


def _read(sym, tf, tag):
    f=CACHE/f"{sym.replace('/','_').replace(':','_')}_{tf}_{tag}.csv"
    if not f.exists(): return None
    df=pd.read_csv(f,index_col='ts',parse_dates=True); df.index=pd.to_datetime(df.index,utc=True)
    return df.rename(columns={'o':'o','h':'h','l':'l','c':'c','v':'v'})


def load_daily(sym, period):
    return _read(sym,'1d','2020_2025' if period=='5y' else '2025_2026oos')

def load_entry(sym, tf, period):
    if tf=='1h':
        return _read(sym,'1h','2020_2025' if period=='5y' else '2025_2026oos')
    df=_read(sym,'15m','2021_2026')          # 15m: arquivo unico, fatiado por periodo
    if df is None: return None
    if period=='5y': return df[(df.index>=datetime(2021,1,1,tzinfo=timezone.utc)) & (df.index<=datetime(2025,10,6,tzinfo=timezone.utc))]
    return df[(df.index>=datetime(2025,11,1,tzinfo=timezone.utc)) & (df.index<=datetime(2026,6,29,tzinfo=timezone.utc))]


def adx(df, period):
    h=df['h']; l=df['l']; c=df['c']; up=h.diff(); dn=-l.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    at=tr.ewm(alpha=1/period,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/at
    mdi=100*pd.Series(mdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/at
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period,adjust=False).mean()


def build(sym, tf, period, P):
    dfd=load_daily(sym,period); df=load_entry(sym,tf,period)
    if dfd is None or df is None or len(df)<P['LB']+P['ATR_P']+50: return None
    ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
    slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
    dfd['regime']='NEUTRAL'
    dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
    dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
    c=df['c']
    tr=pd.concat([(df['h']-df['l']),(df['h']-c.shift(1)).abs(),(df['l']-c.shift(1)).abs()],axis=1).max(axis=1)
    df=df.copy()
    df['atr']=tr.ewm(com=P['ATR_P']-1,adjust=False).mean()
    df['adx']=adx(df,P['ADX_P'])
    df['regime']=dfd['regime'].shift(1).reindex(df.index,method='ffill')
    return df


def gen_lateral(df, P):
    n=len(df); h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; adx_a=df['adx'].values; reg=df['regime'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan)
    tp=np.full(n,np.nan); bl=np.full(n,np.nan)
    LB=P['LB']; ADXM=20; BUF=0.10
    start_i=max(LB+5, P['ADX_P']+5, P['ATR_P']+5, 50)
    for i in range(start_i,n):
        if reg[i]!='NEUTRAL': continue
        a=atr[i]
        if np.isnan(a) or a<=0: continue
        if np.isnan(adx_a[i-1]) or adx_a[i-1]>ADXM: continue
        rl=np.min(l[i-LB:i]); rh=np.max(h[i-LB:i])
        if rl<=0 or rh<=rl: continue
        buf=BUF*a; cl=c[i]; pc=c[i-1]; height=rh-rl
        if cl>rh+buf and pc<=rh and cl>rl:
            side[i]='LONG'; entry[i]=cl; sl[i]=rl; tp[i]=cl+height; bl[i]=rh
        elif cl<rl-buf and pc>=rl and rh>cl:
            side[i]='SHORT'; entry[i]=cl; sl[i]=rh; tp[i]=cl-height; bl[i]=rl
    return side,entry,sl,tp,bl


def prep(tf, period, P):
    arrs={}
    for sym in PAIRS:
        df=build(sym,tf,period,P)
        if df is None: continue
        s,e,sl,tp,bl=gen_lateral(df,P)
        arrs[sym]=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_bl=bl)
    return arrs


def run(arrs, P, start, end):
    midx=None
    for sym,d in arrs.items():
        midx=d.index if midx is None else midx.union(d.index)
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,
                'tp':dd['_tp'].values,'bl':dd['_bl'].values}
    INVAL=P['INVAL']; TIMEOUT=P['TIMEOUT']; CD=P['CD']
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; monthly={}; cur_month=None; trades=[]
    def close_all(k):
        nonlocal balance
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]; pos=positions[sym]
            if np.isnan(c): c=pos['entry']
            e=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
            nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r
            balance+=nr*pos['risk_usd']; trades.append(nr); del positions[sym]
    for k in range(len(midx)):
        ts=midx[k]; mk=ts.strftime('%Y-%m')
        if cur_month is None: cur_month=mk
        if mk!=cur_month:
            close_all(k); monthly[cur_month]=balance-MONTHLY_BASE
            balance=MONTHLY_BASE; peak=MONTHLY_BASE; cur_month=mk
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
                if pos['age']<=INVAL and not np.isnan(pos['bl']):
                    inside=(c<pos['bl']) if pos['side']=='LONG' else (c>pos['bl'])
                    if inside: nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
                if not closed and pos['age']>=TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']; trades.append(nr); del positions[sym]; cooldown_until[sym]=k+CD
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sym in A:
            if sym in positions or k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            if side=='LONG' and n_long>=MAX_PER_SIDE: continue
            if side=='SHORT' and n_short>=MAX_PER_SIDE: continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'age':0,'risk_px':risk_px,
                            'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px,'bl':d['bl'][k]}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    monthly[cur_month]=balance-MONTHLY_BASE
    return monthly, max_dd, trades


def summ(res):
    monthly, ddmax, trades = res
    yr={}
    for mk,v in monthly.items(): yr[mk[:4]]=yr.get(mk[:4],0)+v
    tot=sum(monthly.values()); tr=np.array(trades); n=len(tr)
    wins=tr[tr>0.03]
    return dict(tot=tot, yr=yr, monthly=monthly, anos=sum(1 for v in yr.values() if v>0),
                meses=sum(1 for v in monthly.values() if v>0), nm=len(monthly), dd=ddmax,
                n=n, wr=len(wins)/n*100 if n else 0)


# ── Configs ──────────────────────────────────────────────────────────────────
P_1H  =dict(tf='1h', LB=30,  ADX_P=14, ATR_P=14, INVAL=5,  TIMEOUT=48,  CD=4)
P_15S =dict(tf='15m',LB=30,  ADX_P=14, ATR_P=14, INVAL=5,  TIMEOUT=48,  CD=4)   # mesmas barras
P_15T =dict(tf='15m',LB=120, ADX_P=56, ATR_P=56, INVAL=20, TIMEOUT=192, CD=16)  # tempo-equiv (x4)
CONFIGS=[('LATERAL 1h (baseline)',P_1H),('LATERAL 15m mesmas-barras',P_15S),('LATERAL 15m tempo-equiv',P_15T)]

S5=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc))
S26=(datetime(2026,1,1,tzinfo=timezone.utc), datetime(2026,6,29,tzinfo=timezone.utc))

print('A carregar dados (cache)...')
data={}
for nome,P in CONFIGS:
    a5=prep(P['tf'],'5y',P); a26=prep(P['tf'],'oos',P)
    data[nome]=(a5,a26)
    print(f'  {nome:<28} 5y:{len(a5)} pares  OOS:{len(a26)} pares')

print('\n'+'='*98)
print('  LATERAL: 1h vs 15m — tese de frequencia | regime diario | 500/mes | custos 0.14% RT')
print('='*98)
print('\n  '+f"{'config':<28}{'5anos':>9}{'anos+':>7}{'DD':>7}{'n':>6}{'WR':>5}{'||':>4}{'2026':>8}{'m+':>6}{'DD':>7}{'n':>5}")
print('  '+'-'*94)
res={}
for nome,P in CONFIGS:
    a5,a26=data[nome]
    s5=summ(run(a5,P,*S5)); s26=summ(run(a26,P,*S26)); res[nome]=(s5,s26)
    print('  '+f"{nome:<28}{s5['tot']:>+9.0f}{s5['anos']:>5}/5{s5['dd']:>6.1f}%{s5['n']:>6}{s5['wr']:>5.0f}%{'||':>4}"
          f"{s26['tot']:>+8.0f}{s26['meses']:>3}/{s26['nm']:<2}{s26['dd']:>6.1f}%{s26['n']:>5}")

b5,b26=res['LATERAL 1h (baseline)']
print('\n'+'='*98)
print('  Leitura: "n" = nº de trades (frequencia). Vencer = mais total SEM piorar OOS/DD.')
for nome,_ in CONFIGS[1:]:
    s5,s26=res[nome]
    v5='OK' if s5['tot']>b5['tot'] else 'nao'; v26='OK' if s26['tot']>b26['tot'] else 'nao'
    print(f"  {nome:<28} 5anos {s5['tot']:+.0f} vs {b5['tot']:+.0f} [{v5}] | OOS {s26['tot']:+.0f} vs {b26['tot']:+.0f} [{v26}] | trades x{s5['n']/max(b5['n'],1):.1f}")

print('\n  [MENSAL OOS 2026] baseline 1h vs melhor 15m')
best=max(CONFIGS[1:], key=lambda x: sum(res[x[0]][1]['monthly'].values()))
mb=b26['monthly']; mw=res[best[0]][1]['monthly']; meses=sorted(set(mb)|set(mw))
print('  '+f"{'mes':<14}"+''.join(f"{m[5:]:>7}" for m in meses)+f"{'TOT':>8}")
print('  '+f"{'1h':<14}"+''.join(f"{mb.get(m,0):>+7.0f}" for m in meses)+f"{sum(mb.values()):>+8.0f}")
print('  '+f"{best[0][8:]:<14}"+''.join(f"{mw.get(m,0):>+7.0f}" for m in meses)+f"{sum(mw.values()):>+8.0f}")
print('\n  Custos 0.14% RT embutidos (fee pesa mais quanto mais trades). Bot NAO tocado.')
