"""
backtest_lateral_rr.py
Pergunta do Rafa: "o R:R está desfavoravel — quando ganha, ganha pouco; quando perde,
perde muito. Quero 3:1." Testa as DUAS leituras da ideia, na LATERAL (a peca robusta):

  A) FILTRO de RR minimo — mantem o alvo da estrategia (altura do range) e DESCARTA os
     rompimentos cuja geometria da RR baixo. Nao muda o alvo, so escolhe melhor a entrada.
  B) ALVO FORCADO — impoe TP = N x risco (1.5:1, 2:1, 3:1), como pedido literalmente.
     O stop continua no lado oposto do range.

GEOMETRIA (importante): na lateral o RR e SEMPRE < 1 por construcao.
  LONG : entry = close (ACIMA do topo), stop = fundo do range, alvo = entry + altura.
         risco = close - fundo  >  altura = alvo  ->  RR < 1 sempre.
  SHORT: simetrico.
Quanto MAIS longe do range o preco rompe, PIOR o RR (o stop fica mais distante).
Logo, o filtro de RR minimo = "nao perseguir rompimento que ja correu demais".
Um min_rr >= 1.0 apaga TODOS os trades — o teste mostra isso em vez de eu afirmar.

Baseline = lateral como esta no bot (invalidacao rh-5, TP fixo, sem trailing).
1H, regime diario NEUTRAL, 10 pares, 5 anos + OOS 2026. 500/mes reset. MAX 3/lado.
Custos 0.14% RT. Levantamento mensal. Bot rodando NAO tocado.
Uso: python backtests/backtest_lateral_rr.py
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
       'DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT','AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
LB=30; ADX_P=14; ATR_P=14; ADX_MAX=20; BUF=0.10
INVAL=5; TIMEOUT=48; CD=4

CACHE=Path(__file__).parent/'cache'


def _read(sym, tf, tag):
    f=CACHE/f"{sym.replace('/','_').replace(':','_')}_{tf}_{tag}.csv"
    if not f.exists(): return None
    df=pd.read_csv(f,index_col='ts',parse_dates=True); df.index=pd.to_datetime(df.index,utc=True)
    return df


def adx(df, period=ADX_P):
    h=df['h']; l=df['l']; c=df['c']; up=h.diff(); dn=-l.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    at=tr.ewm(alpha=1/period,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/at
    mdi=100*pd.Series(mdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/at
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period,adjust=False).mean()


def build(sym, period):
    tag='2020_2025' if period=='5y' else '2025_2026oos'
    dfd=_read(sym,'1d',tag); df=_read(sym,'1h',tag)
    if dfd is None or df is None or len(df)<LB+ATR_P+60: return None
    ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
    slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
    dfd['regime']='NEUTRAL'
    dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
    dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
    c=df['c']
    tr=pd.concat([(df['h']-df['l']),(df['h']-c.shift(1)).abs(),(df['l']-c.shift(1)).abs()],axis=1).max(axis=1)
    df=df.copy()
    df['atr']=tr.ewm(com=ATR_P-1,adjust=False).mean()
    df['adx']=adx(df)
    df['regime']=dfd['regime'].shift(1).reindex(df.index,method='ffill')
    return df


def gen(df, min_rr: float, force_rr: float):
    """min_rr>0  → descarta sinais com RR configurado < min_rr (alvo = altura do range).
       force_rr>0 → alvo passa a ser force_rr × risco (stop continua no lado oposto)."""
    n=len(df); h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; adx_a=df['adx'].values; reg=df['regime'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan)
    tp=np.full(n,np.nan); bl=np.full(n,np.nan); rrv=np.full(n,np.nan)
    start_i=max(LB+5, ADX_P+5, ATR_P+5, 50)
    for i in range(start_i,n):
        if reg[i]!='NEUTRAL': continue
        a=atr[i]
        if np.isnan(a) or a<=0: continue
        if np.isnan(adx_a[i-1]) or adx_a[i-1]>ADX_MAX: continue
        rl=np.min(l[i-LB:i]); rh=np.max(h[i-LB:i])
        if rl<=0 or rh<=rl: continue
        buf=BUF*a; cl=c[i]; pc=c[i-1]; height=rh-rl
        s=e=st=t=b=None
        if cl>rh+buf and pc<=rh and cl>rl:
            s,e,st,b = 'LONG', cl, rl, rh
            t = cl+height
        elif cl<rl-buf and pc>=rl and rh>cl:
            s,e,st,b = 'SHORT', cl, rh, rl
            t = cl-height
        if s is None: continue
        risk=abs(e-st)
        if risk<=0: continue
        rr_cfg = abs(t-e)/risk                       # RR da estrategia (altura/risco)
        if min_rr>0 and rr_cfg < min_rr: continue    # (A) filtro de geometria
        if force_rr>0:                               # (B) alvo forcado = N x risco
            t = e + force_rr*risk if s=='LONG' else e - force_rr*risk
        side[i]=s; entry[i]=e; sl[i]=st; tp[i]=t; bl[i]=b
        rrv[i]=abs(t-e)/risk
    return side,entry,sl,tp,bl,rrv


def prep(period, min_rr, force_rr):
    arrs={}
    for sym in PAIRS:
        df=build(sym,period)
        if df is None: continue
        s,e,sl,tp,bl,rr=gen(df,min_rr,force_rr)
        arrs[sym]=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_bl=bl,_rr=rr)
    return arrs


def run(arrs, start, end):
    midx=None
    for sym,d in arrs.items():
        midx=d.index if midx is None else midx.union(d.index)
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,
                'tp':dd['_tp'].values,'bl':dd['_bl'].values,'rr':dd['_rr'].values}
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; monthly={}; cur_month=None
    trades=[]; rrs=[]
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
            rrs.append(d['rr'][k])
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    monthly[cur_month]=balance-MONTHLY_BASE
    return monthly, max_dd, trades, rrs


def summ(res):
    monthly, ddmax, trades, rrs = res
    yr={}
    for mk,v in monthly.items(): yr[mk[:4]]=yr.get(mk[:4],0)+v
    tot=sum(monthly.values()); tr=np.array(trades); n=len(tr)
    wins=tr[tr>0.03]
    return dict(tot=tot, yr=yr, monthly=monthly, anos=sum(1 for v in yr.values() if v>0),
                meses=sum(1 for v in monthly.values() if v>0), nm=len(monthly), dd=ddmax,
                n=n, wr=len(wins)/n*100 if n else 0,
                rr_med=float(np.mean(rrs)) if rrs else 0.0)


S5=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc))
S26=(datetime(2026,1,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc))

CONFIGS = [
    ('BASELINE (bot atual)',      0.0, 0.0),
    ('A) filtro RR>=0.70',        0.70, 0.0),
    ('A) filtro RR>=0.80',        0.80, 0.0),
    ('A) filtro RR>=0.85',        0.85, 0.0),
    ('A) filtro RR>=0.90',        0.90, 0.0),
    ('A) filtro RR>=0.95',        0.95, 0.0),
    ('A) filtro RR>=1.00',        1.00, 0.0),
    ('A) filtro RR>=3.00 (pedido)', 3.00, 0.0),
    ('B) alvo forcado 1.5:1',     0.0, 1.5),
    ('B) alvo forcado 2.0:1',     0.0, 2.0),
    ('B) alvo forcado 2.5:1',     0.0, 2.5),
    ('B) alvo forcado 3.0:1',     0.0, 3.0),
]

print('A carregar (cache)...')
print()
print('='*104)
print('  LATERAL — R:R: filtro de geometria (A) vs alvo forcado (B) | 500/mes | custos 0.14% RT')
print('='*104)
print('\n  '+f"{'config':<30}{'5anos':>9}{'anos+':>7}{'DD':>7}{'n':>6}{'WR':>6}{'RRmed':>7}{'||':>4}{'2026':>8}{'m+':>6}{'n':>5}")
print('  '+'-'*100)

base=None
for nome, min_rr, force_rr in CONFIGS:
    a5=prep('5y',min_rr,force_rr); a26=prep('oos',min_rr,force_rr)
    s5=summ(run(a5,*S5)); s26=summ(run(a26,*S26))
    if base is None: base=(s5,s26)
    mark=''
    if s5['n']==0: mark='  <= ZERO TRADES'
    elif s5['tot']>base[0]['tot'] and s26['tot']>base[1]['tot']: mark='  <= bate nos 2'
    print('  '+f"{nome:<30}{s5['tot']:>+9.0f}{s5['anos']:>5}/5{s5['dd']:>6.1f}%{s5['n']:>6}{s5['wr']:>5.0f}%"
          f"{s5['rr_med']:>7.2f}{'||':>4}{s26['tot']:>+8.0f}{s26['meses']:>3}/{s26['nm']:<2}{s26['n']:>5}{mark}")

print('\n'+'='*104)
print('  Leitura: RRmed = R:R medio dos trades aceites. n = nº de trades.')
print('  (A) so escolhe melhor a entrada — o alvo continua a altura do range.')
print('  (B) afasta o alvo -> o WR TEM de cair. So compensa se a queda for menor que o ganho.')
print('  Custos 0.14% RT embutidos. Bot rodando NAO tocado.')
