"""
backtest_parciais_alvo.py
Ideia do Rafa: manter o bot exactamente como esta, mas ir realizando parciais pelo
caminho — ao bater 25% do alvo clica no botao 25%, aos 50% no botao 50%, aos 75%
no botao 75%. O que sobra corre ate ao alvo/stop com as regras actuais.

DUAS LEITURAS (dao resultados bem diferentes):
  'botao'      : semantica REAL do dashboard — close_partial() fecha `fraction` da
                 posicao ACTUAL. 25% de 100 -> 50% de 75 -> 75% de 37.5  => sobra 9.4%
  'cumulativo' : vender ate ter realizado 25%/50%/75% do ORIGINAL  => sobra 25%

MECANICA:
  progresso = (preco - entry) / (tp - entry)   [espelhado no short]
  marco atingido quando a MAXIMA da vela (long) alcanca o preco do marco.
  Cada fatia realiza ao preco do marco e paga a sua parte das taxas (f * fee_r) —
  o total de taxas continua a ser RT sobre o notional, distribuido pelas fatias.

CONSERVADOR: se a vela tocar o STOP e um marco, assume-se o STOP primeiro (nao da
para saber a ordem dentro da vela). Isto penaliza ligeiramente as parciais — de
proposito, para nao inflar o resultado.

Baseline = bot actual. Sinais, filtros, invalidacao, BE/trailing: tudo igual.
5 anos + OOS 2026. 500/mes reset. MAX 3/lado. Custos 0.14% RT. Bot NAO tocado.
Uso: python backtests/backtest_parciais_alvo.py
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

MONTHLY_BASE=500.0; RISK_PCT=1.0; MAX_PER_SIDE=3
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD; B_ATR_AVG=BEAR.ATR_AVG_PERIOD
B_COOLDOWN=BEAR.COOLDOWN_BARS; B_RR=BEAR.RR_CAP; B_TRAIL=BEAR.TRAIL_ATR
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
BSEL_LB=30; BSEL_ADX=20; BSEL_STRUCT=20
U_SWING_N=10; U_CHOCH_BARS=12; U_CHOCH_REF=15; U_MIN_SWEEP=0.05; U_RR=2.5; U_TRAIL=2.0
U_COOLDOWN=3; U_SLOPE_MIN=0.006; U_EMA_TREND=100
L_LOOKBACK=30; L_ADX_MAX=20; L_BUFFER=0.1; L_COOLDOWN=4; L_TIMEOUT=48; L_INVAL_N=5

CACHE_DIR=Path(__file__).parent/'cache'; CACHE_DIR.mkdir(exist_ok=True)
ex=ccxt.binanceusdm({'enableRateLimit':True})


def _fetch(sym, tf, start_dt, end_dt, tag):
    safe=sym.replace('/','_').replace(':','_'); cache=CACHE_DIR/f'{safe}_{tf}_{tag}.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True); df.index=pd.to_datetime(df.index,utc=True); return df
    since=int((start_dt-timedelta(days=5)).timestamp()*1000); end_ms=int(end_dt.timestamp()*1000); rows=[]
    while True:
        b=ex.fetch_ohlcv(sym,tf,since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v']); df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=df.set_index('ts').sort_index()[lambda d:d.index<=end_dt]; df.to_csv(cache); return df


def fetch_5y(sym, tf):
    return _fetch(sym, tf, datetime(2020,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc), '2020_2025')
def fetch_26(sym, tf):
    return _fetch(sym, tf, datetime(2025,9,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc), '2025_2026oos')


def adx(df, period=14):
    h=df['h']; l=df['l']; c=df['c']; up=h.diff(); dn=-l.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    at=tr.ewm(alpha=1/period,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/at
    mdi=100*pd.Series(mdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/at
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period,adjust=False).mean()


def load(fetch_fn):
    pdata={}; btcb=None
    for sym in PAIRS:
        dfd=fetch_fn(sym,'1d').copy(); df1=fetch_fn(sym,'1h').copy()
        if len(df1)<300: continue
        ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
        slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
        dfd['regime']='NEUTRAL'
        dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
        dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
        dfd['slope_d']=slope
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['atr_avg']=df1['atr'].rolling(B_ATR_AVG).mean()
        df1['adx']=adx(df1,14)
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['slope_d']=dfd['slope_d'].shift(1).reindex(df1.index,method='ffill')
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        pdata[sym]=df1
        if sym.startswith('BTC'): btcb=(df1['regime']=='BULL')
    return pdata, btcb


def gen(df, btc_local):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    reg=df['regime'].values; slope=df['slope_d'].values; ema_t=df['ema_trend'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object); blevel=np.full(n,np.nan)
    start_i=max(B_ATR_AVG+BSEL_LB+5, U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, L_LOOKBACK+5, 210)
    for i in range(start_i,n):
        r=reg[i]; a=atr[i]
        if np.isnan(a) or a<=0: continue
        if r=='NEUTRAL':
            if np.isnan(adx_a[i-1]) or adx_a[i-1]>L_ADX_MAX: continue
            rl=np.min(l[i-L_LOOKBACK:i]); rh=np.max(h[i-L_LOOKBACK:i])
            if rl<=0 or rh<=rl: continue
            buf=L_BUFFER*a; cl=c[i]; pc=c[i-1]; height=rh-rl
            if cl>rh+buf and pc<=rh:
                if cl>rl: side[i]='LONG'; entry[i]=cl; sl[i]=rl; tp[i]=cl+height; strat[i]='lat'; blevel[i]=rh
            elif cl<rl-buf and pc>=rl:
                if rh>cl: side[i]='SHORT'; entry[i]=cl; sl[i]=rh; tp[i]=cl-height; strat[i]='lat'; blevel[i]=rl
        elif r=='BULL':
            if not btc_local[i]: continue
            if np.isnan(slope[i]) or slope[i]<U_SLOPE_MIN: continue
            if np.isnan(ema_t[i]) or c[i]<=ema_t[i]: continue
            av=atr_avg[i]
            if not np.isnan(av) and a<av: continue
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
        elif r=='BEAR':
            cl=c[i]; op=o[i]; av=atr_avg[i]
            if np.isnan(av): continue
            low_n=np.min(l[i-BSEL_LB:i])
            if not (cl<low_n and cl<op and a>av): continue
            if np.isnan(adx_a[i-1]) or adx_a[i-1]>BSEL_ADX: continue
            stop=np.max(h[i-BSEL_STRUCT:i+1])+0.1*a; risk=stop-cl
            if risk>0: side[i]='SHORT'; entry[i]=cl; sl[i]=stop; tp[i]=cl-B_RR*risk; strat[i]='bear'
    return side,entry,sl,tp,strat,blevel


def prep(pdata, btcb):
    arrs={}
    for sym,df in pdata.items():
        s,e,sl,tp,st,bl=gen(df, btcb.reindex(df.index).fillna(False).values)
        arrs[sym]=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st,_bl=bl)
    return arrs


GEST={'bull':dict(be=0.40,trail=U_TRAIL,cd=U_COOLDOWN),
      'bear':dict(be=0.40,trail=B_TRAIL,cd=B_COOLDOWN),
      'lat' :dict(be=9.99,trail=0.0,cd=L_COOLDOWN)}

def run(arrs, start, end, marcos=None, modo='botao'):
    """marcos: [(progresso_do_alvo, fracao), ...] ou None = baseline.
    modo 'botao'      -> fracao da posicao ACTUAL (semantica real do dashboard)
    modo 'cumulativo' -> fracao ja vendida do ORIGINAL ao atingir o marco."""
    MARCOS = marcos or []
    midx=None
    for sym,d in arrs.items():
        midx=d.index if midx is None else midx.union(d.index)
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,'tp':dd['_tp'].values,
                'strat':dd['_st'].values,'bl':dd['_bl'].values}
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; monthly={}; cur_month=None
    trades=[]; n_parciais=0

    def r_slice(pos, f, px):
        """R liquido de uma fatia `f` fechada em `px` (paga a sua quota de taxas)."""
        e=pos['entry']; risk=pos['risk_px']
        g=((px-e) if pos['side']=='LONG' else (e-px))/risk
        return f*(g - pos['fee_r'])

    def finalize(pos, px):
        return pos['realized_r'] + r_slice(pos, pos['remaining'], px)

    def close_all(k):
        nonlocal balance
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]; pos=positions[sym]
            if np.isnan(c): c=pos['entry']
            nr=finalize(pos,c)
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
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]; atr=d['atr'][k]
            risk=pos['risk_px']; e=pos['entry']; a=atr if not np.isnan(atr) else risk
            closed=False; nr=0.0
            # ── 1) STOP e ALVO primeiro (conservador: se a vela tocou o stop, e stop) ──
            if pos['side']=='LONG':
                if lo<=pos['cur']: nr=finalize(pos,pos['cur']); closed=True
                elif hi>=pos['tp']: nr=finalize(pos,pos['tp']); closed=True
            else:
                if hi>=pos['cur']: nr=finalize(pos,pos['cur']); closed=True
                elif lo<=pos['tp']: nr=finalize(pos,pos['tp']); closed=True
            # ── 2) Parciais nos marcos (so se a vela nao encerrou) ──
            if not closed and MARCOS:
                for idx,(prog,frac) in enumerate(MARCOS):
                    if idx in pos['marcos_feitos']: continue
                    mp = e + prog*(pos['tp']-e)          # preco do marco (serve p/ os 2 lados)
                    atingiu = (hi>=mp) if pos['side']=='LONG' else (lo<=mp)
                    if not atingiu: continue
                    if modo=='botao':
                        f = pos['remaining']*frac         # fracao da posicao ACTUAL
                    else:                                  # cumulativo
                        alvo_vendido = frac                # ja vendido no total
                        f = max(0.0, alvo_vendido - (1.0-pos['remaining']))
                    f = min(f, pos['remaining'])
                    if f<=0: continue
                    pos['realized_r'] += r_slice(pos,f,mp)
                    pos['remaining'] -= f
                    pos['marcos_feitos'].add(idx); n_parciais+=1
                if pos['remaining']<=1e-9:
                    nr=pos['realized_r']; closed=True
            # ── 3) Gestao do restante (BE/trailing) ──
            if not closed and pos['tmode']=='atr':
                if pos['side']=='LONG':
                    if not pos['be_done'] and (c-e)/risk>=pos['be']*((pos['tp']-e)/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c-pos['trail']*a
                        if cand>pos['cur']: pos['cur']=cand
                else:
                    if not pos['be_done'] and (e-c)/risk>=pos['be']*((e-pos['tp'])/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c+pos['trail']*a
                        if cand<pos['cur']: pos['cur']=cand
            if not closed and pos['manage']=='fixed':
                pos['age']+=1
                if pos['age']<=L_INVAL_N and not np.isnan(pos['bl']):
                    inside=(c<pos['bl']) if pos['side']=='LONG' else (c>pos['bl'])
                    if inside: nr=finalize(pos,c); closed=True
                if not closed and pos['age']>=L_TIMEOUT:
                    nr=finalize(pos,c); closed=True
            if closed:
                balance+=nr*pos['risk_usd']; trades.append(nr); del positions[sym]; cooldown_until[sym]=k+pos['cd']
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sym in A:
            if sym in positions or k<=cooldown_until[sym]: continue
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
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px,
                            'tmode':'fixed' if strat=='lat' else 'atr','be':g['be'],'trail':g['trail'],
                            'manage':'fixed' if strat=='lat' else 'trail','cd':g['cd'],
                            'strat':strat,'bl':d['bl'][k],
                            'remaining':1.0,'realized_r':0.0,'marcos_feitos':set()}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    monthly[cur_month]=balance-MONTHLY_BASE
    return monthly, max_dd, trades, n_parciais


def summ(res):
    monthly, ddmax, trades, npar = res
    yr={}
    for mk,v in monthly.items(): yr[mk[:4]]=yr.get(mk[:4],0)+v
    tot=sum(monthly.values()); tr=np.array(trades); n=len(tr)
    wins=tr[tr>0.03]
    return dict(tot=tot, yr=yr, monthly=monthly, anos=sum(1 for v in yr.values() if v>0),
                meses=sum(1 for v in monthly.values() if v>0), nm=len(monthly), dd=ddmax,
                n=n, wr=len(wins)/n*100 if n else 0, npar=npar)


print('A carregar 5 anos + OOS 2026 (cache)...')
p5,b5=load(fetch_5y); a5=prep(p5,b5)
p26,b26=load(fetch_26); a26=prep(p26,b26)
print(f'  {len(a5)} pares OK')
S5=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc))
S26=(datetime(2026,1,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc))

M=[0.25,0.50,0.75]   # marcos: 25%, 50%, 75% do caminho ao alvo (100% = o TP fecha o resto)

def botao(f1,f2,f3): return [(M[0],f1),(M[1],f2),(M[2],f3)]

CFG=[
  ('BASELINE (bot actual)',            None,                      'botao'),
  ('fixo 20% em cada marco',           botao(0.20,0.20,0.20),     'botao'),
  ('fixo 25% em cada marco',           botao(0.25,0.25,0.25),     'botao'),
  ('fixo 33% em cada marco',           botao(0.33,0.33,0.33),     'botao'),
  ('fixo 50% em cada marco',           botao(0.50,0.50,0.50),     'botao'),
  ('ASCENDENTE 25/50/75',              botao(0.25,0.50,0.75),     'botao'),
  ('DESCENDENTE 75/50/25',             botao(0.75,0.50,0.25),     'botao'),
  ('cumulativo 25/50/75 (sobra 25%)',  botao(0.25,0.50,0.75),     'cumulativo'),
]

def sobra(cfg, modo):
    if not cfg: return 1.0
    if modo=='cumulativo': return 1.0-cfg[-1][1]
    r=1.0
    for _,f in cfg: r*= (1-f)
    return r

print('\n'+'='*104)
print('  Parciais nos marcos 25%/50%/75% do alvo | fracao = clique do botao (da posicao ACTUAL)')
print('='*104)
print('\n  '+f"{'config':<34}{'sobra':>7}{'5anos':>9}{'anos+':>7}{'DD':>7}{'WR':>6}{'||':>4}{'2026':>8}{'m+':>6}{'DD':>7}")
print('  '+'-'*100)
res={}
for nome, cfg, modo in CFG:
    s5=summ(run(a5,*S5,marcos=cfg,modo=modo)); s26=summ(run(a26,*S26,marcos=cfg,modo=modo))
    res[nome]=(s5,s26)
    print('  '+f"{nome:<34}{sobra(cfg,modo)*100:>6.1f}%{s5['tot']:>+9.0f}{s5['anos']:>5}/5{s5['dd']:>6.1f}%{s5['wr']:>5.0f}%{'||':>4}"
          f"{s26['tot']:>+8.0f}{s26['meses']:>3}/{s26['nm']:<2}{s26['dd']:>6.1f}%")

b5s,b26s=res['BASELINE (bot actual)']
print('\n'+'='*104)
print('  DELTA vs baseline (custo/ganho de cada esquema)')
print('  '+f"{'config':<34}{'delta 5anos':>13}{'delta OOS':>12}{'delta DD':>11}{'delta WR':>10}")
for nome,cfg,modo in CFG[1:]:
    s5,s26=res[nome]
    print('  '+f"{nome:<34}{s5['tot']-b5s['tot']:>+13.0f}{s26['tot']-b26s['tot']:>+12.0f}"
          f"{s5['dd']-b5s['dd']:>+10.1f}%{s5['wr']-b5s['wr']:>+9.0f}pp")

a5s=res['ASCENDENTE 25/50/75'][0]; d5s=res['DESCENDENTE 75/50/25'][0]
a26s=res['ASCENDENTE 25/50/75'][1]; d26s=res['DESCENDENTE 75/50/25'][1]
print('\n  VENDER CEDO vs TARDE (ambos deixam 9.4% a correr — so muda o TIMING):')
print(f"    ASCENDENTE (vende tarde): {a5s['tot']:+.0f} 5anos | {a26s['tot']:+.0f} OOS")
print(f"    DESCENDENTE (vende cedo): {d5s['tot']:+.0f} 5anos | {d26s['tot']:+.0f} OOS")
print(f"    -> diferenca: {a5s['tot']-d5s['tot']:+.0f} (5anos) | {a26s['tot']-d26s['tot']:+.0f} (OOS)")
print('\n  Conservador: vela que toca stop E marco conta como STOP. Bot NAO tocado.')
