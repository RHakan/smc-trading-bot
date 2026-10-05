"""
backtest_maxside_confirma.py
Curiosidade do Rafa: "e se o bot fizer EXACTAMENTE o contrario? Onde da LONG, abre SHORT."

INVERSAO = mesma entrada, mesmas DISTANCIAS, lado espelhado:
    side  : LONG <-> SHORT
    stop  : 2*entry - stop_original   (espelha em torno da entrada)
    alvo  : 2*entry - alvo_original
Logo o risco e o alvo mantem a MAGNITUDE (RR identico) — testa-se so a DIRECAO.
Na lateral, como o stop original e longe (lado oposto do range) e o alvo perto
(altura), o trade invertido fica com STOP LONGE e ALVO PERTO, como o Rafa notou.

Sinais IDENTICOS ao bot (mesmo decisor, mesmas 3 estrategias, mesmos filtros).
So a direcao muda no momento de abrir.

A INVALIDACAO nao transfere: ela corta quando o rompimento falha (preco fecha de
volta ao range) — mas num trade invertido isso e o cenario BOM. Por isso rodamos:
  - INVERSO (invalidacao espelhada) = "tudo exactamente igual, so a direcao"
  - INVERSO sem invalidacao          = a leitura honesta (a regra nao faz sentido)

NOTA: inverter NAO transforma -X em +X. As taxas (0.14% RT) pagam-se nos dois lados,
e a saida nao e simetrica (levar stop != bater alvo). Por isso o resultado nao e o
simetrico do baseline — e e isso que torna o teste interessante.

5 anos + OOS 2026. 500/mes reset. MAX 3/lado. Levantamento mensal. Bot NAO tocado.
Uso: python backtests/backtest_inverso.py
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
    """Sinais IDENTICOS ao bot — a inversao acontece só ao abrir (ver run())."""
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


def run(arrs, start, end, invert_strats=frozenset(), use_inval=True, MAX_PER_SIDE=3):
    """invert_strats: conjunto de estrategias a inverter ('lat','bull','bear').
    Vazio = baseline. Como o decisor mapeia regime->estrategia 1:1, inverter uma
    estrategia E inverter o regime dela sao a MESMA coisa (BULL<->bull_smc, etc)."""
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
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]; atr=d['atr'][k]
            risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']; a=atr if not np.isnan(atr) else risk
            closed=False; nr=0.0
            if pos['side']=='LONG':
                if lo<=pos['cur']: nr=(pos['cur']-e)/risk-fee_r; closed=True
                elif hi>=pos['tp']: nr=(pos['tp']-e)/risk-fee_r; closed=True
                elif pos['tmode']=='atr':
                    if not pos['be_done'] and (c-e)/risk>=pos['be']*((pos['tp']-e)/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c-pos['trail']*a
                        if cand>pos['cur']: pos['cur']=cand
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
                elif pos['tmode']=='atr':
                    if not pos['be_done'] and (e-c)/risk>=pos['be']*((e-pos['tp'])/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c+pos['trail']*a
                        if cand<pos['cur']: pos['cur']=cand
            if not closed and pos['manage']=='fixed':
                pos['age']+=1
                if use_inval and pos['age']<=L_INVAL_N and not np.isnan(pos['bl']):
                    inside=(c<pos['bl']) if pos['side']=='LONG' else (c>pos['bl'])
                    if inside:
                        nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
                if not closed and pos['age']>=L_TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']; trades.append(nr); del positions[sym]; cooldown_until[sym]=k+pos['cd']
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sym in A:
            if sym in positions or k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]; strat=d['strat'][k]; bl=d['bl'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue

            if strat in invert_strats:
                # Espelha em torno da entrada: mesma magnitude de risco e alvo, lado oposto.
                side = 'SHORT' if side=='LONG' else 'LONG'
                stop = 2*e - stop
                tp   = 2*e - tp
                if not np.isnan(bl): bl = 2*e - bl

            if side=='LONG' and n_long>=MAX_PER_SIDE: continue
            if side=='SHORT' and n_short>=MAX_PER_SIDE: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            g=GEST[strat]
            tmode='fixed' if strat=='lat' else 'atr'
            manage='fixed' if strat=='lat' else 'trail'
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'be_done':False,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px,
                            'tmode':tmode,'be':g['be'],'trail':g['trail'],'manage':manage,'cd':g['cd'],
                            'strat':strat,'bl':bl}
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


print('A carregar 5 anos + OOS 2026 (cache)...')
p5,b5=load(fetch_5y); a5=prep(p5,b5)
p26,b26=load(fetch_26); a26=prep(p26,b26)
print(f'  {len(a5)} pares OK')
S5=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc))
S26=(datetime(2026,1,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc))


# ============================================================================
# BATERIA DE CONFIRMACAO do max_per_side=2 — a mesma que a invalidacao passou.
#   [1] PLATO        — vizinhanca suave e contigua, nao um bico de faca
#   [2] SUB-PERIODOS — o efeito existe nas DUAS metades dos 5 anos
#   [3] OOS mes a mes — nao depende de 1 mes de sorte
#   [4] GRADE ANUAL  — que ano fica negativo no cap 2 (o 4/5 vs 5/5)
# ============================================================================
print()
print('='*100)
print('  CONFIRMACAO: max_per_side=2 | v3 completo | 500/mes | risco 1%')
print('='*100)
print()
print('  [1] PLATO — a regiao toda tem de ser boa e suave')
print('  '+f"{'max/lado':<10}{'5anos':>9}{'anos+':>7}{'DD':>8}{'ret/DD':>8}{'||':>4}{'2026':>8}{'m+':>6}{'DD':>8}")
print('  '+'-'*64)
R={}
for m in [1,2,3,4,5,6]:
    s5=summ(run(a5,*S5,MAX_PER_SIDE=m)); s26=summ(run(a26,*S26,MAX_PER_SIDE=m)); R[m]=(s5,s26)
    rd=s5['tot']/s5['dd'] if s5['dd'] else 0
    print('  '+f"{m:<10}{s5['tot']:>+9.0f}{s5['anos']:>5}/5{s5['dd']:>7.1f}%{rd:>8.1f}{'||':>4}"
          f"{s26['tot']:>+8.0f}{s26['meses']:>3}/{s26['nm']:<2}{s26['dd']:>7.1f}%")
print()
print('  [2] SUB-PERIODOS — cap2 vs cap3 nas duas metades dos 5 anos')
H1=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2023,6,30,23,tzinfo=timezone.utc))
H2=(datetime(2023,7,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc))
print('  '+f"{'periodo':<22}{'cap3 tot':>10}{'cap3 DD':>9}{'||':>4}{'cap2 tot':>10}{'cap2 DD':>9}{'delta':>8}{'dDD':>8}")
print('  '+'-'*82)
for nome,per in [('2021-01 .. 2023-06',H1),('2023-07 .. 2025-10',H2)]:
    c3=summ(run(a5,*per,MAX_PER_SIDE=3)); c2=summ(run(a5,*per,MAX_PER_SIDE=2))
    print('  '+f"{nome:<22}{c3['tot']:>+10.0f}{c3['dd']:>8.1f}%{'||':>4}{c2['tot']:>+10.0f}{c2['dd']:>8.1f}%"
          f"{c2['tot']-c3['tot']:>+8.0f}{c2['dd']-c3['dd']:>+7.1f}%")
print()
print('  [3] OOS 2026 MES A MES — cap5 (o bot hoje) vs cap3 vs cap2')
m5=R[5][1]['monthly']; m3=R[3][1]['monthly']; m2=R[2][1]['monthly']
meses=sorted(set(m5)|set(m3)|set(m2))
print('  '+f"{'mes':<12}"+''.join(f'{m[5:]:>7}' for m in meses)+f"{'TOT':>8}")
for nome,mm in [('cap5 (hoje)',m5),('cap3',m3),('cap2',m2)]:
    print('  '+f"{nome:<12}"+''.join(f'{mm.get(m,0):>+7.0f}' for m in meses)+f"{sum(mm.values()):>+8.0f}")
print()
print('  [4] GRADE ANUAL — de onde vem o 4/5 do cap2')
print('  '+f"{'ano':<8}{'cap3':>10}{'cap2':>10}{'delta':>9}")
y3=R[3][0]['yr']; y2=R[2][0]['yr']
for y in sorted(set(y3)|set(y2)):
    flag=' <= vira negativo' if (y3.get(y,0)>0 and y2.get(y,0)<0) else ''
    print('  '+f"{y:<8}{y3.get(y,0):>+10.0f}{y2.get(y,0):>+10.0f}{y2.get(y,0)-y3.get(y,0):>+9.0f}{flag}")
print()
print('='*100)
c2_5,c2_26=R[2]; c3_5,c3_26=R[3]
plato=sum(1 for m in [1,2,3] if R[m][0]['tot']>0 and R[m][1]['tot']>0)
print(f"  Custo 5 anos: {c2_5['tot']-c3_5['tot']:+.0f}  |  Ganho OOS: {c2_26['tot']-c3_26['tot']:+.0f}")
print(f"  DD: {c3_5['dd']:.1f}% -> {c2_5['dd']:.1f}%  |  ret/DD: {c3_5['tot']/c3_5['dd']:.1f} -> {c2_5['tot']/c2_5['dd']:.1f}")
print(f"  WR: {c3_5['wr']:.0f}% -> {c2_5['wr']:.0f}% (o cap nao toca no sinal)")
print(f"  Plato caps 1-3 positivo nos DOIS periodos: {plato}/3")
