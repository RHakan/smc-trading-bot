"""
backtest_be30_todas.py
Piso de proteção nas 3 estratégias: ao atingir >=30% do caminho ao alvo, move o
stop para ENTRY + TAXA (zero a zero). Objetivo do Rafa: minimizar perdas sem
tocar nos ganhos — o alvo e o trailing das direcionais ficam intactos; só se
corta a cauda de prejuízo quando o trade já andou a favor.

baseline : gestão atual (lateral TP/SL fixos; bull/bear BE@1.0R + trail 2xATR)
be30     : as 3 -> BE a 30% do caminho, stop em entry+taxa (net 0 se voltar).
           direcionais mantêm o trailing depois do BE (ganho não é limitado);
           lateral mantém TP fixo, só ganha o piso de BE.

Mede o que importa: total, ano a ano, WR, DD, meses+, E a decomposição
perda/ganho (soma dos trades negativos vs positivos, nº de trades salvos em 0).
Sinais = sistema v3 com filtro BTC aprovado. 5 anos, 500/mês reset. Cache 2020-2025.
Uso: python backtests/backtest_be30_todas.py
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
L_LOOKBACK=30; L_ADX_MAX=20; L_BUFFER=0.1; L_COOLDOWN=4; L_TIMEOUT=48
SCRATCH=0.03   # |net R| <= isto conta como "zero a zero" (salvo pelo piso)

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


print('Piso de proteção BE@30%+taxa nas 3 estratégias | sistema v3, 5 anos, 500/mês reset')
print('A carregar dados (cache 2020-2025)...')
pair_data={}; btc_bull=None
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
        df1['atr_avg']=df1['atr'].rolling(B_ATR_AVG).mean()
        df1['adx']=adx(df1,14)
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['slope_d']=dfd['slope_d'].shift(1).reindex(df1.index,method='ffill')
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        df1['ema20']=c.ewm(span=20,adjust=False).mean()
        df1['ema50']=c.ewm(span=50,adjust=False).mean()
        pair_data[sym]=df1
        if sym.startswith('BTC'):
            btc_bull=(df1['regime']=='BULL')
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen(df, btc_local):
    """Sinais do sistema v3 com filtro BTC (bull só em BTC BULL). Não muda por modo."""
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    reg=df['regime'].values; slope=df['slope_d'].values; ema_t=df['ema_trend'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object)
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
                if cl>rl: side[i]='LONG'; entry[i]=cl; sl[i]=rl; tp[i]=cl+height; strat[i]='lat'
            elif cl<rl-buf and pc>=rl:
                if rh>cl: side[i]='SHORT'; entry[i]=cl; sl[i]=rh; tp[i]=cl-height; strat[i]='lat'
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
    return side,entry,sl,tp,strat


def gest_for(mode):
    if mode=='baseline':
        return {'bull':dict(tmode='atr',ecol=None,eact=0.0,be=0.40,trail=U_TRAIL,manage='trail',cd=U_COOLDOWN),
                'bear':dict(tmode='atr',ecol=None,eact=0.0,be=0.40,trail=B_TRAIL,manage='trail',cd=B_COOLDOWN),
                'lat' :dict(tmode='fixed',ecol=None,eact=0.0,be=9.99,trail=0.0,manage='fixed',cd=L_COOLDOWN)}
    # prog_<far>_<near>: trailing PROGRESSIVO. Distância = interpola de (far/10)R longe
    # do alvo até (near/10)R colado ao alvo, em função do % do caminho. Ideia do Rafa:
    # folga larga cedo (deixa correr), aperta perto do TP (protege a reversão na chegada).
    if mode.startswith('prog_'):
        _,f,nr=mode.split('_'); dfar=float(f)/10.0; dnear=float(nr)/10.0
        base=dict(tmode='prog',ecol=None,eact=0.0,dfar=dfar,dnear=dnear,be=9.99,trail=0.0,manage='prog')
        return {'bull':{**base,'cd':U_COOLDOWN},'bear':{**base,'cd':B_COOLDOWN},'lat':{**base,'cd':L_COOLDOWN}}
    # rtrail<N>: step trailing de distância FIXA = (N/10)R atrás do topo (ratchet).
    # Ideia do Rafa: ao avançar, o stop sobe mantendo a mesma folga em R. rtrail10=1.0R
    # (no entry o stop = SL inicial; a +1R vai a BE; a +2R trava +1R...). rtrail15=1.5R (folgado).
    if mode.startswith('rtrail'):
        mult=float(mode.replace('rtrail',''))/10.0
        base=dict(tmode='rtrail',ecol=None,eact=0.0,rmult=mult,be=9.99,trail=0.0,manage='rtrail')
        return {'bull':{**base,'cd':U_COOLDOWN},'bear':{**base,'cd':B_COOLDOWN},'lat':{**base,'cd':L_COOLDOWN}}
    # ema<N>[_<act>]: stop segue a EMA-N (ratchet a partir do SL inicial), exit se
    # o preço perde a média. act = % do caminho a partir do qual a EMA passa a valer
    # (antes disso o SL estrutural dá espaço). Ex.: ema20, ema50, ema20_50 (=ativa a 50%).
    parts=mode.split('_'); ecol=parts[0]                       # 'ema20' ou 'ema50'
    eact=float(parts[1])/100.0 if len(parts)>1 else 0.0
    base=dict(tmode='ema',ecol=ecol,eact=eact,be=9.99,trail=0.0,manage='ematrail')
    return {'bull':{**base,'cd':U_COOLDOWN},'bear':{**base,'cd':B_COOLDOWN},'lat':{**base,'cd':L_COOLDOWN}}


def run(mode, start, end):
    GEST=gest_for(mode)
    midx=None; arrs={}
    for sym,df in pair_data.items():
        btc_local=btc_bull.reindex(df.index).fillna(False).values
        s,e,sl,tp,st=gen(df, btc_local)
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st)
        midx=d.index if midx is None else midx.union(d.index); arrs[sym]=d
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'ema20':dd['ema20'].values,'ema50':dd['ema50'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,'tp':dd['_tp'].values,'strat':dd['_st'].values}
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; monthly={}; cur_month=None
    trades=[]   # net R de cada trade fechado (p/ WR e decomposição perda/ganho)
    def record(nr): trades.append(nr)
    def close_all(k):
        nonlocal balance
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]; pos=positions[sym]
            if np.isnan(c): c=pos['entry']
            e=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
            nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r
            balance+=nr*pos['risk_usd']; record(nr); del positions[sym]
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
                    if not pos['be_done'] and (c-e)/risk>=pos['be']*((pos['tp']-e)/risk):
                        pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c-pos['trail']*a
                        if cand>pos['cur']: pos['cur']=cand
                elif pos['tmode']=='ema':
                    if (c-e)/risk>=pos['eact']*((pos['tp']-e)/risk):
                        ev=d[pos['ecol']][k]                    # EMA segue por baixo do preço
                        if not np.isnan(ev) and ev>pos['cur'] and ev<c: pos['cur']=ev
                elif pos['tmode']=='rtrail':
                    cand=c-pos['rmult']*risk                   # distância fixa em R atrás do topo
                    if cand>pos['cur']: pos['cur']=cand
                elif pos['tmode']=='prog':
                    pr=min(max((c-e)/(pos['tp']-e),0.0),1.0)   # % do caminho ao alvo
                    D=pos['dfar']*(1-pr)+pos['dnear']*pr       # folga encolhe perto do TP
                    cand=c-D*risk
                    if cand>pos['cur']: pos['cur']=cand
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
                elif pos['tmode']=='atr':
                    if not pos['be_done'] and (e-c)/risk>=pos['be']*((e-pos['tp'])/risk):
                        pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c+pos['trail']*a
                        if cand<pos['cur']: pos['cur']=cand
                elif pos['tmode']=='ema':
                    if (e-c)/risk>=pos['eact']*((e-pos['tp'])/risk):
                        ev=d[pos['ecol']][k]                    # EMA segue por cima do preço
                        if not np.isnan(ev) and ev<pos['cur'] and ev>c: pos['cur']=ev
                elif pos['tmode']=='rtrail':
                    cand=c+pos['rmult']*risk                   # distância fixa em R atrás do topo
                    if cand<pos['cur']: pos['cur']=cand
                elif pos['tmode']=='prog':
                    pr=min(max((e-c)/(e-pos['tp']),0.0),1.0)
                    D=pos['dfar']*(1-pr)+pos['dnear']*pr
                    cand=c+D*risk
                    if cand<pos['cur']: pos['cur']=cand
            if not closed and pos['manage']=='fixed':
                pos['age']+=1
                if pos['age']>=L_TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']; record(nr)
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
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px,
                            'tmode':g['tmode'],'ecol':g['ecol'],'eact':g['eact'],'rmult':g.get('rmult',0.0),
                            'dfar':g.get('dfar',0.0),'dnear':g.get('dnear',0.0),
                            'be':g['be'],'trail':g['trail'],'manage':g['manage'],'cd':g['cd']}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    monthly[cur_month]=balance-MONTHLY_BASE
    return dict(monthly=monthly, pnl=sum(monthly.values()), max_dd=max_dd, trades=trades)


def evaluate(mode):
    allm={}; tot=0; ddmax=0; alltr=[]
    for yr in YEARS:
        start=datetime(yr,1,1,tzinfo=timezone.utc); end=YEAR_END.get(yr,datetime(yr,12,31,tzinfo=timezone.utc))
        r=run(mode,start,end)
        allm.update(r['monthly']); tot+=r['pnl']; ddmax=max(ddmax,r['max_dd']); alltr+=r['trades']
    yr_pnl={}
    for mk,v in allm.items(): yr_pnl[mk[:4]]=yr_pnl.get(mk[:4],0)+v
    tr=np.array(alltr)
    n=len(tr)
    wins=tr[tr>SCRATCH]; losses=tr[tr<-SCRATCH]; scratch=tr[np.abs(tr)<=SCRATCH]
    return dict(tot=tot, anos=sum(1 for v in yr_pnl.values() if v>0),
                meses=sum(1 for v in allm.values() if v>0), nm=len(allm), ddmax=ddmax, yr_pnl=yr_pnl,
                n=n, wr=len(wins)/n*100 if n else 0, nwin=len(wins), nloss=len(losses), nscr=len(scratch),
                sum_win=wins.sum(), sum_loss=losses.sum(), avg_loss=losses.mean() if len(losses) else 0)


MODES=[('baseline','baseline (atual)'),('rtrail15','step fixo 1.5R (ref)'),
       ('prog_25_05','progr 2.5R→0.5R'),('prog_20_03','progr 2.0R→0.3R'),
       ('prog_30_05','progr 3.0R→0.5R'),('prog_20_05','progr 2.0R→0.5R')]
print('='*104)
print('  TRAILING PROGRESSIVO — folga larga longe do alvo, aperta perto do TP | v3, 5 anos, 500/mês')
print('='*104)
print('  '+f"{'Config':<18}"+''.join(f"{str(y)[2:]+YEAR_CTX[y][:1]:>7}" for y in YEARS)+
      f"{'TOTAL':>8}{'anos+':>7}{'DD':>7}")
print('  '+'-'*80)
res={}
for mode,lbl in MODES:
    ev=evaluate(mode); res[mode]=ev
    print('  '+f"{lbl:<18}"+''.join(f"{ev['yr_pnl'].get(str(y),0):>+7.0f}" for y in YEARS)+
          f"{ev['tot']:>+8.0f}{ev['anos']:>5}/5{ev['ddmax']:>6.1f}%")

b=res['baseline']
print('\n'+'='*104)
print('  TRADE-OFF vs baseline: cortou dor (Δperdas>0) mais do que comeu lucro (Δganhos<0)?')
print('  '+'-'*80)
print(f"  {'':<18}{'WR':>6}{'salvos~0':>10}{'perdas R':>10}{'ganhos R':>10}{'Δperda':>9}{'Δganho':>9}{'ΔUSDC':>8}")
print(f"  {'baseline':<18}{b['wr']:>5.0f}%{b['nscr']:>10}{b['sum_loss']:>+10.0f}{b['sum_win']:>+10.0f}{'—':>9}{'—':>9}{'—':>8}")
for mode,lbl in MODES[1:]:
    e=res[mode]
    dl=e['sum_loss']-b['sum_loss']; dw=e['sum_win']-b['sum_win']; dt=e['tot']-b['tot']
    print(f"  {lbl:<18}{e['wr']:>5.0f}%{e['nscr']:>10}{e['sum_loss']:>+10.0f}{e['sum_win']:>+10.0f}"
          f"{dl:>+9.0f}{dw:>+9.0f}{dt:>+8.0f}")
print('='*104)
best=max(res.items(), key=lambda x:x[1]['tot'])
rd=lambda e: e['tot']/e['ddmax'] if e['ddmax'] else 0
print(f"  Maior total: {best[0]} ({best[1]['tot']:+.0f}). Baseline = {b['tot']:+.0f}.")
print('  ret/DD:  '+' | '.join(f"{lbl.split(' ')[0]} {rd(res[m]):.1f}" for m,lbl in MODES))
print('  progr X→Y = folga X R longe do alvo, encolhendo linearmente até Y R colado ao TP.')
print('  Custos 0.14% RT já embutidos. 2026 OOS não incluído (cache 2020-2025).')
