"""
backtest_v3_candle_trail.py
Sistema v3 — gestao de saida: TRAILING POR VELA (ideia do Rafa) vs ATR atual.

Ideia do Rafa: apos abrir, mover o stop a cada vela para 0.30% "contra" a vela
anterior. Short: stop = close[vela-1]*(1+m). Long: stop = close[vela-1]*(1-m).
Move so a favor. Se a vela rompe -> sai (reversao/respiro); o Decisor fica livre
para reabrir. Captura o movimento de perto e encurta a duracao das ordens.

Aplicado so as DIRECIONAIS (bull/bear); a lateral mantem TP fixo (altura do range).
Comparado com a gestao atual (BE 70% + trailing 2xATR).

Variacoes: margem 0.2/0.3/0.5%; referencia close vs high(short)/low(long);
ativacao imediata (0R) vs so depois de 0.5R/1R de lucro (deixa respirar no inicio).

Sistema v3, Decisor diario (o do bot), 5 anos (2021-2025), 500/mes reset.
Metricas: total, anos+, DD, WR, duracao media das direcionais. Cache _2020_2025.
Uso: python backtests/backtest_v3_candle_trail.py
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
B_COOLDOWN=BEAR.COOLDOWN_BARS; B_RR=BEAR.RR_CAP; B_BE=BEAR.BE_TRIGGER_PCT; B_TRAIL=BEAR.TRAIL_ATR
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
BSEL_LB=30; BSEL_ADX=20; BSEL_STRUCT=20
U_SWING_N=10; U_CHOCH_BARS=12; U_CHOCH_REF=15; U_MIN_SWEEP=0.05; U_RR=2.5; U_BE=0.70; U_TRAIL=2.0
U_COOLDOWN=3; U_SLOPE_MIN=0.006; U_EMA_TREND=100
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


print('Sistema v3 — trailing por VELA (ideia do Rafa) vs ATR atual | 5 anos, 500/mes reset')
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
        dfd['slope_d']=slope
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['atr_avg']=df1['atr'].rolling(B_ATR_AVG).mean()
        df1['adx']=adx(df1,14)
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['slope_d']=dfd['slope_d'].shift(1).reindex(df1.index,method='ffill')
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen(df):
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


CD={'bull':U_COOLDOWN,'bear':B_COOLDOWN,'lat':L_COOLDOWN}


def run(mode, margin, ref, activate, start, end):
    """mode: 'atr' (BE+trail 2xATR) | 'candle' (trailing por vela) — so nas direcionais."""
    midx=None; arrs={}
    for sym,df in pair_data.items():
        s,e,sl,tp,st=gen(df)
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st)
        midx=d.index if midx is None else midx.union(d.index); arrs[sym]=d
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,'tp':dd['_tp'].values,'strat':dd['_st'].values}
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; monthly={}; cur_month=None
    wins=0; ntr=0; dur_dir=[]
    def close_all(k):
        nonlocal balance,wins,ntr
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): c=positions[sym]['entry']
            pos=positions[sym]; e=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
            nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r
            balance+=nr*pos['risk_usd']; ntr+=1; wins+= 1 if nr>0 else 0
            if pos['strat']!='lat': dur_dir.append(k-pos['k_in'])
            del positions[sym]
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
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
            if not closed and pos['manage']=='atr':
                prog=(c-e)/risk if pos['side']=='LONG' else (e-c)/risk
                if not pos['be_done'] and prog>=pos['be']*pos['rr']: pos['cur']=e; pos['be_done']=True
                if pos['be_done']:
                    if pos['side']=='LONG':
                        cand=c-U_TRAIL*a
                        if cand>pos['cur']: pos['cur']=cand
                    else:
                        cand=c+B_TRAIL*a
                        if cand<pos['cur']: pos['cur']=cand
            elif not closed and pos['manage']=='candle':
                prog=(c-e)/risk if pos['side']=='LONG' else (e-c)/risk
                if prog>=activate:   # so aperta apos 'activate' R de lucro
                    baseval=c if ref=='close' else (hi if pos['side']=='SHORT' else lo)
                    if pos['side']=='LONG':
                        cand=baseval*(1-margin)
                        if cand>pos['cur']: pos['cur']=cand
                    else:
                        cand=baseval*(1+margin)
                        if cand<pos['cur']: pos['cur']=cand
            elif not closed and pos['manage']=='fixed':
                pos['age']+=1
                if pos['age']>=L_TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']; ntr+=1; wins+= 1 if nr>0 else 0
                if pos['strat']!='lat': dur_dir.append(k-pos['k_in'])
                del positions[sym]; cooldown_until[sym]=k+CD[pos['strat']]
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
            mng='fixed' if strat=='lat' else mode
            rr=(U_RR if strat=='bull' else B_RR)
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'be_done':False,'age':0,'k_in':k,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px,
                            'strat':strat,'be':(U_BE if strat=='bull' else B_BE),'rr':rr,'manage':mng}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    monthly[cur_month]=balance-MONTHLY_BASE
    return dict(monthly=monthly, pnl=sum(monthly.values()), max_dd=max_dd,
                wr=(wins/ntr*100 if ntr else 0), ntr=ntr, dur=(np.median(dur_dir) if dur_dir else 0))


def evaluate(mode, margin=0.003, ref='close', activate=0.0):
    allm={}; tot=0; ddmax=0; wins=0; ntr=0; durs=[]
    for yr in YEARS:
        start=datetime(yr,1,1,tzinfo=timezone.utc); end=YEAR_END.get(yr,datetime(yr,12,31,tzinfo=timezone.utc))
        r=run(mode,margin,ref,activate,start,end)
        allm.update(r['monthly']); tot+=r['pnl']; ddmax=max(ddmax,r['max_dd'])
        wins+=r['wr']*r['ntr']/100; ntr+=r['ntr']; durs.append(r['dur'])
    yr_pnl={}
    for mk,v in allm.items(): yr_pnl[mk[:4]]=yr_pnl.get(mk[:4],0)+v
    anos=sum(1 for v in yr_pnl.values() if v>0)
    meses=sum(1 for v in allm.values() if v>0)
    return dict(tot=tot,anos=anos,meses=meses,n=len(allm),ddmax=ddmax,
                wr=(wins/ntr*100 if ntr else 0),ntr=ntr,dur=np.median(durs),monthly=allm)


print('='*104)
print('  GESTAO DE SAIDA — ATR atual vs trailing por vela (dur = duracao mediana das direcionais, h)')
print('='*104)
print(f"  {'Config':<30} | {'TOTAL':>7} | {'anos+':>6} | {'meses+':>9} | {'WR':>5} | {'DD':>6} | {'trades':>7} | {'dur':>5}")
print('  '+'-'*100)
def show(lbl, ev):
    print(f"  {lbl:<30} | {ev['tot']:>+7.0f} | {ev['anos']:>4}/5 | {ev['meses']:>3}/{ev['n']:<3} | "
          f"{ev['wr']:>4.1f}% | {ev['ddmax']:>5.1f}% | {ev['ntr']:>7} | {ev['dur']:>4.0f}h")

base=evaluate('atr'); show('ATR atual (BE70 + trail 2x)', base)
print('\n  Trailing por vela — margem (ref=close, ativacao imediata):')
for m in [0.002,0.003,0.005]:
    show(f'candle {m*100:.1f}% close imediato', evaluate('candle',margin=m,ref='close',activate=0.0))
print('\n  Trailing por vela — referencia high/low da vela (margem 0.3%):')
show('candle 0.3% high/low imediato', evaluate('candle',margin=0.003,ref='wick',activate=0.0))
print('\n  Trailing por vela — deixar respirar antes de apertar (margem 0.3% close):')
for act in [0.5,1.0]:
    show(f'candle 0.3% close apos {act}R', evaluate('candle',margin=0.003,ref='close',activate=act))

# melhor por (anos+, depois total)
cands=[('atr',base)]
for m in [0.002,0.003,0.005]: cands.append((f'candle{m}', evaluate('candle',margin=m)))
for act in [0.5,1.0]: cands.append((f'candle_act{act}', evaluate('candle',activate=act)))
bn,best=max(cands, key=lambda x:(x[1]['anos'],x[1]['tot']))
print('\n'+'='*104)
print(f'  MELHOR: {bn}  ->  {best["anos"]}/5 anos+, total {best["tot"]:+.0f}, DD {best["ddmax"]:.1f}%, WR {best["wr"]:.1f}%, dur {best["dur"]:.0f}h')
print('='*104)
ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
months=best['monthly']; yrs=sorted(set(int(mk[:4]) for mk in months))
print(f"  {'Ano':<6} | "+' '.join(f'{m:>5}' for m in ML)+f" | {'Soma':>7}")
print('  '+'-'*90)
for yr in yrs:
    vals=[months.get(f'{yr}-{m+1:02d}') for m in range(12)]
    cells=' '.join((f'{v:>+5.0f}' if v is not None else '    .') for v in vals)
    print(f"  {yr:<6} | {cells} | {sum(v for v in vals if v is not None):>+7.0f}")
print('\n  Nota: candle-trail so nas direcionais (bull/bear); lateral mantem TP fixo.')
print('  ref=wick usa high(short)/low(long) da vela; ref=close usa o fecho. Custos 0.14% RT.')
