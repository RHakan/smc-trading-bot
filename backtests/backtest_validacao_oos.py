"""
backtest_validacao_oos.py
VALIDACAO OUT-OF-SAMPLE do sistema completo (o teste que separa edge de overfit).

Todos os filtros (min_score, RSI dinamico, guard BTC) foram tunados em jun25-jun26.
Aqui rodamos o sistema com TODOS os parametros CONGELADOS em periodos que eles
NUNCA viram: 2024 inteiro e 2023 (18/06->fim). Se aguentar -> robusto. Se desabar
-> overfit, e recalibramos AGORA (em backtest, nao com dinheiro real).

Parametros CONGELADOS (identicos ao melhor de jun25-jun26):
  min_score=0.2 | max_per_side=3 | RSI dinamico bear W200 k2.0 | guard BTC RSI<25

Usa cache _1h_2020_2025.csv / _1d_2020_2025.csv (do backtest_bull_smc/_dev_hist).
Inclui levantamento MENSAL por periodo. Uso: python backtests/backtest_validacao_oos.py
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

# Periodos OUT-OF-SAMPLE (nenhum parametro foi tunado aqui)
PERIODS=[
    {'name':'OOS 2024 (ano inteiro)','start':datetime(2024,1,1,tzinfo=timezone.utc),'end':datetime(2024,12,31,tzinfo=timezone.utc)},
    {'name':'OOS 2023 (jun-dez)','start':datetime(2023,6,18,tzinfo=timezone.utc),'end':datetime(2023,12,31,tzinfo=timezone.utc)},
    {'name':'OOS 2025 (jan-out)','start':datetime(2025,1,1,tzinfo=timezone.utc),'end':datetime(2025,10,6,tzinfo=timezone.utc)},
]
FETCH_END=datetime(2025,10,6,tzinfo=timezone.utc)
INITIAL_BALANCE=5000.0; RISK_PCT=1.0
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
B_LOOKBACK=BEAR.LOW_LOOKBACK; B_ATR_AVG=BEAR.ATR_AVG_PERIOD; B_ATR_STOP=BEAR.ATR_STOP_MULT
B_COOLDOWN=BEAR.COOLDOWN_BARS; B_RR=BEAR.RR_CAP; B_BE=BEAR.BE_TRIGGER_PCT; B_TRAIL=BEAR.TRAIL_ATR
U_SWING_N=10; U_CHOCH_BARS=12; U_CHOCH_REF=15; U_MIN_SWEEP=0.05; U_RR=2.5; U_BE=0.70; U_TRAIL=2.0
U_COOLDOWN=3; U_SLOPE_MIN=0.006; U_EMA_TREND=100
L_LOOKBACK=30; L_ADX_MAX=20; L_BUFFER=0.1; L_COOLDOWN=4; L_TIMEOUT=48
# ── CONGELADOS ──
MIN_SCORE=0.2; MAX_PER_SIDE=3; BEAR_RSI_W=200; BEAR_RSI_K=2.0; BTC_GUARD_RSI=25

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


print('VALIDACAO OUT-OF-SAMPLE — parametros CONGELADOS de jun25-jun26')
print(f'  min_score={MIN_SCORE} side={MAX_PER_SIDE} RSI-din W{BEAR_RSI_W}/k{BEAR_RSI_K} guardBTC<{BTC_GUARD_RSI}')
print('  Referencia in-sample (jun25-jun26): +204%, DD 30.4%, dor -1668\n')
print('A carregar dados (cache 2020-2025; 1d baixa se faltar)...')
pair_data={}; btc_rsi_full=None
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
        rsi=rsi_calc(c,14); df1['rsi']=rsi
        df1['rsi_ma']=rsi.rolling(BEAR_RSI_W).mean(); df1['rsi_sd']=rsi.rolling(BEAR_RSI_W).std()
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['slope_d']=dfd['slope_d'].shift(1).reindex(df1.index,method='ffill')
        df1['low_n']=df1['l'].shift(1).rolling(B_LOOKBACK).min()
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        df1['btc_bull']=(df1['regime']=='BULL')
        pair_data[sym]=df1
        if sym.startswith('BTC'): btc_rsi_full=rsi
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()

GEST={'bear':dict(be=B_BE,trail=B_TRAIL,manage='trail',cd=B_COOLDOWN),
      'bull':dict(be=U_BE,trail=U_TRAIL,manage='trail',cd=U_COOLDOWN),
      'lat': dict(be=0,trail=0,manage='fixed',cd=L_COOLDOWN)}
ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']


def gen_all(df):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    reg=df['regime'].values; slope=df['slope_d'].values; low_n=df['low_n'].values; ema_t=df['ema_trend'].values
    rsi=df['rsi'].values; rsi_ma=df['rsi_ma'].values; rsi_sd=df['rsi_sd'].values; btc_local=df['btc_bull'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object); score=np.full(n,np.nan)
    start_i=max(B_ATR_AVG+B_LOOKBACK+5, U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, L_LOOKBACK+5, BEAR_RSI_W+5, 210)
    for i in range(start_i,n):
        r=reg[i]; a=atr[i]
        if np.isnan(a) or a<=0: continue
        if r=='BEAR':
            cl=c[i]; op=o[i]; av=atr_avg[i]; ln=low_n[i]
            if not any(np.isnan(v) for v in (cl,av,ln)) and cl<ln and cl<op and a>av:
                ri=rsi[i]
                if not np.isnan(rsi_ma[i]) and not np.isnan(rsi_sd[i]) and not np.isnan(ri):
                    if ri < rsi_ma[i]-BEAR_RSI_K*rsi_sd[i]: continue
                stop=cl+B_ATR_STOP*a; risk=stop-cl
                if risk>0:
                    side[i]='SHORT'; entry[i]=cl; sl[i]=stop; tp[i]=cl-B_RR*risk; strat[i]='bear'; score[i]=(ln-cl)/a
            continue
        if r=='BULL':
            av=atr_avg[i]
            if (not np.isnan(av) and a<av): continue
            if np.isnan(slope[i]) or slope[i]<U_SLOPE_MIN: continue
            if not btc_local[i]: continue
            if np.isnan(ema_t[i]) or c[i]<=ema_t[i]: continue
            best=None
            for j in range(i-1,max(i-U_CHOCH_BARS-1,U_SWING_N+U_CHOCH_REF)-1,-1):
                swing_low=np.min(l[j-U_SWING_N:j])
                if not (l[j]<swing_low and c[j]>swing_low): continue
                if U_MIN_SWEEP>0 and (swing_low-l[j])/swing_low*100<U_MIN_SWEEP: continue
                ref_high=np.max(h[j-U_CHOCH_REF:j])
                if np.any(c[j+1:i]>ref_high): continue
                if c[i]>ref_high:
                    al=np.min(l[j:i+1]); risk=c[i]-al
                    if risk>0 and risk/c[i]<=0.10: best=(c[i],al,ref_high); break
            if best:
                e,stop,rh=best; risk=e-stop
                side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=e+U_RR*risk; strat[i]='bull'; score[i]=(e-rh)/a
            continue
        if r=='NEUTRAL':
            if np.isnan(adx_a[i-1]) or adx_a[i-1]>L_ADX_MAX: continue
            rl=np.min(l[i-L_LOOKBACK:i]); rh=np.max(h[i-L_LOOKBACK:i])
            if rl<=0 or rh<=rl: continue
            buf=L_BUFFER*a; cl=c[i]; pc=c[i-1]; height=rh-rl
            if cl>rh+buf and pc<=rh:
                stop=rl; risk=cl-stop
                if risk>0: side[i]='LONG'; entry[i]=cl; sl[i]=stop; tp[i]=cl+height; strat[i]='lat'; score[i]=(cl-rh)/a
            elif cl<rl-buf and pc>=rl:
                stop=rh; risk=stop-cl
                if risk>0: side[i]='SHORT'; entry[i]=cl; sl[i]=stop; tp[i]=cl-height; strat[i]='lat'; score[i]=(rl-cl)/a
    return side,entry,sl,tp,strat,score


def run_period(period):
    midx=None
    arrs={}
    for sym,df in pair_data.items():
        s,e,sl,tp,st,sc=gen_all(df)
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st,_sc=sc)
        midx=d.index if midx is None else midx.union(d.index)
        arrs[sym]=d
    midx=midx[(midx>=period['start'])&(midx<=period['end'])]
    btc_rsi=btc_rsi_full.reindex(midx).values
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,
                'tp':dd['_tp'].values,'strat':dd['_st'].values,'score':dd['_sc'].values}
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]
    N=len(midx)
    for k in range(N):
        ts=midx[k]
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]; atr=d['atr'][k]
            risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']; a=atr if not np.isnan(atr) else risk
            closed=False; nr=0.0
            if pos['side']=='LONG':
                if lo<=pos['cur']: nr=(pos['cur']-e)/risk-fee_r; closed=True
                elif hi>=pos['tp']: nr=(pos['tp']-e)/risk-fee_r; closed=True
                elif pos['manage']=='trail':
                    if not pos['be_done'] and (c-e)/risk>=pos['be']*((pos['tp']-e)/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done']:
                        cand=c-pos['trail']*a
                        if cand>pos['cur']: pos['cur']=cand
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
                elif pos['manage']=='trail':
                    if not pos['be_done'] and (e-c)/risk>=pos['be']*((e-pos['tp'])/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done']:
                        cand=c+pos['trail']*a
                        if cand<pos['cur']: pos['cur']=cand
            if not closed and pos['manage']=='fixed':
                pos['age']+=1
                if pos['age']>=L_TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']
                trades.append({'ts':ts,'nr':nr,'strat':pos['strat'],'pnl':nr*pos['risk_usd']})
                del positions[sym]; cooldown_until[sym]=k+pos['cd']
        rb=btc_rsi[k]; block_shorts=(not np.isnan(rb)) and rb<BTC_GUARD_RSI
        cands=[]
        for sym in A:
            if sym in positions: continue
            if k<=cooldown_until[sym]: continue
            d=A[sym]; sidev=d['side'][k]
            if sidev is None or (isinstance(sidev,float) and np.isnan(sidev)): continue
            if block_shorts and sidev=='SHORT': continue
            sc=d['score'][k]
            if np.isnan(sc) or sc<MIN_SCORE: continue
            cands.append((sc,sym,sidev))
        cands.sort(reverse=True)
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sc,sym,sidev in cands:
            if sidev=='LONG' and n_long>=MAX_PER_SIDE: continue
            if sidev=='SHORT' and n_short>=MAX_PER_SIDE: continue
            d=A[sym]; e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]; strat=d['strat'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            g=GEST[strat]
            positions[sym]={'side':sidev,'entry':e,'cur':stop,'tp':tp,'be_done':False,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px,
                            'strat':strat,'be':g['be'],'trail':g['trail'],'manage':g['manage'],'cd':g['cd']}
            if sidev=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return dict(balance=balance,trades=trades,max_dd=max_dd)


for p in PERIODS:
    res=run_period(p); t=res['trades']
    print('='*92)
    print(f"  {p['name']}  ({p['start'].date()} -> {p['end'].date()})")
    print('='*92)
    if not t:
        print('  sem trades\n'); continue
    w=[x for x in t if x['nr']>0]; gw=sum(x['nr'] for x in w); gl=abs(sum(x['nr'] for x in t if x['nr']<=0))
    ret=(res['balance']/INITIAL_BALANCE-1)*100
    by={}
    for x in t:
        mk=x['ts'].strftime('%Y-%m'); by.setdefault(mk,{'pnl':0,'t':0,'bear':0}); by[mk]['pnl']+=x['pnl']; by[mk]['t']+=1
        if x['strat']=='bear': by[mk]['bear']+=x['pnl']
    negs=[d['pnl'] for d in by.values() if d['pnl']<0]
    print(f"  Retorno {ret:+.1f}% | {len(t)}t | WR {len(w)/len(t)*100:.1f}% | PF {gw/gl if gl>0 else 99:.2f} | "
          f"DD {res['max_dd']:.1f}% | meses- {len(negs)} | dor {sum(negs):+.0f}")
    run_bal=INITIAL_BALANCE
    print(f"\n  {'Mes':<8} | {'Trades':>6} | {'P&L mes':>10} | {'BEAR':>9} | {'Saldo':>11}")
    print('  '+'-'*54)
    for mk in sorted(by):
        d=by[mk]; run_bal+=d['pnl']; yr,mo=int(mk[:4]),int(mk[5:]); flag=' <<<' if d['pnl']<0 else ''
        print(f"  {ML[mo-1]}/{yr%100:02d} | {d['t']:>6} | {d['pnl']:>+10.0f} | {d['bear']:>+9.0f} | {run_bal:>9,.0f}{flag}")
    print()

print('='*92)
print('  VEREDICTO: se os 3 periodos OOS forem positivos com PF>1 e DD parecido -> ROBUSTO.')
print('  Se algum desabar (PF<1, DD explode) -> overfit; recalibrar com menos parametros.')
print('  Custos 0.14% round-trip. Risco 1%/trade. Bull e Lateral HIPOTETICAS.')
