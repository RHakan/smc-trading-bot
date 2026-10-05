"""
backtest_decisor.py
Ataca o DECISOR — e testa o modelo de capital do Rafa (500 USDC + reset mensal).

Capital: comeca CADA MES com 500 USDC (saca excedente / repoe perda no fim do mes).
  -> drawdown limitado ao que UM mes perde; lucros protegidos. P&L total = soma dos
     meses. Risco 1% do saldo corrente (compoe intra-mes, reseta no mes).

Decisor (regime EMA20 diaria + slope) — variantes testadas:
  slope_thresh : exige inclinacao maior p/ BEAR/BULL (mais estrito = mais NEUTRAL)
  bear_route   : para onde vai o regime BEAR ->
       'bear13' : bear_v13 (shorts de rompimento)  [atual]
       'lat'    : breakout (a logica robusta) tambem no BEAR
       'off'    : nao opera no BEAR

Fixos: NEUTRAL->lateral breakout, BULL->bull SMC+filtros. Validado em 5 anos.
Inclui levantamento MENSAL. Usa cache _2020_2025.
Uso: python backtests/backtest_decisor.py
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
ATR_PERIOD=BEAR.ATR_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS
B_LOOKBACK=BEAR.LOW_LOOKBACK; B_ATR_AVG=BEAR.ATR_AVG_PERIOD; B_ATR_STOP=BEAR.ATR_STOP_MULT
B_COOLDOWN=BEAR.COOLDOWN_BARS; B_RR=BEAR.RR_CAP; B_BE=BEAR.BE_TRIGGER_PCT; B_TRAIL=BEAR.TRAIL_ATR
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


print('DECISOR + modelo de capital do Rafa (500 USDC/mes, reset mensal)')
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
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['atr_avg']=df1['atr'].rolling(B_ATR_AVG).mean()
        df1['adx']=adx(df1,14)
        # daily reindexado (shift 1) p/ classificar regime com slope_thresh variavel
        df1['d_close']=dfd['c'].shift(1).reindex(df1.index,method='ffill')
        df1['d_ema']=ema_d.shift(1).reindex(df1.index,method='ffill')
        df1['d_slope']=slope.shift(1).reindex(df1.index,method='ffill')
        df1['low_n']=df1['l'].shift(1).rolling(B_LOOKBACK).min()
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def regime_of(d_close,d_ema,d_slope,thresh):
    if np.isnan(d_close) or np.isnan(d_ema) or np.isnan(d_slope): return 'NEUTRAL'
    if d_close<d_ema and d_slope<-thresh: return 'BEAR'
    if d_close>d_ema and d_slope> thresh: return 'BULL'
    return 'NEUTRAL'


def breakout_signal(h,l,c,atr_i,i,adx_a):
    """Breakout de range (2 lados). Devolve (side,entry,sl,tp) ou None."""
    if np.isnan(adx_a[i-1]) or adx_a[i-1]>L_ADX_MAX: return None
    rl=np.min(l[i-L_LOOKBACK:i]); rh=np.max(h[i-L_LOOKBACK:i])
    if rl<=0 or rh<=rl: return None
    buf=L_BUFFER*atr_i; cl=c[i]; pc=c[i-1]; height=rh-rl
    if cl>rh+buf and pc<=rh:
        if cl-rl>0: return ('LONG',cl,rl,cl+height)
    elif cl<rl-buf and pc>=rl:
        if rh-cl>0: return ('SHORT',cl,rh,cl-height)
    return None


def gen(df, slope_thresh, bear_route):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    slope=df['d_slope'].values; dclose=df['d_close'].values; dema=df['d_ema'].values
    low_n=df['low_n'].values; ema_t=df['ema_trend'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object)
    bull_bull=(np.array([regime_of(dclose[i],dema[i],slope[i],slope_thresh) for i in range(n)])=='BULL')
    start_i=max(B_ATR_AVG+B_LOOKBACK+5, U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, L_LOOKBACK+5, 210)
    for i in range(start_i,n):
        a=atr[i]
        if np.isnan(a) or a<=0: continue
        r=regime_of(dclose[i],dema[i],slope[i],slope_thresh)
        if r=='NEUTRAL':
            sig=breakout_signal(h,l,c,a,i,adx_a)
            if sig: side[i],entry[i],sl[i],tp[i]=sig; strat[i]='lat'
        elif r=='BULL':
            if np.isnan(slope[i]) or slope[i]<U_SLOPE_MIN: continue
            if not bull_bull[i]: continue
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
                    if risk>0 and risk/c[i]<=0.10: best=(c[i],al); break
            if best:
                e,stop=best; risk=e-stop
                side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=e+U_RR*risk; strat[i]='bull'
        elif r=='BEAR':
            if bear_route=='off': continue
            if bear_route=='lat':
                sig=breakout_signal(h,l,c,a,i,adx_a)
                if sig: side[i],entry[i],sl[i],tp[i]=sig; strat[i]='bear'
            else:  # bear13
                cl=c[i]; op=o[i]; av=atr_avg[i]; ln=low_n[i]
                if not any(np.isnan(v) for v in (cl,av,ln)) and cl<ln and cl<op and a>av:
                    stop=cl+B_ATR_STOP*a; risk=stop-cl
                    if risk>0: side[i]='SHORT'; entry[i]=cl; sl[i]=stop; tp[i]=cl-B_RR*risk; strat[i]='bear'
    return side,entry,sl,tp,strat


GEST={'bull':dict(be=U_BE,trail=U_TRAIL,manage='trail',cd=U_COOLDOWN),
      'bear':dict(be=B_BE,trail=B_TRAIL,manage='trail',cd=B_COOLDOWN),
      'lat': dict(be=0,trail=0,manage='fixed',cd=L_COOLDOWN)}


def run(slope_thresh, bear_route, start, end):
    midx=None; arrs={}
    for sym,df in pair_data.items():
        s,e,sl,tp,st=gen(df, slope_thresh, bear_route)
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st)
        midx=d.index if midx is None else midx.union(d.index); arrs[sym]=d
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,'tp':dd['_tp'].values,'strat':dd['_st'].values}
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}
    monthly={}; cur_month=None; bear_pnl=0.0
    def close_all(k):
        nonlocal balance, bear_pnl
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): c=positions[sym]['entry']
            pos=positions[sym]; e=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
            nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r
            balance+=nr*pos['risk_usd']
            if pos['strat']=='bear': bear_pnl+=nr*pos['risk_usd']
            del positions[sym]
    for k in range(len(midx)):
        ts=midx[k]; mk=ts.strftime('%Y-%m')
        if cur_month is None: cur_month=mk
        if mk!=cur_month:                      # virou o mes -> fecha, registra, reseta
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
                if pos['strat']=='bear': bear_pnl+=nr*pos['risk_usd']
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
                            'strat':strat,'be':g['be'],'trail':g['trail'],'manage':g['manage'],'cd':g['cd']}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    monthly[cur_month]=balance-MONTHLY_BASE
    return dict(monthly=monthly, pnl=sum(monthly.values()), max_dd=max_dd, bear_pnl=bear_pnl)


ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']

def evaluate(slope_thresh, bear_route):
    allm={}; tot=0; ddmax=0; bearp=0
    for yr in YEARS:
        start=datetime(yr,1,1,tzinfo=timezone.utc); end=YEAR_END.get(yr,datetime(yr,12,31,tzinfo=timezone.utc))
        r=run(slope_thresh, bear_route, start, end)
        allm.update(r['monthly']); tot+=r['pnl']; ddmax=max(ddmax,r['max_dd']); bearp+=r['bear_pnl']
    pos=sum(1 for v in allm.values() if v>0); neg=sum(1 for v in allm.values() if v<0)
    return dict(monthly=allm, tot=tot, pos=pos, neg=neg, n=len(allm), bearp=bearp, ddmax=ddmax)


CONFIGS=[
    ('atual: thr0.001 bear13', 0.001,'bear13'),
    ('estrito: thr0.004 bear13',0.004,'bear13'),
    ('thr0.001 BEAR->breakout', 0.001,'lat'),
    ('thr0.004 BEAR->breakout', 0.004,'lat'),
    ('thr0.001 BEAR off',       0.001,'off'),
]

print('='*100)
print('  DECISOR — capital 500/mes com reset. P&L = soma dos meses (cada um sobre 500)')
print('='*100)
print(f"  {'Config':<26} | {'TOTAL':>8} | {'meses+':>10} | {'BEAR slot':>10} | {'DDmes':>7}")
print('  '+'-'*78)
res=[]
for name,thr,route in CONFIGS:
    ev=evaluate(thr,route)
    print(f"  {name:<26} | {ev['tot']:>+8.0f} | {ev['pos']:>3}/{ev['n']:<2} (+) | {ev['bearp']:>+10.0f} | {ev['ddmax']:>6.1f}%")
    res.append((name,ev))

best=max(res, key=lambda x:(x[1]['pos']/x[1]['n'], x[1]['tot']))
name,ev=best
print('\n'+'='*100)
print(f'  MELHOR (consistencia): {name}  ->  {ev["pos"]}/{ev["n"]} meses+ | total {ev["tot"]:+.0f} | DDmes {ev["ddmax"]:.1f}%')
print('='*100)
print('  P&L mensal (cada mes comeca com 500):')
yrs=sorted(set(int(mk[:4]) for mk in ev['monthly']))
print(f"  {'Ano':<6} | "+' '.join(f'{ML[m]:>5}' for m in range(12))+f" | {'Soma':>7}")
print('  '+'-'*90)
for yr in yrs:
    vals=[ev['monthly'].get(f'{yr}-{m+1:02d}') for m in range(12)]
    cells=' '.join((f'{v:>+5.0f}' if v is not None else '    .') for v in vals)
    s=sum(v for v in vals if v is not None)
    print(f"  {yr:<6} | {cells} | {s:>+7.0f}")
print(f"\n  Total {ev['tot']:+.0f} USDC em {ev['n']} meses (base 500/mes) | media {ev['tot']/ev['n']:+.1f}/mes")
print('  Custos 0.14% round-trip. Bear sem edge (ver matriz 5 anos). Bull/Lateral HIPOTETICAS.')
