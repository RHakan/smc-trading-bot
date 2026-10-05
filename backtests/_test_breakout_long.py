import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

START = datetime(2026,1,1,tzinfo=timezone.utc)
END   = datetime(2026,6,24,tzinfo=timezone.utc)
ex = ccxt.binanceusdm({'enableRateLimit': True})

def fetch(sym, tf, extra):
    since = int((START - timedelta(days=extra)).timestamp()*1000)
    end_ms = int(END.timestamp()*1000)
    rows=[]
    while True:
        b=ex.fetch_ohlcv(sym,tf,since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms or len(b)==0: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    return df.set_index('ts').sort_index()[lambda d: d.index < END]

PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','BNB/USDT:USDT',
       'ADA/USDT:USDT','AVAX/USDT:USDT','DOGE/USDT:USDT','DOT/USDT:USDT',
       'XRP/USDT:USDT','LINK/USDT:USDT']

print('Carregando dados...')
data = {}
for sym in PAIRS:
    dfd = fetch(sym,'1d',30).copy()
    dfd['ema']=dfd['c'].ewm(span=20,adjust=False).mean()
    dfd['slope']=(dfd['ema']-dfd['ema'].shift(5))/dfd['ema'].shift(5)
    dfd['regime']='NEUTRAL'
    dfd.loc[(dfd['c']>dfd['ema'])&(dfd['slope']>0),'regime']='BULL'
    dfd.loc[(dfd['c']<dfd['ema'])&(dfd['slope']<0),'regime']='BEAR'
    df1=fetch(sym,'1h',5).copy()
    o4=(df1['o']+df1['h']+df1['l']+df1['c'])/4
    tr=pd.concat([(df1['h']-df1['l']),(df1['h']-df1['c'].shift(1)).abs(),(df1['l']-df1['c'].shift(1)).abs()],axis=1).max(axis=1)
    df1['atr']=tr.ewm(com=13,adjust=False).mean()
    df1['atr_avg48']=df1['atr'].rolling(48).mean()
    d=o4.diff(); g=d.where(d>0,0.).ewm(com=13,adjust=False).mean(); lv=(-d.where(d<0,0.)).ewm(com=13,adjust=False).mean()
    df1['rsi']=100-(100/(1+g/lv.replace(0,np.nan)))
    df1['low4']=df1['l'].shift(1).rolling(4).min()
    df1['high4']=df1['h'].shift(1).rolling(4).max()
    df1['high12']=df1['h'].shift(1).rolling(12).max()
    df1['high24']=df1['h'].shift(1).rolling(24).max()
    reg_s=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
    df1['regime']=reg_s
    data[sym]=df1
    time.sleep(0.05)
print('OK\n')

FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2; COOLDOWN=8; ATR_STOP=1.0; RR=3.0
mlabels={'2026-01':'Jan','2026-02':'Fev','2026-03':'Mar',
         '2026-04':'Abr','2026-05':'Mai','2026-06':'Jun'}

def sim_long(df, ib, cl, sl):
    risk=cl-sl; tp=cl+RR*risk; fee_r=cl*RT/risk
    cur=sl; be=False
    for j in range(ib+1, min(ib+200, len(df))):
        bar=df.iloc[j]; lo=float(bar['l']); hi=float(bar['h']); c=float(bar['c'])
        atr=float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if lo<=cur: return (cur-cl)/risk - fee_r
        if hi>=tp: return RR - fee_r
        if not be and (c-cl)/risk>=0.8: cur=cl; be=True
        if be:
            cand=c-1.5*atr
            if cand>cur: cur=cand
    return (float(df.iloc[min(ib+199,len(df)-1)]['c'])-cl)/risk - fee_r

def sim_short(df, ib, cl, sl):
    risk=sl-cl; tp=cl-RR*risk; fee_r=cl*RT/risk
    cur=sl; be=False
    for j in range(ib+1, min(ib+200, len(df))):
        bar=df.iloc[j]; lo=float(bar['l']); hi=float(bar['h']); c=float(bar['c'])
        atr=float(bar['atr']) if not np.isnan(bar['atr']) else risk
        if hi>=cur: return (cl-cur)/risk - fee_r
        if lo<=tp: return RR - fee_r
        if not be and (cl-c)/risk>=0.8: cur=cl; be=True
        if be:
            cand=c+1.5*atr
            if cand<cur: cur=cand
    return (cl-float(df.iloc[min(ib+199,len(df)-1)]['c']))/risk - fee_r

# LONG breakout configs
configs_long = [
    ('high4',  False, 'BULL breakout 4H-high (sem ATR)'),
    ('high4',  True,  'BULL breakout 4H-high + ATR>avg'),
    ('high12', True,  'BULL breakout 12H-high + ATR>avg'),
    ('high24', True,  'BULL breakout 24H-high + ATR>avg'),
]

print('LONGS em BULL regime - breakout momentum')
print('-'*72)
for (high_col, use_atr, label) in configs_long:
    all_trades=[]
    for sym, df1 in data.items():
        last_bar=-9999
        start_i=max(55, next((i for i,ts in enumerate(df1.index) if ts>=START), 55))
        for i in range(start_i, len(df1)-1):
            if i-last_bar<COOLDOWN: continue
            ts_bar=df1.index[i]
            if ts_bar<START: continue
            r=df1.iloc[i]; regime=str(r['regime'])
            if regime!='BULL': continue
            cl=float(r['c']); op=float(r['o']); atr=float(r['atr'])
            atr_avg=float(r['atr_avg48']); hi_n=float(r[high_col])
            rsi=float(r['rsi'])
            if any(np.isnan(v) for v in [cl,atr,atr_avg,hi_n,rsi]): continue
            if use_atr and atr < atr_avg: continue
            if rsi > 75: continue
            if cl > hi_n and cl > op:
                sl=cl-ATR_STOP*atr
                if sl>=cl: continue
                nr=sim_long(df1, i, cl, sl)
                all_trades.append({'ts':ts_bar,'nr':nr})
                last_bar=i
    if not all_trades:
        print(label + ' - 0 trades'); continue
    nr_all=[t['nr'] for t in all_trades]
    wr=sum(1 for r in nr_all if r>0)/len(nr_all)*100
    avgr=sum(nr_all)/len(nr_all)
    gw=sum(r for r in nr_all if r>0); gl=abs(sum(r for r in nr_all if r<0))
    pf=gw/gl if gl>0 else 99
    print(label+': '+str(len(nr_all))+' t | WR '+str(round(wr,1))+'% | avg '+str(round(avgr,3))+'R | PF '+str(round(pf,2)))

print()
print('COMBINADO: BEAR SHORT breakout (ATR>avg) + BULL LONG breakout 4H (ATR>avg)')
print('='*72)
all_trades_comb=[]
for sym, df1 in data.items():
    last_bar=-9999
    start_i=max(55, next((i for i,ts in enumerate(df1.index) if ts>=START), 55))
    for i in range(start_i, len(df1)-1):
        if i-last_bar<COOLDOWN: continue
        ts_bar=df1.index[i]
        if ts_bar<START: continue
        r=df1.iloc[i]; regime=str(r['regime'])
        cl=float(r['c']); op=float(r['o']); atr=float(r['atr'])
        atr_avg=float(r['atr_avg48']); rsi=float(r['rsi'])
        l4=float(r['low4']); h4=float(r['high4'])
        if any(np.isnan(v) for v in [cl,atr,atr_avg,l4,h4,rsi]): continue
        if atr < atr_avg: continue
        sig=None
        if regime=='BEAR' and cl < l4 and cl < op:
            sl=cl+ATR_STOP*atr
            if sl>cl: sig=('short',cl,sl)
        elif regime=='BULL' and cl > h4 and cl > op and rsi < 75:
            sl=cl-ATR_STOP*atr
            if sl<cl: sig=('long',cl,sl)
        if not sig: continue
        side,entry,sl_price=sig
        nr=sim_short(df1,i,entry,sl_price) if side=='short' else sim_long(df1,i,entry,sl_price)
        all_trades_comb.append({'ts':ts_bar,'side':side,'nr':nr})
        last_bar=i

all_trades_comb.sort(key=lambda t: t['ts'])
capital=100.0; months={}
for t in all_trades_comb:
    mk=t['ts'].strftime('%Y-%m')
    pnl=t['nr']*capital*0.01; capital+=pnl
    if mk not in months:
        months[mk]={'cnt':0,'wins':0,'nr_sum':0.0,'longs':0,'shorts':0,'start':capital-pnl}
    months[mk]['end_cap']=capital; months[mk]['cnt']+=1
    if t['nr']>0: months[mk]['wins']+=1
    months[mk]['nr_sum']+=t['nr']
    if t['side']=='long': months[mk]['longs']+=1
    else: months[mk]['shorts']+=1

tw=0; tl=0
for mk in sorted(months.keys()):
    m=months[mk]; lbl=mlabels.get(mk,mk)
    pct=(m['end_cap']/m['start']-1)*100
    wr=m['wins']/m['cnt']*100; avgr=m['nr_sum']/m['cnt']
    tw+=m['wins']; tl+=m['cnt']-m['wins']
    mark='+' if pct>0 else ''
    print(lbl+': '+str(m['cnt'])+' t (L:'+str(m['longs'])+' S:'+str(m['shorts'])+') WR '+str(round(wr,1))+'% avg '+str(round(avgr,3))+'R '+mark+str(round(pct,1))+'% cap $'+str(round(m['end_cap'],2)))

nr_all=[t['nr'] for t in all_trades_comb]
gw=sum(r for r in nr_all if r>0); gl=abs(sum(r for r in nr_all if r<0))
pf=gw/gl if gl>0 else 99
print('TOTAL: '+str(len(nr_all))+' t | WR '+str(round(tw/(tw+tl)*100,1))+'% | avg '+str(round(sum(nr_all)/len(nr_all),3))+'R | PF '+str(round(pf,2)))
print('$100 -> $'+str(round(capital,2))+' ('+str(round((capital/100-1)*100,1))+'%)')
