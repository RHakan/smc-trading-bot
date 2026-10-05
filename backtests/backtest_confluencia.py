"""
backtest_confluencia.py
Estrategia de REVERSAO por CONFLUENCIA — baseada no script TradingView do Rafa.

Script original (RSI + EMA200 + Bollinger %b):
  buyConditionDaily  = close < ema200 and rsi < 30 and bbr < 0
  sellConditionDaily = rsi > 70 and bbr > 1 and close > ema200 and mayer > 1.3
  onde bbr (%b) = (src - lowerBB)/(upperBB - lowerBB)
       bbr < 0 -> preco ABAIXO da banda inteira ; bbr > 1 -> ACIMA

A diferenca face ao mean-reversion que falhou (PF 0.93): aquele entrava em
QUALQUER toque com UMA condicao. Este exige VARIOS extremos alinhados —
RSI extremo + BB %b extremo + (opc) contexto de tendencia + (opc) Mayer.
A confluencia E o filtro de ruido que o Rafa pediu.

Objetivo deste passo: confirmar se a CONFLUENCIA cria EDGE (PF>1) no 1H, onde o
mean-reversion simples nao criou. Se sim, proximo passo e descer para 15m (onde
ha frequencia) — a confluencia limpa o ruido do TF baixo.

Motor de portfolio realista (saldo 5000, correlacao). Periodo jun/2025->jun/2026.
Uso: python backtests/backtest_confluencia.py
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

START=datetime(2025,6,1,tzinfo=timezone.utc); END=datetime(2026,6,29,tzinfo=timezone.utc)
INITIAL_BALANCE=5000.0; RISK_PCT=1.0; PERIOD_DAYS=(END-START).days; TARGET_PER_DAY=40.0
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
COOLDOWN=4; TIMEOUT=48
BB_N=20; BB_K=2.0; RSI_P=14

CACHE_DIR=Path(__file__).parent/'cache'; CACHE_DIR.mkdir(exist_ok=True)
ex=ccxt.binanceusdm({'enableRateLimit':True})


def fetch(sym, tf, extra_days):
    safe=sym.replace('/','_').replace(':','_'); cache=CACHE_DIR/f'{safe}_{tf}_jun25_jun26.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True); return df
    since=int((START-timedelta(days=extra_days)).timestamp()*1000); end_ms=int(END.timestamp()*1000); rows=[]
    while True:
        b=ex.fetch_ohlcv(sym,tf,since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=df.set_index('ts').sort_index()[lambda d:d.index<=END]; df.to_csv(cache); return df


def rsi_calc(series, period=14):
    delta=series.diff()
    up=delta.clip(lower=0).ewm(com=period-1,adjust=False).mean()
    dn=(-delta.clip(upper=0)).ewm(com=period-1,adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)


print('Estrategia REVERSAO por CONFLUENCIA (RSI + BB %b + EMA + Mayer) — script do Rafa')
print(f'  Periodo: {START:%d/%m/%Y} -> {END:%d/%m/%Y} ({PERIOD_DAYS} dias) | saldo {INITIAL_BALANCE:.0f} | 1H')
print('  Objetivo: confirmar EDGE (PF>1) com confluencia, antes de descer p/ 15m\n')
print('A carregar dados (cache)...')

pair_data={}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd=fetch(sym,'1d',40).copy(); df1=fetch(sym,'1h',25).copy()
        if len(df1)<300: print('SEM DADOS'); continue
        ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
        slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
        dfd['regime']='NEUTRAL'
        dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
        dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
        o,h,l,c=df1['o'],df1['h'],df1['l'],df1['c']
        ohlc4=(o+h+l+c)/4
        tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        basis=ohlc4.rolling(BB_N).mean(); dev=BB_K*ohlc4.rolling(BB_N).std()
        df1['bb_mid']=basis; df1['bb_up']=basis+dev; df1['bb_lo']=basis-dev
        df1['bbr']=(ohlc4-(basis-dev))/((basis+dev)-(basis-dev))
        df1['rsi']=rsi_calc(ohlc4, RSI_P)
        for p in (100,200): df1[f'ema{p}']=c.ewm(span=p,adjust=False).mean()
        df1['mayer']=c/c.rolling(200).mean()   # Mayer Multiple no TF
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()

master_index=None
for sym,df in pair_data.items():
    master_index=df.index if master_index is None else master_index.union(df.index)
master_index=master_index[(master_index>=START)&(master_index<=END)]


def gen_signals(df, bbr_lo, bbr_hi, rsi_lo, rsi_hi, ctx_ema, tp_mode, stop_atr, rr_cap,
                mayer_buy, mayer_sell, regime_filter):
    n=len(df)
    c=df['c'].values; atr=df['atr'].values; reg=df['regime'].values
    bbr=df['bbr'].values; rsi=df['rsi'].values
    mid=df['bb_mid'].values; up=df['bb_up'].values; lo=df['bb_lo'].values
    ema=df[f'ema{ctx_ema}'].values if ctx_ema else None
    mayer=df['mayer'].values
    side=np.array([None]*n,dtype=object)
    entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    for i in range(210, n):
        if regime_filter!='ALL' and reg[i]!=regime_filter: continue
        a=atr[i]; cl=c[i]
        if np.isnan(a) or np.isnan(bbr[i]) or np.isnan(rsi[i]) or np.isnan(mid[i]): continue
        # LONG — extremo inferior
        long_ok = bbr[i] < bbr_lo and rsi[i] < rsi_lo
        if ema is not None: long_ok = long_ok and not np.isnan(ema[i]) and cl < ema[i]
        if mayer_buy>0: long_ok = long_ok and not np.isnan(mayer[i]) and mayer[i] < mayer_buy
        if long_ok:
            e=cl; stop=e-stop_atr*a; risk=e-stop
            if risk>0:
                target = mid[i] if tp_mode=='mid' else up[i]
                if rr_cap>0: target=min(target, e+rr_cap*risk)
                if target>e:
                    side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=target
                    continue
        # SHORT — extremo superior
        short_ok = bbr[i] > bbr_hi and rsi[i] > rsi_hi
        if ema is not None: short_ok = short_ok and not np.isnan(ema[i]) and cl > ema[i]
        if mayer_sell>0: short_ok = short_ok and not np.isnan(mayer[i]) and mayer[i] > mayer_sell
        if short_ok:
            e=cl; stop=e+stop_atr*a; risk=stop-e
            if risk>0:
                target = mid[i] if tp_mode=='mid' else lo[i]
                if rr_cap>0: target=max(target, e-rr_cap*risk)
                if target<e:
                    side[i]='SHORT'; entry[i]=e; sl[i]=stop; tp[i]=target
    return side, entry, sl, tp


def build_arrays(**kw):
    A={}
    for sym,df in pair_data.items():
        s,e,sl,tp=gen_signals(df, **kw)
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp).reindex(master_index)
        A[sym]={'h':d['h'].values,'l':d['l'].values,'c':d['c'].values,
                'side':d['_s'].values,'entry':d['_e'].values,'sl':d['_sl'].values,'tp':d['_tp'].values}
    return A


def run_portfolio(A, max_concurrent=None):
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]; max_conc=0
    for k in range(len(master_index)):
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]
            risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']
            closed=False; nr=0.0
            if pos['side']=='LONG':
                if lo<=pos['sl']: nr=-1-fee_r; closed=True
                elif hi>=pos['tp']: nr=(pos['tp']-e)/risk-fee_r; closed=True
            else:
                if hi>=pos['sl']: nr=-1-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
            if not closed:
                pos['age']+=1
                if pos['age']>=TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']
                trades.append({'ts':master_index[k],'nr':nr,'side':pos['side']})
                del positions[sym]; cooldown_until[sym]=k+COOLDOWN
        for sym in A:
            if sym in positions: continue
            if max_concurrent is not None and len(positions)>=max_concurrent: break
            if k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            positions[sym]={'side':side,'entry':e,'sl':stop,'tp':tp,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px}
        max_conc=max(max_conc,len(positions))
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return {'balance':balance,'trades':trades,'max_dd':max_dd,'max_conc':max_conc}


def show(label, res, base=False):
    t=res['trades']; bal=res['balance']; mark='*' if base else ' '
    if not t: print(f'  {label:<32}{mark}: sem trades'); return res
    w=[x for x in t if x['nr']>0]; gw=sum(x['nr'] for x in w); gl=abs(sum(x['nr'] for x in t if x['nr']<=0))
    pnl=bal-INITIAL_BALANCE; lng=[x for x in t if x['side']=='LONG']; sht=[x for x in t if x['side']=='SHORT']
    print(f'  {label:<32}{mark}: {pnl:>+8.0f} USDC | {pnl/PERIOD_DAYS:>+6.1f}/dia | {len(t):>4}t | '
          f'WR {len(w)/len(t)*100:>4.1f}% | PF {gw/gl if gl>0 else 99:>4.2f} | DD {res["max_dd"]:>4.1f}% | L{len(lng)}/S{len(sht)}')
    return res


# Base = a logica do Rafa (bbr<0 / >1, rsi 30/70, contexto ema200), em qualquer regime
BASE=dict(bbr_lo=0.0, bbr_hi=1.0, rsi_lo=30, rsi_hi=70, ctx_ema=200,
          tp_mode='mid', stop_atr=1.5, rr_cap=0, mayer_buy=0, mayer_sell=0, regime_filter='ALL')

print('='*120)
print('  REVERSAO POR CONFLUENCIA — script do Rafa no motor de portfolio (1H)')
print('='*120)
show('BASE (logica do Rafa)', run_portfolio(build_arrays(**BASE)), base=True)

print('\nSweep 1 — extremo do %b (bbr): mais negativo = mais extremo/seletivo:')
for blo,bhi in [(0.05,0.95),(0.0,1.0),(-0.05,1.05),(-0.1,1.1)]:
    show(f'bbr {blo}/{bhi}', run_portfolio(build_arrays(**{**BASE,'bbr_lo':blo,'bbr_hi':bhi})))

print('\nSweep 2 — extremo do RSI:')
for rlo,rhi in [(35,65),(30,70),(25,75),(20,80)]:
    show(f'rsi {rlo}/{rhi}', run_portfolio(build_arrays(**{**BASE,'rsi_lo':rlo,'rsi_hi':rhi})))

print('\nSweep 3 — contexto de tendencia (filtro EMA):')
for ce in [0,100,200]:
    show(f'ctx_ema={ce if ce else "off"}', run_portfolio(build_arrays(**{**BASE,'ctx_ema':ce})))

print('\nSweep 4 — alvo (mid=media | band=banda oposta):')
for tm in ['mid','band']:
    show(f'tp_mode={tm}', run_portfolio(build_arrays(**{**BASE,'tp_mode':tm})))

print('\nSweep 5 — stop (xATR):')
for sa in [1.0,1.5,2.0,2.5]:
    show(f'stop_atr={sa}', run_portfolio(build_arrays(**{**BASE,'stop_atr':sa})))

print('\nSweep 6 — regime (ALL vs so NEUTRAL vs alinhar com tendencia):')
for rf in ['ALL','NEUTRAL','BULL','BEAR']:
    show(f'regime={rf}', run_portfolio(build_arrays(**{**BASE,'regime_filter':rf})))

print('\nSweep 7 — Mayer (valuation): so compra barato / vende caro:')
for mb,ms in [(0,0),(0.95,1.05),(0.9,1.1)]:
    show(f'mayer {mb}/{ms}', run_portfolio(build_arrays(**{**BASE,'mayer_buy':mb,'mayer_sell':ms})))

print('\nCombinacoes (confluencia forte):')
combos=[
    dict(bbr_lo=0.0,  bbr_hi=1.0,  rsi_lo=30, rsi_hi=70, ctx_ema=200, tp_mode='mid',  stop_atr=2.0, rr_cap=0, mayer_buy=0,    mayer_sell=0,    regime_filter='ALL'),
    dict(bbr_lo=-0.05,bbr_hi=1.05, rsi_lo=25, rsi_hi=75, ctx_ema=200, tp_mode='mid',  stop_atr=2.0, rr_cap=0, mayer_buy=0,    mayer_sell=0,    regime_filter='ALL'),
    dict(bbr_lo=0.0,  bbr_hi=1.0,  rsi_lo=30, rsi_hi=70, ctx_ema=0,   tp_mode='mid',  stop_atr=2.0, rr_cap=0, mayer_buy=0,    mayer_sell=0,    regime_filter='NEUTRAL'),
    dict(bbr_lo=-0.05,bbr_hi=1.05, rsi_lo=25, rsi_hi=75, ctx_ema=200, tp_mode='band', stop_atr=2.0, rr_cap=3, mayer_buy=0,    mayer_sell=0,    regime_filter='ALL'),
    dict(bbr_lo=0.0,  bbr_hi=1.0,  rsi_lo=30, rsi_hi=70, ctx_ema=200, tp_mode='mid',  stop_atr=2.0, rr_cap=0, mayer_buy=0.95, mayer_sell=1.05, regime_filter='ALL'),
]
res=[]
for p in combos:
    lbl=f"bbr{p['bbr_lo']}/{p['bbr_hi']} rsi{p['rsi_lo']}/{p['rsi_hi']} {p['regime_filter'][:4]}"
    res.append((p, show(lbl, run_portfolio(build_arrays(**p)))))

best_p,best_r=max(res, key=lambda x:x[1]['balance'])
pnl=best_r['balance']-INITIAL_BALANCE
print('\n'+'='*120)
print(f'  MELHOR: {best_p}')
print(f'  -> {pnl:+.0f} USDC | {pnl/PERIOD_DAYS:+.1f}/dia | DD {best_r["max_dd"]:.1f}%')
print('='*120)
print('  REGRA: so vale descer p/ 15m se a confluencia mostrar EDGE (PF>1) aqui.')
print('  Se PF>1 mas resultado baixo = falta frequencia -> 15m resolve. Se PF<1 = sinal sem edge.')
print('  Custos 0.14% round-trip. Baseado no script TradingView do Rafa.')
