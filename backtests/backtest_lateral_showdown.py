"""
backtest_lateral_showdown.py
Duelo de teses para a LATERAL, no mesmo motor de portfolio:

  TESE A — BREAKOUT do range (MOMENTUM): para de fazer fade. Detecta o range e
           SEGUE o rompimento (compra rompe resistencia / vende rompe suporte).
           Gestao de momentum: deixa correr (RR alto / trailing).
           Alinhada com a lei descoberta (cripto paga momentum).

  TESE B — REVERSAO em VALUATION extremo (Mayer): a unica veia de reversao que
           nao perdeu (PF~1). So opera em extremos profundos de Mayer (close/SMA200)
           + RSI + BB %b. Gestao mean-reversion (volta a media).

Pergunta: qual das duas tem o melhor resultado? (PF e USDC/dia)

Motor de portfolio realista (saldo 5000, correlacao). Periodo jun/2025->jun/2026.
Uso: python backtests/backtest_lateral_showdown.py
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
INITIAL_BALANCE=5000.0; RISK_PCT=1.0; PERIOD_DAYS=(END-START).days
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
COOLDOWN=4; TIMEOUT=48; BB_N=20; BB_K=2.0; RSI_P=14
BE_PCT=0.70; TRAIL_ATR=2.0   # gestao de momentum (breakout)

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


def adx(df, period=14):
    h=df['h']; l=df['l']; c=df['c']
    up=h.diff(); dn=-l.diff()
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


print('LATERAL SHOWDOWN — Breakout (momentum) vs Reversao-Mayer (valuation)')
print(f'  Periodo: {START:%d/%m/%Y} -> {END:%d/%m/%Y} ({PERIOD_DAYS} dias) | saldo {INITIAL_BALANCE:.0f} | 1H\n')
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
        o,h,l,c=df1['o'],df1['h'],df1['l'],df1['c']; ohlc4=(o+h+l+c)/4
        tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['adx']=adx(df1,14)
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        basis=ohlc4.rolling(BB_N).mean(); dev=BB_K*ohlc4.rolling(BB_N).std()
        df1['bb_mid']=basis; df1['bbr']=(ohlc4-(basis-dev))/((basis+dev)-(basis-dev))
        df1['rsi']=rsi_calc(ohlc4,RSI_P)
        df1['ema200']=c.ewm(span=200,adjust=False).mean()
        df1['mayer']=c/c.rolling(200).mean()
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()

master_index=None
for sym,df in pair_data.items():
    master_index=df.index if master_index is None else master_index.union(df.index)
master_index=master_index[(master_index>=START)&(master_index<=END)]


# ── TESE A: breakout do range ─────────────────────────────────────────────────
def gen_breakout(df, lookback, adx_max, stop_mode, rr_cap, buffer_atr, regime_filter):
    n=len(df); h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; adx_a=df['adx'].values; reg=df['regime'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    for i in range(lookback+5, n):
        if regime_filter!='ALL' and reg[i]!=regime_filter: continue
        a=atr[i]
        if np.isnan(a): continue
        rl=np.min(l[i-lookback:i]); rh=np.max(h[i-lookback:i])
        if rl<=0 or rh<=rl: continue
        # consolidacao previa (estava sem tendencia forte)
        if adx_max>0 and (np.isnan(adx_a[i-1]) or adx_a[i-1]>adx_max): continue
        buf=buffer_atr*a; cl=c[i]; pc=c[i-1]; height=rh-rl
        # LONG: rompe a resistencia (fresco: barra anterior estava dentro)
        if cl>rh+buf and pc<=rh:
            e=cl
            if stop_mode=='mid': stop=(rl+rh)/2
            elif stop_mode=='opposite': stop=rl
            else: stop=rh-1.0*a   # 'edge': stop logo abaixo da borda rompida
            risk=e-stop
            if risk>0:
                target=e+rr_cap*risk if rr_cap>0 else e+height
                if target>e: side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=target; continue
        # SHORT: rompe o suporte
        if cl<rl-buf and pc>=rl:
            e=cl
            if stop_mode=='mid': stop=(rl+rh)/2
            elif stop_mode=='opposite': stop=rh
            else: stop=rl+1.0*a
            risk=stop-e
            if risk>0:
                target=e-rr_cap*risk if rr_cap>0 else e-height
                if target<e: side[i]='SHORT'; entry[i]=e; sl[i]=stop; tp[i]=target
    return side,entry,sl,tp


# ── TESE B: reversao em valuation extremo ─────────────────────────────────────
def gen_reversao(df, bbr_lo, bbr_hi, rsi_lo, rsi_hi, mayer_buy, mayer_sell, stop_atr, regime_filter):
    n=len(df); c=df['c'].values; atr=df['atr'].values; reg=df['regime'].values
    bbr=df['bbr'].values; rsi=df['rsi'].values; mid=df['bb_mid'].values
    ema=df['ema200'].values; mayer=df['mayer'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    for i in range(210,n):
        if regime_filter!='ALL' and reg[i]!=regime_filter: continue
        a=atr[i]; cl=c[i]
        if np.isnan(a) or np.isnan(bbr[i]) or np.isnan(rsi[i]) or np.isnan(mid[i]) or np.isnan(mayer[i]): continue
        if bbr[i]<bbr_lo and rsi[i]<rsi_lo and cl<ema[i] and mayer[i]<mayer_buy:
            e=cl; stop=e-stop_atr*a; risk=e-stop
            if risk>0 and mid[i]>e: side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=mid[i]; continue
        if bbr[i]>bbr_hi and rsi[i]>rsi_hi and cl>ema[i] and mayer[i]>mayer_sell:
            e=cl; stop=e+stop_atr*a; risk=stop-e
            if risk>0 and mid[i]<e: side[i]='SHORT'; entry[i]=e; sl[i]=stop; tp[i]=mid[i]
    return side,entry,sl,tp


def build_arrays(gen_fn, **kw):
    A={}
    for sym,df in pair_data.items():
        s,e,sl,tp=gen_fn(df, **kw)
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp).reindex(master_index)
        A[sym]={'h':d['h'].values,'l':d['l'].values,'c':d['c'].values,'atr':d['atr'].values,
                'side':d['_s'].values,'entry':d['_e'].values,'sl':d['_sl'].values,'tp':d['_tp'].values}
    return A


def run_portfolio(A, manage='fixed', max_concurrent=None):
    """manage='fixed' (TP/SL fixos) ou 'trail' (BE+trailing, p/ momentum)."""
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]; max_conc=0
    for k in range(len(master_index)):
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]; atr=d['atr'][k]
            risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']; a=atr if not np.isnan(atr) else risk
            closed=False; nr=0.0
            if pos['side']=='LONG':
                if lo<=pos['cur']: nr=(pos['cur']-e)/risk-fee_r; closed=True
                elif hi>=pos['tp']: nr=(pos['tp']-e)/risk-fee_r; closed=True
                elif manage=='trail':
                    if not pos['be'] and (c-e)/risk>=BE_PCT*((pos['tp']-e)/risk): pos['cur']=e; pos['be']=True
                    if pos['be']:
                        cand=c-TRAIL_ATR*a
                        if cand>pos['cur']: pos['cur']=cand
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
                elif manage=='trail':
                    if not pos['be'] and (e-c)/risk>=BE_PCT*((e-pos['tp'])/risk): pos['cur']=e; pos['be']=True
                    if pos['be']:
                        cand=c+TRAIL_ATR*a
                        if cand<pos['cur']: pos['cur']=cand
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
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'be':False,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px}
        max_conc=max(max_conc,len(positions))
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return {'balance':balance,'trades':trades,'max_dd':max_dd,'max_conc':max_conc}


def show(label, res, base=False):
    t=res['trades']; bal=res['balance']; mark='*' if base else ' '
    if not t: print(f'  {label:<34}{mark}: sem trades'); return res
    w=[x for x in t if x['nr']>0]; gw=sum(x['nr'] for x in w); gl=abs(sum(x['nr'] for x in t if x['nr']<=0))
    pnl=bal-INITIAL_BALANCE; lng=[x for x in t if x['side']=='LONG']; sht=[x for x in t if x['side']=='SHORT']
    print(f'  {label:<34}{mark}: {pnl:>+8.0f} USDC | {pnl/PERIOD_DAYS:>+6.1f}/dia | {len(t):>4}t | '
          f'WR {len(w)/len(t)*100:>4.1f}% | PF {gw/gl if gl>0 else 99:>4.2f} | DD {res["max_dd"]:>4.1f}% | L{len(lng)}/S{len(sht)}')
    return res


best={}

# ════════════════ TESE A — BREAKOUT (momentum, gestao trailing) ════════════════
print('='*122)
print('  TESE A — BREAKOUT do range (momentum)')
print('='*122)
BA=dict(lookback=30, adx_max=25, stop_mode='mid', rr_cap=0, buffer_atr=0.1, regime_filter='NEUTRAL')
resA=[]
def runA(lbl, manage='trail', **over):
    r=show(lbl, run_portfolio(build_arrays(gen_breakout, **{**BA,**over}), manage=manage)); resA.append(r); return r
runA('BASE (proj. altura, trail)', manage='trail')
print('\n  lookback:')
for lb in [20,30,50]: runA(f'lookback={lb}', lookback=lb)
print('\n  adx_max (consolidacao previa):')
for ax in [0,20,25]: runA(f'adx_max={ax if ax else "off"}', adx_max=ax)
print('\n  stop_mode:')
for sm in ['mid','opposite','edge']: runA(f'stop={sm}', stop_mode=sm)
print('\n  alvo (rr_cap=0 -> projecao da altura | senao RR fixo):')
for rr in [0,1.5,2.0,3.0]: runA(f'rr_cap={rr if rr else "altura"}', rr_cap=rr)
print('\n  regime:')
for rf in ['NEUTRAL','ALL']: runA(f'regime={rf}', regime_filter=rf)
print('\n  gestao (trail=deixa correr | fixed=TP fixo):')
for mg in ['trail','fixed']: runA(f'manage={mg}', manage=mg)
bA=max(resA, key=lambda r:r['balance']); best['BREAKOUT']=bA

# ════════════════ TESE B — REVERSAO valuation (Mayer, mean-rev) ════════════════
print('\n'+'='*122)
print('  TESE B — REVERSAO em valuation extremo (Mayer)')
print('='*122)
BB_=dict(bbr_lo=0.0,bbr_hi=1.0,rsi_lo=30,rsi_hi=70,mayer_buy=0.9,mayer_sell=1.1,stop_atr=2.0,regime_filter='ALL')
resB=[]
def runB(lbl, **over):
    r=show(lbl, run_portfolio(build_arrays(gen_reversao, **{**BB_,**over}), manage='fixed')); resB.append(r); return r
runB('BASE (mayer 0.9/1.1)', )
print('\n  aperto do Mayer (mais extremo = mais seletivo):')
for mb,ms in [(0.95,1.05),(0.9,1.1),(0.85,1.15),(0.8,1.2)]: runB(f'mayer {mb}/{ms}', mayer_buy=mb, mayer_sell=ms)
print('\n  RSI:')
for rl,rh in [(30,70),(25,75),(20,80)]: runB(f'rsi {rl}/{rh}', rsi_lo=rl, rsi_hi=rh)
print('\n  stop_atr:')
for sa in [1.5,2.0,2.5]: runB(f'stop_atr={sa}', stop_atr=sa)
print('\n  regime:')
for rf in ['ALL','NEUTRAL','BEAR']: runB(f'regime={rf}', regime_filter=rf)
bB=max(resB, key=lambda r:r['balance']); best['REVERSAO']=bB

# ════════════════ VEREDICTO ════════════════
print('\n'+'='*122)
print('  VEREDICTO — melhor de cada tese')
print('='*122)
for nome,r in best.items():
    t=r['trades']; w=[x for x in t if x['nr']>0]
    gw=sum(x['nr'] for x in w); gl=abs(sum(x['nr'] for x in t if x['nr']<=0))
    pnl=r['balance']-INITIAL_BALANCE
    print(f'  {nome:<10}: {pnl:>+8.0f} USDC | {pnl/PERIOD_DAYS:>+6.1f}/dia | {len(t)}t | '
          f'WR {len(w)/len(t)*100:.1f}% | PF {gw/gl if gl>0 else 99:.2f} | DD {r["max_dd"]:.1f}%')
vencedor=max(best.items(), key=lambda x:x[1]['balance'])
print(f'\n  >>> MELHOR: {vencedor[0]} <<<')
print('='*122)
print('  Custos 0.14% round-trip. Risco 1%/trade. Lateral HIPOTETICA.')
