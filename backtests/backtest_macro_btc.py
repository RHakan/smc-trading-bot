"""
backtest_macro_btc.py
Filtro MACRO pelo BTC — ataca a correlacao dos repiques amplos.

O RSI dinamico (ideia do Rafa) maximizou o retorno (+197%) mas NAO reduziu o DD
(~31% em tudo) nem matou os piores meses (Mar/26 bear -1727). Causa: os piores
meses sao repiques de mercado AMPLOS — quando o BTC vira, todos os alts viram
juntos. O RSI de cada alt isolado nao ve a reversao correlacionada.

Solucao: usar o BTC como TERMOMETRO. Quando o BTC da sinal de repique, NAO abrir
novos shorts em NENHUM par (corta a correlacao na raiz).

Gatilhos macro testados (btc_guard):
  rsi_low   — BTC sobrevendido (RSI BTC < X): mercado esticado, repique provavel
  reversal  — BTC RSI virou para cima (cruzou acima de X vindo de baixo): reversao iniciada

Base: sistema + qualidade (min0.2, side3) + RSI dinamico no bear (W200, k2.0).
So afeta SHORTS (bear/lat short). Longs intactos. Inclui levantamento MENSAL.
Periodo jun/2025->jun/2026. Uso: python backtests/backtest_macro_btc.py
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
MIN_SCORE=0.2; MAX_PER_SIDE=3
BEAR_RSI_W=200; BEAR_RSI_K=2.0   # RSI dinamico do bear (melhor do teste anterior)

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


print('FILTRO MACRO pelo BTC — corta shorts quando o lider sinaliza repique')
print(f'  Periodo: {START:%d/%m/%Y} -> {END:%d/%m/%Y} | base: qualidade + RSI dinamico bear (W{BEAR_RSI_W} k{BEAR_RSI_K})\n')
print('A carregar dados (cache)...')
pair_data={}; btc_bull=None; btc_rsi_s=None
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
        dfd['slope_d']=slope
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['atr_avg']=df1['atr'].rolling(B_ATR_AVG).mean()
        df1['adx']=adx(df1,14)
        rsi=rsi_calc(c,14); df1['rsi']=rsi
        df1[f'rsi_ma{BEAR_RSI_W}']=rsi.rolling(BEAR_RSI_W).mean()
        df1[f'rsi_sd{BEAR_RSI_W}']=rsi.rolling(BEAR_RSI_W).std()
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['slope_d']=dfd['slope_d'].shift(1).reindex(df1.index,method='ffill')
        df1['low_n']=df1['l'].shift(1).rolling(B_LOOKBACK).min()
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        pair_data[sym]=df1
        if sym.startswith('BTC'):
            btc_bull=(df1['regime']=='BULL'); btc_rsi_s=rsi
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()

master_index=None
for sym,df in pair_data.items():
    master_index=df.index if master_index is None else master_index.union(df.index)
master_index=master_index[(master_index>=START)&(master_index<=END)]
btc_rsi_arr=btc_rsi_s.reindex(master_index).values   # RSI do BTC alinhado ao indice mestre


def gen_all(df, btc_local):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    reg=df['regime'].values; slope=df['slope_d'].values; low_n=df['low_n'].values; ema_t=df['ema_trend'].values
    rsi=df['rsi'].values; rsi_ma=df[f'rsi_ma{BEAR_RSI_W}'].values; rsi_sd=df[f'rsi_sd{BEAR_RSI_W}'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object); score=np.full(n,np.nan)
    start_i=max(B_ATR_AVG+B_LOOKBACK+5, U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, L_LOOKBACK+5, 210)
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


GEST={'bear':dict(be=B_BE,trail=B_TRAIL,manage='trail',cd=B_COOLDOWN),
      'bull':dict(be=U_BE,trail=U_TRAIL,manage='trail',cd=U_COOLDOWN),
      'lat': dict(be=0,trail=0,manage='fixed',cd=L_COOLDOWN)}

# pre-computa sinais 1x (nao dependem do guard, que e aplicado na entrada)
A={}
for sym,df in pair_data.items():
    btc_local=btc_bull.reindex(df.index).fillna(False).values
    s,e,sl,tp,st,sc=gen_all(df, btc_local)
    d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st,_sc=sc).reindex(master_index)
    A[sym]={'h':d['h'].values,'l':d['l'].values,'c':d['c'].values,'atr':d['atr'].values,
            'side':d['_s'].values,'entry':d['_e'].values,'sl':d['_sl'].values,
            'tp':d['_tp'].values,'strat':d['_st'].values,'score':d['_sc'].values}

# guard macro pre-computado por barra (depende so do BTC)
def make_guard(mode, lo, hi):
    """Devolve array bool: True = BLOQUEAR novos shorts nesta barra."""
    g=np.zeros(len(master_index), dtype=bool)
    if mode=='off': return g
    prev=np.nan
    for k in range(len(master_index)):
        rb=btc_rsi_arr[k]
        if np.isnan(rb): prev=rb; continue
        if mode=='rsi_low':
            g[k]= rb < lo
        elif mode=='reversal':
            # BTC RSI cruzou acima de hi vindo de baixo de lo recentemente => reversao
            g[k]= (not np.isnan(prev)) and prev < hi and rb >= hi and rb < lo+25
        prev=rb
    # 'reversal' bloqueia por algumas barras apos o gatilho
    if mode=='reversal':
        block=np.zeros_like(g); hold=0
        for k in range(len(g)):
            if g[k]: hold=12
            block[k]= hold>0; hold=max(0,hold-1)
        return block
    return g


def run(guard):
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]
    for k in range(len(master_index)):
        ts=master_index[k]
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
        block_shorts=guard[k]
        cands=[]
        for sym in A:
            if sym in positions: continue
            if k<=cooldown_until[sym]: continue
            d=A[sym]; sidev=d['side'][k]
            if sidev is None or (isinstance(sidev,float) and np.isnan(sidev)): continue
            if block_shorts and sidev=='SHORT': continue   # GUARD: corta shorts no repique macro
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


ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
def monthly(trades):
    m={}
    for x in trades: m.setdefault(x['ts'].strftime('%Y-%m'),0.0); m[x['ts'].strftime('%Y-%m')]+=x['pnl']
    return m
def line(label, res, base=False):
    t=res['trades']; bal=res['balance']; mark='*' if base else ' '
    if not t: print(f'  {label:<26}{mark}: sem trades'); return res
    m=monthly(t); negs=[v for v in m.values() if v<0]; pnl=bal-INITIAL_BALANCE
    print(f'  {label:<26}{mark}: {pnl/INITIAL_BALANCE*100:>+6.1f}% ({pnl:>+6.0f}) | {len(t):>4}t | '
          f'meses- {len(negs):>2} | dor {sum(negs):>+7.0f} | DD {res["max_dd"]:>4.1f}%')
    return res


print('='*100)
print('  GUARD MACRO BTC (bloqueia novos SHORTS quando o BTC sinaliza repique)')
print('='*100)
base=line('SEM guard', run(make_guard('off',0,0)), base=True)

print('\nGuard = BTC sobrevendido (RSI BTC < limiar):')
for lo in [25,30,35,40]:
    line(f'btc_rsi<{lo}', run(make_guard('rsi_low',lo,0)))

print('\nGuard = BTC reversao (RSI BTC cruza acima de hi, vindo de baixo):')
for lo,hi in [(35,40),(40,45),(45,50)]:
    line(f'reversal {lo}->{hi}', run(make_guard('reversal',lo,hi)))

# melhor por (total + dor)
configs=[('rsi_low',25,0),('rsi_low',30,0),('rsi_low',35,0),('rsi_low',40,0),
         ('reversal',35,40),('reversal',40,45),('reversal',45,50)]
results=[]
for mode,lo,hi in configs:
    r=run(make_guard(mode,lo,hi)); m=monthly(r['trades']); dor=sum(v for v in m.values() if v<0)
    results.append((((r['balance']-INITIAL_BALANCE)+dor), mode,lo,hi, r))
results.sort(reverse=True, key=lambda x:x[0])
_,mode,lo,hi,best=results[0]

print('\n'+'='*100)
print(f'  MELHOR GUARD: {mode} ({lo}/{hi})')
print('='*100)
by={}
for x in best['trades']:
    mk=x['ts'].strftime('%Y-%m'); by.setdefault(mk,{'pnl':0,'bear':0,'t':0})
    by[mk]['pnl']+=x['pnl']; by[mk]['t']+=1
    if x['strat']=='bear': by[mk]['bear']+=x['pnl']
run_bal=INITIAL_BALANCE
print(f"  {'Mes':<8} | {'Trades':>6} | {'P&L mes':>10} | {'BEAR':>9} | {'Saldo':>11}")
print('  '+'-'*56)
for mk in sorted(by):
    d=by[mk]; run_bal+=d['pnl']; yr,mo=int(mk[:4]),int(mk[5:]); flag=' <<<' if d['pnl']<0 else ''
    print(f"  {ML[mo-1]}/{yr%100:02d} | {d['t']:>6} | {d['pnl']:>+10.0f} | {d['bear']:>+9.0f} | {run_bal:>9,.0f}{flag}")

bm=monthly(base['trades']); bn=sum(v for v in bm.values() if v<0)
mm=monthly(best['trades']); mn=sum(v for v in mm.values() if v<0)
print(f"\n  Dor:    SEM guard {bn:+.0f}  ->  com guard {mn:+.0f}")
print(f"  Total:  SEM guard {base['balance']-INITIAL_BALANCE:+.0f}  ->  com guard {best['balance']-INITIAL_BALANCE:+.0f}")
print(f"  DD:     SEM guard {base['max_dd']:.1f}%  ->  com guard {best['max_dd']:.1f}%")
print(f"  Meses-: SEM guard {sum(1 for v in bm.values() if v<0)}  ->  com guard {sum(1 for v in mm.values() if v<0)}")
print('\n  Custos 0.14% round-trip. Bull e Lateral HIPOTETICAS.')
