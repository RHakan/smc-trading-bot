"""
backtest_sistema_qualidade.py
SISTEMA COMPLETO + PRIORIZACAO POR QUALIDADE das entradas.

O sistema abre trades demais e indiscriminados (1672t, PF1.07) e os tetos
anteriores deram resultado enviesado (escolhiam por ORDEM de par). Aqui:

  1. Cada sinal ganha um SCORE de conviccao (quao decisivo e o rompimento,
     medido em ATRs):
        bear : (low_n - close)/atr   — quao fundo rompeu o suporte
        bull : (close - ref_high)/atr — quao decisivo o CHoCH
        lat  : (rompimento da borda)/atr
  2. min_score — descarta sinais fracos (sobe a qualidade media).
  3. Quando varios sinais competem por vaga, o motor abre os de MAIOR score
     primeiro (ranking por merito — elimina o vies de ordem-de-par).

Objetivo: menos trades, melhores; subir PF e ret/DD. Honesto: pode reduzir o
resultado total (menos trades) mas com curva muito mais saudavel.

Inclui LEVANTAMENTO MENSAL (trades/mes + P&L/mes) em cada config — pedido do Rafa.
Periodo jun/2025->jun/2026. Uso: python backtests/backtest_sistema_qualidade.py
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
INITIAL_BALANCE=5000.0; PERIOD_DAYS=(END-START).days
N_MONTHS=13
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


print('SISTEMA COMPLETO + PRIORIZACAO POR QUALIDADE')
print(f'  Periodo: {START:%d/%m/%Y} -> {END:%d/%m/%Y} ({PERIOD_DAYS} dias) | saldo {INITIAL_BALANCE:.0f}\n')
print('A carregar dados (cache)...')
pair_data={}; btc_bull=None
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
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['slope_d']=dfd['slope_d'].shift(1).reindex(df1.index,method='ffill')
        df1['low_n']=df1['l'].shift(1).rolling(B_LOOKBACK).min()
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        pair_data[sym]=df1
        if sym.startswith('BTC'): btc_bull=(df1['regime']=='BULL')
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()

master_index=None
for sym,df in pair_data.items():
    master_index=df.index if master_index is None else master_index.union(df.index)
master_index=master_index[(master_index>=START)&(master_index<=END)]


def gen_all(df, btc_local):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    reg=df['regime'].values; slope=df['slope_d'].values; low_n=df['low_n'].values; ema_t=df['ema_trend'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object); score=np.full(n,np.nan)
    start_i=max(B_ATR_AVG+B_LOOKBACK+5, U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, L_LOOKBACK+5, 210)
    for i in range(start_i,n):
        r=reg[i]; a=atr[i]
        if np.isnan(a) or a<=0: continue
        if r=='BEAR':
            cl=c[i]; op=o[i]; av=atr_avg[i]; ln=low_n[i]
            if not any(np.isnan(v) for v in (cl,av,ln)) and cl<ln and cl<op and a>av:
                stop=cl+B_ATR_STOP*a; risk=stop-cl
                if risk>0:
                    side[i]='SHORT'; entry[i]=cl; sl[i]=stop; tp[i]=cl-B_RR*risk; strat[i]='bear'
                    score[i]=(ln-cl)/a   # conviccao: quao fundo rompeu o suporte
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
                side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=e+U_RR*risk; strat[i]='bull'
                score[i]=(e-rh)/a   # conviccao do CHoCH
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

A={}
for sym,df in pair_data.items():
    btc_local=btc_bull.reindex(df.index).fillna(False).values
    s,e,sl,tp,st,sc=gen_all(df, btc_local)
    d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st,_sc=sc).reindex(master_index)
    A[sym]={'h':d['h'].values,'l':d['l'].values,'c':d['c'].values,'atr':d['atr'].values,
            'side':d['_s'].values,'entry':d['_e'].values,'sl':d['_sl'].values,
            'tp':d['_tp'].values,'strat':d['_st'].values,'score':d['_sc'].values}


def run(min_score=0.0, max_total=None, max_per_side=None, risk_pct=1.0):
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
        # candidatos rankeados por score (merito, nao ordem de par)
        cands=[]
        for sym in A:
            if sym in positions: continue
            if k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            sc=d['score'][k]
            if np.isnan(sc) or sc<min_score: continue
            cands.append((sc, sym, side))
        cands.sort(reverse=True)   # maior score primeiro
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sc,sym,side in cands:
            if max_total is not None and len(positions)>=max_total: break
            if max_per_side is not None:
                if side=='LONG' and n_long>=max_per_side: continue
                if side=='SHORT' and n_short>=max_per_side: continue
            d=A[sym]; e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]; strat=d['strat'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            g=GEST[strat]
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'be_done':False,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(risk_pct/100.0),'fee_r':e*RT/risk_px,
                            'strat':strat,'be':g['be'],'trail':g['trail'],'manage':g['manage'],'cd':g['cd']}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return dict(balance=balance,trades=trades,max_dd=max_dd)


def line(label, res, base=False):
    t=res['trades']; bal=res['balance']; mark='*' if base else ' '
    if not t: print(f'  {label:<30}{mark}: sem trades'); return res
    w=[x for x in t if x['nr']>0]; gw=sum(x['nr'] for x in w); gl=abs(sum(x['nr'] for x in t if x['nr']<=0))
    pnl=bal-INITIAL_BALANCE; ret=pnl/INITIAL_BALANCE*100
    rdd=ret/res['max_dd'] if res['max_dd']>0 else 0
    print(f'  {label:<30}{mark}: {ret:>+6.1f}% | {len(t):>4}t ({len(t)/N_MONTHS:>4.0f}/mes) | '
          f'{pnl/N_MONTHS:>+6.0f}/mes | PF {gw/gl if gl>0 else 99:>4.2f} | DD {res["max_dd"]:>4.1f}% | ret/DD {rdd:>4.2f}')
    return res


def tabela_mensal(res):
    months={}
    for x in res['trades']:
        months.setdefault(x['ts'].strftime('%Y-%m'),[]).append(x)
    ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
    run_bal=INITIAL_BALANCE
    print(f"\n  {'Mes':<8} | {'Trades':>6} | {'P&L mes':>11} | {'Saldo':>12}")
    print('  '+'-'*46)
    for mk in sorted(months):
        xs=months[mk]; pnl=sum(x['pnl'] for x in xs); run_bal+=pnl
        yr,mo=int(mk[:4]),int(mk[5:])
        print(f"  {ML[mo-1]}/{yr%100:02d} | {len(xs):>6} | {pnl:>+10.0f} | {run_bal:>10,.0f}")


print('='*112)
print('  PRIORIZACAO POR QUALIDADE (min_score em ATRs | trades/mes e P&L/mes em cada linha)')
print('='*112)
line('SEM filtro (min_score=0)', run(min_score=0.0), base=True)

print('\nMin score de conviccao (descarta sinais fracos):')
for ms in [0.0, 0.1, 0.2, 0.3, 0.5]:
    line(f'min_score={ms}', run(min_score=ms))

print('\nQualidade + teto por lado (ranking ja remove o vies):')
for ms,ps in [(0.2,2),(0.2,3),(0.3,2),(0.3,3)]:
    line(f'min{ms} side{ps}', run(min_score=ms, max_per_side=ps))

print('\nQualidade + teto + subir risco (magnitude com DD controlado):')
for ms,ps,rp in [(0.3,2,1.0),(0.3,2,2.0),(0.3,2,3.0),(0.2,3,2.0)]:
    line(f'min{ms} side{ps} risk{rp}%', run(min_score=ms, max_per_side=ps, risk_pct=rp))

# escolhe a melhor por ret/DD (operabilidade) entre as positivas
configs=[(0.2,None,1.0),(0.3,None,1.0),(0.3,2,1.0),(0.3,2,2.0),(0.2,3,2.0),(0.3,3,1.0)]
results=[]
for ms,ps,rp in configs:
    r=run(min_score=ms, max_per_side=ps, risk_pct=rp)
    ret=(r['balance']/INITIAL_BALANCE-1)*100
    rdd=ret/r['max_dd'] if r['max_dd']>0 else -99
    results.append((rdd, ret, ms, ps, rp, r))
results.sort(reverse=True)
rdd,ret,ms,ps,rp,best=results[0]

print('\n'+'='*112)
print(f'  MELHOR (por ret/DD): min_score={ms} max_per_side={ps} risk={rp}%')
print(f'  -> {ret:+.1f}% | {best["balance"]-INITIAL_BALANCE:+.0f} USDC | DD {best["max_dd"]:.1f}% | ret/DD {rdd:.2f}')
print('='*112)
tabela_mensal(best)
print('\n  Custos 0.14% round-trip. Bull e Lateral HIPOTETICAS. Risco por trade variavel.')
