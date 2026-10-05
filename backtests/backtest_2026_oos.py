"""
backtest_2026_oos.py
"E se eu tivesse ligado o bot em JANEIRO/2026 com 500 USDC?" — B vs C vs A(atual),
nos DOIS modelos de capital do Rafa:

  RESET       : comeca TODO mes com 500 (saca lucro / repoe perda no fim do mes).
                P&L total = soma dos meses.
  ACUMULATIVO : entra com 500 em janeiro e deixa compor — sem saques nem aportes.

BONUS metodologico: 2026 e OUT-OF-SAMPLE GENUINO — todas as calibracoes (sistema
v3, variantes B/C do Decisor) usaram 2021-2025. Jan-jun/2026 nunca foi visto.
Se o sistema aguentar aqui, e evidencia forte de robustez; se desabar, red flag.

Variantes do Decisor (sistema v3 completo: lateral + bear_breakout + bull_smc):
  A) Diario fechado (ATUAL no bot) — lag 24-48h
  B) Diario continuo (EMA480+slope120 no 1H) — sem lag, mesma lentidao
  C) 4H fechado (EMA20+slope5 em 4H) — mais reativo

Riscos: 1% (referencia dos comparativos) e 4% (config live do Rafa).
Periodo: 01/01/2026 -> 29/06/2026 (fim do cache de dados).
Usa cache _jun25_jun26. Uso: python backtests/backtest_2026_oos.py
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

FETCH_START=datetime(2025,6,1,tzinfo=timezone.utc)   # cache jun25_jun26
FETCH_END  =datetime(2026,6,29,tzinfo=timezone.utc)
TEST_START =datetime(2026,1,1,tzinfo=timezone.utc)
TEST_END   =FETCH_END
MONTHLY_BASE=500.0; MAX_PER_SIDE=3
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD; B_ATR_AVG=BEAR.ATR_AVG_PERIOD
B_COOLDOWN=BEAR.COOLDOWN_BARS; B_RR=BEAR.RR_CAP; B_BE=BEAR.BE_TRIGGER_PCT; B_TRAIL=BEAR.TRAIL_ATR
R_THRESH=regime_mod.SLOPE_THRESH
BSEL_LB=30; BSEL_ADX=20; BSEL_STRUCT=20
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
    since=int((FETCH_START-timedelta(days=extra_days)).timestamp()*1000)
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


def classify(price, ema, slope, thresh=R_THRESH):
    reg=np.full(len(price),'NEUTRAL',dtype=object)
    reg[(price<ema)&(slope<-thresh)]='BEAR'
    reg[(price>ema)&(slope> thresh)]='BULL'
    return reg


def regime_variants(df1, dfd):
    out={}; c1=df1['c']
    ema_d=dfd['c'].ewm(span=20,adjust=False).mean()
    slope_d=(ema_d-ema_d.shift(5))/ema_d.shift(5)
    regd=pd.Series(classify(dfd['c'].values, ema_d.values, slope_d.values), index=dfd.index)
    out['A_diario_atual']=regd.shift(1).reindex(df1.index, method='ffill').fillna('NEUTRAL').values
    ema_b=c1.ewm(span=480,adjust=False).mean()
    slope_b=(ema_b-ema_b.shift(120))/ema_b.shift(120)
    out['B_diario_continuo']=classify(c1.values, ema_b.values, slope_b.values)
    df4=c1.resample('4h').last().dropna()
    ema4=df4.ewm(span=20,adjust=False).mean()
    slope4=(ema4-ema4.shift(5))/ema4.shift(5)
    reg4=pd.Series(classify(df4.values, ema4.values, slope4.values), index=df4.index)
    out['C_4h_fechado']=reg4.shift(1).reindex(df1.index, method='ffill').fillna('NEUTRAL').values
    return out


print('OOS 2026 — "e se eu tivesse ligado o bot em janeiro com 500 USDC?"')
print(f'  Teste: {TEST_START:%d/%m/%Y} -> {TEST_END:%d/%m/%Y} (fim do cache) | sistema v3')
print('  2026 = out-of-sample GENUINO (nada foi calibrado nestes dados)\n')
print('A carregar dados (cache jun25-jun26)...')
pair_data={}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd=fetch(sym,'1d',40).copy(); df1=fetch(sym,'1h',25).copy()
        if len(df1)<800: print('SEM DADOS'); continue
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['atr_avg']=df1['atr'].rolling(B_ATR_AVG).mean()
        df1['adx']=adx(df1,14)
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        ema_d=dfd['c'].ewm(span=20,adjust=False).mean()
        slope_d=(ema_d-ema_d.shift(5))/ema_d.shift(5)
        df1['slope_d']=slope_d.shift(1).reindex(df1.index,method='ffill')
        pair_data[sym]=(df1, regime_variants(df1, dfd))
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen(df, reg):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    slope=df['slope_d'].values; ema_t=df['ema_trend'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object)
    start_i=max(B_ATR_AVG+BSEL_LB+5, U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, L_LOOKBACK+5, 610)
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


GEST={'bull':dict(be=U_BE,trail=U_TRAIL,manage='trail',cd=U_COOLDOWN),
      'bear':dict(be=B_BE,trail=B_TRAIL,manage='trail',cd=B_COOLDOWN),
      'lat': dict(be=0,trail=0,manage='fixed',cd=L_COOLDOWN)}


def run(variant, capital_mode, risk_pct):
    """capital_mode: 'reset' (500/mes) | 'acum' (500 compondo, sem aportes)."""
    midx=None; arrs={}
    for sym,(df,regs) in pair_data.items():
        s,e,sl,tp,st=gen(df, regs[variant])
        d=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st)
        midx=d.index if midx is None else midx.union(d.index); arrs[sym]=d
    midx=midx[(midx>=TEST_START)&(midx<=TEST_END)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,'tp':dd['_tp'].values,'strat':dd['_st'].values}
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; monthly={}; cur_month=None; month_start_bal=MONTHLY_BASE
    trades=0
    def close_all(k):
        nonlocal balance
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): c=positions[sym]['entry']
            pos=positions[sym]; e=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
            nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r
            balance+=nr*pos['risk_usd']; del positions[sym]
    for k in range(len(midx)):
        ts=midx[k]; mk=ts.strftime('%Y-%m')
        if cur_month is None: cur_month=mk; month_start_bal=balance
        if mk!=cur_month:
            if capital_mode=='reset':
                close_all(k); monthly[cur_month]=balance-MONTHLY_BASE
                balance=MONTHLY_BASE; peak=MONTHLY_BASE
            else:
                monthly[cur_month]=balance-month_start_bal
            cur_month=mk; month_start_bal=balance
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
                            'risk_px':risk_px,'risk_usd':balance*(risk_pct/100.0),'fee_r':e*RT/risk_px,
                            'strat':strat,'be':g['be'],'trail':g['trail'],'manage':g['manage'],'cd':g['cd']}
            trades+=1
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    if cur_month is not None:
        monthly[cur_month]=(balance-MONTHLY_BASE) if capital_mode=='reset' else (balance-month_start_bal)
    final=(MONTHLY_BASE+sum(monthly.values())) if capital_mode=='reset' else balance
    return dict(monthly=monthly, final=final, pnl=sum(monthly.values()), max_dd=max_dd, trades=trades)


VARIANTS=['A_diario_atual','B_diario_continuo','C_4h_fechado']
LABELS={'A_diario_atual':'A) Diario (ATUAL)','B_diario_continuo':'B) Diario continuo','C_4h_fechado':'C) 4H fechado'}
ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']

for risk in [1.0, 4.0]:
    print('='*100)
    print(f'  RISCO {risk:.0f}%/trade — jan a jun/2026, entrada com 500 USDC')
    print('='*100)
    print(f"  {'Decisor':<22} | {'-- RESET (500/mes) --':^26} | {'-- ACUMULATIVO (compoe) --':^30}")
    print(f"  {'':<22} | {'P&L total':>10} {'meses+':>8} {'DD':>5} | {'saldo final':>12} {'ret%':>7} {'DD':>6} | {'trades':>6}")
    print('  '+'-'*96)
    detail={}
    for v in VARIANTS:
        r_res=run(v,'reset',risk); r_acu=run(v,'acum',risk)
        detail[v]=(r_res,r_acu)
        mp=sum(1 for x in r_res['monthly'].values() if x>0)
        print(f"  {LABELS[v]:<22} | {r_res['pnl']:>+10.0f} {mp:>4}/{len(r_res['monthly']):<3} {r_res['max_dd']:>4.0f}% | "
              f"{r_acu['final']:>12,.0f} {(r_acu['final']/MONTHLY_BASE-1)*100:>+6.1f}% {r_acu['max_dd']:>5.1f}% | {r_res['trades']:>6}")
    print()
    print('  Mensal (RESET, cada mes sobre 500):')
    hdr='  '+f"{'Decisor':<22}"+''.join(f"{ML[m]:>8}" for m in range(6))
    print(hdr)
    for v in VARIANTS:
        r_res,_=detail[v]
        row='  '+f"{LABELS[v]:<22}"
        for m in range(1,7):
            val=r_res['monthly'].get(f'2026-{m:02d}')
            row+=f"{val:>+8.0f}" if val is not None else f"{'.':>8}"
        print(row)
    print()
    print('  Mensal (ACUMULATIVO, P&L do mes sobre saldo corrente):')
    print(hdr)
    for v in VARIANTS:
        _,r_acu=detail[v]
        row='  '+f"{LABELS[v]:<22}"
        for m in range(1,7):
            val=r_acu['monthly'].get(f'2026-{m:02d}')
            row+=f"{val:>+8.0f}" if val is not None else f"{'.':>8}"
        print(row)
    print()

print('  Notas: dados ate 29/06/2026 (jun incompleto por 1 dia). 2026 e OOS genuino —')
print('  nenhum parametro foi calibrado nestes dados. Custos 0.14% round-trip. max_per_side=3.')
