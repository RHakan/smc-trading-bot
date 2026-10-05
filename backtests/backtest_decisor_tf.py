"""
backtest_decisor_tf.py
VELOCIDADE do Decisor — dados diarios (atual, lag 24-48h) vs variantes mais rapidas.

Contexto: o Decisor ja roda a cada hora, mas olha candles DIARIOS e descarta a
vela em formacao -> um movimento de hoje so entra na conta amanha. Isso deixa o
bot cego aos repiques intra-regime (rallies de 24-48h que o Rafa marcou no grafico).
Ja sabemos que o extremo oposto (estrutura 1H) REPROVA por whipsaw. Aqui testamos
o meio-termo, no sistema v3 completo (lateral + bear melhorado + bull):

  A) DIARIO fechado (ATUAL)  : EMA20 diaria + slope 5 velas, vela em formacao
                               descartada. Lag ate 24-48h.
  B) DIARIO continuo (1H)    : EMA480 + slope 120 barras no 1H — mesma lentidao,
                               mas atualiza A CADA HORA (sem lag, sem repaint).
  C) 4H fechado              : EMA20 de 4H + slope 5 velas (reage em ~2-3 dias).
  D) 4H continuo (rapido)    : EMA80 + slope 20 barras no 1H (reage em horas —
                               proximo do que ja falhou; incluido como controle).

Mede: P&L total, meses+, DD, e Nº DE ENTRADAS POR ESTRATEGIA (a pergunta do Rafa:
"faria diferenca nos numeros de entradas?") + distribuicao de regime (% barras).
Capital 500/mes reset, risco 1% (4% so escala a magnitude), max_per_side 3.
5 anos (2021-2025). Usa cache _2020_2025. Uso: python backtests/backtest_decisor_tf.py
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
FETCH_END=datetime(2025,10,6,tzinfo=timezone.utc)
MONTHLY_BASE=500.0; RISK_PCT=1.0; MAX_PER_SIDE=3
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD; B_ATR_AVG=BEAR.ATR_AVG_PERIOD
B_COOLDOWN=BEAR.COOLDOWN_BARS; B_RR=BEAR.RR_CAP; B_BE=BEAR.BE_TRIGGER_PCT; B_TRAIL=BEAR.TRAIL_ATR
R_THRESH=regime_mod.SLOPE_THRESH   # 0.001 — mesmo threshold em todas as variantes
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


def classify(price, ema, slope, thresh=R_THRESH):
    reg=np.full(len(price),'NEUTRAL',dtype=object)
    reg[(price<ema)&(slope<-thresh)]='BEAR'
    reg[(price>ema)&(slope> thresh)]='BULL'
    return reg


def regime_variants(df1, dfd):
    """Devolve dict variante -> array de regime por barra 1H (todos causais)."""
    out={}
    c1=df1['c']

    # A) DIARIO fechado (atual): EMA20 D + slope 5D, shift(1), ffill no 1H
    ema_d=dfd['c'].ewm(span=20,adjust=False).mean()
    slope_d=(ema_d-ema_d.shift(5))/ema_d.shift(5)
    regd=pd.Series(classify(dfd['c'].values, ema_d.values, slope_d.values), index=dfd.index)
    out['A_diario_atual']=regd.shift(1).reindex(df1.index, method='ffill').fillna('NEUTRAL').values

    # B) DIARIO continuo: EMA480 + slope 120 no 1H (mesma lentidao, atualiza a cada hora)
    ema_b=c1.ewm(span=480,adjust=False).mean()
    slope_b=(ema_b-ema_b.shift(120))/ema_b.shift(120)
    out['B_diario_continuo']=classify(c1.values, ema_b.values, slope_b.values)

    # C) 4H fechado: resample -> EMA20 + slope 5 velas 4H, shift(1), ffill
    df4=df1['c'].resample('4h').last().dropna()
    ema4=df4.ewm(span=20,adjust=False).mean()
    slope4=(ema4-ema4.shift(5))/ema4.shift(5)
    reg4=pd.Series(classify(df4.values, ema4.values, slope4.values), index=df4.index)
    out['C_4h_fechado']=reg4.shift(1).reindex(df1.index, method='ffill').fillna('NEUTRAL').values

    # D) 4H continuo (rapido — controle, perto do que ja falhou)
    ema_f=c1.ewm(span=80,adjust=False).mean()
    slope_f=(ema_f-ema_f.shift(20))/ema_f.shift(20)
    out['D_4h_continuo']=classify(c1.values, ema_f.values, slope_f.values)

    return out


print('DECISOR — velocidade dos dados (diario com lag vs continuo vs 4H) no sistema v3')
print('A carregar dados (cache 2020-2025)...')
pair_data={}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd=fetch(sym,'1d').copy(); df1=fetch(sym,'1h').copy()
        if len(df1)<500: print('SEM DADOS'); continue
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['atr_avg']=df1['atr'].rolling(B_ATR_AVG).mean()
        df1['adx']=adx(df1,14)
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        # slope diario p/ filtro da bull (fiel ao bot: sempre do diario fechado)
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
    start_i=max(B_ATR_AVG+BSEL_LB+5, U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, L_LOOKBACK+5, 490)
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


def run(variant, start, end):
    midx=None; arrs={}
    for sym,(df,regs) in pair_data.items():
        s,e,sl,tp,st=gen(df, regs[variant])
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
    counts={'lat':0,'bull':0,'bear':0}
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
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px,
                            'strat':strat,'be':g['be'],'trail':g['trail'],'manage':g['manage'],'cd':g['cd']}
            counts[strat]+=1
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    monthly[cur_month]=balance-MONTHLY_BASE
    return dict(monthly=monthly, pnl=sum(monthly.values()), max_dd=max_dd, counts=counts)


VARIANTS=['A_diario_atual','B_diario_continuo','C_4h_fechado','D_4h_continuo']
LABELS={'A_diario_atual':'A) Diario fechado (ATUAL)','B_diario_continuo':'B) Diario continuo (s/ lag)',
        'C_4h_fechado':'C) 4H fechado','D_4h_continuo':'D) 4H continuo (rapido)'}

# Distribuicao de regime (BTC como referencia)
print('='*100)
print('  DISTRIBUICAO DE REGIME (BTC, % das barras 1H em 2021-2025)')
print('='*100)
btc_df,btc_regs=pair_data['BTC/USDT:USDT']
mask=(btc_df.index>=datetime(2021,1,1,tzinfo=timezone.utc))
for v in VARIANTS:
    r=pd.Series(btc_regs[v][np.asarray(mask)])
    pct=r.value_counts(normalize=True)*100
    # trocas de regime (uma medida de whipsaw)
    switches=int((r!=r.shift(1)).sum())
    print(f"  {LABELS[v]:<30}: BEAR {pct.get('BEAR',0):>4.1f}% | BULL {pct.get('BULL',0):>4.1f}% | "
          f"NEUTRAL {pct.get('NEUTRAL',0):>4.1f}% | trocas de regime: {switches}")

print('\n'+'='*100)
print('  SISTEMA v3 COMPLETO com cada Decisor (500/mes reset, risco 1%)')
print('='*100)
print(f"  {'Decisor':<30} | {'TOTAL':>7} | {'meses+':>9} | {'DDmes':>6} | {'ENTRADAS lat/bull/bear':>24}")
print('  '+'-'*92)
results={}
for v in VARIANTS:
    allm={}; tot=0; ddmax=0; cnt={'lat':0,'bull':0,'bear':0}
    for yr in YEARS:
        start=datetime(yr,1,1,tzinfo=timezone.utc); end=YEAR_END.get(yr,datetime(yr,12,31,tzinfo=timezone.utc))
        r=run(v,start,end)
        allm.update(r['monthly']); tot+=r['pnl']; ddmax=max(ddmax,r['max_dd'])
        for s in cnt: cnt[s]+=r['counts'][s]
    pos=sum(1 for x in allm.values() if x>0)
    results[v]=dict(tot=tot,pos=pos,n=len(allm),ddmax=ddmax,cnt=cnt,monthly=allm)
    print(f"  {LABELS[v]:<30} | {tot:>+7.0f} | {pos:>3}/{len(allm):<3} | {ddmax:>5.1f}% | "
          f"{cnt['lat']:>6} /{cnt['bull']:>5} /{cnt['bear']:>5}")

best=max(results, key=lambda v:(results[v]['pos'], results[v]['tot']))
print('\n'+'='*100)
print(f"  MELHOR: {LABELS[best]}  ->  total {results[best]['tot']:+.0f} | {results[best]['pos']}/{results[best]['n']} meses+ | DD {results[best]['ddmax']:.1f}%")
print('='*100)
print('  Leitura: mais entradas != melhor. O que importa: total, meses+, DD.')
print('  D (rapido) e o controle — se ganhar, contradiz o teste da estrutura 1H; se perder, confirma.')
print('  Custos 0.14% round-trip. Bull sempre filtra pelo slope diario (fiel ao bot).')
