"""
backtest_lateral_wyckoff.py
Estrategia LATERAL v2 — Wyckoff Spring + Upthrust (os dois lados do range).

Fundamentada no material do Rafa (Guia Pratico Metodo Wyckoff, Eduardo Custodio):
  "Fase C: mercado faz o falso rompimento... Spring no fundo e UT/UTAD no topo.
   Essa e a melhor fase para fazer compras ou vendas, o risco retorno e muito
   positivo. Stop na minima e primeiro alvo no topo do range."

Porque o mean-reversion (BB+RSI) falhou e isto deve funcionar:
  Mean-reversion : entrava em QUALQUER toque, stop arbitrario (banda-ATR), alvo
                   curto (a media). Ganhava pouco, perdia GRANDE no rompimento.
  Wyckoff        : entra so no FALSO rompimento confirmado (varre liquidez e volta),
                   stop ESTRUTURAL na minima/maxima do sweep (pequeno), alvo no
                   LADO OPOSTO do range (RR 3-5). Inverte o perfil: perde pouco,
                   ganha grande. E a mesma mecanica do sweep+CHoCH que ja deu edge
                   na ALTA — agora nas duas bordas do range.

  SPRING   (LONG) : low < suporte do range  E  close > suporte  -> compra.
                    stop = low do spring - folga ; alvo = topo do range.
  UPTHRUST (SHORT): high > resistencia       E  close < resistencia -> vende.
                    stop = high do upthrust + folga ; alvo = fundo do range.

Deteccao de RANGE genuino (o NEUTRAL do Decisor nao basta — inclui transicoes):
  - regime NEUTRAL (Decisor) E
  - ADX < adx_max (sem tendencia forte = range de verdade) E
  - largura do range dentro de uma faixa sa (nem estreito nem explosivo)

Motor de portfolio realista (saldo 5000, correlacao). Periodo jun/2025->jun/2026.
Mede USDC/dia vs meta 40/dia. Reporta Spring(long) vs Upthrust(short) em separado.
Uso: python backtests/backtest_lateral_wyckoff.py
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
COOLDOWN=4; TIMEOUT=72   # range pode demorar a resolver -> timeout maior

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
    """ADX de Wilder. Alto = tendencia; baixo = range."""
    h=df['h']; l=df['l']; c=df['c']
    up=h.diff(); dn=-l.diff()
    plus_dm  = np.where((up>dn)&(up>0), up, 0.0)
    minus_dm = np.where((dn>up)&(dn>0), dn, 0.0)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di =100*pd.Series(plus_dm, index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    minus_di=100*pd.Series(minus_dm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


print('Estrategia LATERAL v2 — Wyckoff Spring + Upthrust (dois lados do range)')
print(f'  Periodo: {START:%d/%m/%Y} -> {END:%d/%m/%Y} ({PERIOD_DAYS} dias) | saldo {INITIAL_BALANCE:.0f}')
print(f'  Meta: {TARGET_PER_DAY:.0f} USDC/dia\n')
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
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-df1['c'].shift(1)).abs(),
                      (df1['l']-df1['c'].shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['adx']=adx(df1, 14)
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['vol_avg']=df1['v'].rolling(48).mean()
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()

master_index=None
for sym,df in pair_data.items():
    master_index=df.index if master_index is None else master_index.union(df.index)
master_index=master_index[(master_index>=START)&(master_index<=END)]


def gen_wyckoff_signals(df, lookback, adx_max, min_pierce, stop_atr, rr_cap,
                        tp_mode, vol_mult, neutral_only):
    """Spring (long) + Upthrust (short) num range detectado."""
    n=len(df)
    h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; adx_a=df['adx'].values; reg=df['regime'].values
    vol=df['v'].values; vavg=df['vol_avg'].values
    side=np.array([None]*n,dtype=object)
    entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)

    for i in range(lookback+5, n):
        if neutral_only and reg[i]!='NEUTRAL': continue
        if adx_max>0 and (np.isnan(adx_a[i]) or adx_a[i]>adx_max): continue
        a=atr[i]
        if np.isnan(a): continue
        # Range das barras anteriores
        rl=np.min(l[i-lookback:i]); rh=np.max(h[i-lookback:i])
        if rl<=0 or rh<=rl: continue
        width=(rh-rl)/rl*100
        if width<1.0 or width>40.0: continue   # nem estreito nem explosivo
        cl=c[i]; lo=l[i]; hi=h[i]
        vol_ok = (vol_mult<=0) or (not np.isnan(vavg[i]) and vol[i]>=vol_mult*vavg[i])

        # SPRING -> LONG: varreu o suporte e fechou acima
        if lo<rl and cl>rl:
            if min_pierce>0 and (rl-lo)/rl*100 < min_pierce: pass
            elif vol_ok:
                e=cl; stop=lo-stop_atr*a; risk=e-stop
                if risk>0:
                    target = rh if tp_mode=='opposite' else (rl+rh)/2
                    if rr_cap>0: target=min(target, e+rr_cap*risk)
                    if target>e:
                        side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=target
                        continue
        # UPTHRUST -> SHORT: varreu a resistencia e fechou abaixo
        if hi>rh and cl<rh:
            if min_pierce>0 and (hi-rh)/rh*100 < min_pierce: pass
            elif vol_ok:
                e=cl; stop=hi+stop_atr*a; risk=stop-e
                if risk>0:
                    target = rl if tp_mode=='opposite' else (rl+rh)/2
                    if rr_cap>0: target=max(target, e-rr_cap*risk)
                    if target<e:
                        side[i]='SHORT'; entry[i]=e; sl[i]=stop; tp[i]=target
    return side, entry, sl, tp


def build_arrays(**kw):
    A={}
    for sym,df in pair_data.items():
        s,e,sl,tp=gen_wyckoff_signals(df, **kw)
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
    if not t: print(f'  {label:<30}{mark}: sem trades'); return res
    w=[x for x in t if x['nr']>0]; gw=sum(x['nr'] for x in w); gl=abs(sum(x['nr'] for x in t if x['nr']<=0))
    pnl=bal-INITIAL_BALANCE; lng=[x for x in t if x['side']=='LONG']; sht=[x for x in t if x['side']=='SHORT']
    print(f'  {label:<30}{mark}: {pnl:>+8.0f} USDC | {pnl/PERIOD_DAYS:>+6.1f}/dia | {len(t):>4}t | '
          f'WR {len(w)/len(t)*100:>4.1f}% | PF {gw/gl if gl>0 else 99:>4.2f} | DD {res["max_dd"]:>4.1f}% | '
          f'L{len(lng)}/S{len(sht)}')
    return res


BASE=dict(lookback=50, adx_max=25, min_pierce=0.05, stop_atr=0.25, rr_cap=0,
          tp_mode='opposite', vol_mult=0, neutral_only=True)

print('='*120)
print('  LATERAL v2 — Wyckoff Spring+Upthrust no motor de portfolio (regime NEUTRAL + range ADX)')
print('='*120)
show('BASE', run_portfolio(build_arrays(**BASE)), base=True)

print('\nSweep 1 — lookback (janela do range):')
for lb in [30,50,80,120]:
    show(f'lookback={lb}', run_portfolio(build_arrays(**{**BASE,'lookback':lb})))

print('\nSweep 2 — adx_max (quao "range" tem de ser; menor=mais estrito):')
for ax in [0,20,25,30]:
    show(f'adx_max={ax if ax else "off"}', run_portfolio(build_arrays(**{**BASE,'adx_max':ax})))

print('\nSweep 3 — alvo (opposite=lado oposto | mid=meio do range):')
for tm in ['opposite','mid']:
    show(f'tp_mode={tm}', run_portfolio(build_arrays(**{**BASE,'tp_mode':tm})))

print('\nSweep 4 — RR_CAP (limita o alvo; 0=sem limite, vai ao lado oposto):')
for rr in [0,2.0,3.0,5.0]:
    show(f'rr_cap={rr if rr else "off"}', run_portfolio(build_arrays(**{**BASE,'rr_cap':rr})))

print('\nSweep 5 — folga do stop alem do sweep (xATR):')
for sa in [0.1,0.25,0.5,1.0]:
    show(f'stop_atr={sa}', run_portfolio(build_arrays(**{**BASE,'stop_atr':sa})))

print('\nSweep 6 — filtro de volume no sweep (Wyckoff: absorcao):')
for vm in [0,1.2,1.5,2.0]:
    show(f'vol_mult={vm if vm else "off"}', run_portfolio(build_arrays(**{**BASE,'vol_mult':vm})))

print('\nSweep 7 — operar so em NEUTRAL vs qualquer regime (com range estrito):')
for no in [True, False]:
    show(f'neutral_only={no}', run_portfolio(build_arrays(**{**BASE,'neutral_only':no})))

print('\nCombinacoes:')
combos=[
    dict(lookback=50, adx_max=25, min_pierce=0.05, stop_atr=0.25, rr_cap=3.0, tp_mode='opposite', vol_mult=0,   neutral_only=True),
    dict(lookback=50, adx_max=20, min_pierce=0.10, stop_atr=0.25, rr_cap=0,   tp_mode='opposite', vol_mult=1.2, neutral_only=True),
    dict(lookback=80, adx_max=25, min_pierce=0.05, stop_atr=0.5,  rr_cap=3.0, tp_mode='opposite', vol_mult=0,   neutral_only=True),
    dict(lookback=30, adx_max=20, min_pierce=0.05, stop_atr=0.25, rr_cap=2.0, tp_mode='mid',      vol_mult=0,   neutral_only=True),
    dict(lookback=50, adx_max=25, min_pierce=0.10, stop_atr=0.25, rr_cap=4.0, tp_mode='opposite', vol_mult=1.5, neutral_only=True),
    dict(lookback=80, adx_max=20, min_pierce=0.10, stop_atr=0.5,  rr_cap=0,   tp_mode='opposite', vol_mult=1.2, neutral_only=False),
]
res=[]
for p in combos:
    lbl=f"lb{p['lookback']} adx{p['adx_max']} rr{p['rr_cap']} {p['tp_mode'][:3]} v{p['vol_mult']}"
    res.append((p, show(lbl, run_portfolio(build_arrays(**p)))))

best_p,best_r=max(res, key=lambda x:x[1]['balance'])
pnl=best_r['balance']-INITIAL_BALANCE
print('\n'+'='*120)
print(f'  MELHOR: {best_p}')
print(f'  -> {pnl:+.0f} USDC | {pnl/PERIOD_DAYS:+.1f}/dia | DD {best_r["max_dd"]:.1f}% (meta {TARGET_PER_DAY:.0f}/dia)')
print('='*120)
print('  Se tiver edge (PF>1) mas faltar resultado, proximo passo: 15m (mais ranges) ou + pares.')
print('  Custos 0.14% round-trip. HIPOTETICA. Risco 1%/trade. Baseado no Guia Wyckoff do Rafa.')
