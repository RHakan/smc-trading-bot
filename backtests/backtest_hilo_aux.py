"""
backtest_hilo_aux.py
Testa 3 usos do HiLo Activator (Gann) COMO AUXILIO ao bot ATUAL (v3), lado a lado.

O baseline aqui NAO e o bot antigo: e o bot ATUAL, ja com a invalidacao de breakout
falho na lateral (rh, 5 velas) que ja foi validada e implementada. Ou seja, medimos se
o HiLo ADICIONA algo POR CIMA do que ja e o melhor sistema validado.

Os 3 modos (cada um isolado, tudo o resto igual ao baseline):
  1) FILTRO   : so abre o trade se o HiLo (1h) ja estiver a favor da direcao.
                Corta entradas ruins -> melhor prior (mesma logica da invalidacao).
  2) TRAIL    : conduz as DIRECIONAIS (bull/bear) com a linha do HiLo em vez do ATR,
                depois do BE. Lateral fica igual. Mesma familia de trailing ja
                reprovada no OOS — teste limpo pra tirar a duvida.
  3) VIES     : so abre o trade se o HiLo do DIARIO estiver a favor (confirma regime).
                Provavelmente redundante com o EMA20 diario do decisor.

HiLo Activator: SMA(highs, N) e SMA(lows, N). Vira pra CIMA quando o fechamento
cruza acima da SMA dos highs; vira pra BAIXO quando cruza abaixo da SMA dos lows.
A "linha" seguida (trailing) e a SMA dos lows quando em alta (piso) / SMA dos highs
quando em baixa (teto). N testado: 3 (o que o Rafa citou), 5, 8.

Direcionais SEMPRE no baseline validado (BE@1.0R + ATR), exceto no modo TRAIL.
Lateral SEMPRE com invalidacao rh-5. 500/mes reset. Custos 0.14% RT.
Uso: python backtests/backtest_hilo_aux.py
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

MONTHLY_BASE=500.0; RISK_PCT=1.0; MAX_PER_SIDE=3
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
       'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
       'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD; B_ATR_AVG=BEAR.ATR_AVG_PERIOD
B_COOLDOWN=BEAR.COOLDOWN_BARS; B_RR=BEAR.RR_CAP; B_TRAIL=BEAR.TRAIL_ATR
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH
BSEL_LB=30; BSEL_ADX=20; BSEL_STRUCT=20
U_SWING_N=10; U_CHOCH_BARS=12; U_CHOCH_REF=15; U_MIN_SWEEP=0.05; U_RR=2.5; U_TRAIL=2.0
U_COOLDOWN=3; U_SLOPE_MIN=0.006; U_EMA_TREND=100
L_LOOKBACK=30; L_ADX_MAX=20; L_BUFFER=0.1; L_COOLDOWN=4; L_TIMEOUT=48
L_INVAL_N=5   # invalidacao de breakout falho na lateral (bot atual)

NS_HILO=[3,5,8]   # periodos do HiLo a testar (3 = o citado pelo Rafa)

CACHE_DIR=Path(__file__).parent/'cache'; CACHE_DIR.mkdir(exist_ok=True)
ex=ccxt.binanceusdm({'enableRateLimit':True})


def _fetch(sym, tf, start_dt, end_dt, tag):
    safe=sym.replace('/','_').replace(':','_'); cache=CACHE_DIR/f'{safe}_{tf}_{tag}.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True); df.index=pd.to_datetime(df.index,utc=True); return df
    since=int((start_dt-timedelta(days=5)).timestamp()*1000); end_ms=int(end_dt.timestamp()*1000); rows=[]
    while True:
        b=ex.fetch_ohlcv(sym,tf,since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v']); df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=df.set_index('ts').sort_index()[lambda d:d.index<=end_dt]; df.to_csv(cache); return df


def fetch_5y(sym, tf):
    return _fetch(sym, tf, datetime(2020,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc), '2020_2025')
def fetch_26(sym, tf):
    return _fetch(sym, tf, datetime(2025,9,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc), '2025_2026oos')


def adx(df, period=14):
    h=df['h']; l=df['l']; c=df['c']; up=h.diff(); dn=-l.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/period,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    mdi=100*pd.Series(mdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period,adjust=False).mean()


def hilo(h, l, c, n):
    """HiLo Activator. Retorna (dir, line):
      dir  : +1 alta, -1 baixa, 0 indefinido.
      line : piso (SMA lows) em alta, teto (SMA highs) em baixa — a linha de trailing.
    Decisao de virada usa a SMA da barra ANTERIOR (sem lookahead); a linha usa a SMA
    da barra atual (conhecida no fechamento)."""
    sma_h=pd.Series(h).rolling(n).mean().values
    sma_l=pd.Series(l).rolling(n).mean().values
    N=len(c); dirn=np.zeros(N); line=np.full(N,np.nan); cur=0
    for i in range(N):
        if i<1 or np.isnan(sma_h[i-1]) or np.isnan(sma_l[i-1]):
            dirn[i]=cur; continue
        if c[i]>sma_h[i-1]: cur=1
        elif c[i]<sma_l[i-1]: cur=-1
        dirn[i]=cur
        if cur==1 and not np.isnan(sma_l[i]): line[i]=sma_l[i]
        elif cur==-1 and not np.isnan(sma_h[i]): line[i]=sma_h[i]
    return dirn, line


def load(fetch_fn):
    pdata={}; btcb=None
    for sym in PAIRS:
        dfd=fetch_fn(sym,'1d').copy(); df1=fetch_fn(sym,'1h').copy()
        if len(df1)<300: continue
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
        df1['ema_trend']=c.ewm(span=U_EMA_TREND,adjust=False).mean()
        # HiLo 1h (dir + line) e HiLo diario (so dir, mapeado pro 1h) para cada N
        hv=df1['h'].values; lv=df1['l'].values; cv=df1['c'].values
        hvd=dfd['h'].values; lvd=dfd['l'].values; cvd=dfd['c'].values
        for n in NS_HILO:
            d1,ln1=hilo(hv,lv,cv,n)
            df1[f'hd{n}']=d1; df1[f'hl{n}']=ln1
            dd,_=hilo(hvd,lvd,cvd,n)
            df1[f'hdd{n}']=pd.Series(dd,index=dfd.index).shift(1).reindex(df1.index,method='ffill').values
        pdata[sym]=df1
        if sym.startswith('BTC'): btcb=(df1['regime']=='BULL')
    return pdata, btcb


def gen(df, btc_local):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    reg=df['regime'].values; slope=df['slope_d'].values; ema_t=df['ema_trend'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object); blevel=np.full(n,np.nan)
    start_i=max(B_ATR_AVG+BSEL_LB+5, U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, L_LOOKBACK+5, 210)
    for i in range(start_i,n):
        r=reg[i]; a=atr[i]
        if np.isnan(a) or a<=0: continue
        if r=='NEUTRAL':
            if np.isnan(adx_a[i-1]) or adx_a[i-1]>L_ADX_MAX: continue
            rl=np.min(l[i-L_LOOKBACK:i]); rh=np.max(h[i-L_LOOKBACK:i])
            if rl<=0 or rh<=rl: continue
            buf=L_BUFFER*a; cl=c[i]; pc=c[i-1]; height=rh-rl
            if cl>rh+buf and pc<=rh:
                if cl>rl: side[i]='LONG'; entry[i]=cl; sl[i]=rl; tp[i]=cl+height; strat[i]='lat'; blevel[i]=rh
            elif cl<rl-buf and pc>=rl:
                if rh>cl: side[i]='SHORT'; entry[i]=cl; sl[i]=rh; tp[i]=cl-height; strat[i]='lat'; blevel[i]=rl
        elif r=='BULL':
            if not btc_local[i]: continue
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
    return side,entry,sl,tp,strat,blevel


def prep(pdata, btcb):
    arrs={}
    for sym,df in pdata.items():
        s,e,sl,tp,st,bl=gen(df, btcb.reindex(df.index).fillna(False).values)
        arrs[sym]=df.assign(_s=s,_e=e,_sl=sl,_tp=tp,_st=st,_bl=bl)
    return arrs


# Gestao baseline das direcionais (BE@1.0R + ATR). Lateral fixa+invalidacao.
GEST={'bull':dict(be=0.40,trail=U_TRAIL,cd=U_COOLDOWN),
      'bear':dict(be=0.40,trail=B_TRAIL,cd=B_COOLDOWN),
      'lat' :dict(be=9.99,trail=0.0,cd=L_COOLDOWN)}


def run(arrs, mode, hn, start, end):
    """mode: 'base' | 'filtro' | 'trail' | 'vies'. hn: periodo do HiLo (3/5/8).
    Baseline = bot atual (lateral com invalidacao rh-5, direcionais BE+ATR)."""
    midx=None
    for sym,d in arrs.items():
        midx=d.index if midx is None else midx.union(d.index)
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,'tp':dd['_tp'].values,
                'strat':dd['_st'].values,'bl':dd['_bl'].values,
                'hd':dd[f'hd{hn}'].values,'hl':dd[f'hl{hn}'].values,'hdd':dd[f'hdd{hn}'].values}
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; monthly={}; cur_month=None
    trades=[]
    def close_all(k):
        nonlocal balance
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]; pos=positions[sym]
            if np.isnan(c): c=pos['entry']
            e=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
            nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r
            balance+=nr*pos['risk_usd']; trades.append(nr); del positions[sym]
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
                elif pos['tmode']=='atr':
                    if not pos['be_done'] and (c-e)/risk>=pos['be']*((pos['tp']-e)/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c-pos['trail']*a
                        if cand>pos['cur']: pos['cur']=cand
                elif pos['tmode']=='hilo':
                    if not pos['be_done'] and (c-e)/risk>=pos['be']*((pos['tp']-e)/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done']:
                        cand=d['hl'][k]
                        if not np.isnan(cand) and cand>pos['cur']: pos['cur']=cand
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
                elif pos['tmode']=='atr':
                    if not pos['be_done'] and (e-c)/risk>=pos['be']*((e-pos['tp'])/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c+pos['trail']*a
                        if cand<pos['cur']: pos['cur']=cand
                elif pos['tmode']=='hilo':
                    if not pos['be_done'] and (e-c)/risk>=pos['be']*((e-pos['tp'])/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done']:
                        cand=d['hl'][k]
                        if not np.isnan(cand) and cand<pos['cur']: pos['cur']=cand
            if not closed and pos['manage']=='fixed':
                pos['age']+=1
                # invalidacao de breakout falho (bot atual): fechou de volta pro range
                if pos['age']<=L_INVAL_N and not np.isnan(pos['bl']):
                    inside=(c<pos['bl']) if pos['side']=='LONG' else (c>pos['bl'])
                    if inside:
                        nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
                if not closed and pos['age']>=L_TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']; trades.append(nr); del positions[sym]; cooldown_until[sym]=k+pos['cd']
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sym in A:
            if sym in positions: continue
            if k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            # ── gate HiLo (filtro 1h / vies diario) ──
            if mode=='filtro':
                hd=d['hd'][k]
                if (side=='LONG' and hd!=1) or (side=='SHORT' and hd!=-1): continue
            elif mode=='vies':
                hd=d['hdd'][k]
                if (side=='LONG' and hd!=1) or (side=='SHORT' and hd!=-1): continue
            if side=='LONG' and n_long>=MAX_PER_SIDE: continue
            if side=='SHORT' and n_short>=MAX_PER_SIDE: continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]; strat=d['strat'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            g=GEST[strat]
            # gestao: lateral sempre fixa+inval; direcionais atr, exceto modo TRAIL -> hilo
            if strat=='lat': tmode='fixed'; manage='fixed'
            elif mode=='trail': tmode='hilo'; manage='trail'
            else: tmode='atr'; manage='trail'
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'be_done':False,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px,
                            'tmode':tmode,'be':g['be'],'trail':g['trail'],
                            'manage':manage,'cd':g['cd'],'strat':strat,'bl':d['bl'][k]}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    monthly[cur_month]=balance-MONTHLY_BASE
    return monthly, max_dd, trades


def summ(res):
    monthly, ddmax, trades = res
    yr={}
    for mk,v in monthly.items(): yr[mk[:4]]=yr.get(mk[:4],0)+v
    tot=sum(monthly.values()); tr=np.array(trades); n=len(tr)
    wins=tr[tr>0.03]; losses=tr[tr<-0.03]
    return dict(tot=tot, yr=yr, monthly=monthly, anos=sum(1 for v in yr.values() if v>0),
                meses=sum(1 for v in monthly.values() if v>0), nm=len(monthly), dd=ddmax,
                n=n, wr=len(wins)/n*100 if n else 0, sum_win=wins.sum(), sum_loss=losses.sum())


print('A carregar 5 anos (cache) + 2026 OOS (fetch)...')
p5,b5=load(fetch_5y); a5=prep(p5,b5); print(f'  5 anos: {len(a5)} pares OK')
p26,b26=load(fetch_26); a26=prep(p26,b26); print(f'  2026 OOS: {len(a26)} pares OK')
S5=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc))
S26=(datetime(2026,1,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc))

print('\n'+'='*100)
print('  HiLo como AUXILIO ao bot ATUAL (baseline ja inclui invalidacao rh-5) | 500/mes reset')
print('='*100)

base5=summ(run(a5,'base',3,*S5)); base26=summ(run(a26,'base',3,*S26))

def linha(nome,s5,s26,mark=''):
    print('  '+f"{nome:<16}{s5['tot']:>+10.0f}{s5['anos']:>5}/5{s5['dd']:>6.1f}%{s5['wr']:>6.0f}%{'||':>4}"
          f"{s26['tot']:>+9.0f}{s26['meses']:>3}/{s26['nm']:<2}{s26['dd']:>6.1f}%{mark}")

print('\n  '+f"{'modo':<16}{'5anos':>10}{'anos+':>7}{'DD':>7}{'WR':>6}{'||':>4}{'2026':>9}{'m+':>6}{'DD':>7}")
print('  '+'-'*78)
linha('BASELINE', base5, base26, '  <= bot atual')

best=None
for mode,rot in [('filtro','FILTRO'),('trail','TRAIL'),('vies','VIES')]:
    print('  '+'-'*78)
    for hn in NS_HILO:
        s5=summ(run(a5,mode,hn,*S5)); s26=summ(run(a26,mode,hn,*S26))
        # candidato so conta se bate baseline nos DOIS periodos
        ok=s5['tot']>base5['tot'] and s26['tot']>base26['tot']
        mark='  <= bate nos 2' if ok else ''
        linha(f'{rot} N={hn}', s5, s26, mark)
        if ok and (best is None or (s5['tot']+s26['tot'])>best[3]):
            best=(mode,hn,(s5,s26),s5['tot']+s26['tot'])

print('\n'+'='*100)
if best is None:
    print('  VEREDITO: NENHUM uso do HiLo bate o baseline nos DOIS periodos (5 anos E OOS 2026).')
    print('  Ou seja: o HiLo, nas 3 formas testadas, nao adiciona edge por cima do bot atual.')
    print('  Coerente com a licao — apertar/confirmar em cripto 1h tende a reprovar no OOS.')
else:
    mode,hn,(w5,w26),_=best
    print(f'  MELHOR: {mode.upper()} N={hn} bate o baseline nos 5 anos E no OOS 2026.')
    print(f'    5 anos: {base5["tot"]:+.0f} -> {w5["tot"]:+.0f} ({w5["tot"]-base5["tot"]:+.0f})  '
          f'DD {base5["dd"]:.1f}% -> {w5["dd"]:.1f}%')
    print(f'    2026  : {base26["tot"]:+.0f} -> {w26["tot"]:+.0f} ({w26["tot"]-base26["tot"]:+.0f})  '
          f'DD {base26["dd"]:.1f}% -> {w26["dd"]:.1f}%')
    print('\n  [MENSAL OOS 2026] baseline vs melhor')
    mb=base26['monthly']; mw=w26['monthly']; meses=sorted(set(mb)|set(mw))
    print('  '+f"{'mes':<10}"+''.join(f"{m[5:]:>7}" for m in meses)+f"{'TOT':>8}")
    print('  '+f"{'baseline':<10}"+''.join(f"{mb.get(m,0):>+7.0f}" for m in meses)+f"{sum(mb.values()):>+8.0f}")
    print('  '+f"{mode:<10}"+''.join(f"{mw.get(m,0):>+7.0f}" for m in meses)+f"{sum(mw.values()):>+8.0f}")

print('\n  [MENSAL 5 ANOS] baseline (por ano)')
for y in sorted(base5['yr']): print(f'    {y}: {base5["yr"][y]:+.0f}')
print('\n  Custos 0.14% RT embutidos. HiLo Activator (SMA highs/lows). Bot rodando NAO tocado.')
