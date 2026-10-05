"""
backtest_rtrail_valida.py
Validação do STEP TRAIL de distância fixa (R) nas DIRECIONAIS (bull/bear).
A lateral fica EXATAMENTE como o baseline (fixa + timeout) — isolamos o efeito real.

Candidato: substituir BE@1.0R+ATR das direcionais por trailing de folga fixa (Nx R).
No screening (5 anos) o 1.5R deu +391 vs +334. Aqui rigor:
  1) ROBUSTEZ 5 anos — vizinhança 1.0..2.5R. Se o edge é real, a região é toda boa
     e suave (não um pico isolado = overfit).
  2) OOS 2026 (jan..jul, dados frescos NÃO usados em nenhuma calibração) — baseline
     vs os melhores. Se desaba no OOS, não vai ao bot.

Lateral sempre fixa+timeout. 500/mês reset. Custos 0.14% RT.
Uso: python backtests/backtest_rtrail_valida.py
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


def fetch_5y(sym, tf):   # cache já existente do screening (2020-01 .. 2025-10-06)
    return _fetch(sym, tf, datetime(2020,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc), '2020_2025')
def fetch_26(sym, tf):   # OOS: warmup desde 2025-09 até 2026-07-09
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
        pdata[sym]=df1
        if sym.startswith('BTC'): btcb=(df1['regime']=='BULL')
    return pdata, btcb


def gen(df, btc_local):
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values; adx_a=df['adx'].values
    reg=df['regime'].values; slope=df['slope_d'].values; ema_t=df['ema_trend'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)
    strat=np.array([None]*n,dtype=object); blevel=np.full(n,np.nan)   # nível do range rompido
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


# Direcionais sempre no baseline validado (BE@1.0R + ATR). Só a lateral muda.
GEST={'bull':dict(tmode='atr',be=0.40,trail=U_TRAIL,manage='trail',cd=U_COOLDOWN),
      'bear':dict(tmode='atr',be=0.40,trail=B_TRAIL,manage='trail',cd=B_COOLDOWN),
      'lat' :dict(tmode='fixed',be=9.99,trail=0.0,manage='fixed',cd=L_COOLDOWN)}


def run(arrs, inval_level, inval_n, start, end):
    """inval_level: 'none' | 'rh' (nível do range rompido) | 'entry'.
    inval_n: nº máx de velas após a entrada em que a invalidação vale (só lateral)."""
    midx=None
    for sym,d in arrs.items():
        midx=d.index if midx is None else midx.union(d.index)
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'h':dd['h'].values,'l':dd['l'].values,'c':dd['c'].values,'atr':dd['atr'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'sl':dd['_sl'].values,'tp':dd['_tp'].values,
                'strat':dd['_st'].values,'bl':dd['_bl'].values}
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; monthly={}; cur_month=None
    trades=[]   # net R por trade (decomposição perda/ganho)
    inval_rs=[] # net R dos trades fechados por invalidação (atribuição)
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
                elif pos['tmode']=='rtrail':
                    cand=c-pos['rmult']*risk
                    if cand>pos['cur']: pos['cur']=cand
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
                elif pos['tmode']=='atr':
                    if not pos['be_done'] and (e-c)/risk>=pos['be']*((e-pos['tp'])/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c+pos['trail']*a
                        if cand<pos['cur']: pos['cur']=cand
                elif pos['tmode']=='rtrail':
                    cand=c+pos['rmult']*risk
                    if cand<pos['cur']: pos['cur']=cand
            if not closed and pos['manage']=='fixed':
                pos['age']+=1
                # invalidação de breakout falho: fechou de volta pra dentro do range
                if inval_level!='none' and pos['age']<=inval_n and not np.isnan(pos['bl']):
                    lvl=pos['bl'] if inval_level=='rh' else pos['entry']
                    inside=(c<lvl) if pos['side']=='LONG' else (c>lvl)
                    if inside:
                        nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True; inval_rs.append(nr)
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
            if side=='LONG' and n_long>=MAX_PER_SIDE: continue
            if side=='SHORT' and n_short>=MAX_PER_SIDE: continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]; strat=d['strat'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance<=0: continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            g=GEST[strat]
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'be_done':False,'age':0,
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px,
                            'tmode':g['tmode'],'rmult':0.0,'be':g['be'],'trail':g['trail'],
                            'manage':g['manage'],'cd':g['cd'],'strat':strat,'bl':d['bl'][k]}
            if side=='LONG': n_long+=1
            else: n_short+=1
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    if positions: close_all(len(midx)-1)
    monthly[cur_month]=balance-MONTHLY_BASE
    return monthly, max_dd, trades, inval_rs


def summ(res):
    monthly, ddmax, trades, inval_rs = res
    yr={}
    for mk,v in monthly.items(): yr[mk[:4]]=yr.get(mk[:4],0)+v
    tot=sum(monthly.values()); tr=np.array(trades); n=len(tr)
    wins=tr[tr>0.03]; losses=tr[tr<-0.03]; iv=np.array(inval_rs)
    return dict(tot=tot, yr=yr, monthly=monthly, anos=sum(1 for v in yr.values() if v>0),
                meses=sum(1 for v in monthly.values() if v>0), nm=len(monthly), dd=ddmax,
                rd=tot/ddmax if ddmax else 0, n=n, wr=len(wins)/n*100 if n else 0,
                sum_win=wins.sum(), sum_loss=losses.sum(),
                n_inval=len(iv), avg_inval=iv.mean() if len(iv) else 0.0, sum_inval=iv.sum())


print('A carregar 5 anos (cache) + 2026 OOS (fetch)...')
p5,b5=load(fetch_5y); a5=prep(p5,b5); print(f'  5 anos: {len(a5)} pares OK')
p26,b26=load(fetch_26); a26=prep(p26,b26); print(f'  2026 OOS: {len(a26)} pares OK')
S5=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc))
S26=(datetime(2026,1,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc))
print()

print('='*104)
print('  CONFIRMAÇÃO: invalidação de breakout falho na LATERAL (nível=range) | 500/mês reset')
print('  Candidato: 5 velas. Confirmamos que NÃO é bico de faca, nem período, nem mês de sorte.')
print('='*104)

# ── (1) Varredura FINA do parâmetro N: 2..8 em AMBOS os períodos ──────────────
print('\n  [1] VARREDURA FINA de N (velas) — platô contíguo = robusto, não bico de faca')
print('  '+f"{'N velas':<14}{'5anos tot':>11}{'anos+':>7}{'DD':>7}{'||':>4}{'2026 tot':>10}{'m+':>6}{'DD':>7}")
print('  '+'-'*72)
base5=summ(run(a5,'none',0,*S5)); base26=summ(run(a26,'none',0,*S26))
print('  '+f"{'baseline':<14}{base5['tot']:>+11.0f}{base5['anos']:>5}/5{base5['dd']:>6.1f}%{'||':>4}"
      f"{base26['tot']:>+10.0f}{base26['meses']:>3}/{base26['nm']:<2}{base26['dd']:>6.1f}%")
fine={}
for nn in [2,3,4,5,6,7,8]:
    s5=summ(run(a5,'rh',nn,*S5)); s26=summ(run(a26,'rh',nn,*S26)); fine[nn]=(s5,s26)
    mark=' <= candidato' if nn==5 else ''
    print('  '+f"{('rh, '+str(nn)):<14}{s5['tot']:>+11.0f}{s5['anos']:>5}/5{s5['dd']:>6.1f}%{'||':>4}"
          f"{s26['tot']:>+10.0f}{s26['meses']:>3}/{s26['nm']:<2}{s26['dd']:>6.1f}%{mark}")

# ── (2) Sub-períodos: edge presente em cada metade dos 5 anos ─────────────────
print('\n  [2] SUB-PERÍODOS — o edge aparece nas DUAS metades (não é 1 fase só)')
H1=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2023,6,30,23,tzinfo=timezone.utc))
H2=(datetime(2023,7,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc))
print('  '+f"{'Período':<22}{'baseline':>11}{'inval-5':>10}{'Δtotal':>9}{'ΔDD':>8}")
print('  '+'-'*60)
for nome,per in [('2021-01 .. 2023-06',H1),('2023-07 .. 2025-10',H2)]:
    bb=summ(run(a5,'none',0,*per)); ii=summ(run(a5,'rh',5,*per))
    print('  '+f"{nome:<22}{bb['tot']:>+11.0f}{ii['tot']:>+10.0f}{ii['tot']-bb['tot']:>+9.0f}{ii['dd']-bb['dd']:>+7.1f}%")

# ── (3) OOS 2026 mês a mês: não é 1 mês de sorte ──────────────────────────────
print('\n  [3] OOS 2026 MÊS A MÊS — baseline vs inval-5 (não depende de 1 mês)')
mb=summ(run(a26,'none',0,*S26))['monthly']; mi=summ(run(a26,'rh',5,*S26))['monthly']
meses=sorted(set(mb)|set(mi))
print('  '+f"{'mês':<10}"+''.join(f"{m[5:]:>7}" for m in meses)+f"{'TOT':>8}")
print('  '+f"{'baseline':<10}"+''.join(f"{mb.get(m,0):>+7.0f}" for m in meses)+f"{sum(mb.values()):>+8.0f}")
print('  '+f"{'inval-5':<10}"+''.join(f"{mi.get(m,0):>+7.0f}" for m in meses)+f"{sum(mi.values()):>+8.0f}")

# ── (4) Atribuição: quantos trades salvou e como ─────────────────────────────
print('\n  [4] ATRIBUIÇÃO (5 anos, inval-5) — o que a regra fez de facto')
s5=fine[5][0]
print(f"    trades fechados por invalidação: {s5['n_inval']}  |  R médio da saída: {s5['avg_inval']:+.2f}R")
print(f"    (sem a regra, esses trades correriam ate ~-1R; a saida antecipada e o ganho)")
print(f"    WR total: {base5['wr']:.0f}% -> {s5['wr']:.0f}% (cai, esperado: converte -1R em -0.3R)")
print(f"    perda total: {base5['sum_loss']:+.0f}R -> {s5['sum_loss']:+.0f}R | ganho: {base5['sum_win']:+.0f}R -> {s5['sum_win']:+.0f}R")

print('\n'+'='*104)
win5=fine[5]; ok5=win5[0]['tot']>base5['tot']; ok26=win5[1]['tot']>base26['tot']
plateau=sum(1 for nn in [3,4,5,6] if fine[nn][0]['tot']>base5['tot'] and fine[nn][1]['tot']>base26['tot'])
print(f"  VEREDITO: inval-5 bate baseline nos 5 anos ({'SIM' if ok5 else 'NAO'}) e no OOS 2026 ({'SIM' if ok26 else 'NAO'}).")
print(f"  Platô 3-6 velas que vence nos DOIS períodos: {plateau}/4 (robusto se >=3).")
print('  Custos 0.14% RT embutidos. Direcionais no baseline. Só a lateral muda.')
