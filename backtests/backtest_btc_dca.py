"""
backtest_btc_dca.py
Extensao da curiosidade anterior (backtest_btc_denominado.py): "e se, alem dos
500 iniciais, eu colocasse mais 200 TODO MES em BTC, e rodasse a estrategia?"

Isto e o plano REAL do Rafa (aportes do freela) testado com dados historicos de
verdade, em vez da formula teorica usada na conversa sobre "quanto tempo ate 89k"
(aquela assumia uma taxa de retorno fixa; aqui usamos a sequencia real de trades
do motor v3 + precos reais do BTC).

4 VARIANTES (mesmo calendario de aportes: 500 no dia 1 + 200/mes):
  A) DCA HODL         — so comprar BTC todo mes e SEGURAR, sem operar. Controlo.
  B) BTC DCA COMPOUND  — aporta + a estrategia compoe SEM reset (a pergunta literal).
  C) BTC DCA RESET     — aporta + todo mes reseta o saldo de trading ao equivalente
                          em BTC do total ja aportado (protege o principal, filosofia
                          identica ao modelo EUR atual, so que denominado em BTC).
  D) EUR RESET c/ aportes — o modelo ATUAL (reset mensal, sem exposicao a BTC) mas
                          com o MESMO calendario de aportes de 200/mes. E a
                          comparacao mais justa: mesmo dinheiro entrando, com e
                          sem BTC.

Reporta drawdown de duas formas: peco-a-vale BRUTO (pode enganar quando ha
aportes, porque cada aporte levanta o pico legitimamente) e drawdown ABAIXO DO
TOTAL APORTADO ATE ALI (a pergunta que importa a um investidor real: "cheguei a
valer menos do que pus?").

Periodo 2021-01-01 .. fim do cache (~jul/2026), MAX_PER_SIDE=2 (config atual).
Custos 0.14% RT. Bot rodando NAO tocado.
Uso: python backtests/backtest_btc_dca.py
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
L_LOOKBACK=30; L_ADX_MAX=20; L_BUFFER=0.1; L_COOLDOWN=4; L_TIMEOUT=48; L_INVAL_N=5
MAX_PER_SIDE=2

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
def fetch_oos(sym, tf):
    return _fetch(sym, tf, datetime(2025,9,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc), '2025_2026oos')


def adx(df, period=14):
    h=df['h']; l=df['l']; c=df['c']; up=h.diff(); dn=-l.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    at=tr.ewm(alpha=1/period,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/at
    mdi=100*pd.Series(mdm,index=df.index).ewm(alpha=1/period,adjust=False).mean()/at
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period,adjust=False).mean()


def load():
    pdata={}; btcb=None
    for sym in PAIRS:
        dfd1=fetch_5y(sym,'1d'); dfd2=fetch_oos(sym,'1d')
        df11=fetch_5y(sym,'1h'); df12=fetch_oos(sym,'1h')
        dfd=pd.concat([dfd1,dfd2[dfd2.index>dfd1.index.max()]])
        df1=pd.concat([df11,df12[df12.index>df11.index.max()]])
        if len(df1)<300: continue
        ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
        slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
        dfd=dfd.copy(); dfd['regime']='NEUTRAL'
        dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
        dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
        dfd['slope_d']=slope
        c=df1['c']
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-c.shift(1)).abs(),(df1['l']-c.shift(1)).abs()],axis=1).max(axis=1)
        df1=df1.copy()
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


GEST={'bull':dict(be=0.40,trail=U_TRAIL,cd=U_COOLDOWN),
      'bear':dict(be=0.40,trail=B_TRAIL,cd=B_COOLDOWN),
      'lat' :dict(be=9.99,trail=0.0,cd=L_COOLDOWN)}


def gerar_sequencia_trades(arrs, start, end):
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
    positions={}; cooldown_until={s:-1 for s in A}
    seq=[]
    def close_all(k):
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]; pos=positions[sym]
            if np.isnan(c): c=pos['entry']
            e=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
            nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r
            seq.append((midx[k], nr)); del positions[sym]
    for k in range(len(midx)):
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
            else:
                if hi>=pos['cur']: nr=(e-pos['cur'])/risk-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk-fee_r; closed=True
                elif pos['tmode']=='atr':
                    if not pos['be_done'] and (e-c)/risk>=pos['be']*((e-pos['tp'])/risk): pos['cur']=e; pos['be_done']=True
                    if pos['be_done'] and pos['trail']>0:
                        cand=c+pos['trail']*a
                        if cand<pos['cur']: pos['cur']=cand
            if not closed and pos['manage']=='fixed':
                pos['age']+=1
                if pos['age']<=L_INVAL_N and not np.isnan(pos['bl']):
                    inside=(c<pos['bl']) if pos['side']=='LONG' else (c>pos['bl'])
                    if inside: nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
                if not closed and pos['age']>=L_TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                seq.append((midx[k], nr)); del positions[sym]; cooldown_until[sym]=k+pos['cd']
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sym in A:
            if sym in positions or k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            if side=='LONG' and n_long>=MAX_PER_SIDE: continue
            if side=='SHORT' and n_short>=MAX_PER_SIDE: continue
            e=d['entry'][k]; stop=d['sl'][k]; tp=d['tp'][k]; strat=d['strat'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp): continue
            risk_px=abs(e-stop)
            if risk_px<=0: continue
            g=GEST[strat]
            positions[sym]={'side':side,'entry':e,'cur':stop,'tp':tp,'be_done':False,'age':0,
                            'risk_px':risk_px,'fee_r':e*RT/risk_px,
                            'tmode':'fixed' if strat=='lat' else 'atr',
                            'be':g['be'],'trail':g['trail'],'manage':'fixed' if strat=='lat' else 'trail',
                            'cd':g['cd'],'strat':strat,'bl':d['bl'][k]}
            if side=='LONG': n_long+=1
            else: n_short+=1
    close_all(len(midx)-1)
    seq.sort(key=lambda x: x[0])
    return seq


# ── Aportes mensais + 4 formas de contabilidade ───────────────────────────────

def _meses_no_periodo(start, end):
    """Lista de (ano,mes) do 1o ao ultimo mes do periodo, para agendar aportes."""
    meses=[]; y,m = start.year, start.month
    while (y,m) <= (end.year, end.month):
        meses.append((y,m)); m+=1
        if m>12: m=1; y+=1
    return meses


def dca_hodl(btc_price, start, end, base_eur, aporte_eur):
    """So compra BTC no dia 1 e todo mes, sem operar. Controlo."""
    meses = _meses_no_periodo(start, end)
    btc_total=0.0; aportado=0.0; curva=[]
    for i,(y,m) in enumerate(meses):
        ts = pd.Timestamp(year=y, month=m, day=1, tz='UTC')
        p = btc_price.asof(ts)
        if np.isnan(p): continue
        add_eur = base_eur if i==0 else aporte_eur
        btc_total += add_eur/p; aportado += add_eur
        curva.append((ts, btc_total*p, aportado))
    p_fim = btc_price.iloc[-1]
    return btc_total*p_fim, aportado, curva


def btc_dca_compound(seq, btc_price, start, end, base_eur, aporte_eur, risk_pct, reset_to_principal=False):
    """Aporta 500 no dia 1 + aporte_eur todo mes (tudo convertido em BTC ao preco
    do momento), soma ao bankroll de trading. Trades compoem em cima do bankroll.
    reset_to_principal=True: no INICIO de cada mes, ANTES do aporte, reseta o
    bankroll ao equivalente BTC do total ja aportado (protege ganhos/perdas de
    trading acumulados, mantendo so o principal + o novo aporte — filosofia
    identica ao modelo EUR atual, denominado em BTC)."""
    meses = _meses_no_periodo(start, end)
    p0 = btc_price.asof(pd.Timestamp(year=meses[0][0], month=meses[0][1], day=1, tz='UTC'))
    bankroll_btc = 0.0; aportado = 0.0
    trades_por_mes = {}
    for ts, nr in seq:
        mk = (ts.year, ts.month)
        trades_por_mes.setdefault(mk, []).append((ts, nr))

    curva=[]   # (timestamp, valor_eur, aportado_ate_aqui)
    for i,(y,m) in enumerate(meses):
        ts_mes = pd.Timestamp(year=y, month=m, day=1, tz='UTC')
        p_mes = btc_price.asof(ts_mes)
        if np.isnan(p_mes): p_mes = p0
        if reset_to_principal and i>0:
            bankroll_btc = aportado / p_mes    # reseta ao valor investido, nao ao lucro
        add_eur = base_eur if i==0 else aporte_eur
        bankroll_btc += add_eur/p_mes
        aportado += add_eur
        for ts, nr in trades_por_mes.get((y,m), []):
            bankroll_btc += nr*(bankroll_btc*risk_pct/100.0)
            p_now = btc_price.asof(ts)
            curva.append((ts, bankroll_btc*(p_now if not np.isnan(p_now) else p_mes), aportado))
    p_fim = btc_price.iloc[-1]
    final_eur = bankroll_btc*p_fim
    return final_eur, aportado, curva


def eur_reset_com_aportes(seq, start, end, base_eur, aporte_eur, risk_pct):
    """Modelo ATUAL (reset mensal em EUR), mas a base cresce com os aportes:
    mes 1 reseta a 500, mes 2 a 700, mes 3 a 900... (principal acumulado)."""
    meses = _meses_no_periodo(start, end)
    trades_por_mes = {}
    for ts, nr in seq:
        mk = (ts.year, ts.month)
        trades_por_mes.setdefault(mk, []).append((ts, nr))
    aportado=0.0; valor_total=0.0; curva=[]
    for i,(y,m) in enumerate(meses):
        add_eur = base_eur if i==0 else aporte_eur
        aportado += add_eur
        balance = aportado    # reseta ao total aportado ate aqui (protege o principal)
        for ts, nr in trades_por_mes.get((y,m), []):
            balance += nr*(balance*risk_pct/100.0)
            # valor TOTAL da conta = saldo do mes corrente (ja inclui o principal
            # acumulado) + lucro RETIDO de meses anteriores (valor_total). O bug
            # anterior usava so "balance-aportado" (so o lucro deste mes),
            # esquecendo o principal inteiro -- dava DD absurdo (-113%).
            curva.append((ts, balance + valor_total, aportado))
        valor_total += (balance - aportado)   # lucro/prejuizo do mes fica retido (nao reseta)
    return aportado + valor_total, aportado, curva


def max_dd_sobre_aportado(curva):
    """DD que importa a um investidor real: cheguei a valer MENOS do que pus?"""
    pior_pct = 0.0; pior_abs = 0.0
    for ts, valor, aportado in curva:
        if aportado<=0: continue
        diff = (valor-aportado)/aportado*100
        if diff < pior_pct: pior_pct = diff; pior_abs = valor-aportado
    return pior_pct, pior_abs


print('A carregar dados (cache, periodo continuo 2021..2026)...')
pdata, btcb = load()
arrs = prep(pdata, btcb)
print(f'  {len(arrs)} pares OK')

S=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc))
seq = gerar_sequencia_trades(arrs, *S)
btc_df = arrs['BTC/USDT:USDT']
btc_price = btc_df['c'][(btc_df.index>=S[0])&(btc_df.index<=S[1])]
n_meses = len(_meses_no_periodo(*S))
print(f'  {len(seq)} trades | {n_meses} meses de aportes (500 no dia 1 + 200/mes)')

BASE=500.0; APORTE=200.0; RISK=2.0   # risco 2% = o risco AO VIVO do Rafa

print('\n'+'='*104)
print(f'  500 no dia 1 + {APORTE:.0f}/mes em BTC, {n_meses} meses, risco {RISK:.0f}% | MAX_PER_SIDE={MAX_PER_SIDE}')
print('='*104)

total_aportado = BASE + APORTE*(n_meses-1)
print(f'\n  Total aportado ao longo do periodo: {total_aportado:,.0f} EUR ({BASE:.0f} inicial + {n_meses-1} x {APORTE:.0f})')

a_final, a_aportado, a_curva = dca_hodl(btc_price, *S, BASE, APORTE)
b_final, b_aportado, b_curva = btc_dca_compound(seq, btc_price, *S, BASE, APORTE, RISK, reset_to_principal=False)
c_final, c_aportado, c_curva = btc_dca_compound(seq, btc_price, *S, BASE, APORTE, RISK, reset_to_principal=True)
d_final, d_aportado, d_curva = eur_reset_com_aportes(seq, *S, BASE, APORTE, RISK)

print('\n  '+f"{'variante':<34}{'final':>12}{'aportado':>11}{'lucro':>11}{'multiplo':>10}")
print('  '+'-'*80)
for nome, final, aport in [
    ('A) DCA HODL (so BTC, sem operar)', a_final, a_aportado),
    ('B) BTC DCA + COMPOUND (sem reset)', b_final, b_aportado),
    ('C) BTC DCA + RESET ao principal',   c_final, c_aportado),
    ('D) EUR RESET c/ aportes (modelo atual)', d_final, d_aportado),
]:
    print('  '+f"{nome:<34}{final:>12,.0f}{aport:>11,.0f}{final-aport:>+11,.0f}{final/aport:>9.2f}x")

print('\n  DRAWDOWN — duas leituras (bruto engana quando ha aportes a levantar o pico)')
print('  '+f"{'variante':<34}{'DD abaixo do aportado':>24}{'pior momento':>16}")
for nome, curva in [('B) BTC DCA + COMPOUND', b_curva), ('C) BTC DCA + RESET', c_curva), ('D) EUR RESET c/ aportes', d_curva)]:
    pct, abs_ = max_dd_sobre_aportado(curva)
    print('  '+f"{nome:<34}{pct:>+23.1f}%{abs_:>+16,.0f}")

print('\n  Custos 0.14% RT embutidos. Precos BTC/USDT como proxy de BTC/EUR. Bot rodando NAO tocado.')
