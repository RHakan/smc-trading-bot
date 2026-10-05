"""
backtest_btc_denominado.py
Curiosidade do Rafa: "e se os 500 estivessem em BTC (nao EUR/USDC) e a estrategia
rodasse por cima, ha 5 anos atras?"

A ideia central: o MOTOR de sinais (lateral/bull/bear, filtros, invalidacao,
BE/trailing) e IDENTICO ao bot atual — R por trade e um RACIO (ganho/risco),
nao depende de que MOEDA o saldo esta denominado. O que muda e so a CONTABILIDADE
do saldo: em vez de EUR com reset mensal a 500, o saldo vive em BTC.

4 VARIANTES (mesma sequencia de trades em todas, so muda a contabilidade):
  1) BASELINE      — o modelo atual: 500 EUR, reset mensal, sem exposicao a BTC.
  2) BTC HODL       — comprar 500 EUR de BTC ha 5 anos e SEGURAR, sem operar.
                      Isola quanto vem so da valorizacao do BTC.
  3) BTC COMPOUND   — 500 EUR convertidos em BTC no dia 1, e o saldo (em BTC)
                      COMPOE trade a trade, sem reset. O que "deixar rodar" da.
  4) BTC RESET      — como o (3), mas reseta TODO MES para o equivalente em BTC
                      de 500 EUR NAQUELE mes (== o modelo de capital do Rafa,
                      so que denominado em BTC em vez de EUR).

Testado em varios niveis de risco/trade (0.5/1/2/4%) na variante COMPOUND, porque
com juros compostos o risco% NAO escala linearmente no resultado (risco alto
compondo pode ate destruir um edge positivo por "volatility drag" — o mesmo
principio do Kelly criterion). Isto e relevante porque o risco AO VIVO do Rafa
e 2%.

Periodo: 2021-01-01 .. fim do cache OOS (~jul/2026), ~5.5 anos. MAX_PER_SIDE=2
(config atual do bot). Custos 0.14% RT. Bot rodando NAO tocado.
Uso: python backtests/backtest_btc_denominado.py
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
MAX_PER_SIDE=2   # config ATUAL do bot (adotada 17/07/2026)

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
    """Junta os dois caches (5 anos + OOS) num unico periodo continuo 2021..2026."""
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
    """Corre o motor v3 completo (MAX_PER_SIDE=2, sem denominacao de saldo — so
    para extrair a sequencia real de trades) e devolve [(timestamp_fecho, net_r), ...]
    em ordem cronologica. R e um racio: nao depende de que moeda o saldo esta."""
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
    seq=[]   # (timestamp, net_r)

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


# ── Contabilidade: aplica a MESMA sequencia de trades sob 4 denominacoes ──────

def eur_reset_mensal(seq, base=500.0, risk_pct=1.0):
    """Baseline: EUR, reset mensal (o modelo atual)."""
    balance=base; cur_month=None; monthly={}
    for ts, nr in seq:
        mk=ts.strftime('%Y-%m')
        if cur_month is None: cur_month=mk
        if mk!=cur_month:
            monthly[cur_month]=balance-base; balance=base; cur_month=mk
        balance += nr*(balance*risk_pct/100.0)
    monthly[cur_month]=balance-base
    return sum(monthly.values()), monthly


def btc_compound(seq, btc_price, base_eur=500.0, risk_pct=1.0, reset_monthly=False):
    """Saldo em BTC. compound=deixa correr; reset_monthly=reseta todo mes ao
    equivalente EUR->BTC daquele mes (usa o preco do BTC no 1o candle do mes)."""
    p0 = btc_price.iloc[0]
    bankroll_btc = base_eur / p0
    cur_month=None; monthly_eur={}
    start_bankroll = bankroll_btc
    for ts, nr in seq:
        mk = ts.strftime('%Y-%m')
        if cur_month is None: cur_month=mk
        if reset_monthly and mk!=cur_month:
            p_now = btc_price.asof(ts)
            eur_now = bankroll_btc*p_now if not np.isnan(p_now) else None
            monthly_eur[cur_month] = (eur_now - base_eur) if eur_now is not None else 0.0
            bankroll_btc = base_eur / (p_now if not np.isnan(p_now) else p0)
            cur_month = mk
        bankroll_btc += nr*(bankroll_btc*risk_pct/100.0)
    p_end = btc_price.iloc[-1]
    final_eur = bankroll_btc*p_end
    if reset_monthly:
        monthly_eur[cur_month] = final_eur - base_eur
    return {
        'bankroll_btc_final': bankroll_btc, 'bankroll_btc_inicial': start_bankroll,
        'p_inicio': p0, 'p_fim': p_end, 'final_eur': final_eur,
        'lucro_eur': final_eur-base_eur, 'multiplo_btc': bankroll_btc/start_bankroll,
        'monthly_eur': monthly_eur,
    }


print('A carregar dados (cache, periodo continuo 2021..2026)...')
pdata, btcb = load()
arrs = prep(pdata, btcb)
print(f'  {len(arrs)} pares OK')

S=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc))
print(f'  periodo: {S[0].date()} .. {S[1].date()}  (~{(S[1]-S[0]).days/365.25:.1f} anos)')

seq = gerar_sequencia_trades(arrs, *S)
print(f'  {len(seq)} trades gerados pelo motor v3 (MAX_PER_SIDE={MAX_PER_SIDE})')

btc_df = arrs['BTC/USDT:USDT']
btc_price = btc_df['c'][(btc_df.index>=S[0])&(btc_df.index<=S[1])]
p0, p1 = btc_price.iloc[0], btc_price.iloc[-1]
print(f'  BTC: {p0:,.0f} -> {p1:,.0f} USDT  ({(p1/p0-1)*100:+.0f}% no periodo)')

print('\n'+'='*98)
print('  E SE OS 500 ESTIVESSEM EM BTC? | mesma sequencia de trades, contabilidade diferente')
print('='*98)

RISK=1.0
base_eur, mb = eur_reset_mensal(seq, 500.0, RISK)
hodl_eur = 500.0 * (p1/p0)
comp = btc_compound(seq, btc_price, 500.0, RISK, reset_monthly=False)
rst  = btc_compound(seq, btc_price, 500.0, RISK, reset_monthly=True)

print(f"\n  Risco por trade usado abaixo: {RISK:.1f}% (referencia do projeto — o teu risco AO VIVO e 2%,")
print(f"  ver sweep de risco mais abaixo, o resultado NAO escala linear sob juros compostos)")
print('\n  '+f"{'variante':<32}{'final (EUR eq.)':>18}{'lucro':>12}{'multiplo':>10}")
print('  '+'-'*74)
print('  '+f"{'1) BASELINE (EUR, reset/mes)':<32}{500+base_eur:>18,.0f}{base_eur:>+12,.0f}{(500+base_eur)/500:>9.2f}x")
print('  '+f"{'2) BTC HODL (so comprar e segurar)':<32}{hodl_eur:>18,.0f}{hodl_eur-500:>+12,.0f}{hodl_eur/500:>9.2f}x")
print('  '+f"{'3) BTC COMPOUND (sem reset)':<32}{comp['final_eur']:>18,.0f}{comp['lucro_eur']:>+12,.0f}{comp['final_eur']/500:>9.2f}x")
print('  '+f"{'4) BTC RESET mensal':<32}{500+sum(rst['monthly_eur'].values()):>18,.0f}{sum(rst['monthly_eur'].values()):>+12,.0f}{(500+sum(rst['monthly_eur'].values()))/500:>9.2f}x")

print(f"\n  DECOMPOSICAO da variante (3) BTC COMPOUND:")
print(f"    BTC comprado no dia 1  : {comp['bankroll_btc_inicial']:.6f} BTC (a {comp['p_inicio']:,.0f})")
print(f"    BTC no fim             : {comp['bankroll_btc_final']:.6f} BTC ({comp['multiplo_btc']:.2f}x em BTC — isto e o EDGE puro)")
print(f"    Convertido a hoje ({comp['p_fim']:,.0f}): {comp['final_eur']:,.0f} EUR")
print(f"    Quanto veio SO da valorizacao do BTC (hodl): {hodl_eur-500:+,.0f}")
print(f"    Quanto veio SO do edge (compound_eur - hodl_eur): {comp['final_eur']-hodl_eur:+,.0f}")

print(f"\n  [MENSAL] BTC RESET — ultimos 12 meses")
meses = sorted(rst['monthly_eur'].keys())[-12:]
for m in meses:
    print(f"    {m}: {rst['monthly_eur'][m]:+8.0f}")

print('\n'+'='*98)
print('  SWEEP DE RISCO — variante COMPOUND (sem reset) NAO escala linear com risco%')
print('  (juros compostos: risco alto por trade pode ate destruir edge positivo — volatility drag)')
print('='*98)
print('  '+f"{'risco/trade':<14}{'final EUR':>14}{'multiplo':>10}{'multiplo BTC':>14}")
for r in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
    c = btc_compound(seq, btc_price, 500.0, r, reset_monthly=False)
    print('  '+f"{str(r)+'%':<14}{c['final_eur']:>14,.0f}{c['final_eur']/500:>9.2f}x{c['multiplo_btc']:>13.2f}x")

print('\n  Custos 0.14% RT embutidos. Precos BTC/USDT como proxy de BTC/EUR. Bot rodando NAO tocado.')
