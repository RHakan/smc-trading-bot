"""
backtest_supertrend_hma.py
Rafa trouxe um indicador do TradingView ("Smart Trend Predictor X v4.0 Hybrid").
A maior parte dele repete coisas ja reprovadas neste projeto (RR forcado,
divergencias RSI = reversao, score de confluencia com 9 filtros empilhados).

A UNICA peca mecanicamente nova: o motor de tendencia central — HMA(14) + banda
ATR(14)*2.0, ratchet tipo SuperTrend/Chandelier. Vira de direcao quando o
FECHAMENTO cruza a banda (nao toque intrabar — fidelidade ao indicador original,
diferente do resto do bot que usa toque de high/low).

TESTE ISOLADO (Rafa aprovou): so este motor, SEM score de confluencia, SEM
divergencia/dip-pump (reversao ja reprovada neste projeto), SEM alvo forcado
(RR forcado ja reprovado — backtest_lateral_rr.py). A saida e a NATURAL do
mecanismo: fica na posicao ate a propria tendencia virar (o trail e o stop
E o alvo ao mesmo tempo, por construcao).

2 variantes:
  RAW      — sem filtro de regime, exatamente como o indicador roda sozinho.
  FILTRADO — so LONG se o regime diario (EMA20+slope, o Decisor do bot) diz
             BULL, so SHORT se diz BEAR. Testa se precisa do filtro pra funcionar
             (como aconteceu com quase tudo neste projeto).

R por trade = distancia entry -> trail NO MOMENTO da entrada (a referencia de
risco natural do mecanismo, mesmo que o trail depois se mova a favor).

5 anos + OOS 2026, 500/mes reset, risco 1% (referencia do projeto), MAX_PER_SIDE=3.
Custos 0.14% RT. Comparado contra o baseline v3 no MESMO motor de portfolio.
Bot rodando NAO tocado.
Uso: python backtests/backtest_supertrend_hma.py
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
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH

# ── Parametros do indicador (defaults originais do Pine) ──────────────────────
MA_LEN=14; ATR_LEN=14; ATR_MULT=2.0; MIN_BARS=3

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


# ── Indicadores: HMA, ATR, trail SuperTrend-like ──────────────────────────────

def _wma(s: pd.Series, length: int) -> pd.Series:
    w = np.arange(1, length+1)
    return s.rolling(length).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)


def hma(s: pd.Series, length: int) -> pd.Series:
    half = max(1, length//2)
    sq = max(1, int(round(np.sqrt(length))))
    raw = 2*_wma(s, half) - _wma(s, length)
    return _wma(raw, sq)


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h,l,c = df['h'], df['l'], df['c']
    tr = pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(com=period-1, adjust=False).mean()


def supertrend_hma(smooth: np.ndarray, atr: np.ndarray, close: np.ndarray):
    """Replica EXATA da logica Pine (linhas 86-97 do indicador):
    var trail=0.0, var dir=1; ratchet — so move a favor, vira no cruzamento
    do FECHAMENTO. Devolve (dirn, trail) arrays."""
    n = len(close)
    trail = np.full(n, np.nan)
    dirn = np.ones(n, dtype=int)
    cur_trail = 0.0; cur_dir = 1; started = False
    for i in range(n):
        if np.isnan(smooth[i]) or np.isnan(atr[i]):
            dirn[i] = cur_dir
            continue
        band = atr[i]*ATR_MULT
        if not started:
            cur_trail = smooth[i]-band if cur_dir==1 else smooth[i]+band
            started = True
        if cur_dir == 1:
            cand = smooth[i]-band
            cur_trail = max(cur_trail, cand)
            if close[i] < cur_trail:
                cur_dir = -1; cur_trail = smooth[i]+band
        else:
            cand = smooth[i]+band
            cur_trail = min(cur_trail, cand)
            if close[i] > cur_trail:
                cur_dir = 1; cur_trail = smooth[i]-band
        trail[i] = cur_trail; dirn[i] = cur_dir
    return dirn, trail


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
    pdata={}
    for sym in PAIRS:
        dfd1=fetch_5y(sym,'1d'); dfd2=fetch_26(sym,'1d')
        df11=fetch_5y(sym,'1h'); df12=fetch_26(sym,'1h')
        dfd=pd.concat([dfd1,dfd2[dfd2.index>dfd1.index.max()]])
        df1=pd.concat([df11,df12[df12.index>df11.index.max()]])
        if len(df1)<300: continue
        ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
        slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
        dfd=dfd.copy(); dfd['regime']='NEUTRAL'
        dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
        dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
        df1=df1.copy()
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')

        smooth = hma(df1['c'], MA_LEN).values
        atr_v  = calc_atr(df1, ATR_LEN).values
        dirn, trail = supertrend_hma(smooth, atr_v, df1['c'].values)
        df1['dir']=dirn; df1['trail']=trail
        pdata[sym]=df1
    return pdata


def gen(df):
    n=len(df); c=df['c'].values; dirn=df['dir'].values; trail=df['trail'].values; reg=df['regime'].values
    side=np.array([None]*n,dtype=object); entry=np.full(n,np.nan); risk_ref=np.full(n,np.nan)
    last_flip = -10_000
    for i in range(1, n):
        if np.isnan(trail[i]) or np.isnan(trail[i-1]): continue
        flip_up   = dirn[i]==1  and dirn[i-1]==-1
        flip_down = dirn[i]==-1 and dirn[i-1]==1
        if not (flip_up or flip_down): continue
        if i - last_flip < MIN_BARS: continue
        last_flip = i
        if flip_up:
            side[i]='LONG'
        else:
            side[i]='SHORT'
        entry[i]=c[i]; risk_ref[i]=abs(c[i]-trail[i])
    return side, entry, risk_ref


def prep(pdata):
    arrs={}
    for sym,df in pdata.items():
        s,e,rr=gen(df)
        arrs[sym]=df.assign(_s=s,_e=e,_rr=rr)
    return arrs


def run(arrs, start, end, filtrado=False):
    midx=None
    for sym,d in arrs.items():
        midx=d.index if midx is None else midx.union(d.index)
    midx=midx[(midx>=start)&(midx<=end)]
    A={}
    for sym,d in arrs.items():
        dd=d.reindex(midx)
        A[sym]={'c':dd['c'].values,'dir':dd['dir'].values,'trail':dd['trail'].values,
                'side':dd['_s'].values,'entry':dd['_e'].values,'rr':dd['_rr'].values,
                'regime':dd['regime'].values}
    balance=MONTHLY_BASE; peak=MONTHLY_BASE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; monthly={}; cur_month=None; trades=[]

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
            pos=positions[sym]; risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']
            closed=False; nr=0.0
            # saida NATURAL do mecanismo: a propria tendencia vira (dir flip).
            # Fechamento-a-fechamento, fiel ao indicador original (nao toque intrabar).
            cur_dir = d['dir'][k]
            if (pos['side']=='LONG' and cur_dir==-1) or (pos['side']=='SHORT' and cur_dir==1):
                nr=((c-e) if pos['side']=='LONG' else (e-c))/risk-fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']; trades.append(nr); del positions[sym]; cooldown_until[sym]=k+2
        n_long=sum(1 for p in positions.values() if p['side']=='LONG')
        n_short=sum(1 for p in positions.values() if p['side']=='SHORT')
        for sym in A:
            if sym in positions or k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            if filtrado:
                reg = d['regime'][k]
                if side=='LONG'  and reg!='BULL': continue
                if side=='SHORT' and reg!='BEAR': continue
            if side=='LONG' and n_long>=MAX_PER_SIDE: continue
            if side=='SHORT' and n_short>=MAX_PER_SIDE: continue
            e=d['entry'][k]; risk_px=d['rr'][k]
            if np.isnan(e) or np.isnan(risk_px) or risk_px<=0 or balance<=0: continue
            positions[sym]={'side':side,'entry':e,'risk_px':risk_px,
                            'risk_usd':balance*(RISK_PCT/100.0),'fee_r':e*RT/risk_px}
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
    wins=tr[tr>0.03]
    return dict(tot=tot, yr=yr, monthly=monthly, anos=sum(1 for v in yr.values() if v>0),
                meses=sum(1 for v in monthly.values() if v>0), nm=len(monthly), dd=ddmax,
                n=n, wr=len(wins)/n*100 if n else 0)


print('A carregar dados + calculando HMA/ATR/trail (cache OHLC, indicadores computados agora)...')
p5 = load()
S5=(datetime(2021,1,1,tzinfo=timezone.utc), datetime(2025,10,6,tzinfo=timezone.utc))
a5 = prep(p5)
print(f'  5 anos: {len(a5)} pares OK')

p26 = load()   # mesma funcao recalcula tudo (cache OHLC reaproveitado, so o indicador roda de novo)
S26=(datetime(2026,1,1,tzinfo=timezone.utc), datetime(2026,7,9,tzinfo=timezone.utc))
a26 = prep(p26)
print(f'  2026 OOS: {len(a26)} pares OK')

print('\n'+'='*98)
print('  SuperTrend/HMA isolado (motor central do indicador TV) | 500/mes | risco 1% | custos 0.14% RT')
print('='*98)
print('\n  '+f"{'config':<22}{'5anos':>9}{'anos+':>7}{'DD':>7}{'n':>6}{'WR':>6}{'||':>4}{'2026':>8}{'m+':>6}{'DD':>7}{'n':>5}")
print('  '+'-'*88)

res={}
for nome, filt in [('RAW (sem filtro)', False), ('FILTRADO (regime v3)', True)]:
    s5=summ(run(a5, *S5, filtrado=filt)); s26=summ(run(a26, *S26, filtrado=filt))
    res[nome]=(s5,s26)
    print('  '+f"{nome:<22}{s5['tot']:>+9.0f}{s5['anos']:>5}/5{s5['dd']:>6.1f}%{s5['n']:>6}{s5['wr']:>5.0f}%{'||':>4}"
          f"{s26['tot']:>+8.0f}{s26['meses']:>3}/{s26['nm']:<2}{s26['dd']:>6.1f}%{s26['n']:>5}")

print('\n'+'='*98)
print('  Para referencia, o baseline v3 (mesmo motor 500/mes/risco 1%) nesta janela: +347 (5 anos) / +60 (OOS)')
print('  Veredito: candidato so vale a pena se bater os DOIS numeros nos DOIS periodos.')
for nome,(s5,s26) in res.items():
    ok = s5['tot']>347 and s26['tot']>60
    print(f"    {nome:<22} 5anos {s5['tot']:+.0f} vs 347 | OOS {s26['tot']:+.0f} vs 60  -> {'bate os dois' if ok else 'NAO bate'}")
print('\n  Saida = fechamento cruza o trail (fiel ao indicador, NAO toque intrabar). Bot rodando NAO tocado.')
