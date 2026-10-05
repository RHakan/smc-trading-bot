"""
backtest_lateral_dev.py
Banco de desenvolvimento da estrategia LATERAL (mean-reversion).

O Decisor classifica NEUTRAL quando o mercado nao tem tendencia clara (~70% do
tempo). Ate agora o bot ficava DE FORA nesse regime. Esta estrategia ataca-o:
em range, comprar perto do suporte / vender perto da resistencia, sair na media.

Logica (regime NEUTRAL):
  Bollinger Bands (SMA n +/- k*desvio) definem o range dinamico.
  LONG  : preco estica abaixo da banda inferior e reverte  -> compra o desconto
  SHORT : preco estica acima da banda superior e reverte    -> vende o premio
  TP    : media central (mid) — o "valor justo" do range
  STOP  : alem da banda + folga ATR (se rompeu o range, nao era range)
  Sem trailing/BE — mean-reversion e mecanico: entra no extremo, sai na media.

Gestao DIFERENTE das direcionais: RR costuma ser <1 mas WR alto. O edge vem da
frequencia de acertos, nao de deixar correr.

Testado no motor de portfolio realista (saldo 5000, correlacao), periodo
jun/2025 -> jun/2026. Mede USDC/DIA vs meta de 40/dia.

⚠️  HIPOTETICA. Se promissora no 1H, proximo passo e testar 15m (mais setups).
Uso: python backtests/backtest_lateral_dev.py
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

START = datetime(2025, 6,  1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 29, tzinfo=timezone.utc)
INITIAL_BALANCE = 5000.0
RISK_PCT        = 1.0
PERIOD_DAYS     = (END - START).days
TARGET_PER_DAY  = 40.0

PAIRS = ['BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
         'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
         'AVAX/USDT:USDT','DOT/USDT:USDT']
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=BEAR.ATR_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH

# Gestao mean-reversion
COOLDOWN = 4          # barras entre trades no mesmo par
TIMEOUT  = 48         # sai a mercado se nao reverteu em TIMEOUT barras

# Base
BB_N=20; BB_K=2.0; RSI_P=14

CACHE_DIR=Path(__file__).parent/'cache'; CACHE_DIR.mkdir(exist_ok=True)
ex=ccxt.binanceusdm({'enableRateLimit':True})


def fetch(sym, tf, extra_days):
    safe=sym.replace('/','_').replace(':','_')
    cache=CACHE_DIR/f'{safe}_{tf}_jun25_jun26.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True); return df
    since=int((START-timedelta(days=extra_days)).timestamp()*1000)
    end_ms=int(END.timestamp()*1000); rows=[]
    while True:
        b=ex.fetch_ohlcv(sym,tf,since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=df.set_index('ts').sort_index()[lambda d:d.index<=END]
    df.to_csv(cache); return df


def rsi(series, period=14):
    delta=series.diff()
    up=delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    dn=(-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs=up/dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


print('Banco de desenvolvimento da LATERAL (mean-reversion)')
print(f'  Periodo: {START:%d/%m/%Y} -> {END:%d/%m/%Y} ({PERIOD_DAYS} dias) | saldo {INITIAL_BALANCE:.0f}')
print(f'  Meta: {TARGET_PER_DAY:.0f} USDC/dia => {TARGET_PER_DAY*PERIOD_DAYS:,.0f} USDC no periodo\n')
print('A carregar dados (cache)...')

pair_data={}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd=fetch(sym,'1d',40).copy()
        df1=fetch(sym,'1h',25).copy()
        if len(df1)<300: print('SEM DADOS'); continue
        ema_d=dfd['c'].ewm(span=R_EMA,adjust=False).mean()
        slope=(ema_d-ema_d.shift(R_SLOPE))/ema_d.shift(R_SLOPE)
        dfd['regime']='NEUTRAL'
        dfd.loc[(dfd['c']<ema_d)&(slope<-R_THRESH),'regime']='BEAR'
        dfd.loc[(dfd['c']>ema_d)&(slope> R_THRESH),'regime']='BULL'
        tr=pd.concat([(df1['h']-df1['l']),(df1['h']-df1['c'].shift(1)).abs(),
                      (df1['l']-df1['c'].shift(1)).abs()],axis=1).max(axis=1)
        df1['atr']=tr.ewm(com=ATR_PERIOD-1,adjust=False).mean()
        df1['regime']=dfd['regime'].shift(1).reindex(df1.index,method='ffill')
        df1['rsi']=rsi(df1['c'], RSI_P)
        df1['bb_mid']=df1['c'].rolling(BB_N).mean()
        df1['bb_std']=df1['c'].rolling(BB_N).std()
        pair_data[sym]=df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()

master_index=None
for sym,df in pair_data.items():
    master_index=df.index if master_index is None else master_index.union(df.index)
master_index=master_index[(master_index>=START)&(master_index<=END)]


def gen_lateral_signals(df, bb_k, rsi_low, rsi_high, confirm, stop_atr, tp_mode):
    """
    Sinais mean-reversion. Devolve side/entry/sl/tp por barra.
    tp_mode: 'mid' (media central) ou 'band' (banda oposta).
    confirm: True = exige reversao (close volta para dentro da banda).
    """
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; reg=df['regime'].values; rsi_a=df['rsi'].values
    mid=df['bb_mid'].values; std=df['bb_std'].values

    side=np.array([None]*n,dtype=object)
    entry=np.full(n,np.nan); sl=np.full(n,np.nan); tp=np.full(n,np.nan)

    for i in range(BB_N+2, n):
        if reg[i]!='NEUTRAL': continue
        m=mid[i]; s=std[i]; a=atr[i]
        if np.isnan(m) or np.isnan(s) or np.isnan(a) or s<=0: continue
        lower=m-bb_k*s; upper=m+bb_k*s
        cl=c[i]; lo=l[i]; hi=h[i]; pc=c[i-1]

        # LONG — esticou abaixo da banda inferior
        long_raw  = lo < lower
        long_conf = (pc < lower) and (cl > lower)   # reverteu para dentro
        long_ok = (long_conf if confirm else long_raw and cl < m)
        if long_ok and (rsi_low<=0 or rsi_a[i] < rsi_low):
            e=cl; stop=lower - stop_atr*a
            target = m if tp_mode=='mid' else upper
            if stop < e < target:
                side[i]='LONG'; entry[i]=e; sl[i]=stop; tp[i]=target
                continue

        # SHORT — esticou acima da banda superior
        short_raw  = hi > upper
        short_conf = (pc > upper) and (cl < upper)
        short_ok = (short_conf if confirm else short_raw and cl > m)
        if short_ok and (rsi_high<=0 or rsi_a[i] > rsi_high):
            e=cl; stop=upper + stop_atr*a
            target = m if tp_mode=='mid' else lower
            if target < e < stop:
                side[i]='SHORT'; entry[i]=e; sl[i]=stop; tp[i]=target
    return side, entry, sl, tp


def build_arrays(**kw):
    A={}
    for sym,df in pair_data.items():
        side,entry,sl,tp=gen_lateral_signals(df, **kw)
        d=df.assign(_s=side,_e=entry,_sl=sl,_tp=tp).reindex(master_index)
        A[sym]={'h':d['h'].values,'l':d['l'].values,'c':d['c'].values,
                'side':d['_s'].values,'entry':d['_e'].values,
                'sl':d['_sl'].values,'tp':d['_tp'].values}
    return A


def run_portfolio(A, max_concurrent=None):
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}; trades=[]; max_conc=0
    for k in range(len(master_index)):
        # saidas
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]; hi=d['h'][k]; lo=d['l'][k]
            risk=pos['risk_px']; fee_r=pos['fee_r']; e=pos['entry']
            closed=False; nr=0.0
            if pos['side']=='LONG':
                if lo<=pos['sl']: nr=-1-fee_r; closed=True
                elif hi>=pos['tp']: nr=(pos['tp']-e)/risk - fee_r; closed=True
            else:
                if hi>=pos['sl']: nr=-1-fee_r; closed=True
                elif lo<=pos['tp']: nr=(e-pos['tp'])/risk - fee_r; closed=True
            if not closed:
                pos['age']+=1
                if pos['age']>=TIMEOUT:
                    nr=((c-e) if pos['side']=='LONG' else (e-c))/risk - fee_r; closed=True
            if closed:
                balance+=nr*pos['risk_usd']; trades.append({'ts':master_index[k],'nr':nr})
                del positions[sym]; cooldown_until[sym]=k+COOLDOWN
        # entradas
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
                            'risk_px':risk_px,'risk_usd':balance*(RISK_PCT/100.0),
                            'fee_r':e*RT/risk_px}
        max_conc=max(max_conc,len(positions))
        peak=max(peak,balance); dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return {'balance':balance,'trades':trades,'max_dd':max_dd,'max_conc':max_conc}


def show(label, res, base=False):
    t=res['trades']; bal=res['balance']; mark='*' if base else ' '
    if not t: print(f'  {label:<32}{mark}: sem trades'); return res
    w=[x for x in t if x['nr']>0]; gw=sum(x['nr'] for x in w); gl=abs(sum(x['nr'] for x in t if x['nr']<=0))
    pnl=bal-INITIAL_BALANCE; perday=pnl/PERIOD_DAYS
    print(f'  {label:<32}{mark}: {pnl:>+8.0f} USDC | {perday:>+6.1f}/dia | {len(t):>4}t | '
          f'WR {len(w)/len(t)*100:>4.1f}% | PF {gw/gl if gl>0 else 99:>4.2f} | DD {res["max_dd"]:>4.1f}%')
    return res


print('='*116)
print('  DESENVOLVIMENTO DA LATERAL — mean-reversion isolada no motor de portfolio (regime NEUTRAL)')
print(f'  (meta {TARGET_PER_DAY:.0f}/dia | 1H | os dois lados: compra suporte + vende resistencia)')
print('='*116)

BASE=dict(bb_k=2.0, rsi_low=0, rsi_high=0, confirm=True, stop_atr=1.0, tp_mode='mid')
show('BASE', run_portfolio(build_arrays(**BASE)), base=True)

print('\nSweep 1 — bb_k (largura do range):')
for k in [1.5, 2.0, 2.5, 3.0]:
    show(f'bb_k={k}', run_portfolio(build_arrays(**{**BASE,'bb_k':k})))

print('\nSweep 2 — confirmacao de reversao:')
for cf in [False, True]:
    show(f'confirm={cf}', run_portfolio(build_arrays(**{**BASE,'confirm':cf})))

print('\nSweep 3 — filtro RSI (extremo):')
for lo,hi in [(0,0),(35,65),(30,70),(25,75)]:
    show(f'rsi={lo}/{hi}', run_portfolio(build_arrays(**{**BASE,'rsi_low':lo,'rsi_high':hi})))

print('\nSweep 4 — stop alem da banda (xATR):')
for sa in [0.5, 1.0, 1.5, 2.0]:
    show(f'stop_atr={sa}', run_portfolio(build_arrays(**{**BASE,'stop_atr':sa})))

print('\nSweep 5 — alvo (mid = media | band = banda oposta):')
for tm in ['mid','band']:
    show(f'tp_mode={tm}', run_portfolio(build_arrays(**{**BASE,'tp_mode':tm})))

print('\nCombinacoes:')
combos=[
    dict(bb_k=2.0, rsi_low=30, rsi_high=70, confirm=True,  stop_atr=1.0, tp_mode='mid'),
    dict(bb_k=2.5, rsi_low=30, rsi_high=70, confirm=True,  stop_atr=1.0, tp_mode='mid'),
    dict(bb_k=2.0, rsi_low=25, rsi_high=75, confirm=True,  stop_atr=1.5, tp_mode='mid'),
    dict(bb_k=2.5, rsi_low=25, rsi_high=75, confirm=True,  stop_atr=1.5, tp_mode='band'),
    dict(bb_k=2.0, rsi_low=35, rsi_high=65, confirm=False, stop_atr=1.0, tp_mode='mid'),
    dict(bb_k=2.5, rsi_low=30, rsi_high=70, confirm=True,  stop_atr=1.5, tp_mode='mid'),
]
res=[]
for p in combos:
    lbl=f"k{p['bb_k']} rsi{p['rsi_low']}/{p['rsi_high']} st{p['stop_atr']} {p['tp_mode']}"
    res.append((p, show(lbl, run_portfolio(build_arrays(**p)))))

best_p,best_r=max(res, key=lambda x:x[1]['balance'])
pnl=best_r['balance']-INITIAL_BALANCE
print('\n'+'='*116)
print(f'  MELHOR: {best_p}')
print(f'  -> {pnl:+.0f} USDC | {pnl/PERIOD_DAYS:+.1f}/dia | DD {best_r["max_dd"]:.1f}%  '
      f'(meta {TARGET_PER_DAY:.0f}/dia)')
print('='*116)
print('  Nota: 1H costuma dar poucos setups de range. Se o edge existir mas faltar')
print('  frequencia/resultado, o passo seguinte e descer para 15m (mais oscilacoes).')
print('  Custos 0.14% round-trip. Bull/Lateral HIPOTETICAS. Risco 1%/trade.')
