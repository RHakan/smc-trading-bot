"""
backtest_grid_10moedas.py
SISTEMA 4 — GRID: config campea (v3) testada nas 10 MOEDAS do bot, isoladas.

Config fixa (vencedora da campanha): two-sided, step 0.8%, half-range 10%,
sizing por equity, lev 2x, sem trail. 500 USDC por moeda. 6 periodos (2026=OOS).

Objetivo: ver QUAIS das 10 moedas do bot se qualificam pela regra do Rafa —
positivo em multiplos periodos + pior mes dentro de -250 + sem quebra. Ranking
final por (periodos+, pior-mes>=-250, total).

Custos maker 0.02%/lado; liquidacoes/stop taker+slip 0.07%. 15m 2021-2026.
1a execucao baixa 15m das moedas que faltam (~190k candles cada).
Uso: python backtests/backtest_grid_10moedas.py
"""
import sys, time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

FETCH_START=datetime(2021,1,1,tzinfo=timezone.utc)
FETCH_END  =datetime(2026,6,29,tzinfo=timezone.utc)
CAPITAL=500.0
MAKER=0.0002; TAKER_SLIP=0.0007
STEP=0.008; HALF_RANGE=0.10; LEV=2
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','BNB/USDT:USDT',
       'XRP/USDT:USDT','ADA/USDT:USDT','AVAX/USDT:USDT','DOGE/USDT:USDT',
       'SUI/USDT:USDT','LINK/USDT:USDT']
PERIODS=[('2021',2021),('2022',2022),('2023',2023),('2024',2024),('2025',2025),('2026',2026)]

CACHE_DIR=Path(__file__).parent/'cache'
ex=ccxt.binanceusdm({'enableRateLimit':True})


def fetch_15m(sym):
    safe=sym.replace('/','_').replace(':','_')
    cache=CACHE_DIR/f'{safe}_15m_2021_2026.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True); return df
    print(f'    (baixando 15m de {sym} — ~190k candles)')
    since=int(FETCH_START.timestamp()*1000); end_ms=int(FETCH_END.timestamp()*1000); rows=[]
    while True:
        b=ex.fetch_ohlcv(sym,'15m',since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=df.set_index('ts').sort_index()[lambda d:d.index<=FETCH_END]
    df.to_csv(cache); return df


def run_grid(df):
    h=df['h'].values; l=df['l'].values; c=df['c'].values
    idx=df.index; n=len(df)
    K=max(2,int(HALF_RANGE/STEP))
    cash=CAPITAL; center=c[0]
    long_lv=short_lv=None; lqty=sqty=None; sprice=None; per_level=0.0
    rts=0; broke=False; monthly={}; daily={}

    def equity(cl):
        e=cash
        if lqty is not None: e+=float((lqty*cl).sum())
        if sqty is not None: e+=float((sqty*(sprice-cl)).sum())
        return e

    def remount(cl):
        nonlocal center,long_lv,short_lv,lqty,sqty,sprice,per_level
        center=cl
        long_lv=center*(1-STEP*np.arange(1,K+1))
        short_lv=center*(1+STEP*np.arange(1,K+1))
        lqty=np.zeros(K); sqty=np.zeros(K); sprice=short_lv.copy()
        per_level=max(equity(cl),0.0)*LEV/(2*K)

    remount(c[0])
    for i in range(n):
        if broke:
            daily[idx[i].strftime('%Y-%m-%d')]=cash; monthly[idx[i].strftime('%Y-%m')]=cash; continue
        lo=l[i]; hi=h[i]; cl=c[i]
        tp=long_lv*(1+STEP)
        for k in np.where((lqty>0)&(hi>=tp))[0]:
            proceeds=lqty[k]*tp[k]; cash+=proceeds-MAKER*(lqty[k]*long_lv[k]+proceeds); lqty[k]=0.0; rts+=1
        tpS=short_lv*(1-STEP)
        for k in np.where((sqty>0)&(lo<=tpS))[0]:
            cash+=sqty[k]*(sprice[k]-tpS[k])-MAKER*(sqty[k]*(sprice[k]+tpS[k])); sqty[k]=0.0; rts+=1
        for k in np.where((lqty==0)&(lo<=long_lv))[0]:
            cash-=per_level; lqty[k]=per_level/long_lv[k]
        for k in np.where((sqty==0)&(hi>=short_lv))[0]:
            sqty[k]=per_level/short_lv[k]; sprice[k]=short_lv[k]
        if cl<center*(1-HALF_RANGE) or cl>center*(1+HALF_RANGE):
            if lqty.any(): cash+=float((lqty*cl).sum())*(1-TAKER_SLIP); lqty[:]=0.0
            if sqty.any():
                cash+=float((sqty*(sprice-cl)).sum())-TAKER_SLIP*float((sqty*cl).sum()); sqty[:]=0.0
            remount(cl)
        elif (not lqty.any()) and (not sqty.any()) and abs(cl-center)/center>2*STEP:
            remount(cl)
        eq=equity(cl)
        if eq<=CAPITAL*0.05:
            if lqty.any(): cash+=float((lqty*cl).sum())*(1-TAKER_SLIP); lqty[:]=0.0
            if sqty.any(): cash+=float((sqty*(sprice-cl)).sum())-TAKER_SLIP*float((sqty*cl).sum()); sqty[:]=0.0
            cash=max(cash,0.0); broke=True; eq=cash
        daily[idx[i].strftime('%Y-%m-%d')]=eq; monthly[idx[i].strftime('%Y-%m')]=eq

    mkeys=sorted(monthly); out_m={}; prev=CAPITAL
    for mk in mkeys: out_m[mk]=monthly[mk]-prev; prev=monthly[mk]
    dser=pd.Series(daily).sort_index()
    peak=dser.cummax(); dd=float(((peak-dser)/peak*100).max())
    return dict(monthly=out_m, final=prev, rts=rts, dd=dd, broke=broke)


print('GRID nas 10 moedas do bot | config campea: two-sided 0.8% R10 2x | 500/moeda')
print('A carregar dados 15m (baixa os que faltam na 1a vez)...')
data={}
for sym in PAIRS:
    print(f'  {sym.split("/")[0]:<5}...', end=' ', flush=True)
    try:
        df=fetch_15m(sym)
        if len(df)<10000: print('SEM DADOS'); continue
        data[sym]=df; print(f'OK ({len(df):,})')
    except Exception as e:
        print(f'ERRO ({e})')
print()

print('='*112)
print(f"  {'Moeda':<7}|"+''.join(f"{p[0]:>8}" for p in PERIODS)+
      f"{'TOTAL':>9}{'/dia':>7}{'%/dia':>8}{'DD':>7}{'pior mes':>10}{'quebra':>8}")
print('  '+'-'*110)
rows=[]
for sym in PAIRS:
    df=data.get(sym)
    if df is None: continue
    r=run_grid(df)
    ypnl={}
    for mk,v in r['monthly'].items(): ypnl[int(mk[:4])]=ypnl.get(int(mk[:4]),0)+v
    tot=sum(ypnl.values()); days=(df.index[-1]-df.index[0]).days
    per_day=tot/max(days,1)
    pos=sum(1 for p in PERIODS if ypnl.get(p[1],0)>0)
    worst=min(r['monthly'].values()) if r['monthly'] else 0
    ok='✓' if (worst>=-250 and not r['broke'] and pos>=5) else ' '
    print(f"  {sym.split('/')[0]:<6}{ok}|"+''.join(f"{ypnl.get(p[1],0):>+8.0f}" for p in PERIODS)+
          f"{tot:>+9.0f}{per_day:>+7.2f}{per_day/CAPITAL*100:>+8.3f}{r['dd']:>6.1f}%{worst:>+10.0f}"
          f"{('SIM' if r['broke'] else '-'):>8}")
    rows.append((sym.split('/')[0], pos, worst, tot, per_day, r['dd'], r['broke']))
print()

# ranking: qualificadas primeiro (5+ periodos, pior>=-250, sem quebra), depois total
def qual(x): return x[1]>=5 and x[2]>=-250 and not x[6]
ranked=sorted(rows, key=lambda x:(qual(x), x[3]), reverse=True)
print('='*112)
print('  RANKING (qualificada = 5+ periodos+, pior mes >= -250, sem quebra)')
print('='*112)
quals=[x for x in ranked if qual(x)]
for x in ranked:
    tag='QUALIFICA' if qual(x) else ('QUEBRA' if x[6] else ('pior>-250' if x[2]<-250 else 'incons.'))
    print(f"  {x[0]:<6} {x[1]}/6 periodos+ | total {x[3]:>+7.0f} | {x[4]:>+5.2f}/dia | DD {x[5]:>4.1f}% | pior mes {x[2]:>+6.0f}  -> {tag}")

print()
if len(quals)>=2:
    print(f'  >>> {len(quals)} moedas qualificam. Multi-par possivel — proximo teste: rodar as')
    print(f'      qualificadas em PARALELO com capital dividido (DD agregado real).')
elif len(quals)==1:
    print(f'  >>> So {quals[0][0]} qualifica. Grid single-par confirmado.')
else:
    print('  >>> Nenhuma passa a regra estrita — rever criterio ou manter ETH (melhor da v3).')
print('  Custos maker 0.02%/lado; liquidacoes 0.07%. Config fixa da campanha (v3).')
print('  Registar no GRID_RESULTS.md.')
