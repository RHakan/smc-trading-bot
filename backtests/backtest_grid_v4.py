"""
backtest_grid_v4.py
SISTEMA 4 — GRID v4: TF 5m + EQUITY-TRAILING (o "stop que sobe" do Rafa).

Dois eixos sobre a vencedora (ETH two-sided R10):
  1. TF 5m vs 15m — mais candles = mais fills/dia; no 5m testa steps menores
     (0.4/0.6/0.8%) porque o ruido fino e mais estreito. Custo por RT igual
     (maker 0.04%) -> step menor tem margem liquida menor: o teste diz se compensa.
  2. EQUITY-TRAILING (ideia do Rafa adaptada ao grid): trailing por TRADE mataria
     o motor do grid (o take fixo de 1 step E o lucro). A versao correta e no
     PORTFOLIO: piso = pico_da_equity*(1-trail). Se equity <= piso -> liquida
     TUDO (trava o lucro), remonta o grid e o pico reseta. Corta as caudas de
     "devolucao" (os meses -369/-339 da v3). trail: off | 10% | 20%.

Par: ETH (vencedor; BTC dilui, SOL quebra). Lev 1x e 2x (2x = teto da tolerancia
de -250/mes do Rafa). Periodos 2021..2026 (2026=OOS). 500 USDC.
Uso: python backtests/backtest_grid_v4.py
  (1a execucao baixa ~578k candles 5m do ETH — pode levar ~10 min)
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
HALF_RANGE=0.10
SYM='ETH/USDT:USDT'
PERIODS=[('2021',2021),('2022',2022),('2023',2023),('2024',2024),('2025',2025),('2026',2026)]

CACHE_DIR=Path(__file__).parent/'cache'
ex=ccxt.binanceusdm({'enableRateLimit':True})


def fetch(tf):
    safe=SYM.replace('/','_').replace(':','_')
    cache=CACHE_DIR/f'{safe}_{tf}_2021_2026.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True); return df
    print(f'    (baixando {tf} do ETH 2021-2026 — pode levar ~10 min)')
    since=int(FETCH_START.timestamp()*1000); end_ms=int(FETCH_END.timestamp()*1000); rows=[]
    while True:
        b=ex.fetch_ohlcv(SYM,tf,since=since,limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df=pd.DataFrame(rows,columns=['ts','o','h','l','c','v'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=df.set_index('ts').sort_index()[lambda d:d.index<=FETCH_END]
    df.to_csv(cache); return df


def run_grid(df, step, lev, trail):
    """Grid two-sided R10 no ETH. trail=0 desliga o equity-trailing."""
    h=df['h'].values; l=df['l'].values; c=df['c'].values
    idx=df.index; n=len(df)
    K=max(2,int(HALF_RANGE/step))

    cash=CAPITAL; center=c[0]
    long_lv=short_lv=None; lqty=sqty=None; sprice=None; per_level=0.0
    rts=0; broke=False; locks=0
    monthly={}; daily={}
    hwm=CAPITAL

    def equity(cl):
        e=cash
        if lqty is not None: e+=float((lqty*cl).sum())
        if sqty is not None: e+=float((sqty*(sprice-cl)).sum())
        return e

    def liquidate(cl):
        nonlocal cash
        if lqty is not None and lqty.any():
            cash+=float((lqty*cl).sum())*(1-TAKER_SLIP); lqty[:]=0.0
        if sqty is not None and sqty.any():
            pnl=float((sqty*(sprice-cl)).sum())
            cash+=pnl-TAKER_SLIP*float((sqty*cl).sum()); sqty[:]=0.0

    def remount(cl):
        nonlocal center,long_lv,short_lv,lqty,sqty,sprice,per_level
        center=cl
        long_lv=center*(1-step*np.arange(1,K+1))
        short_lv=center*(1+step*np.arange(1,K+1))
        lqty=np.zeros(K); sqty=np.zeros(K); sprice=short_lv.copy()
        per_level=max(equity(cl),0.0)*lev/(2*K)

    remount(c[0])

    for i in range(n):
        if broke:
            daily[idx[i].strftime('%Y-%m-%d')]=cash
            monthly[idx[i].strftime('%Y-%m')]=cash
            continue
        lo=l[i]; hi=h[i]; cl=c[i]
        tp=long_lv*(1+step)
        for k in np.where((lqty>0)&(hi>=tp))[0]:
            proceeds=lqty[k]*tp[k]
            cash+=proceeds-MAKER*(lqty[k]*long_lv[k]+proceeds)
            lqty[k]=0.0; rts+=1
        tpS=short_lv*(1-step)
        for k in np.where((sqty>0)&(lo<=tpS))[0]:
            pnl=sqty[k]*(sprice[k]-tpS[k])
            cash+=pnl-MAKER*(sqty[k]*(sprice[k]+tpS[k]))
            sqty[k]=0.0; rts+=1
        for k in np.where((lqty==0)&(lo<=long_lv))[0]:
            cash-=per_level; lqty[k]=per_level/long_lv[k]
        for k in np.where((sqty==0)&(hi>=short_lv))[0]:
            sqty[k]=per_level/short_lv[k]; sprice[k]=short_lv[k]
        # saida do range
        if cl<center*(1-HALF_RANGE) or cl>center*(1+HALF_RANGE):
            liquidate(cl); remount(cl)
        elif (not lqty.any()) and (not sqty.any()) and abs(cl-center)/center>2*step:
            remount(cl)
        eq=equity(cl)
        # ── EQUITY-TRAILING (o "stop que sobe") ────────────────────────────────
        if trail>0:
            hwm=max(hwm,eq)
            if eq<=hwm*(1-trail):
                liquidate(cl); remount(cl)
                eq=equity(cl); hwm=eq; locks+=1
        # quebra
        if eq<=CAPITAL*0.05:
            liquidate(cl); cash=max(cash,0.0); broke=True; eq=cash
        daily[idx[i].strftime('%Y-%m-%d')]=eq
        monthly[idx[i].strftime('%Y-%m')]=eq

    mkeys=sorted(monthly); out_m={}; prev=CAPITAL
    for mk in mkeys: out_m[mk]=monthly[mk]-prev; prev=monthly[mk]
    dser=pd.Series(daily).sort_index()
    peak=dser.cummax(); dd=float(((peak-dser)/peak*100).max())
    return dict(monthly=out_m, final=prev, rts=rts, dd=dd, broke=broke, locks=locks)


print('SISTEMA 4 — GRID v4: TF 5m + equity-trailing | ETH two-sided R10 | 500 USDC')
print('A carregar dados...')
data={}
for tf in ['15m','5m']:
    print(f'  {tf}...', end=' ', flush=True)
    try:
        # 15m usa o cache da v1 (nome antigo)
        if tf=='15m':
            safe=SYM.replace('/','_').replace(':','_')
            df=pd.read_csv(CACHE_DIR/f'{safe}_15m_2021_2026.csv',index_col='ts',parse_dates=True)
            df.index=pd.to_datetime(df.index,utc=True)
        else:
            df=fetch(tf)
        data[tf]=df
        print(f'OK ({len(df):,})')
    except Exception as e:
        print(f'ERRO ({e})')
print()

CONFIGS=[]
for tf,steps in [('15m',[0.008]),('5m',[0.004,0.006,0.008])]:
    for step in steps:
        for lev in [1,2]:
            for trail in [0.0,0.10,0.20]:
                CONFIGS.append((tf,step,lev,trail))

print('='*114)
print(f"  {'TF':<5}{'step':>6}{'lev':>5}{'trail':>7} |"+''.join(f"{p[0]:>8}" for p in PERIODS)+
      f"{'TOTAL':>9}{'/dia':>7}{'%/dia':>8}{'DD':>7}{'rt/d':>6}{'locks':>7}")
print('  '+'-'*112)
best=None
for tf,step,lev,trail in CONFIGS:
    df=data.get(tf)
    if df is None: continue
    r=run_grid(df, step, lev, trail)
    ypnl={}
    for mk,v in r['monthly'].items(): ypnl[int(mk[:4])]=ypnl.get(int(mk[:4]),0)+v
    tot=sum(ypnl.values())
    days=(df.index[-1]-df.index[0]).days
    per_day=tot/max(days,1)
    pos=sum(1 for p in PERIODS if ypnl.get(p[1],0)>0)
    tl='off' if trail==0 else f'{int(trail*100)}%'
    print(f"  {tf:<5}{step*100:>5.1f}%{lev:>4}x{tl:>7} |"+''.join(f"{ypnl.get(p[1],0):>+8.0f}" for p in PERIODS)+
          f"{tot:>+9.0f}{per_day:>+7.2f}{per_day/CAPITAL*100:>+8.3f}{r['dd']:>6.1f}%{r['rts']/max(days,1):>6.1f}{r['locks']:>7}")
    # criterio: periodos+, depois pior-mes dentro de -250 (tolerancia), depois total
    worst=min(r['monthly'].values()) if r['monthly'] else 0
    score=(pos, worst>=-250, tot)
    if best is None or score>best[0]:
        best=(score,tf,step,lev,trail,r,ypnl,days)
print()

score,tf,step,lev,trail,r,ypnl,days=best
tot=sum(ypnl.values()); per_day=tot/max(days,1)
worst=min(r['monthly'].values())
tl='off' if trail==0 else f'{int(trail*100)}%'
print('='*114)
print(f'  MELHOR (periodos+ > pior-mes>=-250 > total): {tf} step {step*100:.1f}% {lev}x trail {tl}')
print('='*114)
print(f'  Total {tot:+.0f} | {per_day:+.2f}/dia ({per_day/CAPITAL*100:+.3f}%/dia) | DD {r["dd"]:.1f}% | pior mes {worst:+.0f}')
if per_day>0:
    print(f'  META 100/dia -> capital necessario ~{100/(per_day/CAPITAL):,.0f} USDC')
print(f'\n  Levantamento mensal:')
ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
yrs=sorted(set(int(mk[:4]) for mk in r['monthly']))
print(f"  {'Ano':<6} | "+' '.join(f'{m:>5}' for m in ML)+f" | {'Soma':>7}")
print('  '+'-'*95)
for yr in yrs:
    vals=[r['monthly'].get(f'{yr}-{m+1:02d}') for m in range(12)]
    cells=' '.join((f'{v:>+5.0f}' if v is not None else '    .') for v in vals)
    print(f"  {yr:<6} | {cells} | {sum(v for v in vals if v is not None):>+7.0f}")
print('\n  Ref v3: ETH 15m 2x trail-off = +2130 (0.212%/dia, DD 37.9%) | 3x = +3920 (0.391%/dia, DD 51.7%).')
print('  locks = quantas vezes o equity-trailing travou lucro e remontou o grid.')
print('  Custos maker 0.02%/lado; liquidacoes 0.07%. Registar no GRID_RESULTS.md.')
