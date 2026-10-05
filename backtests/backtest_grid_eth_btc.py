"""
backtest_grid_eth_btc.py
SISTEMA 4 — GRID: ETH sozinho vs ETH+BTC em paralelo (as 2 unicas que qualificam).

Das 10 moedas do bot, so ETH e BTC passaram a regra (5+ periodos+, pior mes>=-250,
sem quebra). Sao complementares em risco: ETH=motor (0.212%/dia, DD 37.9%),
BTC=ancora (0.068%/dia, DD 21.5%). Aqui medimos se combinar baixa o DD do
CONJUNTO (equity agregada dia a dia — descorrelacao real, nao soma de DDs).

Cenarios (config campea: two-sided 0.8% R10 2x):
  ETH 500          — referencia (motor puro)
  BTC 500          — referencia (ancora pura)
  ETH250 + BTC250  — 500 TOTAL dividido (comparacao justa vs ETH500)
  ETH500 + BTC500  — 1000 TOTAL (para quem aloca mais capital)

Metrica-chave: retorno/DD (eficiencia) e pior mes agregado vs tolerancia -250.
Uso: python backtests/backtest_grid_eth_btc.py
"""
import sys, time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MAKER=0.0002; TAKER_SLIP=0.0007
STEP=0.008; HALF_RANGE=0.10; LEV=2
PERIODS=[('2021',2021),('2022',2022),('2023',2023),('2024',2024),('2025',2025),('2026',2026)]
CACHE_DIR=Path(__file__).parent/'cache'


def load(sym):
    safe=sym.replace('/','_').replace(':','_')
    df=pd.read_csv(CACHE_DIR/f'{safe}_15m_2021_2026.csv',index_col='ts',parse_dates=True)
    df.index=pd.to_datetime(df.index,utc=True); return df


def run_grid(df, cap0):
    """Grid two-sided R10 2x. Devolve equity DIARIA (para DD agregado real)."""
    h=df['h'].values; l=df['l'].values; c=df['c'].values
    idx=df.index; n=len(df)
    K=max(2,int(HALF_RANGE/STEP))
    cash=cap0; center=c[0]
    long_lv=short_lv=None; lqty=sqty=None; sprice=None; per_level=0.0
    broke=False; daily={}

    def equity(cl):
        e=cash
        if lqty is not None: e+=float((lqty*cl).sum())
        if sqty is not None: e+=float((sqty*(sprice-cl)).sum())
        return e

    def remount(cl):
        nonlocal center,long_lv,short_lv,lqty,sqty,sprice,per_level
        center=cl
        long_lv=center*(1-STEP*np.arange(1,K+1)); short_lv=center*(1+STEP*np.arange(1,K+1))
        lqty=np.zeros(K); sqty=np.zeros(K); sprice=short_lv.copy()
        per_level=max(equity(cl),0.0)*LEV/(2*K)

    remount(c[0])
    for i in range(n):
        if broke: daily[idx[i]]=cash; continue
        lo=l[i]; hi=h[i]; cl=c[i]
        tp=long_lv*(1+STEP)
        for k in np.where((lqty>0)&(hi>=tp))[0]:
            pr=lqty[k]*tp[k]; cash+=pr-MAKER*(lqty[k]*long_lv[k]+pr); lqty[k]=0.0
        tpS=short_lv*(1-STEP)
        for k in np.where((sqty>0)&(lo<=tpS))[0]:
            cash+=sqty[k]*(sprice[k]-tpS[k])-MAKER*(sqty[k]*(sprice[k]+tpS[k])); sqty[k]=0.0
        for k in np.where((lqty==0)&(lo<=long_lv))[0]:
            cash-=per_level; lqty[k]=per_level/long_lv[k]
        for k in np.where((sqty==0)&(hi>=short_lv))[0]:
            sqty[k]=per_level/short_lv[k]; sprice[k]=short_lv[k]
        if cl<center*(1-HALF_RANGE) or cl>center*(1+HALF_RANGE):
            if lqty.any(): cash+=float((lqty*cl).sum())*(1-TAKER_SLIP); lqty[:]=0.0
            if sqty.any(): cash+=float((sqty*(sprice-cl)).sum())-TAKER_SLIP*float((sqty*cl).sum()); sqty[:]=0.0
            remount(cl)
        elif (not lqty.any()) and (not sqty.any()) and abs(cl-center)/center>2*STEP: remount(cl)
        eq=equity(cl)
        if eq<=cap0*0.05:
            if lqty.any(): cash+=float((lqty*cl).sum())*(1-TAKER_SLIP); lqty[:]=0.0
            if sqty.any(): cash+=float((sqty*(sprice-cl)).sum())-TAKER_SLIP*float((sqty*cl).sum()); sqty[:]=0.0
            cash=max(cash,0.0); broke=True; eq=cash
        daily[idx[i]]=eq
    return pd.Series(daily).sort_index()


print('GRID — ETH sozinho vs ETH+BTC paralelo (as 2 que qualificam) | two-sided 0.8% R10 2x')
print('A carregar dados (cache)...\n')
eth=load('ETH/USDT:USDT'); btc=load('BTC/USDT:USDT')


def metrics(name, cap_total, eth_cap, btc_cap):
    parts=[]
    if eth_cap>0: parts.append(run_grid(eth, eth_cap))
    if btc_cap>0: parts.append(run_grid(btc, btc_cap))
    # equity agregada diaria (soma; dias sem um par usam ffill do ultimo valor)
    agg=None
    for s in parts:
        d=s.resample('1D').last()
        agg=d if agg is None else agg.add(d, fill_value=np.nan)
    agg=agg.ffill().dropna()
    # P&L mensal do conjunto
    monthly={}
    prev=cap_total
    for mk,eqv in agg.groupby(agg.index.strftime('%Y-%m')).last().items():
        monthly[mk]=eqv-prev; prev=eqv
    ypnl={}
    for mk,v in monthly.items(): ypnl[int(mk[:4])]=ypnl.get(int(mk[:4]),0)+v
    tot=sum(ypnl.values()); days=(agg.index[-1]-agg.index[0]).days
    peak=agg.cummax(); dd=float(((peak-agg)/peak*100).max())
    per_day=tot/max(days,1); pos=sum(1 for p in PERIODS if ypnl.get(p[1],0)>0)
    worst=min(monthly.values())
    return dict(name=name,cap=cap_total,tot=tot,per_day=per_day,
                pct=per_day/cap_total*100,dd=dd,pos=pos,worst=worst,
                eff=tot/dd if dd>0 else 0, ypnl=ypnl, monthly=monthly)


SCEN=[
    ('ETH 500 (motor)',          500, 500,   0),
    ('BTC 500 (ancora)',         500,   0, 500),
    ('ETH250+BTC250 (500 tot)', 500, 250, 250),
    ('ETH500+BTC500 (1000)',   1000, 500, 500),
]
res=[metrics(*s) for s in SCEN]

print('='*104)
print(f"  {'Cenario':<24}{'cap':>6} |"+''.join(f"{p[0]:>8}" for p in PERIODS)+
      f"{'TOT':>8}{'%/dia':>8}{'DD':>7}{'pior':>7}{'ret/DD':>8}")
print('  '+'-'*102)
for r in res:
    print(f"  {r['name']:<24}{r['cap']:>6} |"+''.join(f"{r['ypnl'].get(p[1],0):>+8.0f}" for p in PERIODS)+
          f"{r['tot']:>+8.0f}{r['pct']:>+8.3f}{r['dd']:>6.1f}%{r['worst']:>+7.0f}{r['eff']:>8.1f}")

print()
eth_only=res[0]; combo=res[2]
print('='*104)
print('  ETH sozinho vs ETH+BTC (mesmo capital 500):')
print('='*104)
print(f"  DD:      ETH {eth_only['dd']:.1f}%  ->  ETH+BTC {combo['dd']:.1f}%   ({combo['dd']-eth_only['dd']:+.1f} pts)")
print(f"  %/dia:   ETH {eth_only['pct']:+.3f}%  ->  ETH+BTC {combo['pct']:+.3f}%")
print(f"  pior mes:ETH {eth_only['worst']:+.0f}  ->  ETH+BTC {combo['worst']:+.0f}")
print(f"  ret/DD:  ETH {eth_only['eff']:.1f}  ->  ETH+BTC {combo['eff']:.1f}  (maior = mais eficiente)")
print()
if combo['eff']>eth_only['eff']:
    print('  >>> Combinar MELHORA a eficiencia (ret/DD): BTC baixa o DD mais do que corta o retorno.')
else:
    print('  >>> ETH sozinho e mais eficiente. BTC corta retorno mais do que baixa o DD.')
print(f"\n  Nota: ETH sozinho ja respeita a tolerancia (-201 > -250). Combinar e opcional:")
print(f"  troca rendimento por estabilidade. Decisao do Rafa conforme o perfil.")
print('  Custos maker 0.02%/lado; liquidacoes 0.07%. Registar no GRID_RESULTS.md.')
