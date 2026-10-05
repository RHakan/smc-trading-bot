"""
backtest_grid_v3.py
SISTEMA 4 — GRID v3: MULTI-PAR + ALAVANCAGEM medida, sobre a vencedora da v2.

Config base (v2, 6/6 periodos+): two-sided, step 0.8%, half-range 10%, sizing
por equity. Agora:
  1. PORTFOLIO multi-par: capital TOTAL de 500 USDC dividido igualmente entre
     os pares (ETH so | BTC+ETH | BTC+ETH+SOL) — grids independentes, equity
     agregada dia a dia (DD do conjunto medido de verdade).
  2. ALAVANCAGEM 1x / 2x / 3x: per_level = equity*lev/(2K). Honestidade: lev
     multiplica P&L E DD; se a equity de um par cair a <=5% do inicial, o grid
     desse par QUEBRA (liquida e para — proxy de liquidacao na exchange).

Meta da campanha: perseguir 100 USDC/dia — medir %/dia real e capital necessario.
Custos: maker 0.02%/lado; liquidacao/stop taker+slip 0.07%. 15m, 2021-2026.
Uso: python backtests/backtest_grid_v3.py
"""
import sys, time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CAPITAL_TOTAL=500.0
MAKER=0.0002; TAKER_SLIP=0.0007
STEP=0.008; HALF_RANGE=0.10          # config vencedora da v2
PERIODS=[('2021',2021),('2022',2022),('2023',2023),('2024',2024),('2025',2025),('2026',2026)]
PORTFOLIOS=[('ETH', ['ETH/USDT:USDT']),
            ('BTC+ETH', ['BTC/USDT:USDT','ETH/USDT:USDT']),
            ('BTC+ETH+SOL', ['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT'])]
LEVS=[1,2,3]

CACHE_DIR=Path(__file__).parent/'cache'


def fetch_15m(sym):
    safe=sym.replace('/','_').replace(':','_')
    df=pd.read_csv(CACHE_DIR/f'{safe}_15m_2021_2026.csv',index_col='ts',parse_dates=True)
    df.index=pd.to_datetime(df.index,utc=True); return df


def run_grid(df, cap0, lev):
    """Grid two-sided (step/range fixos da v2) com alavancagem. Devolve P&L mensal,
    equity DIARIA (para DD agregado), round-trips e flag de quebra."""
    h=df['h'].values; l=df['l'].values; c=df['c'].values
    idx=df.index; n=len(df)
    K=max(2,int(HALF_RANGE/STEP))

    cash=cap0; center=c[0]
    long_lv=short_lv=None; lqty=sqty=None; sprice=None; per_level=0.0
    rts=0; broke=False
    monthly={}; daily={}

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
        per_level=max(equity(cl),0.0)*lev/(2*K)

    remount(c[0])

    for i in range(n):
        if broke:
            # conta do par morta: equity congelada
            daily[idx[i].strftime('%Y-%m-%d')]=cash
            monthly[idx[i].strftime('%Y-%m')]=cash
            continue
        lo=l[i]; hi=h[i]; cl=c[i]
        # fechos com lucro
        tp=long_lv*(1+STEP)
        for k in np.where((lqty>0)&(hi>=tp))[0]:
            proceeds=lqty[k]*tp[k]
            cash+=proceeds-MAKER*(lqty[k]*long_lv[k]+proceeds)
            lqty[k]=0.0; rts+=1
        tpS=short_lv*(1-STEP)
        for k in np.where((sqty>0)&(lo<=tpS))[0]:
            pnl=sqty[k]*(sprice[k]-tpS[k])
            cash+=pnl-MAKER*(sqty[k]*(sprice[k]+tpS[k]))
            sqty[k]=0.0; rts+=1
        # novas entradas
        for k in np.where((lqty==0)&(lo<=long_lv))[0]:
            cash-=per_level; lqty[k]=per_level/long_lv[k]
        for k in np.where((sqty==0)&(hi>=short_lv))[0]:
            sqty[k]=per_level/short_lv[k]; sprice[k]=short_lv[k]
        # saida do range -> liquida lado perdedor e recentra
        if cl<center*(1-HALF_RANGE) or cl>center*(1+HALF_RANGE):
            if lqty.any():
                inv=float((lqty*cl).sum()); cash+=inv*(1-TAKER_SLIP); lqty[:]=0.0
            if sqty.any():
                pnl=float((sqty*(sprice-cl)).sum())
                cash+=pnl-TAKER_SLIP*float((sqty*cl).sum()); sqty[:]=0.0
            remount(cl)
        elif (not lqty.any()) and (not sqty.any()) and abs(cl-center)/center>2*STEP:
            remount(cl)
        # quebra (proxy de liquidacao)
        eq=equity(cl)
        if eq<=cap0*0.05:
            if lqty.any():
                cash+=float((lqty*cl).sum())*(1-TAKER_SLIP); lqty[:]=0.0
            if sqty.any():
                pnl=float((sqty*(sprice-cl)).sum())
                cash+=pnl-TAKER_SLIP*float((sqty*cl).sum()); sqty[:]=0.0
            cash=max(cash,0.0); broke=True; eq=cash
        daily[idx[i].strftime('%Y-%m-%d')]=eq
        monthly[idx[i].strftime('%Y-%m')]=eq

    # equity de fecho -> P&L por mes
    mkeys=sorted(monthly); out_m={}; prev=cap0
    for mk in mkeys: out_m[mk]=monthly[mk]-prev; prev=monthly[mk]
    dser=pd.Series(daily).sort_index()
    return dict(monthly=out_m, daily=dser, rts=rts, broke=broke, final=prev)


print('SISTEMA 4 — GRID v3: multi-par + alavancagem | base v2 (two-sided 0.8%/R10) | 500 TOTAL')
print('A carregar dados 15m (cache)...')
data={}
for sym in set(s for _,syms in PORTFOLIOS for s in syms):
    try:
        data[sym]=fetch_15m(sym); print(f'  {sym}: OK ({len(data[sym]):,})')
    except Exception as e:
        print(f'  {sym}: ERRO ({e}) — roda a v1 primeiro para gerar o cache')
print()

print('='*112)
print(f"  {'Portfolio':<14}{'lev':>4} |"+''.join(f"{p[0]:>8}" for p in PERIODS)+
      f"{'TOTAL':>9}{'/dia':>7}{'%/dia':>8}{'DD':>7}{'rt/d':>6}{'quebras':>8}")
print('  '+'-'*110)
best=None
for pname,syms in PORTFOLIOS:
    for lev in LEVS:
        cap_each=CAPITAL_TOTAL/len(syms)
        runs=[run_grid(data[s], cap_each, lev) for s in syms]
        # agrega mensal e diario
        allm={}
        for r in runs:
            for mk,v in r['monthly'].items(): allm[mk]=allm.get(mk,0)+v
        dtot=None
        for r in runs:
            dtot=r['daily'] if dtot is None else dtot.add(r['daily'], fill_value=np.nan)
        dtot=dtot.dropna()
        peak=dtot.cummax(); dd=float(((peak-dtot)/peak*100).max())
        ypnl={}
        for mk,v in allm.items(): ypnl[int(mk[:4])]=ypnl.get(int(mk[:4]),0)+v
        tot=sum(ypnl.values()); days=len(dtot)
        per_day=tot/max(days,1); pct_day=per_day/CAPITAL_TOTAL*100
        rts=sum(r['rts'] for r in runs); broke=sum(1 for r in runs if r['broke'])
        pos=sum(1 for p in PERIODS if ypnl.get(p[1],0)>0)
        print(f"  {pname:<14}{lev:>3}x |"+''.join(f"{ypnl.get(p[1],0):>+8.0f}" for p in PERIODS)+
              f"{tot:>+9.0f}{per_day:>+7.2f}{pct_day:>+8.3f}{dd:>6.1f}%{rts/max(days,1):>6.1f}{broke:>8}")
        score=(pos, tot)
        if best is None or score>best[0]:
            best=(score,pname,lev,allm,tot,days,dd)
print()

score,pname,lev,allm,tot,days,dd=best
per_day=tot/max(days,1)
print('='*112)
print(f'  MELHOR: {pname} {lev}x  | {score[0]}/6 periodos+ | total {tot:+.0f} | DD {dd:.1f}%')
print('='*112)
print(f'  {per_day:+.2f} USDC/dia sobre 500 ({per_day/CAPITAL_TOTAL*100:+.3f}%/dia)')
if per_day>0:
    print(f'  META 100/dia -> capital necessario ~{100/(per_day/CAPITAL_TOTAL):,.0f} USDC')
print(f'\n  Levantamento mensal — {pname} {lev}x:')
ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
yrs=sorted(set(int(mk[:4]) for mk in allm))
print(f"  {'Ano':<6} | "+' '.join(f'{m:>5}' for m in ML)+f" | {'Soma':>7}")
print('  '+'-'*95)
for yr in yrs:
    vals=[allm.get(f'{yr}-{m+1:02d}') for m in range(12)]
    cells=' '.join((f'{v:>+5.0f}' if v is not None else '    .') for v in vals)
    print(f"  {yr:<6} | {cells} | {sum(v for v in vals if v is not None):>+7.0f}")
print('\n  Ref v2: ETH 1x = +778 (0.078%/dia), 6/6, DD 22%. Quebra = equity do par <=5% (proxy liquidacao).')
print('  Custos maker 0.02%/lado; liquidacoes 0.07%. Registar no GRID_RESULTS.md.')
