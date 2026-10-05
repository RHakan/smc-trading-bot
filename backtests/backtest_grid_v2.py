"""
backtest_grid_v2.py
SISTEMA 4 — GRID v2: two-sided + sizing por equity + politica por REGIME.

O que a v1 ensinou (GRID_RESULTS.md):
  1. BUG: sizing fixo (500/K) nao acompanha a equity -> apos perdas o grid
     "alavanca" sozinho (SOL: DD 242%). FIX: per_level = equity/K a cada
     (re)montagem do grid.
  2. Grid long-only morre em bear (2022/2026 negativos em TODAS as configs).
     FIX: grid TWO-SIDED em futuros — shorts acima do centro, longs abaixo.
  3. `hold` ganha em bull, `stop` protege em bear -> a politica certa depende
     do REGIME. FIX: modo 'regime' — usa a MESMA EMA/slope do Decisor do bot:
        BULL    -> so lado LONG  (compra os dips; nao shorta forca)
        BEAR    -> so lado SHORT (vende os repiques; nao compra faca)
        NEUTRAL -> two-sided (range = habitat natural do grid)

Modos comparados (mesma grade, mesmos custos):
  long_v1   : v1 (referencia, ja com sizing por equity)
  two_sided : shorts acima + longs abaixo, sempre
  regime    : two-sided com lados ligados/desligados pelo regime (acima)

Saida do range: liquida o lado perdedor (taker+slip 0.07%) e recentra.
Custos maker 0.02%/lado. 15m, BTC/ETH/SOL, 500/par, sem alavancagem.
Periodos: 2021..2025 + 2026 OOS. Uso: python backtests/backtest_grid_v2.py
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
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT']
PERIODS=[('2021',2021),('2022',2022),('2023',2023),('2024',2024),('2025',2025),('2026',2026)]
# Regime no 15m: EMA20 diaria ~ EMA1920 barras 15m; slope de 5 dias = 480 barras
REG_EMA=1920; REG_SLOPE_BARS=480; REG_THRESH=0.001

CACHE_DIR=Path(__file__).parent/'cache'
ex=ccxt.binanceusdm({'enableRateLimit':True})


def fetch_15m(sym):
    safe=sym.replace('/','_').replace(':','_')
    cache=CACHE_DIR/f'{safe}_15m_2021_2026.csv'
    df=pd.read_csv(cache,index_col='ts',parse_dates=True)
    df.index=pd.to_datetime(df.index,utc=True); return df


def run_grid_v2(df, step, half_range, mode):
    """
    Grid v2. mode: 'long_v1' | 'two_sided' | 'regime'.
    Sizing por equity: per_level recalculado a cada (re)montagem do grid.
    """
    h=df['h'].values; l=df['l'].values; c=df['c'].values
    reg=df['regime'].values
    idx=df.index; n=len(df)
    K=max(2,int(half_range/step))

    cash=CAPITAL
    center=c[0]
    long_lv=short_lv=None
    lqty=sqty=None; sprice=None
    per_level=CAPITAL/(2*K if mode!='long_v1' else K)
    rts=0; monthly={}; peak=CAPITAL; max_dd=0.0

    def equity(cl):
        e=cash
        if lqty is not None: e+=float((lqty*cl).sum())
        if sqty is not None: e+=float((sqty*(sprice-cl)).sum())   # pnl dos shorts
        return e

    def remount(cl):
        nonlocal center,long_lv,short_lv,lqty,sqty,sprice,per_level
        center=cl
        long_lv=center*(1-step*np.arange(1,K+1))
        short_lv=center*(1+step*np.arange(1,K+1))
        lqty=np.zeros(K); sqty=np.zeros(K); sprice=short_lv.copy()
        eq=max(equity(cl),1.0)
        per_level=eq/(2*K if mode!='long_v1' else K)

    remount(c[0])

    for i in range(n):
        lo=l[i]; hi=h[i]; cl=c[i]
        allow_long = mode!='regime' or reg[i] in ('BULL','NEUTRAL')
        allow_short = mode in ('two_sided','regime') and (mode!='regime' or reg[i] in ('BEAR','NEUTRAL'))

        # ── fechos com lucro (processa antes de novas entradas) ────────────────
        held=lqty>0
        if held.any():
            tp=long_lv*(1+step)
            for k in np.where(held & (hi>=tp))[0]:
                proceeds=lqty[k]*tp[k]
                cash+=proceeds-MAKER*(lqty[k]*long_lv[k]+proceeds)
                lqty[k]=0.0; rts+=1
        heldS=sqty>0
        if heldS.any():
            tpS=short_lv*(1-step)
            for k in np.where(heldS & (lo<=tpS))[0]:
                pnl=sqty[k]*(sprice[k]-tpS[k])
                cash+=pnl-MAKER*(sqty[k]*(sprice[k]+tpS[k]))
                sqty[k]=0.0; rts+=1

        # ── novas entradas ─────────────────────────────────────────────────────
        if allow_long:
            for k in np.where((lqty==0)&(lo<=long_lv))[0]:
                q=per_level/long_lv[k]
                cash-=per_level; lqty[k]=q
        if allow_short:
            for k in np.where((sqty==0)&(hi>=short_lv))[0]:
                sqty[k]=per_level/short_lv[k]; sprice[k]=short_lv[k]
                # futuros: short nao move cash na abertura (margem implicita)

        # ── saida do range: liquida o lado perdedor e recentra ────────────────
        out_bottom = cl<center*(1-half_range)
        out_top    = cl>center*(1+half_range)
        if out_bottom or out_top:
            # longs em perda no fundo; shorts em perda no topo
            if lqty.any():
                inv=float((lqty*cl).sum()); cash+=inv*(1-TAKER_SLIP); lqty[:]=0.0
            if sqty.any():
                pnl=float((sqty*(sprice-cl)).sum())
                cash+=pnl-TAKER_SLIP*float((sqty*cl).sum()); sqty[:]=0.0
            remount(cl)
        # recentragem "trailing": inventario zero e preco afastou 2 steps
        elif (not lqty.any()) and (not sqty.any()) and abs(cl-center)/center>2*step:
            remount(cl)

        eq=equity(cl)
        peak=max(peak,eq)
        dd=(peak-eq)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
        monthly[idx[i].strftime('%Y-%m')]=eq

    mkeys=sorted(monthly); out_m={}; prev=CAPITAL
    for mk in mkeys: out_m[mk]=monthly[mk]-prev; prev=monthly[mk]
    return dict(monthly=out_m, final=equity(c[-1]), rts=rts, max_dd=max_dd)


print('SISTEMA 4 — GRID v2 (two-sided + sizing por equity + regime) | 500/par | maker 0.04% RT')
print('A carregar dados 15m (cache da v1)...')
data={}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        df=fetch_15m(sym)
        c=df['c']
        ema=c.ewm(span=REG_EMA,adjust=False).mean()
        slope=(ema-ema.shift(REG_SLOPE_BARS))/ema.shift(REG_SLOPE_BARS)
        regv=np.full(len(df),'NEUTRAL',dtype=object)
        regv[(c.values<ema.values)&(slope.values<-REG_THRESH)]='BEAR'
        regv[(c.values>ema.values)&(slope.values> REG_THRESH)]='BULL'
        df['regime']=regv
        data[sym]=df
        print(f'OK ({len(df):,})')
    except Exception as e:
        print(f'ERRO ({e}) — roda a v1 primeiro para gerar o cache 15m')
print()

CONFIGS=[]
for step in [0.008,0.012]:
    for hr in [0.10,0.20]:
        for mode in ['long_v1','two_sided','regime']:
            CONFIGS.append(dict(step=step,half_range=hr,mode=mode))

best=None
for sym,df in data.items():
    print('='*106)
    print(f'  {sym.split("/")[0]} — P&L por periodo (USDC sobre 500, sizing por equity)')
    print('='*106)
    print('  '+f"{'Config':<28}"+''.join(f"{p[0]:>9}" for p in PERIODS)+f"{'TOTAL':>9}{'rt/dia':>8}{'DD':>7}")
    print('  '+'-'*104)
    for cfg in CONFIGS:
        r=run_grid_v2(df, **cfg)
        ypnl={}
        for mk,v in r['monthly'].items():
            ypnl[int(mk[:4])]=ypnl.get(int(mk[:4]),0)+v
        tot=sum(ypnl.values())
        days=(df.index[-1]-df.index[0]).days
        lbl=f"step {cfg['step']*100:.1f}% R{int(cfg['half_range']*100)}% {cfg['mode']}"
        print('  '+f"{lbl:<28}"+''.join(f"{ypnl.get(p[1],0):>+9.0f}" for p in PERIODS)
              +f"{tot:>+9.0f}{r['rts']/max(days,1):>8.1f}{r['max_dd']:>6.1f}%")
        score=(sum(1 for p in PERIODS if ypnl.get(p[1],0)>0), tot)
        if best is None or score>best[0]:
            best=(score,sym,cfg,r,ypnl,days)
print()

score,sym,cfg,r,ypnl,days=best
tot=sum(ypnl.values()); per_day=tot/max(days,1)
print('='*106)
print(f"  MELHOR: {sym.split('/')[0]} step {cfg['step']*100:.1f}% R{int(cfg['half_range']*100)}% {cfg['mode']}  "
      f"| {score[0]}/6 periodos+ | total {tot:+.0f}")
print('='*106)
print(f'  {per_day:+.2f} USDC/dia sobre 500 ({per_day/CAPITAL*100:+.3f}%/dia)')
if per_day>0:
    print(f'  META 100/dia -> capital necessario ~{100/(per_day/CAPITAL):,.0f} USDC')
print(f'\n  Levantamento mensal — {sym.split("/")[0]}, melhor config:')
ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
yrs=sorted(set(int(mk[:4]) for mk in r['monthly']))
print(f"  {'Ano':<6} | "+' '.join(f'{m:>5}' for m in ML)+f" | {'Soma':>7}")
print('  '+'-'*95)
for yr in yrs:
    vals=[r['monthly'].get(f'{yr}-{m+1:02d}') for m in range(12)]
    cells=' '.join((f'{v:>+5.0f}' if v is not None else '    .') for v in vals)
    print(f"  {yr:<6} | {cells} | {sum(v for v in vals if v is not None):>+7.0f}")
print('\n  v1 referencia: ETH 1.2%/R10/stop = +816 (0.081%/dia), 5/6 periodos.')
print('  Custos: maker 0.02%/lado; liquidacao taker+slip 0.07%. Sem alavancagem.')
print('  Registar no GRID_RESULTS.md apos rodar.')
