"""
backtest_grid_v1.py
SISTEMA 4 (novo, separado do v3) — GRID TRADING v1: grid long classico.

GOAL do Rafa: estrategia de alta frequencia (grid/market-making/scalp) que
persiga 100 USDC/dia. Regra anti-ilusao: 100/dia sobre 500 = 20%/dia — nenhum
sistema legitimo faz isso. Medimos o %/dia REAL e calculamos o capital que
seria necessario para os 100/dia. Documentar cada versao em GRID_RESULTS.md,
testando SEMPRE em periodos de mercado diferentes (2021 bull, 2022 bear,
2023 recup, 2024 bull, 2025 misto, 2026 OOS).

v1 = grid long classico (estilo Pionex/Binance grid bot):
  - Range [center*(1-R), center] com K niveis de COMPRA espacados step%.
  - Cada compra no nivel L vira ordem de venda em L*(1+step) -> lucro = step - custos.
  - Preco sobe alem do topo com inventario zerado -> grid RECENTRA para cima (trailing).
  - Preco cai abaixo do fundo:
      hold : segura o inventario ate voltar (risco: capital preso em queda longa)
      stop : liquida tudo a mercado (perda realizada) e recentra em baixo
  - Ordens limit -> custo MAKER 0.02%/lado (round-trip 0.04%). Liquidacao de
    stop usa taker 0.05%+slip.

Timeframe 15m (5.5 anos, ~190k candles/par). Pares: BTC, ETH, SOL (500 USDC cada,
sem alavancagem — lev multiplica P&L e DD linearmente, reportado no final).
Uso: python backtests/backtest_grid_v1.py
"""
import sys, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

FETCH_START=datetime(2021,1,1,tzinfo=timezone.utc)
FETCH_END  =datetime(2026,6,29,tzinfo=timezone.utc)
CAPITAL=500.0
MAKER=0.0002              # 0.02%/lado (ordens limit)
TAKER_SLIP=0.0007         # liquidacao de stop: taker 0.05% + slip 0.02%
PAIRS=['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT']
PERIODS=[('2021 bull',2021),('2022 BEAR',2022),('2023 recup',2023),
         ('2024 bull',2024),('2025 misto',2025),('2026 OOS',2026)]

CACHE_DIR=Path(__file__).parent/'cache'; CACHE_DIR.mkdir(exist_ok=True)
ex=ccxt.binanceusdm({'enableRateLimit':True})


def fetch_15m(sym):
    safe=sym.replace('/','_').replace(':','_')
    cache=CACHE_DIR/f'{safe}_15m_2021_2026.csv'
    if cache.exists():
        df=pd.read_csv(cache,index_col='ts',parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True); return df
    print(f'    (baixando 15m de {sym} — ~190k candles, alguns minutos)')
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


def run_grid(df, step, half_range, stop_mode):
    """
    Grid long em 1 par com CAPITAL fixo. Devolve dict com P&L por ano/mes,
    round-trips, DD sobre equity.
    """
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    idx=df.index
    n=len(df)
    K=max(2,int(half_range/step))          # nº de niveis de compra
    per_level=CAPITAL/K                    # USDC por nivel

    center=c[0]
    buy_lv=center*(1-step*np.arange(1,K+1))
    bought=np.zeros(K,bool); qty=np.zeros(K)
    cash=CAPITAL
    rts=0                                   # round-trips concluidos
    monthly={}; yearly_rt={}                # monthly = equity de FECHO por mes
    peak=CAPITAL; max_dd=0.0

    for i in range(n):
        lo=l[i]; hi=h[i]; cl=c[i]
        # ── vendas primeiro (take 1 step acima dos niveis comprados) ──────────
        if bought.any():
            tp_lv=buy_lv*(1+step)
            sell=bought & (hi>=tp_lv)
            if sell.any():
                for k in np.where(sell)[0]:
                    proceeds=qty[k]*tp_lv[k]
                    fees=MAKER*(qty[k]*buy_lv[k]+proceeds)
                    cash+=proceeds-fees
                    bought[k]=False; qty[k]=0.0
                    rts+=1
                    yr=idx[i].year; yearly_rt[yr]=yearly_rt.get(yr,0)+1
        # ── compras (limit abaixo) ─────────────────────────────────────────────
        fill=(~bought)&(lo<=buy_lv)
        if fill.any():
            for k in np.where(fill)[0]:
                q=per_level/buy_lv[k]
                cash-=per_level          # taxa cobrada na venda (agregada acima)
                qty[k]=q; bought[k]=True
        # ── recentragem para CIMA (inventario zerado e preco acima do centro) ──
        if not bought.any() and cl>center:
            center=cl
            buy_lv=center*(1-step*np.arange(1,K+1))
        # ── fundo do grid ─────────────────────────────────────────────────────
        bottom=center*(1-half_range)
        if cl<bottom:
            if stop_mode=='stop':
                # liquida tudo a mercado e recentra em baixo
                inv_val=float((qty*cl).sum())
                if inv_val>0:
                    cash+=inv_val*(1-TAKER_SLIP)
                bought[:]=False; qty[:]=0.0
                center=cl
                buy_lv=center*(1-step*np.arange(1,K+1))
            # hold: espera voltar (niveis todos comprados, sem acao)
        # ── equity / DD / mensal ───────────────────────────────────────────────
        eq=cash+float((qty*cl).sum())
        peak=max(peak,eq)
        dd=(peak-eq)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
        monthly[idx[i].strftime('%Y-%m')]=eq   # sobrescreve ate ficar o fecho do mes
    # converte equity de fecho por mes -> P&L mensal (diferencas)
    mkeys=sorted(monthly.keys()); out_m={}
    prev=CAPITAL
    for mk in mkeys:
        out_m[mk]=monthly[mk]-prev; prev=monthly[mk]
    final_eq=cash+float((qty*c[-1]).sum())
    return dict(monthly=out_m, final=final_eq, rts=rts, yearly_rt=yearly_rt, max_dd=max_dd)


print('SISTEMA 4 — GRID v1 (grid long classico) | capital 500/par | custo maker 0.04% RT')
print('A carregar dados 15m 2021-2026 (baixa e faz cache na 1a vez — pode demorar)...')
data={}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        df=fetch_15m(sym)
        if len(df)<10000: print('SEM DADOS'); continue
        data[sym]=df
        print(f'OK ({len(df):,} candles)')
    except Exception as e:
        print(f'ERRO ({e})')
print()

CONFIGS=[]
for step in [0.005,0.008,0.012]:
    for hr in [0.10,0.20]:
        for sm in ['hold','stop']:
            CONFIGS.append(dict(step=step,half_range=hr,stop_mode=sm))

best=None
for sym,df in data.items():
    print('='*108)
    print(f'  {sym.split("/")[0]} — P&L por periodo (USDC sobre 500, sem alavancagem)')
    print('='*108)
    hdr='  '+f"{'Config':<26}"+''.join(f"{p[0].split()[0]:>9}" for p in PERIODS)+f"{'TOTAL':>9}{'rt/dia':>8}{'DD':>7}"
    print(hdr); print('  '+'-'*106)
    for cfg in CONFIGS:
        r=run_grid(df, **cfg)
        ypnl={}
        for mk,v in r['monthly'].items():
            ypnl[int(mk[:4])]=ypnl.get(int(mk[:4]),0)+v
        tot=sum(ypnl.values())
        days=(df.index[-1]-df.index[0]).days
        rt_day=r['rts']/max(days,1)
        lbl=f"step {cfg['step']*100:.1f}% R{int(cfg['half_range']*100)}% {cfg['stop_mode']}"
        row='  '+f"{lbl:<26}"+''.join(f"{ypnl.get(p[1],0):>+9.0f}" for p in PERIODS)+f"{tot:>+9.0f}{rt_day:>8.1f}{r['max_dd']:>6.1f}%"
        print(row)
        score=(sum(1 for p in PERIODS if ypnl.get(p[1],0)>0), tot)
        if best is None or score>best[0]:
            best=(score, sym, cfg, r, ypnl, days)
print()

score,sym,cfg,r,ypnl,days=best
tot=sum(ypnl.values())
per_day=tot/max(days,1)
pct_day=per_day/CAPITAL*100
print('='*108)
print(f'  MELHOR (periodos+ depois total): {sym.split("/")[0]} step {cfg["step"]*100:.1f}% R{int(cfg["half_range"]*100)}% {cfg["stop_mode"]}')
print('='*108)
print(f'  Total {tot:+.0f} USDC em {days} dias  ->  {per_day:+.2f} USDC/dia sobre 500  ({pct_day:+.3f}%/dia)')
if per_day>0:
    cap_needed=100/ (per_day/CAPITAL)
    print(f'  META 100/dia: com este %/dia seriam necessarios ~{cap_needed:,.0f} USDC de capital')
    print(f'  (ou alavancagem equivalente — mas lev multiplica o DD na mesma proporcao)')
print(f'\n  Levantamento mensal (regra do Rafa) — {sym.split("/")[0]}, melhor config:')
ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
yrs=sorted(set(int(mk[:4]) for mk in r['monthly']))
print(f"  {'Ano':<6} | "+' '.join(f'{m:>5}' for m in ML)+f" | {'Soma':>7}")
print('  '+'-'*95)
for yr in yrs:
    vals=[r['monthly'].get(f'{yr}-{m+1:02d}') for m in range(12)]
    cells=' '.join((f'{v:>+5.0f}' if v is not None else '    .') for v in vals)
    s=sum(v for v in vals if v is not None)
    print(f"  {yr:<6} | {cells} | {s:>+7.0f}")
print('\n  Notas: 15m, ordens limit (maker 0.02%/lado). stop usa taker+slip 0.07%.')
print('  Alavancagem nao modelada: 2x dobra P&L E DD (liquidacao ignorada — cuidado).')
print('  Proxima versao: registar em backtests/GRID_RESULTS.md e iterar (grid neutro, recentragem, TF 5m).')
