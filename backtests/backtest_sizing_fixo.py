"""
backtest_sizing_fixo.py
PEDIDO DO RAFA: e se cada ordem usar uma FRACAO FIXA DO CAPITAL como posicao (ex: 50%),
com o stop limitado a uma % da POSICAO (ex: 10%)?
  "banca 100, 2 ordens, cada uma com 50 USDC de posicao, stop maximo 5 USDC"

MUDANCA DE MODELO (e a razao de este teste existir):
  HOJE  — risco fixo: arrisca-se sempre 2% da banca; o TAMANHO da posicao e derivado
          (notional = risco / distancia_do_stop). Stop perto => posicao grande.
  TESTE — posicao fixa: a posicao e sempre F% da banca; o RISCO e que passa a variar
          (risco = notional x distancia_do_stop, limitado por um tecto).
  Sao modelos opostos. No primeiro todos os trades doem igual; no segundo os trades
  de stop largo doem MUITO mais que os de stop apertado.

O TECTO DO STOP: limitar a perda a X% da posicao equivale a limitar a distancia do
stop a X% do PRECO. Se a estrategia quiser um stop mais longe, e cortado no tecto —
o que muda o trade (sai mais cedo), nao so a contabilidade.

Sinais IDENTICOS ao bot (auto_v2: lateral + bull_smc + bear_breakout, com decisor,
filtro BTC e invalidacao). So o dimensionamento e os niveis mudam.

NOTA sobre o limite de ordens: o bot NAO esta em 2 ordens no total. O .env tem
MAX_PER_SIDE=2 e MAX_OPEN_POSITIONS=5 — 2 por lado, ate 5 em simultaneo. Testam-se
os dois cenarios, porque com 50% do capital por ordem isso e a diferenca entre
usar 100% e 250% da banca.

Baseline (modelo de risco fixo, 1%/trade): +347 (5 anos) / +60 (OOS), DD 10.1%, 5/5 anos.
Mesma base de 500/mes com reset, custos 0.14% RT. Bot rodando NAO tocado.
Uso: python backtests/backtest_sizing_fixo.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

_s = importlib.util.spec_from_file_location(
    '_base', str(Path(__file__).parent / 'backtest_v3_trail_alvo.py'))
B = importlib.util.module_from_spec(_s); _s.loader.exec_module(B)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

S5 = (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2025, 10, 6, tzinfo=timezone.utc))
S26 = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 9, tzinfo=timezone.utc))

FRACS = [0.25, 0.50, 0.75, 1.00]      # posicao como fracao da banca
TETOS = [0.05, 0.10, 0.15, 0.20]      # stop maximo, como fracao da POSICAO


def run_fixo(arrs, start, end, frac, teto, max_abertas, max_lado):
    """Motor v3 com dimensionamento por fracao fixa do capital.

    Diferenca central face ao B.run: a contabilidade e feita em RETORNO DE PRECO
    (pnl = notional x (retorno - taxa)), nao em unidades de R. E o que o modelo do
    Rafa pede — a posicao e fixa, o risco e consequencia.
    """
    midx = None
    for d in arrs.values():
        midx = d.index if midx is None else midx.union(d.index)
    midx = midx[(midx >= start) & (midx <= end)]
    A = {}
    for sym, d in arrs.items():
        dd = d.reindex(midx)
        A[sym] = {'h': dd['h'].values, 'l': dd['l'].values, 'c': dd['c'].values,
                  'atr': dd['atr'].values, 'side': dd['_s'].values,
                  'entry': dd['_e'].values, 'sl': dd['_sl'].values, 'tp': dd['_tp'].values,
                  'strat': dd['_st'].values, 'bl': dd['_bl'].values}

    bal = B.MONTHLY_BASE; peak = B.MONTHLY_BASE; ddmax = 0.0
    pos = {}; cdu = {s: -1 for s in A}; monthly = {}; cur = None
    trades = []; riscos = []; cortados = 0; total_sig = 0

    def fecha(sym, px):
        nonlocal bal
        p = pos[sym]
        pr = (px - p['e']) / p['e'] * p['d']          # retorno de preco assinado
        pnl = p['notional'] * (pr - B.RT)             # taxa sobre o notional
        bal += pnl
        trades.append(dict(pnl=pnl, pr=pr, notional=p['notional'],
                           risco_pct=p['risco_pct'], strat=p['strat']))
        del pos[sym]

    for k in range(len(midx)):
        mk = midx[k].strftime('%Y-%m')
        if cur is None: cur = mk
        if mk != cur:
            for sym in list(pos):
                c = A[sym]['c'][k]
                fecha(sym, c if not np.isnan(c) else pos[sym]['e'])
            monthly[cur] = bal - B.MONTHLY_BASE
            bal = B.MONTHLY_BASE; peak = B.MONTHLY_BASE; cur = mk

        for sym in list(pos):
            d = A[sym]; c = d['c'][k]
            if np.isnan(c): continue
            p = pos[sym]; hi = d['h'][k]; lo = d['l'][k]
            saiu = None
            if p['d'] > 0:
                if lo <= p['sl']: saiu = p['sl']
                elif hi >= p['tp']: saiu = p['tp']
            else:
                if hi >= p['sl']: saiu = p['sl']
                elif lo <= p['tp']: saiu = p['tp']
            if saiu is None and p['strat'] == 'lat':
                p['age'] += 1
                if p['age'] <= B.L_INVAL_N and not np.isnan(p['bl']):
                    dentro = (c < p['bl']) if p['d'] > 0 else (c > p['bl'])
                    if dentro: saiu = c
                if saiu is None and p['age'] >= B.L_TIMEOUT: saiu = c
            if saiu is not None:
                fecha(sym, saiu); cdu[sym] = k + p['cd']

        nl = sum(1 for p in pos.values() if p['d'] > 0)
        ns = sum(1 for p in pos.values() if p['d'] < 0)
        for sym in A:
            if len(pos) >= max_abertas: break
            if sym in pos or k <= cdu[sym]: continue
            d = A[sym]; sd = d['side'][k]
            if sd is None or (isinstance(sd, float) and np.isnan(sd)): continue
            e = d['entry'][k]; sl = d['sl'][k]; tp = d['tp'][k]; st = d['strat'][k]
            if np.isnan(e) or np.isnan(sl) or np.isnan(tp) or bal <= 0: continue
            dd_ = 1 if sd == 'LONG' else -1
            if dd_ > 0 and nl >= max_lado: continue
            if dd_ < 0 and ns >= max_lado: continue
            total_sig += 1

            dist = abs(e - sl) / e                    # distancia do stop, em % do preco
            if dist <= 0: continue
            if dist > teto:                           # corta no tecto pedido
                dist = teto
                sl = e - dd_ * teto * e
                cortados += 1
            notional = frac * bal
            pos[sym] = {'d': dd_, 'e': e, 'sl': sl, 'tp': tp, 'notional': notional,
                        'risco_pct': dist * notional / bal * 100,   # % da banca em risco
                        'strat': st, 'bl': d['bl'][k], 'age': 0,
                        'cd': B.GEST_BASE[st]['cd']}
            if dd_ > 0: nl += 1
            else: ns += 1
        peak = max(peak, bal)
        ddmax = max(ddmax, (peak - bal) / peak * 100 if peak > 0 else 0)
        if bal <= 0: break

    for sym in list(pos):
        c = A[sym]['c'][len(midx) - 1]
        fecha(sym, c if not np.isnan(c) else pos[sym]['e'])
    monthly[cur] = bal - B.MONTHLY_BASE

    yr = {}
    for m, v in monthly.items(): yr[m[:4]] = yr.get(m[:4], 0) + v
    pn = np.array([t['pnl'] for t in trades]) if trades else np.array([])
    rk = np.array([t['risco_pct'] for t in trades]) if trades else np.array([])
    return dict(tot=sum(monthly.values()), dd=ddmax, n=len(pn),
                wr=float((pn > 0).mean() * 100) if len(pn) else 0.0,
                anos=sum(1 for v in yr.values() if v > 0), nyr=len(yr), yr=yr,
                risco_med=float(rk.mean()) if len(rk) else 0.0,
                risco_max=float(rk.max()) if len(rk) else 0.0,
                cortados=cortados, tsig=total_sig, monthly=monthly)


def main():
    print('\n' + '=' * 100)
    print('  DIMENSIONAMENTO POR FRACAO FIXA DO CAPITAL (auto_v2, sinais intactos)')
    print('=' * 100)
    print('  A carregar...')
    a5 = B.prep(*B.load(B.fetch_5y)); a26 = B.prep(*B.load(B.fetch_26))
    b5 = B.summ(B.run(a5, *S5)); b26 = B.summ(B.run(a26, *S26))
    print(f"  BASELINE (risco fixo 1%/trade): {b5['tot']:+.0f} | {b26['tot']:+.0f} | "
          f"DD {b5['dd']:.1f}% | {b5['anos']}/5 anos | acerto {b5['wr']:.1f}%")
    if abs(b5['tot'] - 347) > 1:
        print('  ABORTADO: motor nao reproduz o baseline.'); sys.exit(1)
    print()

    for rot, mo, ml in [('O TEU CENARIO — max 2 ordens no total', 2, 2),
                        ('CONFIG REAL DO BOT — max 5 no total, 2 por lado', 5, 2)]:
        print('=' * 100)
        print(f'  {rot}')
        print('=' * 100)
        print(f"  {'posicao':<10}{'tecto stop':<12}{'5 anos':>9}{'OOS':>8}{'DD':>8}"
              f"{'anos+':>7}{'acerto':>8}{'risco medio':>13}{'risco max':>11}{'stops cortados':>16}")
        print('  ' + '-' * 94)
        for frac in FRACS:
            for teto in TETOS:
                r5 = run_fixo(a5, *S5, frac, teto, mo, ml)
                r26 = run_fixo(a26, *S26, frac, teto, mo, ml)
                m = ''
                if r5['tot'] > 347 and r26['tot'] > 60:
                    m = '  <= bate nos 2'
                pc = f"{r5['cortados']}/{r5['tsig']} ({r5['cortados']/max(r5['tsig'],1)*100:.0f}%)"
                print(f"  {frac*100:>6.0f}%   {teto*100:>8.0f}%   {r5['tot']:>+9.0f}"
                      f"{r26['tot']:>+8.0f}{r5['dd']:>7.1f}%{r5['anos']:>5}/5"
                      f"{r5['wr']:>7.1f}%{r5['risco_med']:>12.2f}%{r5['risco_max']:>10.2f}%"
                      f"{pc:>16}{m}")
            print()

    print('=' * 100)
    print('  COMO LER')
    print('=' * 100)
    print('  "risco medio" = quanto da BANCA fica em risco num trade tipico (a distancia')
    print('  do stop x o tamanho da posicao). No modelo atual esse numero e sempre 1%')
    print('  no backtest (2% ao vivo). Aqui varia com cada setup — e essa e a diferenca.')
    print('  "stops cortados" = quantos sinais tinham stop mais longe que o tecto e')
    print('  foram encurtados. Quanto maior, mais o tecto esta a mudar a estrategia.')
    print()


if __name__ == '__main__':
    main()
