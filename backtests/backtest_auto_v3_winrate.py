"""
backtest_auto_v3_winrate.py
AUTO v3 — qual a MAIOR taxa de acerto alcancavel em 15m com os sinais que ja temos?

Pedido do Rafa: pegar nas 3 estrategias do v2 (lateral / bull_smc / bear_breakout),
leva-las para 15m e afinar para o acerto mais alto possivel em trades curtos.

AVISO ENTREGUE ANTES DE CORRER (e a razao de este script medir DUAS coisas):
Maximizar so o acerto tem solucao trivial e inutil — alvo minusculo e stop enorme da
90%+ de acerto e destroi a conta. Ja se viu nesta sessao: encurtar alvos subiu o acerto
de 42.5% para 44.8% e baixou o total de +347 para +302.
Por isso este teste devolve a FRONTEIRA acerto x lucro: para cada nivel de acerto,
qual o melhor total possivel. Assim ve-se o maximo real E o preco de cada degrau.

LIMITE FISICO DO 15m: o ATR ronda 0.4% do preco e as taxas sao 0.14% ida-e-volta.
Um alvo abaixo de ~0.4xATR custa mais em taxas do que rende. A fronteira bate nessa
parede — e o script mostra onde.

DESENHO
  Sinais: os MESMOS do v3 (regime diario -> lateral/bull/bear), com os parametros
  convertidos para tempo-equivalente (x4, porque 4 velas de 15m = 1 de 1h). Assim
  testa-se a mesma logica em granularidade fina, nao uma estrategia diferente.
  Saidas: varridas em multiplos de ATR — stop S x ATR, alvo T x ATR.
  A invalidacao da lateral mantem-se (e o que corta os perdedores cedo).

  Reportado por celula: taxa de acerto, total 5 anos, total OOS, DD, nº de trades.
  No fim: a fronteira (melhor total por banda de acerto) e o veredicto contra o v2.

Baseline v2 (1H): 42.5% de acerto, +347 (5 anos) / +60 (OOS), DD 10.1%.
ATENCAO: a janela OOS aqui acaba a 2026-06-29 (fim do cache 15m), ~10 dias mais curta
que a do baseline. Comparacao APROXIMADA e declarada como tal.
10 pares (SUI no lugar de DOT — e o que existe em 15m). 500/mes reset. Custos 0.14% RT.
Bot rodando NAO tocado.
Uso: python backtests/backtest_auto_v3_winrate.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from bot import regime as regime_mod

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MONTHLY_BASE = 500.0
RISK_PCT = 1.0
MAX_PER_SIDE = 3
PAIRS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT',
         'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'LINK/USDT:USDT', 'SOL/USDT:USDT',
         'AVAX/USDT:USDT', 'SUI/USDT:USDT']
FEE = 0.0005; SLIP = 0.0002; RT = (FEE + SLIP) * 2

R_EMA = regime_mod.EMA_PERIOD; R_SLOPE = regime_mod.SLOPE_BARS
R_THRESH = regime_mod.SLOPE_THRESH

# Parametros x4 (4 velas de 15m = 1 vela de 1h) — mesma janela REAL do v2
M = 4
ATR_P = 14 * M; ADX_P = 14 * M; ATR_AVG_P = 48 * M
L_LB = 30 * M; L_ADX_MAX = 20; L_BUF = 0.1; L_CD = 4 * M
L_INVAL = 5 * M; L_TIMEOUT = 48 * M
B_LB = 30 * M; B_ADX = 20; B_STRUCT = 20 * M; B_CD = 3 * M
U_SWING = 10 * M; U_CHOCH = 12 * M; U_REF = 15 * M; U_MIN_SWEEP = 0.05
U_CD = 3 * M; U_SLOPE_MIN = 0.006; U_EMA_TREND = 100 * M

STOPS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]        # S x ATR
ALVOS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # T x ATR

S5 = (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2025, 10, 6, tzinfo=timezone.utc))
S26 = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 6, 29, tzinfo=timezone.utc))
CACHE = Path(__file__).parent / 'cache'


def _read(sym, tf, tag):
    f = CACHE / f"{sym.replace('/', '_').replace(':', '_')}_{tf}_{tag}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, index_col='ts', parse_dates=True)
    d.index = pd.to_datetime(d.index, utc=True)
    return d


def adx(df, period):
    h, l, c = df['h'], df['l'], df['c']
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    at = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / at
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / at
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def load(period):
    a, b = S5 if period == '5y' else S26
    tag_d = '2020_2025' if period == '5y' else '2025_2026oos'
    out = {}; btcb = None
    for sym in PAIRS:
        dfd = _read(sym, '1d', tag_d)
        df = _read(sym, '15m', '2021_2026')
        if dfd is None or df is None:
            continue
        df = df[(df.index >= a) & (df.index <= b)].copy()
        if len(df) < L_LB + ATR_P + 500:
            continue
        ema_d = dfd['c'].ewm(span=R_EMA, adjust=False).mean()
        slope = (ema_d - ema_d.shift(R_SLOPE)) / ema_d.shift(R_SLOPE)
        dfd['regime'] = 'NEUTRAL'
        dfd.loc[(dfd['c'] < ema_d) & (slope < -R_THRESH), 'regime'] = 'BEAR'
        dfd.loc[(dfd['c'] > ema_d) & (slope > R_THRESH), 'regime'] = 'BULL'
        dfd['slope_d'] = slope
        c = df['c']
        tr = pd.concat([(df['h'] - df['l']), (df['h'] - c.shift(1)).abs(),
                        (df['l'] - c.shift(1)).abs()], axis=1).max(axis=1)
        df['atr'] = tr.ewm(com=ATR_P - 1, adjust=False).mean()
        df['atr_avg'] = df['atr'].rolling(ATR_AVG_P).mean()
        df['adx'] = adx(df, ADX_P)
        df['regime'] = dfd['regime'].shift(1).reindex(df.index, method='ffill')
        df['slope_d'] = dfd['slope_d'].shift(1).reindex(df.index, method='ffill')
        df['ema_trend'] = c.ewm(span=U_EMA_TREND, adjust=False).mean()
        out[sym] = df
        if sym.startswith('BTC'):
            btcb = (df['regime'] == 'BULL')
    return out, btcb


def gen(df, btc_local):
    """Sinais v3 em 15m. Devolve tambem o nivel de invalidacao da lateral."""
    n = len(df)
    o = df['o'].values; h = df['h'].values; l = df['l'].values; c = df['c'].values
    atr = df['atr'].values; atr_avg = df['atr_avg'].values; adx_a = df['adx'].values
    reg = df['regime'].values; slope = df['slope_d'].values; ema_t = df['ema_trend'].values
    side = np.array([None] * n, dtype=object); entry = np.full(n, np.nan)
    strat = np.array([None] * n, dtype=object); blevel = np.full(n, np.nan)
    start_i = max(ATR_AVG_P + B_LB, U_SWING + U_REF + U_CHOCH, L_LB, U_EMA_TREND) + 10

    for i in range(start_i, n):
        r = reg[i]; a = atr[i]
        if np.isnan(a) or a <= 0: continue
        if r == 'NEUTRAL':
            if np.isnan(adx_a[i - 1]) or adx_a[i - 1] > L_ADX_MAX: continue
            rl = np.min(l[i - L_LB:i]); rh = np.max(h[i - L_LB:i])
            if rl <= 0 or rh <= rl: continue
            buf = L_BUF * a; cl = c[i]; pc = c[i - 1]
            if cl > rh + buf and pc <= rh and cl > rl:
                side[i] = 'LONG'; entry[i] = cl; strat[i] = 'lat'; blevel[i] = rh
            elif cl < rl - buf and pc >= rl and rh > cl:
                side[i] = 'SHORT'; entry[i] = cl; strat[i] = 'lat'; blevel[i] = rl
        elif r == 'BULL':
            if not btc_local[i]: continue
            if np.isnan(slope[i]) or slope[i] < U_SLOPE_MIN: continue
            if np.isnan(ema_t[i]) or c[i] <= ema_t[i]: continue
            av = atr_avg[i]
            if not np.isnan(av) and a < av: continue
            for j in range(i - 1, max(i - U_CHOCH - 1, U_SWING + U_REF) - 1, -1):
                sw = np.min(l[j - U_SWING:j])
                if not (l[j] < sw and c[j] > sw): continue
                if U_MIN_SWEEP > 0 and (sw - l[j]) / sw * 100 < U_MIN_SWEEP: continue
                rh2 = np.max(h[j - U_REF:j])
                if np.any(c[j + 1:i] > rh2): continue
                if c[i] > rh2:
                    side[i] = 'LONG'; entry[i] = c[i]; strat[i] = 'bull'
                    break
        elif r == 'BEAR':
            cl = c[i]; op = o[i]; av = atr_avg[i]
            if np.isnan(av): continue
            if not (cl < np.min(l[i - B_LB:i]) and cl < op and a > av): continue
            if np.isnan(adx_a[i - 1]) or adx_a[i - 1] > B_ADX: continue
            side[i] = 'SHORT'; entry[i] = cl; strat[i] = 'bear'
    return side, entry, strat, blevel


def prep(pdata, btcb):
    out = {}
    for sym, df in pdata.items():
        s, e, st, bl = gen(df, btcb.reindex(df.index).fillna(False).values)
        out[sym] = df.assign(_s=s, _e=e, _st=st, _bl=bl)
    return out


CD = {'lat': L_CD, 'bull': U_CD, 'bear': B_CD}


def run(arrs, start, end, S, T):
    """Stop = S x ATR, alvo = T x ATR (ambos fixados na entrada)."""
    midx = None
    for d in arrs.values():
        midx = d.index if midx is None else midx.union(d.index)
    midx = midx[(midx >= start) & (midx <= end)]
    A = {}
    for sym, d in arrs.items():
        dd = d.reindex(midx)
        A[sym] = {'h': dd['h'].values, 'l': dd['l'].values, 'c': dd['c'].values,
                  'atr': dd['atr'].values, 'side': dd['_s'].values,
                  'entry': dd['_e'].values, 'strat': dd['_st'].values, 'bl': dd['_bl'].values}
    bal = MONTHLY_BASE; peak = MONTHLY_BASE; ddmax = 0.0
    pos = {}; cd = {s: -1 for s in A}; monthly = {}; cur = None; trades = []

    def fecha(sym, nr):
        nonlocal bal
        bal += nr * pos[sym]['risk_usd']; trades.append(nr); del pos[sym]

    for k in range(len(midx)):
        mk = midx[k].strftime('%Y-%m')
        if cur is None: cur = mk
        if mk != cur:
            for sym in list(pos):
                d = A[sym]; c = d['c'][k]; p = pos[sym]
                if np.isnan(c): c = p['entry']
                nr = ((c - p['entry']) if p['side'] == 'LONG' else (p['entry'] - c)) / p['risk'] - p['fee']
                fecha(sym, nr)
            monthly[cur] = bal - MONTHLY_BASE
            bal = MONTHLY_BASE; peak = MONTHLY_BASE; cur = mk

        for sym in list(pos):
            d = A[sym]; c = d['c'][k]
            if np.isnan(c): continue
            p = pos[sym]; hi = d['h'][k]; lo = d['l'][k]
            e = p['entry']; risk = p['risk']; fee = p['fee']
            done = False; nr = 0.0
            if p['side'] == 'LONG':
                if lo <= p['sl']: nr = (p['sl'] - e) / risk - fee; done = True
                elif hi >= p['tp']: nr = (p['tp'] - e) / risk - fee; done = True
            else:
                if hi >= p['sl']: nr = (e - p['sl']) / risk - fee; done = True
                elif lo <= p['tp']: nr = (e - p['tp']) / risk - fee; done = True
            if not done and p['strat'] == 'lat':
                p['age'] += 1
                if p['age'] <= L_INVAL and not np.isnan(p['bl']):
                    dentro = (c < p['bl']) if p['side'] == 'LONG' else (c > p['bl'])
                    if dentro:
                        nr = ((c - e) if p['side'] == 'LONG' else (e - c)) / risk - fee; done = True
                if not done and p['age'] >= L_TIMEOUT:
                    nr = ((c - e) if p['side'] == 'LONG' else (e - c)) / risk - fee; done = True
            if done:
                fecha(sym, nr); cd[sym] = k + CD[p['strat']] if sym not in pos else cd[sym]

        nl = sum(1 for p in pos.values() if p['side'] == 'LONG')
        ns = sum(1 for p in pos.values() if p['side'] == 'SHORT')
        for sym in A:
            if sym in pos or k <= cd[sym]: continue
            d = A[sym]; sd = d['side'][k]
            if sd is None or (isinstance(sd, float) and np.isnan(sd)): continue
            e = d['entry'][k]; a = d['atr'][k]; st = d['strat'][k]
            if np.isnan(e) or np.isnan(a) or a <= 0 or bal <= 0: continue
            if sd == 'LONG' and nl >= MAX_PER_SIDE: continue
            if sd == 'SHORT' and ns >= MAX_PER_SIDE: continue
            risk = S * a
            sl = e - risk if sd == 'LONG' else e + risk
            tp = e + T * a if sd == 'LONG' else e - T * a
            pos[sym] = {'side': sd, 'entry': e, 'sl': sl, 'tp': tp, 'risk': risk,
                        'fee': e * RT / risk, 'risk_usd': bal * (RISK_PCT / 100.0),
                        'strat': st, 'bl': d['bl'][k], 'age': 0}
            if sd == 'LONG': nl += 1
            else: ns += 1
        peak = max(peak, bal)
        ddmax = max(ddmax, (peak - bal) / peak * 100 if peak > 0 else 0)

    for sym in list(pos):
        p = pos[sym]; c = A[sym]['c'][len(midx) - 1]
        if np.isnan(c): c = p['entry']
        nr = ((c - p['entry']) if p['side'] == 'LONG' else (p['entry'] - c)) / p['risk'] - p['fee']
        fecha(sym, nr)
    monthly[cur] = bal - MONTHLY_BASE
    tr = np.array(trades)
    return dict(tot=sum(monthly.values()), dd=ddmax, n=len(tr),
                wr=float((tr > 0.03).mean() * 100) if len(tr) else 0.0,
                monthly=monthly)


def main():
    print('\n' + '=' * 104)
    print('  AUTO v3 — FRONTEIRA ACERTO x LUCRO em 15m (sinais do v2, saidas varridas)')
    print('=' * 104)
    print('  Baseline v2 (1H): 42.5% de acerto | +347 (5 anos) | +60 (OOS) | DD 10.1%')
    print('  NOTA: OOS aqui acaba a 2026-06-29 (fim do cache 15m) — comparacao aproximada.\n')
    print('  A carregar 15m...')
    a5 = prep(*load('5y')); a26 = prep(*load('oos'))
    atrpct = np.median([np.nanmedian(d['atr'].values / d['c'].values) for d in a5.values()])
    print(f'  {len(a5)} pares | ATR mediano em 15m = {atrpct*100:.3f}% do preco')
    print(f'  Taxa ida-e-volta = {RT*100:.2f}% = {RT/atrpct:.2f} x ATR')
    print(f'  >> Um alvo abaixo de {RT/atrpct:.2f} xATR nao paga sequer as taxas.\n')

    print(f"  {'stop':<8}{'alvo':<8}{'R:R':>6}{'acerto':>9}{'5 anos':>9}{'OOS':>8}"
          f"{'DD':>7}{'trades':>9}")
    print('  ' + '-' * 66)
    res = []
    for S in STOPS:
        for T in ALVOS:
            r5 = run(a5, *S5, S, T); r26 = run(a26, *S26, S, T)
            res.append(dict(S=S, T=T, rr=T / S, wr=r5['wr'], tot=r5['tot'],
                            oos=r26['tot'], dd=r5['dd'], n=r5['n'],
                            m5=r5['monthly'], m26=r26['monthly']))
            flag = ''
            if T < RT / atrpct:
                flag = '  taxa > alvo'
            elif r5['tot'] > 347 and r26['tot'] > 60:
                flag = '  <= bate o v2 nos 2'
            print(f"  {S:<8.1f}{T:<8.2f}{T/S:>6.2f}{r5['wr']:>8.1f}%{r5['tot']:>+9.0f}"
                  f"{r26['tot']:>+8.0f}{r5['dd']:>6.1f}%{r5['n']:>9}{flag}")
        print()

    # ── FRONTEIRA ────────────────────────────────────────────────────────────
    print('=' * 104)
    print('  FRONTEIRA — melhor resultado possivel para cada banda de acerto')
    print('=' * 104)
    print(f"  {'banda de acerto':<20}{'melhor 5 anos':>15}{'OOS':>9}{'stop/alvo':>13}"
          f"{'R:R':>7}{'DD':>7}")
    print('  ' + '-' * 72)
    bandas = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
    for lo, hi in bandas:
        cel = [r for r in res if lo <= r['wr'] < hi]
        if not cel:
            print(f"  {f'{lo}-{hi}%':<20}{'(nenhuma celula)':>15}")
            continue
        b = max(cel, key=lambda r: r['tot'])
        print(f"  {f'{lo}-{hi}%':<20}{b['tot']:>+15.0f}{b['oos']:>+9.0f}"
              f"{f'{b[chr(83)]}/{b[chr(84)]}':>13}{b['rr']:>7.2f}{b['dd']:>6.1f}%")
    print()
    mx = max(res, key=lambda r: r['wr'])
    print(f"  ACERTO MAXIMO ALCANCAVEL: {mx['wr']:.1f}%  (stop {mx['S']}xATR / alvo {mx['T']}xATR)")
    print(f"    -> total 5 anos {mx['tot']:+.0f} | OOS {mx['oos']:+.0f} | DD {mx['dd']:.1f}%"
          f" | {mx['n']} trades")
    lucr = [r for r in res if r['tot'] > 0 and r['oos'] > 0]
    if lucr:
        bwr = max(lucr, key=lambda r: r['wr'])
        print(f"\n  MAIOR ACERTO **ainda lucrativo nos dois periodos**: {bwr['wr']:.1f}%")
        print(f"    stop {bwr['S']}xATR / alvo {bwr['T']}xATR (R:R {bwr['rr']:.2f})"
              f" | 5 anos {bwr['tot']:+.0f} | OOS {bwr['oos']:+.0f} | DD {bwr['dd']:.1f}%")
    melhor = max(res, key=lambda r: r['tot'] + r['oos'])
    print(f"\n  MELHOR RESULTADO (independente do acerto): {melhor['tot']:+.0f} / {melhor['oos']:+.0f}"
          f"  com acerto de {melhor['wr']:.1f}%  (stop {melhor['S']}x / alvo {melhor['T']}x)")
    print()

    bate = [r for r in res if r['tot'] > 347 and r['oos'] > 60]
    print('=' * 104)
    print('  VEREDICTO vs AUTO v2')
    print('=' * 104)
    if bate:
        print(f'  {len(bate)} celula(s) batem o v2 nos dois periodos:')
        for r in sorted(bate, key=lambda x: -(x['tot'] + x['oos']))[:5]:
            print(f"    stop {r['S']}x / alvo {r['T']}x  R:R {r['rr']:.2f}  acerto {r['wr']:.1f}%"
                  f"  -> {r['tot']:+.0f} / {r['oos']:+.0f}  DD {r['dd']:.1f}%")
        print('\n  Estas celulas ainda TEM de passar a bateria de confirmacao')
        print('  (planalto, sub-periodos, ano a ano, mes a mes) antes de valerem alguma coisa.')
    else:
        print('  NENHUMA celula bate o v2 nos dois periodos. O 15m nao produz um v3 melhor')
        print('  do que o v2 em 1H — nem no total, nem com acerto alto.')
    print()


if __name__ == '__main__':
    main()
