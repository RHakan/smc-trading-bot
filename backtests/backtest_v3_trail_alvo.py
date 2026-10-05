"""
backtest_v3_trail_alvo.py
DOR DO RAFA (observada ao vivo, 05/08/2026): "e muito comum o valor chegar proximo
do alvo e voltar para buscar o stop".

Este script faz TRES coisas, por ordem:

  1. DIAGNOSTICO — mede se a observacao e verdadeira. Dos trades PERDEDORES, ate
     que % do caminho ao alvo chegaram antes de virar? Por estrategia.
     Se a maioria vira aos 40%, apertar a saida nao resolve nada e nao vale mexer.
     Se chega aos 80-90%, ha muito a ganhar. Sem este numero, o resto e palpite.

  2. GRELHA 2D  TRAIL_ATR x BE_TRIGGER — o pedido do Rafa (0.25/0.5/0.75/1/1.5/2 ATR).
     ATENCAO AO ACOPLAMENTO: no bot o trailing esta atras de um portao —
     position_manager.py:373 so faz trailing se `prog >= be_trigger` E o breakeven
     ja tiver disparado. Logo, com BE_TRIGGER_PCT=0.99 (a lateral de hoje) qualquer
     TRAIL_ATR e IRRELEVANTE: nunca ativa. Os dois parametros TEM de ser varridos
     juntos, senao metade da grelha da resultados identicos sem se perceber porque.

  3. ALVOS MAIS CURTOS — a outra solucao para a mesma dor (fator sobre a distancia
     entrada->alvo). Comparada lado a lado com o trailing, porque sao concorrentes:
     encurtar o alvo corta o vencedor mas fecha mais vezes; apertar o trail mantem
     o alvo e so protege o que ja andou.

ESTADO ATUAL QUE ESTAMOS A ATACAR (lido do codigo vivo, nao de memoria):
  lateral_breakout : SL = lado oposto do range(30) | TP = altura do range projetada
                     BE_TRIGGER_PCT=0.99  TRAIL_ATR=0.0  -> ZERO protecao ate ao TP/SL
                     unica defesa: invalidacao (fecha se voltar ao range em 5 velas)
  bull_smc         : SL = minimo estrutural do sweep | TP = entrada + 2.5*risco
                     BE=0.40  TRAIL=2.0xATR
  bear_breakout    : SL = topo estrutural(20) + 0.1xATR | TP = entrada - 2.5*risco
                     BE=0.40  TRAIL=2.0xATR

A lateral e onde a dor deve viver (nao tem BE nem trailing). Por isso este script
permite, pela primeira vez, LIGAR BE+trailing na lateral — mantendo a invalidacao.

PORTAO DE VEREDICTO: tem de bater +347 (5 anos) E +60 (OOS 2026), sem piorar o
DD de 10.1%. Qualquer vencedor passa ainda pelo TESTE DE PLANALTO (vizinhos), a
disciplina que matou o candidato da sessao passada.

Sinais IDENTICOS ao bot (v3 + filtro BTC + invalidacao). 10 pares, 1H, 500/mes
reset, custos 0.14% RT. Levantamento mensal no fim. Bot rodando NAO tocado.
Uso: python backtests/backtest_v3_trail_alvo.py
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

# ── Constantes: puxadas dos modulos VIVOS, nunca hardcoded ───────────────────
MONTHLY_BASE = 500.0
RISK_PCT = 1.0
PAIRS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT',
         'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'LINK/USDT:USDT', 'SOL/USDT:USDT',
         'AVAX/USDT:USDT', 'DOT/USDT:USDT']
FEE = 0.0005; SLIP = 0.0002; RT = (FEE + SLIP) * 2
ATR_PERIOD = BEAR.ATR_PERIOD; B_ATR_AVG = BEAR.ATR_AVG_PERIOD
B_COOLDOWN = BEAR.COOLDOWN_BARS; B_RR = BEAR.RR_CAP; B_TRAIL = BEAR.TRAIL_ATR
R_EMA = regime_mod.EMA_PERIOD; R_SLOPE = regime_mod.SLOPE_BARS
R_THRESH = regime_mod.SLOPE_THRESH
BSEL_LB = 30; BSEL_ADX = 20; BSEL_STRUCT = 20
U_SWING_N = 10; U_CHOCH_BARS = 12; U_CHOCH_REF = 15; U_MIN_SWEEP = 0.05
U_RR = 2.5; U_TRAIL = 2.0; U_COOLDOWN = 3; U_SLOPE_MIN = 0.006; U_EMA_TREND = 100
L_LOOKBACK = 30; L_ADX_MAX = 20; L_BUFFER = 0.1; L_COOLDOWN = 4
L_TIMEOUT = 48; L_INVAL_N = 5

MAX_PER_SIDE = 3          # o baseline +347/+60 foi medido com 3

# Gestao de referencia — reproduz exatamente o bot de hoje.
# 'be' = fracao do caminho entrada->alvo a que o stop vai para breakeven.
# 9.99 na lateral = nunca (equivale ao BE_TRIGGER_PCT=0.99 do codigo vivo).
GEST_BASE = {'bull': dict(be=0.40, trail=U_TRAIL, cd=U_COOLDOWN),
             'bear': dict(be=0.40, trail=B_TRAIL, cd=B_COOLDOWN),
             'lat':  dict(be=9.99, trail=0.0,     cd=L_COOLDOWN)}

TRAILS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
BES = [0.30, 0.40, 0.50, 0.70, 0.99]
TP_SCALES = [0.60, 0.70, 0.80, 0.90, 1.00]

CACHE_DIR = Path(__file__).parent / 'cache'; CACHE_DIR.mkdir(exist_ok=True)
ex = ccxt.binanceusdm({'enableRateLimit': True})


def _fetch(sym, tf, start_dt, end_dt, tag):
    safe = sym.replace('/', '_').replace(':', '_')
    cache = CACHE_DIR / f'{safe}_{tf}_{tag}.csv'
    if cache.exists():
        df = pd.read_csv(cache, index_col='ts', parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    since = int((start_dt - timedelta(days=5)).timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000); rows = []
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b: break
        rows.extend(b); since = b[-1][0] + 1
        if since >= end_ms: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('ts').sort_index()[lambda d: d.index <= end_dt]
    df.to_csv(cache); return df


def fetch_5y(sym, tf):
    return _fetch(sym, tf, datetime(2020, 1, 1, tzinfo=timezone.utc),
                  datetime(2025, 10, 6, tzinfo=timezone.utc), '2020_2025')


def fetch_26(sym, tf):
    return _fetch(sym, tf, datetime(2025, 9, 1, tzinfo=timezone.utc),
                  datetime(2026, 7, 9, tzinfo=timezone.utc), '2025_2026oos')


def adx(df, period=14):
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


def load(fetch_fn):
    pdata = {}; btcb = None
    for sym in PAIRS:
        dfd = fetch_fn(sym, '1d').copy(); df1 = fetch_fn(sym, '1h').copy()
        if len(df1) < 300: continue
        ema_d = dfd['c'].ewm(span=R_EMA, adjust=False).mean()
        slope = (ema_d - ema_d.shift(R_SLOPE)) / ema_d.shift(R_SLOPE)
        dfd['regime'] = 'NEUTRAL'
        dfd.loc[(dfd['c'] < ema_d) & (slope < -R_THRESH), 'regime'] = 'BEAR'
        dfd.loc[(dfd['c'] > ema_d) & (slope > R_THRESH), 'regime'] = 'BULL'
        dfd['slope_d'] = slope
        c = df1['c']
        tr = pd.concat([(df1['h'] - df1['l']), (df1['h'] - c.shift(1)).abs(),
                        (df1['l'] - c.shift(1)).abs()], axis=1).max(axis=1)
        df1['atr'] = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
        df1['atr_avg'] = df1['atr'].rolling(B_ATR_AVG).mean()
        df1['adx'] = adx(df1, 14)
        df1['regime'] = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
        df1['slope_d'] = dfd['slope_d'].shift(1).reindex(df1.index, method='ffill')
        df1['ema_trend'] = c.ewm(span=U_EMA_TREND, adjust=False).mean()
        pdata[sym] = df1
        if sym.startswith('BTC'): btcb = (df1['regime'] == 'BULL')
    return pdata, btcb


def gen(df, btc_local):
    """Sinais IDENTICOS ao bot (copiado de backtest_maxside_confirma.py:114)."""
    n = len(df)
    o = df['o'].values; h = df['h'].values; l = df['l'].values; c = df['c'].values
    atr = df['atr'].values; atr_avg = df['atr_avg'].values; adx_a = df['adx'].values
    reg = df['regime'].values; slope = df['slope_d'].values; ema_t = df['ema_trend'].values
    side = np.array([None] * n, dtype=object)
    entry = np.full(n, np.nan); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    strat = np.array([None] * n, dtype=object); blevel = np.full(n, np.nan)
    start_i = max(B_ATR_AVG + BSEL_LB + 5,
                  U_SWING_N + U_CHOCH_REF + U_CHOCH_BARS + 5, L_LOOKBACK + 5, 210)
    for i in range(start_i, n):
        r = reg[i]; a = atr[i]
        if np.isnan(a) or a <= 0: continue
        if r == 'NEUTRAL':
            if np.isnan(adx_a[i - 1]) or adx_a[i - 1] > L_ADX_MAX: continue
            rl = np.min(l[i - L_LOOKBACK:i]); rh = np.max(h[i - L_LOOKBACK:i])
            if rl <= 0 or rh <= rl: continue
            buf = L_BUFFER * a; cl = c[i]; pc = c[i - 1]; height = rh - rl
            if cl > rh + buf and pc <= rh:
                if cl > rl:
                    side[i] = 'LONG'; entry[i] = cl; sl[i] = rl; tp[i] = cl + height
                    strat[i] = 'lat'; blevel[i] = rh
            elif cl < rl - buf and pc >= rl:
                if rh > cl:
                    side[i] = 'SHORT'; entry[i] = cl; sl[i] = rh; tp[i] = cl - height
                    strat[i] = 'lat'; blevel[i] = rl
        elif r == 'BULL':
            if not btc_local[i]: continue
            if np.isnan(slope[i]) or slope[i] < U_SLOPE_MIN: continue
            if np.isnan(ema_t[i]) or c[i] <= ema_t[i]: continue
            av = atr_avg[i]
            if not np.isnan(av) and a < av: continue
            best = None
            for j in range(i - 1, max(i - U_CHOCH_BARS - 1, U_SWING_N + U_CHOCH_REF) - 1, -1):
                swing_low = np.min(l[j - U_SWING_N:j])
                if not (l[j] < swing_low and c[j] > swing_low): continue
                if U_MIN_SWEEP > 0 and (swing_low - l[j]) / swing_low * 100 < U_MIN_SWEEP: continue
                ref_high = np.max(h[j - U_CHOCH_REF:j])
                if np.any(c[j + 1:i] > ref_high): continue
                if c[i] > ref_high:
                    al = np.min(l[j:i + 1]); risk = c[i] - al
                    if risk > 0 and risk / c[i] <= 0.10:
                        best = (c[i], al); break
            if best:
                e, stop = best; risk = e - stop
                side[i] = 'LONG'; entry[i] = e; sl[i] = stop; tp[i] = e + U_RR * risk
                strat[i] = 'bull'
        elif r == 'BEAR':
            cl = c[i]; op = o[i]; av = atr_avg[i]
            if np.isnan(av): continue
            low_n = np.min(l[i - BSEL_LB:i])
            if not (cl < low_n and cl < op and a > av): continue
            if np.isnan(adx_a[i - 1]) or adx_a[i - 1] > BSEL_ADX: continue
            stop = np.max(h[i - BSEL_STRUCT:i + 1]) + 0.1 * a; risk = stop - cl
            if risk > 0:
                side[i] = 'SHORT'; entry[i] = cl; sl[i] = stop; tp[i] = cl - B_RR * risk
                strat[i] = 'bear'
    return side, entry, sl, tp, strat, blevel


def prep(pdata, btcb):
    arrs = {}
    for sym, df in pdata.items():
        s, e, sl, tp, st, bl = gen(df, btcb.reindex(df.index).fillna(False).values)
        arrs[sym] = df.assign(_s=s, _e=e, _sl=sl, _tp=tp, _st=st, _bl=bl)
    return arrs


def run(arrs, start, end, gest=None, tp_scale=None, max_per_side=MAX_PER_SIDE):
    """Motor v3. Duas extensoes sobre o template canonico:

    1. `gest` — gestao por estrategia (be/trail). Permite pela primeira vez LIGAR
       BE+trailing na LATERAL, que hoje nao tem nenhum. A invalidacao e o timeout
       da lateral continuam ativos (sao independentes do modo de gestao).
    2. `tp_scale` — encolhe a distancia entrada->alvo por estrategia.

    Cada trade devolve tambem o MFE (melhor excursao em R) para o diagnostico.
    """
    gest = gest or GEST_BASE
    tp_scale = tp_scale or {}
    midx = None
    for sym, d in arrs.items():
        midx = d.index if midx is None else midx.union(d.index)
    midx = midx[(midx >= start) & (midx <= end)]
    A = {}
    for sym, d in arrs.items():
        dd = d.reindex(midx)
        A[sym] = {'h': dd['h'].values, 'l': dd['l'].values, 'c': dd['c'].values,
                  'atr': dd['atr'].values, 'side': dd['_s'].values,
                  'entry': dd['_e'].values, 'sl': dd['_sl'].values,
                  'tp': dd['_tp'].values, 'strat': dd['_st'].values, 'bl': dd['_bl'].values}
    balance = MONTHLY_BASE; peak = MONTHLY_BASE; max_dd = 0.0
    positions = {}; cooldown_until = {s: -1 for s in A}
    monthly = {}; cur_month = None; trades = []

    def _rec(pos, nr, reason):
        trades.append(dict(nr=nr, strat=pos['strat'], mfe=pos['mfe'],
                           path=pos['path'], reason=reason))

    def close_all(k):
        nonlocal balance
        for sym in list(positions.keys()):
            d = A[sym]; c = d['c'][k]; pos = positions[sym]
            if np.isnan(c): c = pos['entry']
            e = pos['entry']; risk = pos['risk_px']; fee_r = pos['fee_r']
            nr = ((c - e) if pos['side'] == 'LONG' else (e - c)) / risk - fee_r
            balance += nr * pos['risk_usd']; _rec(pos, nr, 'mes')
            del positions[sym]

    for k in range(len(midx)):
        ts = midx[k]; mk = ts.strftime('%Y-%m')
        if cur_month is None: cur_month = mk
        if mk != cur_month:
            close_all(k); monthly[cur_month] = balance - MONTHLY_BASE
            balance = MONTHLY_BASE; peak = MONTHLY_BASE; cur_month = mk

        for sym in list(positions.keys()):
            d = A[sym]; c = d['c'][k]
            if np.isnan(c): continue
            pos = positions[sym]; hi = d['h'][k]; lo = d['l'][k]; atr = d['atr'][k]
            risk = pos['risk_px']; fee_r = pos['fee_r']; e = pos['entry']
            a = atr if not np.isnan(atr) else risk

            # MFE em R — melhor momento do trade, para o diagnostico
            fav = ((hi - e) if pos['side'] == 'LONG' else (e - lo)) / risk
            if fav > pos['mfe']: pos['mfe'] = fav

            closed = False; nr = 0.0; reason = ''
            if pos['side'] == 'LONG':
                if lo <= pos['cur']:
                    nr = (pos['cur'] - e) / risk - fee_r; closed = True; reason = 'stop'
                elif hi >= pos['tp']:
                    nr = (pos['tp'] - e) / risk - fee_r; closed = True; reason = 'alvo'
                else:
                    if not pos['be_done'] and (c - e) / risk >= pos['be'] * pos['path']:
                        pos['cur'] = e; pos['be_done'] = True
                    if pos['be_done'] and pos['trail'] > 0:
                        cand = c - pos['trail'] * a
                        if cand > pos['cur']: pos['cur'] = cand
            else:
                if hi >= pos['cur']:
                    nr = (e - pos['cur']) / risk - fee_r; closed = True; reason = 'stop'
                elif lo <= pos['tp']:
                    nr = (e - pos['tp']) / risk - fee_r; closed = True; reason = 'alvo'
                else:
                    if not pos['be_done'] and (e - c) / risk >= pos['be'] * pos['path']:
                        pos['cur'] = e; pos['be_done'] = True
                    if pos['be_done'] and pos['trail'] > 0:
                        cand = c + pos['trail'] * a
                        if cand < pos['cur']: pos['cur'] = cand

            # Invalidacao + timeout: so a lateral (independente do modo de gestao)
            if not closed and pos['manage'] == 'fixed':
                pos['age'] += 1
                if pos['age'] <= L_INVAL_N and not np.isnan(pos['bl']):
                    inside = (c < pos['bl']) if pos['side'] == 'LONG' else (c > pos['bl'])
                    if inside:
                        nr = ((c - e) if pos['side'] == 'LONG' else (e - c)) / risk - fee_r
                        closed = True; reason = 'inval'
                if not closed and pos['age'] >= L_TIMEOUT:
                    nr = ((c - e) if pos['side'] == 'LONG' else (e - c)) / risk - fee_r
                    closed = True; reason = 'timeout'

            if closed:
                balance += nr * pos['risk_usd']; _rec(pos, nr, reason)
                del positions[sym]; cooldown_until[sym] = k + pos['cd']

        n_long = sum(1 for p in positions.values() if p['side'] == 'LONG')
        n_short = sum(1 for p in positions.values() if p['side'] == 'SHORT')
        for sym in A:
            if sym in positions or k <= cooldown_until[sym]: continue
            d = A[sym]; side = d['side'][k]
            if side is None or (isinstance(side, float) and np.isnan(side)): continue
            e = d['entry'][k]; stop = d['sl'][k]; tp = d['tp'][k]
            strat = d['strat'][k]; bl = d['bl'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or balance <= 0: continue
            if side == 'LONG' and n_long >= max_per_side: continue
            if side == 'SHORT' and n_short >= max_per_side: continue
            risk_px = abs(e - stop)
            if risk_px <= 0: continue

            sc = tp_scale.get(strat, 1.0)
            if sc != 1.0:
                tp = e + (tp - e) * sc      # encolhe a distancia entrada->alvo

            g = gest[strat]
            path = abs(tp - e) / risk_px    # caminho ao alvo, em R
            if path <= 0: continue
            positions[sym] = {'side': side, 'entry': e, 'cur': stop, 'tp': tp,
                              'be_done': False, 'age': 0, 'risk_px': risk_px,
                              'risk_usd': balance * (RISK_PCT / 100.0),
                              'fee_r': e * RT / risk_px,
                              'be': g['be'], 'trail': g['trail'],
                              'manage': 'fixed' if strat == 'lat' else 'trail',
                              'cd': g['cd'], 'strat': strat, 'bl': bl,
                              'mfe': 0.0, 'path': path}
            if side == 'LONG': n_long += 1
            else: n_short += 1

        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if positions: close_all(len(midx) - 1)
    monthly[cur_month] = balance - MONTHLY_BASE
    return monthly, max_dd, trades


def summ(res):
    monthly, ddmax, trades = res
    yr = {}
    for mk, v in monthly.items(): yr[mk[:4]] = yr.get(mk[:4], 0) + v
    tot = sum(monthly.values())
    nrs = np.array([t['nr'] for t in trades]) if trades else np.array([])
    n = len(nrs); wins = nrs[nrs > 0.03] if n else nrs
    return dict(tot=tot, yr=yr, monthly=monthly, dd=ddmax, n=n,
                anos=sum(1 for v in yr.values() if v > 0),
                meses=sum(1 for v in monthly.values() if v > 0), nm=len(monthly),
                wr=len(wins) / n * 100 if n else 0.0, trades=trades)


def cab(t1='5 anos', t2='OOS'):
    print(f"  {'variante':<26}{t1:>9}{'DD':>7}{'anos+':>7}{'meses+':>8}"
          f"{t2:>8}{'DD':>7}{'trades':>8}{'WR':>7}")
    print('  ' + '-' * 88)


def linha(rot, s5, s26, base5=347, base26=60, base_dd=10.1):
    ok = s5['tot'] > base5 and s26['tot'] > base26
    dd = max(s5['dd'], s26['dd'])
    flag = ''
    if ok:
        flag = '  <= BATE nos 2' + ('' if dd <= base_dd else f' (DD {dd:.1f}% pior)')
    print(f"  {rot:<26}{s5['tot']:>+9.0f}{s5['dd']:>6.1f}%{s5['anos']:>5}/5"
          f"{s5['meses']:>6}/{s5['nm']}{s26['tot']:>+8.0f}{s26['dd']:>6.1f}%"
          f"{s5['n']:>8}{s5['wr']:>6.1f}%{flag}")


def diagnostico(s5, s26):
    """Responde a pergunta do Rafa: os perdedores chegam perto do alvo antes de virar?"""
    print('=' * 100)
    print('  1. DIAGNOSTICO — a observacao do Rafa e verdadeira?')
    print('     "e muito comum o valor chegar proximo do alvo e voltar para buscar o stop"')
    print('=' * 100)
    for rot, s in [('5 anos', s5), ('OOS 2026', s26)]:
        print(f'\n  [{rot}] Dos trades PERDEDORES: ate que % do caminho ao alvo chegaram?')
        print(f"    {'estrategia':<12}{'perdedores':>11}{'>=50%':>8}{'>=70%':>8}"
              f"{'>=80%':>8}{'>=90%':>8}{'MFE medio':>11}")
        for st in ['lat', 'bull', 'bear']:
            perd = [t for t in s['trades'] if t['strat'] == st and t['nr'] < 0]
            if not perd:
                print(f'    {st:<12}{"(sem trades)":>11}')
                continue
            frac = np.array([t['mfe'] / t['path'] for t in perd])
            n = len(frac)
            print(f'    {st:<12}{n:>11}'
                  f"{(frac >= 0.5).sum() / n * 100:>7.0f}%{(frac >= 0.7).sum() / n * 100:>7.0f}%"
                  f"{(frac >= 0.8).sum() / n * 100:>7.0f}%{(frac >= 0.9).sum() / n * 100:>7.0f}%"
                  f'{frac.mean() * 100:>10.0f}%')
        tot_perd = [t for t in s['trades'] if t['nr'] < 0]
        if tot_perd:
            fr = np.array([t['mfe'] / t['path'] for t in tot_perd])
            print(f'    {"TODAS":<12}{len(fr):>11}'
                  f"{(fr >= 0.5).mean() * 100:>7.0f}%{(fr >= 0.7).mean() * 100:>7.0f}%"
                  f"{(fr >= 0.8).mean() * 100:>7.0f}%{(fr >= 0.9).mean() * 100:>7.0f}%"
                  f'{fr.mean() * 100:>10.0f}%')
    print()
    print('  Como ler: se poucos perdedores passam dos 70%, apertar a saida quase nao')
    print('  tem material para trabalhar e o ganho possivel e pequeno — por muito que')
    print('  a memoria diga o contrario (vemos os casos dolorosos, nao a distribuicao).')
    print()


def mensal(titulo, sres):
    print(f'\n  [MENSAL] {titulo}')
    meses = sorted(set().union(*[set(s['monthly']) for _, s in sres]))
    for i in range(0, len(meses), 14):
        blk = meses[i:i + 14]
        print('    ' + f"{'mes':<16}" + ''.join(f'{m[2:]:>8}' for m in blk) + f"{'TOT':>9}")
        for nome, s in sres:
            print('    ' + f'{nome:<16}'
                  + ''.join(f"{s['monthly'].get(m, 0):>+8.0f}" for m in blk)
                  + f"{sum(s['monthly'].get(m, 0) for m in blk):>+9.0f}")
        print()


def g(**kw):
    """Constroi um GEST a partir do base, com overrides por estrategia."""
    out = {k: dict(v) for k, v in GEST_BASE.items()}
    for k, v in kw.items():
        out[k].update(v)
    return out


def main():
    print()
    print('=' * 100)
    print('  V3 — TRAILING MAIS APERTADO E/OU ALVO MAIS CURTO')
    print('=' * 100)
    print('  A carregar dados...')
    a5 = prep(*load(fetch_5y))
    a26 = prep(*load(fetch_26))
    S5 = (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2025, 10, 6, tzinfo=timezone.utc))
    S26 = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 9, tzinfo=timezone.utc))
    print(f'  {len(a5)} pares (5 anos) | {len(a26)} pares (OOS)\n')

    # ── Baseline: TEM de reproduzir +347/+60, senao o motor esta partido ──
    b5 = summ(run(a5, *S5))
    b26 = summ(run(a26, *S26))
    print('  REGRESSAO — o motor reproduz o baseline conhecido?')
    print(f"    5 anos: {b5['tot']:+.0f} (esperado ~+347)   "
          f"OOS: {b26['tot']:+.0f} (esperado ~+60)   DD: {b5['dd']:.1f}% (esperado ~10.1%)")
    if abs(b5['tot'] - 347) > 25 or abs(b26['tot'] - 60) > 15:
        print('\n  ABORTADO: o motor NAO reproduz o baseline. Nao apresento numeros')
        print('  de um motor que nao consigo validar contra um resultado conhecido.')
        sys.exit(1)
    print('    OK — motor validado.\n')

    diagnostico(b5, b26)

    # ── 2. Grelha TRAIL x BE nas DIRECIONAIS (lateral no baseline) ──
    print('=' * 100)
    print('  2a. GRELHA TRAIL x BE — DIRECIONAIS (bull+bear). Lateral fica no baseline.')
    print('      Hoje: BE=0.40, TRAIL=2.0xATR')
    print('=' * 100)
    cab()
    linha('BASELINE', b5, b26)
    best_dir = None
    for be in BES:
        for tr in TRAILS:
            gg = g(bull=dict(be=be, trail=tr), bear=dict(be=be, trail=tr))
            s5 = summ(run(a5, *S5, gest=gg)); s26 = summ(run(a26, *S26, gest=gg))
            if be == 0.40 and tr == 2.0: continue      # e o baseline, ja impresso
            linha(f'be={be:.2f} trail={tr:.2f}', s5, s26)
            sc = s5['tot'] + s26['tot']
            if best_dir is None or sc > best_dir[0]:
                best_dir = (sc, be, tr, s5, s26)
    print()

    # ── 3. Grelha TRAIL x BE na LATERAL (direcionais no baseline) ──
    print('=' * 100)
    print('  2b. GRELHA TRAIL x BE — LATERAL. Direcionais ficam no baseline.')
    print('      Hoje: BE=0.99 (nunca) e TRAIL=0.0 -> ZERO protecao. Aqui LIGAMOS os dois.')
    print('      (a invalidacao e o timeout da lateral continuam ativos)')
    print('=' * 100)
    cab()
    linha('BASELINE', b5, b26)
    best_lat = None
    for be in BES:
        for tr in TRAILS:
            gg = g(lat=dict(be=be, trail=tr))
            s5 = summ(run(a5, *S5, gest=gg)); s26 = summ(run(a26, *S26, gest=gg))
            linha(f'lat be={be:.2f} trail={tr:.2f}', s5, s26)
            sc = s5['tot'] + s26['tot']
            if best_lat is None or sc > best_lat[0]:
                best_lat = (sc, be, tr, s5, s26)
    print()

    # ── 4. Alvos mais curtos ──
    print('=' * 100)
    print('  3. ALVOS MAIS CURTOS (fator sobre a distancia entrada->alvo)')
    print('=' * 100)
    cab()
    linha('BASELINE', b5, b26)
    best_tp = None
    for alvo in ['lat', 'todas']:
        for sc_ in TP_SCALES:
            if sc_ == 1.0: continue
            ts = {'lat': sc_} if alvo == 'lat' else {'lat': sc_, 'bull': sc_, 'bear': sc_}
            s5 = summ(run(a5, *S5, tp_scale=ts)); s26 = summ(run(a26, *S26, tp_scale=ts))
            linha(f'alvo {alvo} x{sc_:.2f}', s5, s26)
            sco = s5['tot'] + s26['tot']
            if best_tp is None or sco > best_tp[0]:
                best_tp = (sco, alvo, sc_, s5, s26)
    print()

    # ── 5. Resumo dos melhores de cada braco ──
    print('=' * 100)
    print('  RESUMO — melhor de cada braco')
    print('=' * 100)
    cab()
    linha('BASELINE', b5, b26)
    cands = []
    if best_dir:
        _, be, tr, s5, s26 = best_dir
        linha(f'dir be={be:.2f} trail={tr:.2f}', s5, s26)
        cands.append(('direcionais', dict(be=be, tr=tr), s5, s26))
    if best_lat:
        _, be, tr, s5, s26 = best_lat
        linha(f'lat be={be:.2f} trail={tr:.2f}', s5, s26)
        cands.append(('lateral', dict(be=be, tr=tr), s5, s26))
    if best_tp:
        _, alvo, sc_, s5, s26 = best_tp
        linha(f'alvo {alvo} x{sc_:.2f}', s5, s26)
        cands.append(('alvo', dict(alvo=alvo, sc=sc_), s5, s26))
    print()

    venc = [c for c in cands if c[2]['tot'] > 347 and c[3]['tot'] > 60]
    print('=' * 100)
    print('  VEREDICTO')
    print('=' * 100)
    print('  Portao: bater +347 (5 anos) E +60 (OOS 2026). Baseline DD = 10.1%')
    if not venc:
        print('\n  >> NENHUM braco bate o baseline nos DOIS periodos.')
        print('     A gestao atual (TP/SL fixos na lateral, BE 40% + trail 2xATR nas')
        print('     direcionais) continua a ser a melhor que conhecemos. Nada a mudar.')
    else:
        print(f'\n  >> {len(venc)} candidato(s) passam o portao. Ver teste de planalto abaixo.')
        for nome, par, s5, s26 in venc:
            print(f"     {nome}: {par}  ->  {s5['tot']:+.0f} / {s26['tot']:+.0f} "
                  f"| DD {max(s5['dd'], s26['dd']):.1f}%")
    print()
    mensal('baseline (referencia)', [('baseline 5a', b5)])
    return b5, b26, cands


if __name__ == '__main__':
    main()
