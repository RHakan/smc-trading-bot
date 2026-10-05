"""
backtest_bull_smc.py
Bull via SMC — Sweep de liquidez vendida + CHoCH (Change of Character)

Testado nos dois bull markets relevantes do crypto:
  BULL1 : 18/06/2023 – 06/10/2025  (ultimo bull market)
  BULL2 : 13/03/2020 – 11/11/2021  (bull anterior)

Sem filtro de regime — os periodos ja sao bull por definicao.
Usa cache local (parquet) para evitar re-download a cada run.

Logica (1H):
  1. SWEEP   : low vai ABAIXO de um swing low recente mas close fecha ACIMA
               (falsa quebra — SM absorveu os stops dos longs)
  2. CHoCH   : nas M barras seguintes, close > maxima de referencia anterior
               (confirma Change of Character — estrutura virou bullish)
  3. ENTRY   : close da barra de CHoCH
  4. STOP    : minima mais baixa entre o sweep e o CHoCH
  5. TARGET  : RR_CAP x risco + breakeven 70% + trailing 2xATR

Uso: python backtests/backtest_bull_smc.py
"""
import sys, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Periodos bull ─────────────────────────────────────────────────────────────
PERIODS = [
    {
        'name'  : 'Bull 2023-2025',
        'start' : datetime(2023,  6, 18, tzinfo=timezone.utc),
        'end'   : datetime(2025, 10,  6, tzinfo=timezone.utc),
    },
    {
        'name'  : 'Bull 2020-2021',
        'start' : datetime(2020,  3, 13, tzinfo=timezone.utc),
        'end'   : datetime(2021, 11, 11, tzinfo=timezone.utc),
    },
]

# Janela total para fetch (cobre os dois periodos + warmup)
FETCH_START = datetime(2020,  1,  1, tzinfo=timezone.utc)
FETCH_END   = datetime(2025, 10,  6, tzinfo=timezone.utc)

# ── Pares e custos ────────────────────────────────────────────────────────────
PAIRS = [
    'BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
    'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT',
    # SOL, AVAX, DOT nao existiam ou tinham liquidez minima em mar/2020
    # sao incluidos mas so terao trades a partir de quando existirem dados
    'SOL/USDT:USDT','AVAX/USDT:USDT','DOT/USDT:USDT',
]
RISK_PCT   = 1.0
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
ATR_PERIOD=14; ATR_AVG_PERIOD=48
COOLDOWN=3; BE_PCT=0.70; TRAIL_ATR=2.0

# ── Parametros base ───────────────────────────────────────────────────────────
BASE = dict(
    swing_n    = 10,    # barras para identificar o swing low (liquidez vendida)
    choch_bars = 8,     # max barras apos sweep para confirmar CHoCH
    choch_ref  = 20,    # janela antes do sweep para a maxima de referencia
    rr_cap     = 2.5,
    min_sweep  = 0.05,  # % minimo de pierce abaixo do swing low (filtra ruido)
    atr_filter = True,  # so opera quando ATR > media (mercado ativo)
)

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

ex = ccxt.binanceusdm({'enableRateLimit': True})


def fetch_1h(sym):
    """Busca 1H de FETCH_START a FETCH_END com cache CSV."""
    safe  = sym.replace('/', '_').replace(':', '_')
    cache = CACHE_DIR / f'{safe}_1h_2020_2025.csv'
    if cache.exists():
        df = pd.read_csv(cache, index_col='ts', parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    since  = int((FETCH_START - timedelta(days=5)).timestamp() * 1000)
    end_ms = int(FETCH_END.timestamp() * 1000)
    rows   = []
    while True:
        b = ex.fetch_ohlcv(sym, '1h', since=since, limit=1000)
        if not b: break
        rows.extend(b); since = b[-1][0] + 1
        if since >= end_ms: break
        time.sleep(0.05)

    df = pd.DataFrame(rows, columns=['ts','o','h','l','c','v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('ts').sort_index()[lambda d: d.index <= FETCH_END]
    df.to_csv(cache)
    return df


# ── Carga de dados ────────────────────────────────────────────────────────────
print('Backtest Bull SMC')
print(f'  BULL1 : {PERIODS[0]["start"].date()} → {PERIODS[0]["end"].date()}')
print(f'  BULL2 : {PERIODS[1]["start"].date()} → {PERIODS[1]["end"].date()}')
print()
print('A carregar dados 1H (usa cache se disponivel)...')
raw = {}; ok_pairs = []

for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        df1 = fetch_1h(sym).copy()
        if len(df1) < 100: print('SEM DADOS'); continue

        tr = pd.concat([
            (df1['h'] - df1['l']),
            (df1['h'] - df1['c'].shift(1)).abs(),
            (df1['l'] - df1['c'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df1['atr']     = tr.ewm(com=ATR_PERIOD-1, adjust=False).mean()
        df1['atr_avg'] = df1['atr'].rolling(ATR_AVG_PERIOD).mean()

        raw[sym] = df1
        ok_pairs.append(sym)
        print('OK (cache)' if (CACHE_DIR / f'{sym.replace("/","_").replace(":","_")}_1h_2020_2025.parquet').exists() else 'OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


# ── Simulacao de posicao long ─────────────────────────────────────────────────
def sim_long(l_arr, h_arr, c_arr, atr_arr, ib, entry, sl_price, rr_cap, n):
    risk = entry - sl_price
    if risk <= 0: return 0.0
    tp    = entry + rr_cap * risk
    fee_r = entry * RT / risk
    cur = sl_price; be = False
    for j in range(ib+1, min(ib+400, n)):
        lo=l_arr[j]; hi=h_arr[j]; c=c_arr[j]
        atr = atr_arr[j] if not np.isnan(atr_arr[j]) else risk
        if lo <= cur: return (cur-entry)/risk - fee_r
        if hi >= tp:  return rr_cap - fee_r
        if not be and (c-entry)/risk >= BE_PCT*rr_cap:
            cur=entry; be=True
        if be:
            cand = c - TRAIL_ATR*atr
            if cand > cur: cur = cand
    return (c_arr[min(ib+399, n-1)]-entry)/risk - fee_r


# ── Estrategia SMC ────────────────────────────────────────────────────────────
def run(swing_n, choch_bars, choch_ref, rr_cap, min_sweep, atr_filter):
    trades = []
    for sym in ok_pairs:
        df      = raw[sym]
        l_arr   = df['l'].values
        h_arr   = df['h'].values
        c_arr   = df['c'].values
        atr_arr = df['atr'].values
        atr_avg = df['atr_avg'].values
        idx     = df.index
        n       = len(df)

        last_bar = -9999
        start_i  = max(swing_n + choch_ref + 20, 60)
        i        = start_i

        while i < n - choch_bars - 2:
            ts = idx[i]

            # So opera dentro dos periodos bull definidos
            in_period = any(p['start'] <= ts <= p['end'] for p in PERIODS)
            if not in_period:
                i += 1; continue

            if i - last_bar < COOLDOWN: i += 1; continue

            atr = atr_arr[i]
            if np.isnan(atr): i += 1; continue
            if atr_filter and not np.isnan(atr_avg[i]) and atr < atr_avg[i]:
                i += 1; continue

            # Swing low: minimo das ultimas swing_n barras (sem a barra atual)
            swing_low = np.min(l_arr[i-swing_n:i])

            # SWEEP: low rompeu abaixo do swing low mas close fechou acima
            lo = l_arr[i]; cl = c_arr[i]
            if not (lo < swing_low and cl > swing_low):
                i += 1; continue

            # Pierce minimo (filtra falsas quebras por ruido)
            if min_sweep > 0 and (swing_low - lo) / swing_low * 100 < min_sweep:
                i += 1; continue

            # Maxima de referencia para o CHoCH (janela antes do sweep)
            ref_high   = np.max(h_arr[i-choch_ref:i])
            actual_low = lo  # rastreia o low mais baixo desde o sweep

            # Procura CHoCH nas proximas choch_bars barras
            found = False
            for j in range(i+1, min(i+choch_bars+1, n-1)):
                actual_low = min(actual_low, l_arr[j])

                if c_arr[j] > ref_high:
                    # CHoCH confirmado — entra no close
                    entry = c_arr[j]
                    sl_p  = actual_low
                    risk  = entry - sl_p

                    # Sanidade: risco maximo de 10%
                    if risk <= 0 or risk / entry > 0.10:
                        break

                    nr     = sim_long(l_arr, h_arr, c_arr, atr_arr, j, entry, sl_p, rr_cap, n)
                    period = next(p['name'] for p in PERIODS if p['start'] <= idx[j] <= p['end'])
                    trades.append({'ts': idx[j], 'period': period, 'nr': nr})
                    last_bar = j
                    i = j + 1
                    found = True
                    break

            if not found:
                i += 1

    return trades


# ── Resumo por periodo ────────────────────────────────────────────────────────
def summarize(trades, period_name):
    ts = [t for t in trades if t['period'] == period_name]
    if not ts: return None
    nr   = [t['nr'] for t in ts]
    wins = [r for r in nr if r > 0]
    gw   = sum(wins); gl = abs(sum(r for r in nr if r <= 0))
    cap  = 100.0; months = {}
    for t in sorted(ts, key=lambda x: x['ts']):
        mk  = t['ts'].strftime('%Y-%m')
        pnl = t['nr'] * cap * (RISK_PCT/100); cap += pnl
        if mk not in months: months[mk] = {'cnt':0,'wins':0,'start':cap-pnl}
        months[mk]['end_cap'] = cap; months[mk]['cnt'] += 1
        if t['nr'] > 0: months[mk]['wins'] += 1
    return {
        't'    : len(nr),
        'wr'   : len(wins)/len(nr)*100,
        'avgr' : sum(nr)/len(nr),
        'pf'   : gw/gl if gl>0 else 99,
        'total': cap-100,
        'months': months,
    }


def run_both(**over):
    trades = run(**{**BASE, **over})
    return {p['name']: summarize(trades, p['name']) for p in PERIODS}


def show(label, s_dict, base=False):
    mark = ' *' if base else '  '
    parts = []
    for pname, s in s_dict.items():
        tag = pname.split()[1]  # "2023-2025" ou "2020-2021"
        if s is None:
            parts.append(f'[{tag}: sem trades      ]')
        else:
            parts.append(
                f'[{tag}: {s["t"]:>3}t  WR{s["wr"]:>4.1f}%  '
                f'avg{s["avgr"]:>+5.3f}R  PF{s["pf"]:>4.2f}  {s["total"]:>+7.1f}%]'
            )
    print(f"  {label:<26}{mark}  " + "  ".join(parts))


# ─────────────────────────────────────────────────────────────────────────────
# SWEEPS
# ─────────────────────────────────────────────────────────────────────────────

print('Sweep 1/5: swing_n (janela do swing low / onde a liquidez se acumula)...')
for sn in [5, 10, 15, 20]:
    show(f'swing_n={sn}', run_both(swing_n=sn), sn == BASE['swing_n'])

print('\nSweep 2/5: choch_bars (max barras para confirmar CHoCH apos sweep)...')
for cb in [5, 8, 10, 15]:
    show(f'choch_bars={cb}', run_both(choch_bars=cb), cb == BASE['choch_bars'])

print('\nSweep 3/5: choch_ref (janela da maxima de referencia para o CHoCH)...')
for cr in [10, 15, 20, 30]:
    show(f'choch_ref={cr}', run_both(choch_ref=cr), cr == BASE['choch_ref'])

print('\nSweep 4/5: min_sweep (% minimo de pierce abaixo do swing low)...')
for ms in [0.0, 0.05, 0.1, 0.2]:
    show(f'min_sweep={ms}%', run_both(min_sweep=ms), ms == BASE['min_sweep'])

print('\nSweep 5/5: RR_CAP...')
for rr in [1.5, 2.0, 2.5, 3.0]:
    show(f'RR_CAP={rr}', run_both(rr_cap=rr), rr == BASE['rr_cap'])

# ─────────────────────────────────────────────────────────────────────────────
# COMBINACOES
# ─────────────────────────────────────────────────────────────────────────────
print('\nCombinacoes:')
combos = [
    dict(swing_n=5,  choch_bars=5,  choch_ref=10, rr_cap=2.0, min_sweep=0.05),
    dict(swing_n=10, choch_bars=8,  choch_ref=20, rr_cap=2.5, min_sweep=0.05),
    dict(swing_n=10, choch_bars=8,  choch_ref=20, rr_cap=2.5, min_sweep=0.10),
    dict(swing_n=15, choch_bars=8,  choch_ref=20, rr_cap=2.5, min_sweep=0.05),
    dict(swing_n=10, choch_bars=10, choch_ref=20, rr_cap=3.0, min_sweep=0.10),
    dict(swing_n=20, choch_bars=8,  choch_ref=30, rr_cap=2.5, min_sweep=0.10),
    dict(swing_n=15, choch_bars=10, choch_ref=20, rr_cap=3.0, min_sweep=0.10),
    dict(swing_n=10, choch_bars=5,  choch_ref=15, rr_cap=2.0, min_sweep=0.20),
    dict(swing_n=20, choch_bars=10, choch_ref=30, rr_cap=3.0, min_sweep=0.05),
    dict(swing_n=15, choch_bars=8,  choch_ref=15, rr_cap=2.5, min_sweep=0.10),
]
results = []
for p in combos:
    s = run_both(**p)
    results.append((p, s))
    lbl = f"sw{p['swing_n']} cb{p['choch_bars']} ref{p['choch_ref']} RR{p['rr_cap']} ms{p['min_sweep']}"
    show(lbl, s)

# ─────────────────────────────────────────────────────────────────────────────
# MELHOR
# ─────────────────────────────────────────────────────────────────────────────
valid = [(p, s) for p, s in results if any(v is not None for v in s.values())]
if valid:
    def score(item):
        # Pondera os dois periodos: queremos positivo em AMBOS
        vals = [v['total'] for v in item[1].values() if v is not None]
        if not vals: return -999
        # Penaliza fortemente se algum periodo for negativo
        if any(v < 0 for v in vals): return sum(vals) * 0.5
        return sum(vals)

    best_p, best_s = max(valid, key=score)
    print(f'\n{"="*90}')
    print(f'  MELHOR: {best_p}')
    print('='*90)

    MLABELS = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
    for pname, s in best_s.items():
        if s is None: continue
        print(f'\n  ── {pname} ──')
        print(f'  {s["t"]} trades | WR {s["wr"]:.1f}% | AvgR {s["avgr"]:+.3f}R | PF {s["pf"]:.2f} | Total {s["total"]:+.1f}%')
        print(f"\n  {'Mes':<8} | {'Trades':>6} | {'WR':>6} | {'Capital':>22}")
        print('  ' + '-'*55)
        for mk in sorted(s['months']):
            m    = s['months'][mk]
            yr_n, mo_n = int(mk[:4]), int(mk[5:])
            wr   = m['wins']/m['cnt']*100 if m['cnt'] > 0 else 0
            pct  = (m['end_cap']/m['start']-1)*100
            mlbl = f"{MLABELS[mo_n-1]}/{yr_n}"
            print(f"  {mlbl:<8} | {m['cnt']:>6} | {wr:>5.1f}% | ${m['end_cap']:>9.2f} ({pct:>+6.1f}%)")

print()
print('Referencia historica (outros testes bull):')
print('  Espelho bull_v1   : -90.6% (2024 calendario)')
print('  Pullback EMA v2.1 : melhor -28.6% (2024 calendario)')
print('  Wyckoff Spring    : +28.5% 2024 | -56.4% 2023  (overfit)')
print()
print('Custos: round-trip 0.14% (taxa 0.05% + slippage 0.02% por lado).')
