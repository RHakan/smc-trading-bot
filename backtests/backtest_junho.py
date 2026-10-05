"""
backtest_junho.py
Backtest da estrategia Bear Market v1.3 EXATAMENTE como esta a correr no bot,
no periodo real em que estiveste a testar: 01/06/2026 -> 27/06/2026 (hoje).

Objectivo: perceber se a sequencia de trades negativos ao vivo seria
esperada pela propria estrategia neste periodo, ou se ha discrepancia
entre o backtest e a execucao real.

Parametros sao importados directamente de bot/strategies/bear_v13.py para
garantir que NAO ha divergencia. Se a estrategia mudar, este backtest segue.

Mostra:
  - Lista trade-a-trade (data, par, entrada, SL, TP, motivo de saida, resultado R)
  - Resumo: total, win rate, soma R, resultado % com alavancagem actual
  - Regime diario de cada par no periodo (para perceber se era mesmo BEAR)
"""
import sys, time
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd

# Importa os parametros REAIS da estrategia — fonte unica da verdade
from bot.strategies import bear_v13 as S

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Periodo de teste ──────────────────────────────────────────────────────────
START = datetime(2026, 6, 1,  tzinfo=timezone.utc)
END   = datetime(2026, 6, 27, 23, 59, tzinfo=timezone.utc)

# ── Alavancagem actual (so afecta a conversao R -> % do capital arriscado) ─────
LEVERAGE     = 3
RISK_PCT     = 1.0    # % do portfolio arriscado por trade (igual ao .env)

# Mesmos pares do bot (uso USDT para dados — preco quase identico ao USDC,
# mas com historico de futuros muito mais robusto na Binance)
PAIRS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
    'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOGE/USDT:USDT',
    'SUI/USDT:USDT', 'LINK/USDT:USDT',
]

FEE = 0.0005; SLIP = 0.0002; RT = (FEE + SLIP) * 2   # custo round-trip em fraccao do preco

# Parametros vindos da estrategia (nao redefinir aqui — vem de S)
ATR_AVG      = S.ATR_AVG_PERIOD
ATR_STOP     = S.ATR_STOP_MULT
COOLDOWN     = S.COOLDOWN_BARS
LOOKBACK     = S.LOW_LOOKBACK
REGIME_EMA   = S.REGIME_EMA_PERIOD
SLOPE_BARS   = S.REGIME_SLOPE_BARS
SLOPE_THRESH = S.REGIME_SLOPE_THRESH
RR_CAP       = S.RR_CAP
BE_TRIGGER_R = S.BE_TRIGGER_PCT * S.RR_CAP   # 0.70 * 2.5 = 1.75R
TRAIL_ATR    = S.TRAIL_ATR

ex = ccxt.binanceusdm({'enableRateLimit': True})


def fetch(sym, tf, extra_days):
    """Carrega OHLCV com 'extra_days' de margem antes de START (para indicadores)."""
    since  = int((START - timedelta(days=extra_days)).timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    rows   = []
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b: break
        rows.extend(b); since = b[-1][0] + 1
        if since >= end_ms: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=['ts','o','h','l','c','v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()[lambda d: d.index <= END]


print(f'Backtest Bear v1.3 — periodo {START:%d/%m/%Y} a {END:%d/%m/%Y}')
print(f'Parametros (de bear_v13.py): lookback={LOOKBACK}  cooldown={COOLDOWN}H  '
      f'RR_CAP={RR_CAP}R  BE={int(S.BE_TRIGGER_PCT*100)}%  TRAIL={TRAIL_ATR}xATR  '
      f'ATR_STOP={ATR_STOP}xATR  momentum=ATR>media{ATR_AVG}H')
print(f'Alavancagem {LEVERAGE}x  |  risco {RISK_PCT}%/trade\n')

print('A carregar dados da Binance...')
raw = {}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    # Daily com margem para EMA20 + slope
    dfd = fetch(sym, '1d', 40).copy()
    dfd['ema']    = dfd['c'].ewm(span=REGIME_EMA, adjust=False).mean()
    dfd['slope']  = (dfd['ema'] - dfd['ema'].shift(SLOPE_BARS)) / dfd['ema'].shift(SLOPE_BARS)
    dfd['regime'] = 'NEUTRAL'
    dfd.loc[(dfd['c'] < dfd['ema']) & (dfd['slope'] < -SLOPE_THRESH), 'regime'] = 'BEAR'
    dfd.loc[(dfd['c'] > dfd['ema']) & (dfd['slope'] >  SLOPE_THRESH), 'regime'] = 'BULL'

    # 1H com margem para ATR média 48H + lookback
    df1 = fetch(sym, '1h', 6).copy()
    tr  = pd.concat([
        (df1['h'] - df1['l']),
        (df1['h'] - df1['c'].shift(1)).abs(),
        (df1['l'] - df1['c'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df1['atr']     = tr.ewm(com=S.ATR_PERIOD - 1, adjust=False).mean()
    df1['atr_avg'] = df1['atr'].rolling(ATR_AVG).mean()
    # Regime do dia ANTERIOR (shift 1) — evita lookahead, igual ao bot
    df1['regime']  = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
    df1['low_n']   = df1['l'].shift(1).rolling(LOOKBACK).min()
    raw[sym] = (df1, dfd)
    time.sleep(0.1)
    print('OK')
print()


def sim_short(df, ib, entry, sl_price):
    """
    Simula a gestao de saida EXATA do position_manager:
    SL inicial -> breakeven a BE_TRIGGER_R -> trailing TRAIL_ATR -> TP em RR_CAP.
    Devolve (resultado_R, motivo_saida, preco_saida).
    """
    risk  = sl_price - entry
    if risk <= 0:
        return 0.0, 'invalido', entry
    tp    = entry - RR_CAP * risk
    fee_r = entry * RT / risk
    cur   = sl_price; be = False
    for j in range(ib + 1, min(ib + 400, len(df))):
        bar = df.iloc[j]
        lo = float(bar['l']); hi = float(bar['h']); c = float(bar['c'])
        atr = float(bar['atr']) if not np.isnan(bar['atr']) else risk
        # Stop / breakeven / trailing — atingido se a vela tocou o stop actual
        if hi >= cur:
            motivo = 'breakeven' if (be and cur == entry) else ('trailing' if be else 'stop_loss')
            return (entry - cur) / risk - fee_r, motivo, cur
        # Take profit
        if lo <= tp:
            return RR_CAP - fee_r, 'take_profit', tp
        # Move para breakeven
        if not be and (entry - c) / risk >= S.BE_TRIGGER_PCT * RR_CAP:
            cur = entry; be = True
        # Trailing depois de breakeven
        if be:
            cand = c + TRAIL_ATR * atr
            if cand < cur: cur = cand
    last_c = float(df.iloc[min(ib + 399, len(df) - 1)]['c'])
    return (entry - last_c) / risk - fee_r, 'timeout', last_c


# ── Simulacao ─────────────────────────────────────────────────────────────────
trades = []
regime_days = {}   # sym -> contagem de dias BEAR/BULL/NEUTRAL no periodo

for sym, (df1, dfd) in raw.items():
    # Conta regimes diarios dentro do periodo
    dmask = (dfd.index >= START) & (dfd.index <= END)
    regime_days[sym] = dfd.loc[dmask, 'regime'].value_counts().to_dict()

    last_bar = -9999
    idx = df1.index
    for i in range(ATR_AVG + LOOKBACK + 2, len(df1) - 1):
        ts_bar = idx[i]
        if ts_bar < START or ts_bar > END:
            continue
        if i - last_bar < COOLDOWN:
            continue
        r = df1.iloc[i]
        if str(r['regime']) != 'BEAR':
            continue
        cl = float(r['c']); op = float(r['o'])
        atr = float(r['atr']); atr_avg = float(r['atr_avg']); low_n = float(r['low_n'])
        if any(np.isnan(v) for v in [cl, atr, atr_avg, low_n]):
            continue
        # Condicoes de entrada — identicas a bear_v13.analyze (momentum = ATR > media)
        if not (cl < low_n and cl < op and atr > atr_avg):
            continue
        sl_p = cl + ATR_STOP * atr
        if sl_p <= cl:
            continue
        nr, motivo, exit_p = sim_short(df1, i, cl, sl_p)
        tp_p = cl - RR_CAP * (sl_p - cl)
        trades.append({
            'ts': ts_bar, 'sym': sym.replace('/USDT:USDT', ''),
            'entry': cl, 'sl': sl_p, 'tp': tp_p,
            'exit': exit_p, 'motivo': motivo, 'nr': nr,
        })
        last_bar = i

trades.sort(key=lambda t: t['ts'])

# ── Lista trade-a-trade ───────────────────────────────────────────────────────
print('=' * 100)
print('  TRADES (cada linha = um SHORT que o bot teria aberto)')
print('=' * 100)
print(f"{'Data/Hora':<17} | {'Par':<5} | {'Entrada':>11} | {'SL':>11} | {'TP':>11} | "
      f"{'Saida':<12} | {'R':>7}")
print('-' * 100)
for t in trades:
    res_col = f"{t['nr']:+.2f}R"
    print(f"  {t['ts']:%d/%m %H:%M}    | {t['sym']:<5} | {t['entry']:>11.4f} | "
          f"{t['sl']:>11.4f} | {t['tp']:>11.4f} | {t['motivo']:<12} | {res_col:>7}")

if not trades:
    print('  (nenhum trade gerado no periodo)')

# ── Resumo ────────────────────────────────────────────────────────────────────
print('\n' + '=' * 100)
print('  RESUMO')
print('=' * 100)

if trades:
    nr_all = [t['nr'] for t in trades]
    wins   = [r for r in nr_all if r > 0]
    losses = [r for r in nr_all if r <= 0]
    soma_r = sum(nr_all)
    wr     = len(wins) / len(nr_all) * 100
    gw     = sum(wins); gl = abs(sum(losses))
    pf     = gw / gl if gl > 0 else 99

    # Capital composto, arriscando RISK_PCT% por trade
    capital = 100.0
    for r in nr_all:
        capital += r * capital * (RISK_PCT / 100)
    total_pct = capital - 100

    # Contagem por motivo de saida
    motivos = {}
    for t in trades:
        motivos[t['motivo']] = motivos.get(t['motivo'], 0) + 1

    print(f"  Total de trades : {len(nr_all)}")
    print(f"  Vitorias        : {len(wins)}  ({wr:.1f}%)")
    print(f"  Derrotas        : {len(losses)}  ({100-wr:.1f}%)")
    print(f"  Soma em R       : {soma_r:+.2f}R")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Resultado capital: {total_pct:+.2f}%  (risco {RISK_PCT}%/trade, composto)")
    print(f"\n  Motivos de saida:")
    for m, c in sorted(motivos.items(), key=lambda x: -x[1]):
        print(f"    {m:<12}: {c}")
else:
    print("  Sem trades — a estrategia nao encontrou setups validos no periodo.")

# ── Regime diario por par ─────────────────────────────────────────────────────
print('\n' + '=' * 100)
print('  REGIME DIARIO NO PERIODO (a estrategia so opera em dias BEAR)')
print('=' * 100)
print(f"{'Par':<6} | {'Dias BEAR':>9} | {'Dias BULL':>9} | {'Dias NEUTRAL':>12}")
print('-' * 100)
for sym in PAIRS:
    s = sym.replace('/USDT:USDT', '')
    rd = regime_days.get(sym, {})
    print(f"  {s:<4} | {rd.get('BEAR',0):>9} | {rd.get('BULL',0):>9} | {rd.get('NEUTRAL',0):>12}")

print('\nNota: dados em USDT (preco ~identico ao USDC que o bot usa).')
print('Custos incluidos: taxa 0.05% + slippage 0.02% por lado (round-trip 0.14%).')
