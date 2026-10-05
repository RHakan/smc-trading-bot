"""
backtest_portfolio_decisor.py
Simulacao de PORTFOLIO do bot COMPLETO, como se tivesse sido ligado em
01/06/2025 e corrido ate hoje (29/06/2026), com:

  - DECISOR ativo (bot/regime.py): por par e por barra, decide o regime
    e despacha a estrategia certa.
        BEAR    -> bear_v13   (estrategia REAL, importada de bot/strategies)
        BULL    -> bull_smc   (HIPOTETICA — sweep+CHoCH, config robusta)
        NEUTRAL -> fica de fora (sem trade)
  - SALDO UNICO de 5000 USDC partilhado entre todos os pares
  - Multiplas posicoes simultaneas (e aqui que a CORRELACAO aparece de verdade —
    risco de 1%/trade vira muito mais quando 5 pares abrem juntos)
  - Gestao de saida IDENTICA nas duas estrategias: BE 70% + trailing 2xATR + RR_CAP

Diferenca-chave face aos backtests anteriores: aqueles testavam cada estrategia
ISOLADA e compunham o capital em sequencia. Este motor e EVENT-DRIVEN no tempo
real — todas as posicoes partilham o mesmo saldo e fecham na ordem cronologica
em que acontecem. E o teste mais proximo do bot a serio.

⚠️  A bull SMC NAO esta implementada nem aprovada. Isto e um "e se". Usa-se a
    config robusta recomendada (nao a agressiva que infla com o bull parabolico).

Uso: python backtests/backtest_portfolio_decisor.py
"""
import sys, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Garante que a raiz do projeto esta no path (o script vive em backtests/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ccxt, numpy as np, pandas as pd

from bot.strategies import bear_v13 as BEAR

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Periodo e saldo ───────────────────────────────────────────────────────────
START = datetime(2025, 6,  1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 29, tzinfo=timezone.utc)

INITIAL_BALANCE = 5000.0    # USDC (saldo da conta de teste)
RISK_PCT        = 1.0       # % do saldo ATUAL arriscado por trade
LEVERAGE        = 3         # so afeta margem/liquidacao; com risco fixo nao muda o P&L

# ── Pares (os mesmos do bot) ──────────────────────────────────────────────────
PAIRS = [
    'BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
    'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
    'AVAX/USDT:USDT','DOT/USDT:USDT',
]

# ── Custos ────────────────────────────────────────────────────────────────────
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2   # round-trip 0.14%

# ── Parametros BEAR (fonte unica: bot/strategies/bear_v13.py) ─────────────────
B_LOOKBACK = BEAR.LOW_LOOKBACK
B_ATR_AVG  = BEAR.ATR_AVG_PERIOD
B_ATR_STOP = BEAR.ATR_STOP_MULT
B_COOLDOWN = BEAR.COOLDOWN_BARS
B_RR       = BEAR.RR_CAP
B_BE_PCT   = BEAR.BE_TRIGGER_PCT
B_TRAIL    = BEAR.TRAIL_ATR
ATR_PERIOD = BEAR.ATR_PERIOD

# Regime (fonte unica)
from bot import regime as regime_mod
R_EMA    = regime_mod.EMA_PERIOD
R_SLOPE  = regime_mod.SLOPE_BARS
R_THRESH = regime_mod.SLOPE_THRESH

# ── Parametros BULL SMC (config ROBUSTA recomendada) ──────────────────────────
U_SWING_N    = 10
U_CHOCH_BARS = 12
U_CHOCH_REF  = 15
U_MIN_SWEEP  = 0.05
U_RR         = 2.5
U_BE_PCT     = 0.70
U_TRAIL      = 2.0
U_COOLDOWN   = 3

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
ex = ccxt.binanceusdm({'enableRateLimit': True})


def fetch(sym, tf, extra_days):
    safe  = sym.replace('/', '_').replace(':', '_')
    cache = CACHE_DIR / f'{safe}_{tf}_jun25_jun26.csv'
    if cache.exists():
        df = pd.read_csv(cache, index_col='ts', parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
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
    df = df.set_index('ts').sort_index()[lambda d: d.index <= END]
    df.to_csv(cache)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CARGA + INDICADORES + PRE-COMPUTACAO DE SINAIS POR PAR
# ─────────────────────────────────────────────────────────────────────────────
print('Backtest PORTFOLIO — bot completo com Decisor')
print(f'  Periodo : {START:%d/%m/%Y} -> {END:%d/%m/%Y}')
print(f'  Saldo   : {INITIAL_BALANCE:.0f} USDC  |  risco {RISK_PCT}%/trade  |  lev {LEVERAGE}x')
print(f'  BEAR    : bear_v13 (real)  RR{B_RR} BE{int(B_BE_PCT*100)}% trail{B_TRAIL}x')
print(f'  BULL    : bull_smc (hipotetica)  sw{U_SWING_N} cb{U_CHOCH_BARS} ref{U_CHOCH_REF} RR{U_RR}')
print()
print('A carregar dados (cache se disponivel)...')

pair_data = {}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd = fetch(sym, '1d', 40).copy()
        df1 = fetch(sym, '1h', 25).copy()
        if len(df1) < 300:
            print('SEM DADOS'); continue

        # Regime diario (shift 1 — usa o dia ja fechado, evita lookahead)
        ema_d  = dfd['c'].ewm(span=R_EMA, adjust=False).mean()
        slope  = (ema_d - ema_d.shift(R_SLOPE)) / ema_d.shift(R_SLOPE)
        dfd['regime'] = 'NEUTRAL'
        dfd.loc[(dfd['c'] < ema_d) & (slope < -R_THRESH), 'regime'] = 'BEAR'
        dfd.loc[(dfd['c'] > ema_d) & (slope >  R_THRESH), 'regime'] = 'BULL'

        # ATR 1H
        tr = pd.concat([
            (df1['h'] - df1['l']),
            (df1['h'] - df1['c'].shift(1)).abs(),
            (df1['l'] - df1['c'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df1['atr']     = tr.ewm(com=ATR_PERIOD-1, adjust=False).mean()
        df1['atr_avg'] = df1['atr'].rolling(B_ATR_AVG).mean()
        df1['regime']  = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
        df1['low_n']   = df1['l'].shift(1).rolling(B_LOOKBACK).min()

        pair_data[sym] = df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen_signals(df):
    """
    Devolve dois arrays alinhados ao indice de df:
      side[i]  -> 'SHORT' (bear) | 'LONG' (bull) | None
      entry/sl -> precos de entrada e stop para a barra i
    O Decisor ja esta embutido: bear so dispara em regime BEAR, bull so em BULL.
    Uma barra nunca gera os dois (regimes mutuamente exclusivos).
    """
    n      = len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values
    low_n=df['low_n'].values; reg=df['regime'].values

    side  = np.array([None]*n, dtype=object)
    entry = np.full(n, np.nan)
    sl    = np.full(n, np.nan)

    start_i = max(B_ATR_AVG + B_LOOKBACK + 5, U_SWING_N + U_CHOCH_REF + U_CHOCH_BARS + 5)

    for i in range(start_i, n):
        r = reg[i]

        # ── BEAR (bear_v13) — regime BEAR ──────────────────────────────────────
        if r == 'BEAR':
            cl=c[i]; op=o[i]; a=atr[i]; av=atr_avg[i]; ln=low_n[i]
            if not any(np.isnan(v) for v in (cl,a,av,ln)):
                if cl < ln and cl < op and a > av:
                    sl_p = cl + B_ATR_STOP * a
                    if sl_p > cl:
                        side[i]='SHORT'; entry[i]=cl; sl[i]=sl_p
            continue

        # ── BULL (sweep + CHoCH) — regime BULL ─────────────────────────────────
        if r == 'BULL':
            a=atr[i]; av=atr_avg[i]
            if np.isnan(a): continue
            if not np.isnan(av) and a < av:   # filtro ATR (mercado ativo)
                continue
            # Procura um SWEEP em alguma das choch_bars barras anteriores cujo
            # ref_high seja rompido AGORA (close[i]) pela 1a vez -> CHoCH.
            best = None
            for j in range(i-1, max(i-U_CHOCH_BARS-1, U_SWING_N+U_CHOCH_REF)-1, -1):
                swing_low = np.min(l[j-U_SWING_N:j])
                # j foi um sweep?
                if not (l[j] < swing_low and c[j] > swing_low):
                    continue
                if U_MIN_SWEEP > 0 and (swing_low - l[j]) / swing_low * 100 < U_MIN_SWEEP:
                    continue
                ref_high = np.max(h[j-U_CHOCH_REF:j])
                # CHoCH so e valido se NENHUMA barra entre j+1 e i-1 ja rompeu ref_high
                if np.any(c[j+1:i] > ref_high):
                    continue
                if c[i] > ref_high:
                    actual_low = np.min(l[j:i+1])
                    risk = c[i] - actual_low
                    if risk > 0 and risk / c[i] <= 0.10:
                        best = (c[i], actual_low)
                        break
            if best:
                side[i]='LONG'; entry[i]=best[0]; sl[i]=best[1]

    return side, entry, sl


print('A gerar sinais (Decisor a despachar por regime)...')
master_index = None
aligned = {}
for sym, df in pair_data.items():
    side, entry, sl = gen_signals(df)
    df = df.assign(sig_side=side, sig_entry=entry, sig_sl=sl)
    aligned[sym] = df
    idx = df.index
    master_index = idx if master_index is None else master_index.union(idx)

master_index = master_index[(master_index >= START) & (master_index <= END)]
print(f'  {len(master_index)} barras 1H no periodo, {len(aligned)} pares\n')

# Re-alinha cada par ao indice mestre (arrays posicionais para velocidade)
A = {}
for sym, df in aligned.items():
    df = df.reindex(master_index)
    A[sym] = {
        'h': df['h'].values, 'l': df['l'].values, 'c': df['c'].values,
        'atr': df['atr'].values, 'reg': df['regime'].values,
        'side': df['sig_side'].values, 'entry': df['sig_entry'].values,
        'sl': df['sig_sl'].values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR EVENT-DRIVEN DE PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────
def update_position(pos, hi, lo, c, atr):
    """
    Avanca UMA barra numa posicao aberta. Gestao identica a bear_v13/SMC:
    SL inicial -> BE a BE_PCT*RR -> trailing TRAIL*ATR -> TP em RR.
    Devolve (fechou?, resultado_R, motivo). Atualiza pos in-place.
    """
    entry=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
    rr=pos['rr']; be_pct=pos['be_pct']; trail=pos['trail']
    a = atr if not np.isnan(atr) else risk

    if pos['side'] == 'SHORT':
        if hi >= pos['cur']:
            motivo = 'breakeven' if (pos['be'] and pos['cur']==entry) else ('trailing' if pos['be'] else 'stop')
            return True, (entry-pos['cur'])/risk - fee_r, motivo
        if lo <= pos['tp']:
            return True, rr - fee_r, 'take_profit'
        if not pos['be'] and (entry-c)/risk >= be_pct*rr:
            pos['cur']=entry; pos['be']=True
        if pos['be']:
            cand = c + trail*a
            if cand < pos['cur']: pos['cur']=cand
    else:  # LONG
        if lo <= pos['cur']:
            motivo = 'breakeven' if (pos['be'] and pos['cur']==entry) else ('trailing' if pos['be'] else 'stop')
            return True, (pos['cur']-entry)/risk - fee_r, motivo
        if hi >= pos['tp']:
            return True, rr - fee_r, 'take_profit'
        if not pos['be'] and (c-entry)/risk >= be_pct*rr:
            pos['cur']=entry; pos['be']=True
        if pos['be']:
            cand = c - trail*a
            if cand > pos['cur']: pos['cur']=cand
    return False, 0.0, None


balance     = INITIAL_BALANCE
peak        = INITIAL_BALANCE
max_dd      = 0.0
positions   = {}                 # sym -> pos dict (posicao aberta)
cooldown_until = {sym: -1 for sym in A}   # indice ate ao qual o par esta em cooldown
trades      = []
equity_curve= []                 # (ts, balance)
max_concurrent = 0

N = len(master_index)
for k in range(N):
    ts = master_index[k]

    # ── 1) SAIDAS primeiro (atualiza saldo antes de novas entradas) ───────────
    for sym in list(positions.keys()):
        d = A[sym]
        hi=d['h'][k]; lo=d['l'][k]; c=d['c'][k]; atr=d['atr'][k]
        if np.isnan(c):   # par sem barra neste ts
            continue
        pos = positions[sym]
        closed, nr, motivo = update_position(pos, hi, lo, c, atr)
        if closed:
            pnl = nr * pos['risk_usd']
            balance += pnl
            trades.append({
                'ts_in': pos['ts_in'], 'ts_out': ts, 'sym': sym.replace('/USDT:USDT',''),
                'strat': pos['strat'], 'nr': nr, 'pnl': pnl, 'motivo': motivo,
                'bal_after': balance,
            })
            del positions[sym]
            cooldown_until[sym] = k + (B_COOLDOWN if pos['strat']=='bear' else U_COOLDOWN)

    # ── 2) ENTRADAS ────────────────────────────────────────────────────────────
    for sym in A:
        if sym in positions:                 # uma posicao por par
            continue
        if k <= cooldown_until[sym]:          # cooldown
            continue
        d = A[sym]
        side = d['side'][k]
        if side is None or (isinstance(side, float) and np.isnan(side)):
            continue
        entry = d['entry'][k]; sl = d['sl'][k]
        if np.isnan(entry) or np.isnan(sl):
            continue
        if balance <= 0:
            continue

        risk_usd = balance * (RISK_PCT/100.0)   # 1% do saldo ATUAL
        if side == 'SHORT':
            risk_px = sl - entry
            tp = entry - B_RR*risk_px
            rr, be_pct, trail, strat = B_RR, B_BE_PCT, B_TRAIL, 'bear'
        else:
            risk_px = entry - sl
            tp = entry + U_RR*risk_px
            rr, be_pct, trail, strat = U_RR, U_BE_PCT, U_TRAIL, 'bull'
        if risk_px <= 0:
            continue

        positions[sym] = {
            'side': side, 'entry': entry, 'cur': sl, 'tp': tp, 'be': False,
            'risk_px': risk_px, 'risk_usd': risk_usd, 'fee_r': entry*RT/risk_px,
            'rr': rr, 'be_pct': be_pct, 'trail': trail, 'strat': strat, 'ts_in': ts,
        }

    max_concurrent = max(max_concurrent, len(positions))

    # ── 3) Equity / drawdown ───────────────────────────────────────────────────
    peak   = max(peak, balance)
    dd     = (peak - balance) / peak * 100 if peak > 0 else 0
    max_dd = max(max_dd, dd)
    equity_curve.append((ts, balance))

# ─────────────────────────────────────────────────────────────────────────────
# RELATORIO
# ─────────────────────────────────────────────────────────────────────────────
print('='*86)
print('  RESULTADO DO PORTFOLIO')
print('='*86)

if not trades:
    print('  Nenhum trade gerado no periodo.')
    sys.exit(0)

nr_all = [t['nr'] for t in trades]
wins   = [t for t in trades if t['nr'] > 0]
ret_pct = (balance/INITIAL_BALANCE - 1) * 100
gw = sum(t['nr'] for t in wins)
gl = abs(sum(t['nr'] for t in trades if t['nr'] <= 0))

bear_t = [t for t in trades if t['strat']=='bear']
bull_t = [t for t in trades if t['strat']=='bull']

def block(name, ts):
    if not ts:
        print(f'  {name:<10}: 0 trades'); return
    w = [t for t in ts if t['nr']>0]
    pnl = sum(t['pnl'] for t in ts)
    print(f'  {name:<10}: {len(ts):>4} trades | WR {len(w)/len(ts)*100:>4.1f}% | '
          f'P&L {pnl:>+10.2f} USDC')

print(f'\n  Saldo inicial : {INITIAL_BALANCE:>12,.2f} USDC')
print(f'  Saldo final   : {balance:>12,.2f} USDC')
print(f'  Retorno       : {ret_pct:>+11.2f}%')
print(f'  Lucro liquido : {balance-INITIAL_BALANCE:>+12,.2f} USDC')
print()
print(f'  Total trades  : {len(trades)}')
print(f'  Win rate      : {len(wins)/len(trades)*100:.1f}%')
print(f'  Profit factor : {gw/gl if gl>0 else 99:.2f}')
print(f'  Soma R        : {sum(nr_all):+.1f}R')
print(f'  Max drawdown  : {max_dd:.1f}%   (pico->vale no saldo)')
print(f'  Max posicoes simultaneas : {max_concurrent}   (<- exposicao de correlacao)')
print()
print('  Contribuicao por estrategia (o Decisor escolheu cada uma por regime):')
block('BEAR', bear_t)
block('BULL', bull_t)

# ── Curva mensal ──────────────────────────────────────────────────────────────
print('\n' + '='*86)
print('  EVOLUCAO MENSAL DO SALDO')
print('='*86)
print(f"  {'Mes':<8} | {'Trades':>6} | {'WR':>5} | {'P&L mes':>14} | {'Saldo fim':>14}")
print('  ' + '-'*70)
months = {}
for t in trades:
    mk = t['ts_out'].strftime('%Y-%m')
    months.setdefault(mk, []).append(t)
ML = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
running = INITIAL_BALANCE
for mk in sorted(months):
    ts = months[mk]
    w  = [t for t in ts if t['nr']>0]
    pnl= sum(t['pnl'] for t in ts)
    running += pnl
    yr, mo = int(mk[:4]), int(mk[5:])
    print(f"  {ML[mo-1]}/{yr%100:02d} | {len(ts):>6} | {len(w)/len(ts)*100:>4.0f}% | "
          f"{pnl:>+12.2f}  | {running:>12,.2f}")

print('\n' + '='*86)
print('  AVISOS (ler antes de tirar conclusoes)')
print('='*86)
print('  • Bull SMC e HIPOTETICA — nao implementada nem aprovada. Config robusta.')
print('  • Saldo unico + posicoes simultaneas = correlacao REAL no drawdown.')
print('  • Risco fixo 1%/trade: a alavancagem nao muda o P&L (so margem/liquidacao).')
print('  • Custos round-trip 0.14% ja incluidos. Dados USDT (~= USDC).')
print('  • Bear v1.3 = codigo real. SMC = replica fiel da logica dos backtests.')
