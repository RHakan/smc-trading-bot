"""
backtest_winrate_maximo.py
PEDIDO DO RAFA: o MAIOR winrate alcancavel em 15m, trades curtos, SEM as amarras do v2.
Sem decisor de regime, sem os filtros do v2, familias de entrada diferentes (incl. as
antigas de reversao). Alvo minimo tem de ser MAIOR que a taxa. Lucro pequeno por trade
e aceitavel.

A COLUNA QUE DECIDE: para cada celula calcula-se o ACERTO NECESSARIO PARA EMPATAR.
  ganho liquido (em ATR) = T - f      perda liquida = S + f      f = RT/atr_pct
  WR_breakeven = (S + f) / ((T - f) + (S + f))
Com stop 8xATR e alvo 0.5xATR isso da 94%. Ou seja: um acerto de 85% ainda perde
dinheiro. E por isso que "maior winrate" so tem sentido lido AO LADO do breakeven —
e as duas colunas aparecem sempre juntas neste relatorio.

FAMILIAS DE ENTRADA (nenhuma usa o decisor de regime)
  rsi   — REVERSAO: RSI(14) < 30 compra, > 70 vende. Extremos tem bounce imediato
          mais provavel que continuacoes — e a familia certa para acerto alto.
  bb    — REVERSAO: fecho abaixo da banda inferior de Bollinger(20,2) compra, acima
          da superior vende.
  dist  — REVERSAO: fecho a mais de K x ATR da EMA(50) — volta a media.
  brk   — CONTINUACAO: o rompimento do v2, sem filtro de regime nem de ADX (controlo,
          para se ver quanto do acerto vem da familia e quanto vem das saidas).

Varridos: alvo T x ATR (sempre acima da taxa), stop S x ATR, e o tempo maximo em barras.
Sem filtro de regime. Sem cooldown longo. Sem invalidacao (nao se aplica fora do v2).

Baseline a bater: v2 em 1H com 42.5% de acerto e +347 (5 anos) / +60 (OOS).
10 pares em 15m, 500/mes reset, custos 0.14% RT. OOS acaba a 2026-06-29 (fim do cache).
Bot rodando NAO tocado.
Uso: python backtests/backtest_winrate_maximo.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MONTHLY_BASE = 500.0; RISK_PCT = 1.0; MAX_ABERTAS = 5
PAIRS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT',
         'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'LINK/USDT:USDT', 'SOL/USDT:USDT',
         'AVAX/USDT:USDT', 'SUI/USDT:USDT']
FEE = 0.0005; SLIP = 0.0002; RT = (FEE + SLIP) * 2

ATR_P = 14; RSI_P = 14; BB_P = 20; BB_K = 2.0; EMA_P = 50; DIST_K = 2.0
FAMILIAS = ['rsi', 'bb', 'dist', 'brk']
ALVOS = [0.30, 0.40, 0.50, 0.75, 1.00]      # x ATR (todos acima da taxa)
STOPS = [1.0, 2.0, 3.0, 5.0, 8.0]           # x ATR
TIMEOUTS = [16, 48]                          # barras de 15m = 4h e 12h
CD = 4

S5 = (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2025, 10, 6, tzinfo=timezone.utc))
S26 = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 6, 29, tzinfo=timezone.utc))
CACHE = Path(__file__).parent / 'cache'


def _read(sym):
    f = CACHE / f"{sym.replace('/', '_').replace(':', '_')}_15m_2021_2026.csv"
    if not f.exists(): return None
    d = pd.read_csv(f, index_col='ts', parse_dates=True)
    d.index = pd.to_datetime(d.index, utc=True)
    return d


def rsi(c, period):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def load(period):
    a, b = S5 if period == '5y' else S26
    out = {}
    for sym in PAIRS:
        df = _read(sym)
        if df is None: continue
        df = df[(df.index >= a) & (df.index <= b)].copy()
        if len(df) < 600: continue
        c = df['c']
        tr = pd.concat([(df['h'] - df['l']), (df['h'] - c.shift(1)).abs(),
                        (df['l'] - c.shift(1)).abs()], axis=1).max(axis=1)
        df['atr'] = tr.ewm(com=ATR_P - 1, adjust=False).mean()
        df['rsi'] = rsi(c, RSI_P)
        ma = c.rolling(BB_P).mean(); sd = c.rolling(BB_P).std()
        df['bb_lo'] = ma - BB_K * sd; df['bb_hi'] = ma + BB_K * sd
        df['ema'] = c.ewm(span=EMA_P, adjust=False).mean()
        df['hh'] = df['h'].rolling(30).max().shift(1)
        df['ll'] = df['l'].rolling(30).min().shift(1)

        n = len(df); cv = c.values; av = df['atr'].values
        sig = {}
        r = df['rsi'].values
        s = np.zeros(n, dtype=np.int8)
        s[(r < 30) & (np.r_[np.nan, r[:-1]] >= 30)] = 1
        s[(r > 70) & (np.r_[np.nan, r[:-1]] <= 70)] = -1
        sig['rsi'] = s

        s = np.zeros(n, dtype=np.int8)
        lo = df['bb_lo'].values; hi = df['bb_hi'].values
        s[cv < lo] = 1; s[cv > hi] = -1
        sig['bb'] = s

        s = np.zeros(n, dtype=np.int8)
        em = df['ema'].values
        with np.errstate(invalid='ignore'):
            s[(em - cv) > DIST_K * av] = 1
            s[(cv - em) > DIST_K * av] = -1
        sig['dist'] = s

        s = np.zeros(n, dtype=np.int8)
        hh = df['hh'].values; ll = df['ll'].values; pc = np.r_[np.nan, cv[:-1]]
        with np.errstate(invalid='ignore'):
            s[(cv > hh) & (pc <= hh)] = 1
            s[(cv < ll) & (pc >= ll)] = -1
        sig['brk'] = s

        df = df.assign(**{f'_{k}': v for k, v in sig.items()})
        out[sym] = df
    return out


def wr_be(S, T, f):
    """Acerto minimo para empatar, em unidades de ATR."""
    g = T - f; p = S + f
    return 100.0 * p / (g + p) if g > 0 else 100.0


def run(arrs, fam, start, end, S, T, tmo):
    midx = None
    for d in arrs.values():
        midx = d.index if midx is None else midx.union(d.index)
    midx = midx[(midx >= start) & (midx <= end)]
    A = {}
    for sym, d in arrs.items():
        dd = d.reindex(midx)
        A[sym] = {'o': dd['o'].values, 'h': dd['h'].values, 'l': dd['l'].values,
                  'c': dd['c'].values, 'atr': dd['atr'].values, 's': dd[f'_{fam}'].values}
    bal = MONTHLY_BASE; peak = MONTHLY_BASE; ddmax = 0.0
    pos = {}; cdu = {s: -1 for s in A}; monthly = {}; cur = None; tr = []

    def fecha(sym, nr):
        nonlocal bal
        bal += nr * pos[sym]['ru']; tr.append(nr); del pos[sym]

    for k in range(1, len(midx)):
        mk = midx[k].strftime('%Y-%m')
        if cur is None: cur = mk
        if mk != cur:
            for sym in list(pos):
                p = pos[sym]; c = A[sym]['c'][k]
                if np.isnan(c): c = p['e']
                fecha(sym, ((c - p['e']) if p['d'] > 0 else (p['e'] - c)) / p['r'] - p['f'])
            monthly[cur] = bal - MONTHLY_BASE
            bal = MONTHLY_BASE; peak = MONTHLY_BASE; cur = mk

        for sym in list(pos):
            d = A[sym]; c = d['c'][k]
            if np.isnan(c): continue
            p = pos[sym]; hi = d['h'][k]; lo = d['l'][k]
            done = False; nr = 0.0
            if p['d'] > 0:
                if lo <= p['sl']: nr = (p['sl'] - p['e']) / p['r'] - p['f']; done = True
                elif hi >= p['tp']: nr = (p['tp'] - p['e']) / p['r'] - p['f']; done = True
            else:
                if hi >= p['sl']: nr = (p['e'] - p['sl']) / p['r'] - p['f']; done = True
                elif lo <= p['tp']: nr = (p['e'] - p['tp']) / p['r'] - p['f']; done = True
            if not done:
                p['age'] += 1
                if p['age'] >= tmo:
                    nr = ((c - p['e']) if p['d'] > 0 else (p['e'] - c)) / p['r'] - p['f']
                    done = True
            if done:
                fecha(sym, nr); cdu[sym] = k + CD

        for sym in A:
            if len(pos) >= MAX_ABERTAS: break
            if sym in pos or k <= cdu[sym]: continue
            d = A[sym]; sg = d['s'][k - 1]
            if sg == 0: continue
            e = d['o'][k]; a = d['atr'][k - 1]
            if np.isnan(e) or np.isnan(a) or a <= 0 or bal <= 0: continue
            r = S * a
            pos[sym] = {'d': int(sg), 'e': e, 'r': r, 'f': e * RT / r,
                        'sl': e - r * sg, 'tp': e + T * a * sg,
                        'ru': bal * (RISK_PCT / 100.0), 'age': 0}
        peak = max(peak, bal)
        ddmax = max(ddmax, (peak - bal) / peak * 100 if peak > 0 else 0)

    for sym in list(pos):
        p = pos[sym]; c = A[sym]['c'][-1]
        if np.isnan(c): c = p['e']
        fecha(sym, ((c - p['e']) if p['d'] > 0 else (p['e'] - c)) / p['r'] - p['f'])
    monthly[cur] = bal - MONTHLY_BASE
    t = np.array(tr)
    return dict(tot=sum(monthly.values()), dd=ddmax, n=len(t),
                wr=float((t > 0).mean() * 100) if len(t) else 0.0)


def main():
    print('\n' + '=' * 104)
    print('  MAIOR WINRATE ALCANCAVEL em 15m — sem decisor, sem filtros do v2')
    print('=' * 104)
    print('  A carregar...')
    a5 = load('5y'); a26 = load('oos')
    atr = float(np.median([np.nanmedian(d['atr'].values / d['c'].values) for d in a5.values()]))
    f = RT / atr
    print(f'  {len(a5)} pares | ATR mediano 15m = {atr*100:.3f}% | taxa = {f:.2f} x ATR')
    print(f'  >> alvo minimo testado = {min(ALVOS)} xATR (acima da taxa, como pediste)\n')

    print(f"  {'fam':<6}{'stop':>6}{'alvo':>6}{'tmo':>5}{'ACERTO':>9}{'precisa':>9}"
          f"{'margem':>8}{'5 anos':>9}{'trades':>8}")
    print('  ' + '-' * 68)
    res = []
    for fam in FAMILIAS:
        for tmo in TIMEOUTS:
            for S in STOPS:
                for T in ALVOS:
                    be = wr_be(S, T, f)
                    r = run(a5, fam, *S5, S, T, tmo)
                    marg = r['wr'] - be
                    res.append(dict(fam=fam, S=S, T=T, tmo=tmo, wr=r['wr'], be=be,
                                    marg=marg, tot=r['tot'], n=r['n'], dd=r['dd']))
                    if marg > -3:      # só imprime as que chegam perto, senão são 200 linhas
                        print(f"  {fam:<6}{S:>6.1f}{T:>6.2f}{tmo:>5}{r['wr']:>8.1f}%"
                              f"{be:>8.1f}%{marg:>+8.1f}{r['tot']:>+9.0f}{r['n']:>8}")
    print('  (só aparecem as células a menos de 3 pontos do breakeven — as outras nem perto)\n')

    print('=' * 104)
    print('  MAIOR ACERTO POR FAMILIA (e o que custa)')
    print('=' * 104)
    print(f"  {'fam':<6}{'melhor acerto':>15}{'precisa':>9}{'margem':>9}{'5 anos':>9}"
          f"{'stop/alvo':>12}{'trades':>8}")
    print('  ' + '-' * 70)
    for fam in FAMILIAS:
        c = [r for r in res if r['fam'] == fam and r['n'] > 200]
        if not c: continue
        b = max(c, key=lambda r: r['wr'])
        sa = f"{b['S']}/{b['T']}"
        print(f"  {fam:<6}{b['wr']:>14.1f}%{b['be']:>8.1f}%{b['marg']:>+9.1f}"
              f"{b['tot']:>+9.0f}{sa:>12}{b['n']:>8}")
    print()

    mx = max([r for r in res if r['n'] > 200], key=lambda r: r['wr'])
    print(f"  >> ACERTO MAXIMO ABSOLUTO: {mx['wr']:.1f}%  ({mx['fam']}, stop {mx['S']}x, "
          f"alvo {mx['T']}x, {mx['tmo']} barras)")
    print(f"     precisa de {mx['be']:.1f}% para empatar -> margem {mx['marg']:+.1f} pontos"
          f" -> total {mx['tot']:+.0f}")
    print()

    viav = [r for r in res if r['marg'] > 0 and r['n'] > 200]
    print('=' * 104)
    print('  CELULAS COM ACERTO ACIMA DO BREAKEVEN (as unicas que podem dar lucro)')
    print('=' * 104)
    if not viav:
        print('  NENHUMA. Em todas as 200 combinacoes o acerto obtido fica ABAIXO do')
        print('  necessario para empatar. Nao e uma questao de afinacao — o acerto alto')
        print('  em 15m custa sempre mais em R:R do que aquilo que traz em frequencia.')
    else:
        print(f"  {'fam':<6}{'stop':>6}{'alvo':>6}{'acerto':>9}{'precisa':>9}{'margem':>8}"
              f"{'5 anos':>9}{'OOS':>8}{'DD':>7}{'trades':>8}")
        print('  ' + '-' * 76)
        for r in sorted(viav, key=lambda x: -x['tot'])[:15]:
            o = run(a26, r['fam'], *S26, r['S'], r['T'], r['tmo'])
            m = '  <= bate o v2 nos 2' if (r['tot'] > 347 and o['tot'] > 60) else ''
            print(f"  {r['fam']:<6}{r['S']:>6.1f}{r['T']:>6.2f}{r['wr']:>8.1f}%"
                  f"{r['be']:>8.1f}%{r['marg']:>+8.1f}{r['tot']:>+9.0f}{o['tot']:>+8.0f}"
                  f"{r['dd']:>6.1f}%{r['n']:>8}{m}")
    print()


if __name__ == '__main__':
    main()
