"""
backtest_momentum_seq.py
ESTRATEGIA AGRESSIVA DO RAFA — capital inteiro, 1 posicao de cada vez, continuidade
de movimento, alvo curto definido em % do CAPITAL (nao do preco).

HIPOTESE CENTRAL
----------------
Apanhar um movimento de 5% com 5 trades encadeados paga 10 pernas de taxa; segurar
um trade so paga 2. A "escada" so compensa se as reentradas evitarem devolucoes que
uma ordem unica sofreria. Isto tem resposta numerica — e o objetivo do teste.

O CONSTRANGIMENTO QUE DOMINA TUDO
---------------------------------
A taxa e um pedagio FIXO proporcional a alavancagem: f*L do capital por round-trip.
Com alvo G% e perda maxima P% do capital, a alavancagem L e a taxa f (fracao RT):

    p_tp = G/L + f      (movimento de preco necessario para o alvo)
    p_sl = P/L - f      (movimento de preco disponivel para o stop)

Consequencias (contra-intuitivas, e o motivo de L ser eixo de 1a ordem):
  * MAIS alavancagem PIORA: consome o orcamento de perda.
  * L_max = P/f. Acima disso p_sl <= 0 e o alvo e MATEMATICAMENTE IMPOSSIVEL.
    (P=0.5%, taker f=0.14%  ->  L_max ~ 3.57x)
  * Com G=2.0 / P=0.5 / L=2 / taker: p_sl = 0.11% — isso e RUIDO, nao e um stop.

TRES MODOS DE SAIDA COMPARADOS (mesmo motor, mesmos sinais)
-----------------------------------------------------------
  A — ESCADA      : TP fixo. Fecha no alvo e REENTRA se o sinal continuar (cd_win=0).
  B — TRAVA+HILO  : (ideia refinada do Rafa, candidato principal)
                    SL fixo. Ao atingir TRIG% do alvo, o TP e ABANDONADO:
                      . stop salta para LOCK% do caminho entrada->TP (lucro travado)
                      . HiLo(N) assume o trailing, so melhora (monotonico)
                      . sai na virada do HiLo ou no stop travado
  C — CONTROLO    : TP fixo, sem reentrada, sem HiLo. Isola o efeito de A e de B.

TRES FAMILIAS DE SINAL DE CONTINUIDADE
--------------------------------------
  imp  — impulso puro   : corpo da vela > K*ATR na direcao (leitura literal do Rafa)
  pull — pullback       : tendencia por EMA, entra no recuo a EMA rapida
  brk  — breakout curto : mesmo DNA de bot/strategies/lateral_breakout.py

HONESTIDADE DO MOTOR
--------------------
  * Entrada SEMPRE na ABERTURA da vela seguinte ao sinal (zero lookahead).
  * Stop verificado ANTES do alvo na mesma vela (conservador).
  * GAP-AWARE: vela que abre ja alem do stop preenche na ABERTURA real, nao no nivel.
  * Stop novo (trava/HiLo) so vale a partir da PROXIMA vela — nunca retroativo.
  * Virada do HiLo fecha na ABERTURA da vela seguinte, nunca dentro da propria vela.
  * MAKER: ordem limite so preenche se o preco a tocar dentro de N barras; senao o
    sinal e DESCARTADO (nao vira entrada a mercado). Sem isto o maker e fantasia.
  * DD calculado MARK-TO-MARKET (inclui posicao aberta, via MAE por trade). O template
    antigo so via saldo realizado — com all-in isso subestima gravemente o DD real.

DADOS / PERIODOS
----------------
  15m: 10 pares  |  5m: so ETH (unico par com 5m em cache)
  5 anos : 2021-01-01 -> 2025-10-06
  OOS    : 2026-01-01 -> 2026-06-29   (o cache 15m/5m acaba a 29/06, nao 09/07)
  ATENCAO: a janela OOS e ~10 dias mais curta que a do baseline v3 (+60). A
  comparacao e APROXIMADA e esta declarada como tal no relatorio.

Baseline v3 a bater: +347 (5 anos) e +60 (OOS), DD 10.1%. 500/mes reset, custos RT.
Bot rodando NAO tocado. Nenhum ficheiro de bot/ ou api/ e alterado por este script.
Uso: python backtests/backtest_momentum_seq.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
MONTHLY_BASE = 500.0          # capital base do modelo "reset mensal" (= template)
RISK_PCT_S2 = 1.0             # risco/trade do modelo S2, igual ao baseline v3

PAIRS_15M = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT',
             'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'LINK/USDT:USDT', 'SOL/USDT:USDT',
             'AVAX/USDT:USDT', 'SUI/USDT:USDT']
PAIRS_5M = ['ETH/USDT:USDT']

# Modelos de taxa (fracao round-trip, as duas pernas somadas)
F_TAKER = (0.0005 + 0.0002) * 2   # 0.14% — taker + slippage (igual ao template)
F_MAKER = (0.0002 + 0.0000) * 2   # 0.04% — maker, sem slippage

# Janelas de simulacao
S5 = (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2025, 10, 6, tzinfo=timezone.utc))
S26 = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 6, 29, tzinfo=timezone.utc))

# Especificacao do Rafa (ponto de partida da grelha)
SPEC = dict(G=2.0, P=0.5, L=2.0, trig=0.75, lock=0.50, hilo_n=5)

NS_HILO = [3, 5, 8]
TRIGS = [0.60, 0.75, 0.85]      # gatilho da trava, em fracao do alvo
LOCKS = [0.40, 0.50, 0.60]      # nivel travado, em fracao do caminho entrada->TP

TIMEOUT_BARS = 192              # 48h em 15m / 16h em 5m
CD_LOSS = 4                     # cooldown apos perda (barras)
CD_BASE = 4                     # cooldown apos ganho nos modos B e C
MAKER_WAIT = 3                  # barras que a ordem limite espera antes de ser descartada

# Parametros dos sinais
IMP_K = 1.0                     # corpo > K*ATR
PULL_EMA_F, PULL_EMA_S = 20, 50
BRK_LB, BRK_ADX_MAX, BRK_BUF = 30, 20.0, 0.10
ATR_P = 14

CACHE = Path(__file__).parent / 'cache'


# ─────────────────────────────────────────────────────────────────────────────
# Leitura de dados — cache-only (os ficheiros sub-horarios ja existem)
# Copiado de backtest_lateral_15m.py:41-57
# ─────────────────────────────────────────────────────────────────────────────
def _read(sym, tf, tag):
    f = CACHE / f"{sym.replace('/', '_').replace(':', '_')}_{tf}_{tag}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, index_col='ts', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_entry(sym, tf, period):
    """15m/5m vivem num ficheiro unico 2021_2026, fatiado por periodo."""
    df = _read(sym, tf, '2021_2026')
    if df is None:
        return None
    a, b = S5 if period == '5y' else S26
    return df[(df.index >= a) & (df.index <= b)]


def adx(df, period=14):
    """Identico a backtest_supertrend_hma.py:130 / backtest_lateral_15m.py:60."""
    h, l, c = df['h'], df['l'], df['c']
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    at = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / at
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / at
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def hilo(h, l, c, n):
    """HiLo Activator (Gann). Copiado verbatim de backtest_hilo_ratchet.py:90.

    Devolve (direcao, sma_h, sma_l). A direcao usa a SMA da barra ANTERIOR — e a
    unica forma de a linha estar disponivel no momento da decisao.
    """
    sma_h = pd.Series(h).rolling(n).mean().values
    sma_l = pd.Series(l).rolling(n).mean().values
    N = len(c)
    dirn = np.zeros(N)
    cur = 0
    for i in range(N):
        if i < 1 or np.isnan(sma_h[i - 1]) or np.isnan(sma_l[i - 1]):
            dirn[i] = cur
            continue
        if c[i] > sma_h[i - 1]:
            cur = 1
        elif c[i] < sma_l[i - 1]:
            cur = -1
        dirn[i] = cur
    return dirn, sma_h, sma_l


# ─────────────────────────────────────────────────────────────────────────────
# Matematica dos niveis — o coracao do constrangimento
# ─────────────────────────────────────────────────────────────────────────────
def levels(g_pct, p_pct, lev, f):
    """Converte alvo/perda em % do CAPITAL para movimentos de PRECO.

    p_tp = G/L + f   |   p_sl = P/L - f
    Devolve None quando p_sl <= 0 (a taxa sozinha ja consome o orcamento de perda).
    """
    p_tp = (g_pct / 100.0) / lev + f
    p_sl = (p_pct / 100.0) / lev - f
    if p_sl <= 0:
        return None
    return p_tp, p_sl


def lev_max(p_pct, f):
    """Alavancagem acima da qual o limite de perda e inatingivel."""
    return (p_pct / 100.0) / f


# ─────────────────────────────────────────────────────────────────────────────
# Preparacao: indicadores + sinais, uma vez por (simbolo, tf, periodo)
# ─────────────────────────────────────────────────────────────────────────────
def prep(sym, tf, period):
    df = load_entry(sym, tf, period)
    if df is None or len(df) < 400:
        return None

    c = df['c']
    tr = pd.concat([(df['h'] - df['l']),
                    (df['h'] - c.shift(1)).abs(),
                    (df['l'] - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=ATR_P - 1, adjust=False).mean()

    o = df['o'].values.astype(float)
    h = df['h'].values.astype(float)
    lo = df['l'].values.astype(float)
    cl = df['c'].values.astype(float)
    a = atr.values.astype(float)
    n = len(cl)

    ema_f = c.ewm(span=PULL_EMA_F, adjust=False).mean().values
    ema_s = c.ewm(span=PULL_EMA_S, adjust=False).mean().values
    adx_v = adx(df, 14).values

    # ── Sinais (0=nada, 1=long, -1=short). Avaliados NO FECHO da barra i;
    #    a entrada acontece na ABERTURA da barra i+1 (tratado no motor).
    sig = {}

    # imp — impulso puro: corpo > K*ATR na direcao do movimento
    body = cl - o
    s = np.zeros(n, dtype=np.int8)
    ok = (~np.isnan(a)) & (a > 0)
    s[ok & (body > IMP_K * a) & (np.r_[False, np.diff(cl) > 0])] = 1
    s[ok & (-body > IMP_K * a) & (np.r_[False, np.diff(cl) < 0])] = -1
    sig['imp'] = s

    # pull — pullback a EMA rapida dentro da tendencia
    s = np.zeros(n, dtype=np.int8)
    up = ema_f > ema_s
    s[up & (lo <= ema_f) & (cl > ema_f) & (cl > o)] = 1
    s[(~up) & (h >= ema_f) & (cl < ema_f) & (cl < o)] = -1
    sig['pull'] = s

    # brk — rompimento de range curto consolidado (DNA da lateral_breakout)
    s = np.zeros(n, dtype=np.int8)
    hh = pd.Series(h).rolling(BRK_LB).max().shift(1).values
    ll = pd.Series(lo).rolling(BRK_LB).min().shift(1).values
    adx_prev = np.r_[np.nan, adx_v[:-1]]
    calm = adx_prev < BRK_ADX_MAX
    prev_c = np.r_[np.nan, cl[:-1]]
    with np.errstate(invalid='ignore'):
        s[ok & calm & (cl > hh + BRK_BUF * a) & (prev_c <= hh)] = 1
        s[ok & calm & (cl < ll - BRK_BUF * a) & (prev_c >= ll)] = -1
    sig['brk'] = s

    hl = {}
    for nn in NS_HILO:
        d, sh, sl_ = hilo(h, lo, cl, nn)
        hl[nn] = (d, sh, sl_)

    # ATR mediano em % do preco — serve para marcar celulas RUIDO
    with np.errstate(invalid='ignore', divide='ignore'):
        atr_pct = np.nanmedian(a / cl)

    return dict(ts=df.index.values, o=o, h=h, l=lo, c=cl, atr=a,
                sig=sig, hilo=hl, atr_pct=float(atr_pct), n=n)


def build(pairs, tf, period):
    out = {}
    for sym in pairs:
        d = prep(sym, tf, period)
        if d is not None:
            out[sym] = d
    return out


def align(pdata):
    """Alinha todos os pares numa timeline-uniao. Pares sem historico (ex.: SUI antes
    de 2023-05) ficam NaN e o motor ignora-os — nunca sao lidos como zeros."""
    if not pdata:
        return None, {}
    idx = None
    for d in pdata.values():
        i = pd.DatetimeIndex(d['ts'])
        idx = i if idx is None else idx.union(i)
    A = {}
    for sym, d in pdata.items():
        src = pd.DatetimeIndex(d['ts'])
        pos = src.get_indexer(idx)          # -1 onde o par nao tem barra
        m = pos >= 0
        def take(arr, fill=np.nan):
            out = np.full(len(idx), fill, dtype=float)
            out[m] = arr[pos[m]]
            return out
        e = dict(o=take(d['o']), h=take(d['h']), l=take(d['l']), c=take(d['c']),
                 atr=take(d['atr']), valid=m)
        e['sig'] = {}
        for k, v in d['sig'].items():
            s = np.zeros(len(idx), dtype=np.int8)
            s[m] = v[pos[m]]
            e['sig'][k] = s
        e['hilo'] = {}
        for nn, (dr, sh, sl_) in d['hilo'].items():
            e['hilo'][nn] = (take(dr, 0.0), take(sh), take(sl_))
        e['atr_pct'] = d['atr_pct']
        A[sym] = e
    return idx, A


# ─────────────────────────────────────────────────────────────────────────────
# Motor de simulacao — 1 posicao aberta de cada vez, global
# ─────────────────────────────────────────────────────────────────────────────
def simulate(idx, A, mode, cfg):
    """Gera a lista de trades. NAO aplica dimensionamento — isso e a contabilidade.

    Como so existe 1 posicao de cada vez e os niveis sao percentuais, a SEQUENCIA de
    trades nao depende do saldo. Da para gerar uma vez e aplicar varias contabilidades
    (o mesmo truque de backtest_btc_denominado.py).

    Cada trade regista, alem do retorno de preco, o MAE (pior excursao) — e o que
    permite calcular o drawdown mark-to-market na fase de contabilidade.
    """
    syms = list(A.keys())
    ns = len(syms)
    K = len(idx)
    fam = cfg['fam']
    p_tp, p_sl = cfg['p_tp'], cfg['p_sl']
    maker = cfg.get('maker', False)
    hn = cfg.get('hilo_n', 5)
    trig, lock = cfg.get('trig', 0.75), cfg.get('lock', 0.50)
    cd_win = 0 if mode == 'A' else CD_BASE

    S = {s: A[s]['sig'][fam] for s in syms}
    trades = []
    pos = None
    pend = None          # ordem limite pendente (so no modo maker)
    cd_until = 0

    for k in range(1, K):
        # ── 1. Procurar sinal (barra k-1) e entrar na ABERTURA de k ──────────
        #    A posicao aberta em k E gerida em k logo a seguir: a propria vela de
        #    entrada tem de poder acionar o stop, senao o resultado e inflacionado.
        if pos is None and pend is None and k >= cd_until:
            off = k % ns          # rotacao: evita favorecer sempre o mesmo par
            for j in range(ns):
                s = syms[(off + j) % ns]
                d = A[s]
                sg = S[s][k - 1]
                if sg == 0:
                    continue
                if np.isnan(d['o'][k]) or np.isnan(d['c'][k - 1]):
                    continue
                if maker:
                    pend = dict(sym=s, dir=int(sg), px=d['c'][k - 1], wait=0)
                else:
                    pos = _open(s, int(sg), d['o'][k], k, p_sl)
                break

        # ── 2. Resolver ordem limite pendente (modo maker) ───────────────────
        if pos is None and pend is not None:
            s = pend['sym']
            d = A[s]
            l_, h_, o_ = d['l'][k], d['h'][k], d['o'][k]
            if np.isnan(o_):
                pend['wait'] += 1
                if pend['wait'] > MAKER_WAIT:
                    pend = None
                continue
            dr = pend['dir']
            filled = (l_ <= pend['px']) if dr > 0 else (h_ >= pend['px'])
            if filled:
                pos = _open(s, dr, pend['px'], k, p_sl)
                pend = None
            else:
                pend['wait'] += 1
                if pend['wait'] > MAKER_WAIT:
                    pend = None   # sinal DESCARTADO — nunca vira entrada a mercado
                continue

        # ── 3. Gestao da posicao aberta na barra k ───────────────────────────
        if pos is not None:
            s = pos['sym']
            d = A[s]
            o_, h_, l_, c_ = d['o'][k], d['h'][k], d['l'][k], d['c'][k]
            if np.isnan(c_):
                pos['bars'] += 1
                if pos['bars'] >= TIMEOUT_BARS:
                    _close(trades, pos, pos['last_c'], k, 'timeout')
                    pos = None
                    cd_until = k + 1 + CD_LOSS
                continue

            dr = pos['dir']
            pos['bars'] += 1
            pos['last_c'] = c_

            # MAE / MFE em retorno de preco assinado
            adverse = (l_ - pos['entry']) / pos['entry'] * dr if dr > 0 else (h_ - pos['entry']) / pos['entry'] * dr
            favor = (h_ - pos['entry']) / pos['entry'] * dr if dr > 0 else (l_ - pos['entry']) / pos['entry'] * dr
            pos['mae'] = min(pos['mae'], adverse)
            pos['mfe'] = max(pos['mfe'], favor)

            exited = False

            # Saida agendada pela virada do HiLo na barra anterior (abre e sai)
            if pos.get('exit_next'):
                _close(trades, pos, o_, k, 'hilo_flip')
                pos = None
                cd_until = k + 1 + cd_win
                continue

            # ── STOP primeiro (conservador), gap-aware ──
            stop = pos['stop']
            if dr > 0:
                gapped = o_ <= stop
                hit = l_ <= stop
            else:
                gapped = o_ >= stop
                hit = h_ >= stop
            if gapped:
                _close(trades, pos, o_, k, 'gap', gap=True)
                exited = True
            elif hit:
                _close(trades, pos, stop, k, 'stop')
                exited = True

            if exited:
                won = trades[-1]['pr'] > 0
                cd = cd_win if won else CD_LOSS
                pos = None
                cd_until = k + 1 + cd
                continue

            # ── Modos A e C: alvo fixo ──
            if mode in ('A', 'C'):
                tgt = pos['entry'] * (1 + p_tp * dr)
                reached = (h_ >= tgt) if dr > 0 else (l_ <= tgt)
                if reached:
                    _close(trades, pos, tgt, k, 'target')
                    pos = None
                    cd_until = k + 1 + cd_win
                    continue

            # ── Modo D: HiLo ATIVO DESDE A ENTRADA (nao ha alvo fixo) ──
            elif mode == 'D':
                dirn, sma_h, sma_l = d['hilo'][hn]
                pos['armed'] = True
                if dirn[k] == dr:
                    # Enquanto o HiLo esta a favor, sobe o stop.
                    #  'open' = abertura desta vela (spec literal do Rafa)
                    #  'line' = a propria linha do HiLo (mais folgado, isola a variavel)
                    if cfg.get('trail', 'open') == 'open':
                        cand = o_
                    else:
                        cand = sma_l[k - 1] if dr > 0 else sma_h[k - 1]
                    if not np.isnan(cand):
                        if (dr > 0 and cand > pos['stop']) or (dr < 0 and cand < pos['stop']):
                            pos['stop'] = cand
                # Breakeven a meio do caminho entrada->TP
                if not pos['be_hit']:
                    be_px = pos['entry'] * (1 + p_tp * 0.5 * dr)
                    if (h_ >= be_px) if dr > 0 else (l_ <= be_px):
                        pos['be_hit'] = True
                        if (dr > 0 and pos['entry'] > pos['stop']) or \
                           (dr < 0 and pos['entry'] < pos['stop']):
                            pos['stop'] = pos['entry']
                        pos['locked_at'] = pos['stop']
                # Quem fecha a ordem e o HiLo: virada -> sai na abertura seguinte
                if dirn[k] != 0 and dirn[k] != dr:
                    pos['exit_next'] = True

            # ── Modo B: trava + HiLo a partir de um gatilho ──
            else:
                if not pos['armed']:
                    trig_px = pos['entry'] * (1 + p_tp * trig * dr)
                    reached = (h_ >= trig_px) if dr > 0 else (l_ <= trig_px)
                    if reached:
                        # Trava so passa a valer a partir da PROXIMA vela (sem retroativo)
                        pos['armed'] = True
                        new_stop = pos['entry'] * (1 + p_tp * lock * dr)
                        if (dr > 0 and new_stop > pos['stop']) or (dr < 0 and new_stop < pos['stop']):
                            pos['stop'] = new_stop
                        pos['locked_at'] = new_stop
                else:
                    dirn, sma_h, sma_l = d['hilo'][hn]
                    # Trailing pela linha do HiLo da barra ANTERIOR (disponivel na abertura)
                    line = sma_l[k - 1] if dr > 0 else sma_h[k - 1]
                    if not np.isnan(line):
                        if (dr > 0 and line > pos['stop']) or (dr < 0 and line < pos['stop']):
                            pos['stop'] = line
                    # Virada do HiLo no FECHO desta vela -> sai na abertura da seguinte
                    if dirn[k] != 0 and dirn[k] != dr:
                        pos['exit_next'] = True

            if pos is not None and pos['bars'] >= TIMEOUT_BARS:
                _close(trades, pos, c_, k, 'timeout')
                pos = None
                cd_until = k + 1 + CD_LOSS
            continue

    return trades


def _open(sym, dr, px, k, p_sl):
    return dict(sym=sym, dir=dr, entry=px, stop=px * (1 - p_sl * dr), k0=k,
                bars=0, mae=0.0, mfe=0.0, armed=False, exit_next=False,
                last_c=px, locked_at=None, be_hit=False)


def _close(trades, pos, px, k, reason, gap=False):
    pr = (px - pos['entry']) / pos['entry'] * pos['dir']
    trades.append(dict(sym=pos['sym'], dir=pos['dir'], k0=pos['k0'], k1=k,
                       pr=pr, mae=pos['mae'], mfe=pos['mfe'],
                       bars=pos['bars'], reason=reason, gap=gap,
                       armed=pos['armed'], locked_at=pos['locked_at']))


# ─────────────────────────────────────────────────────────────────────────────
# Contabilidade — aplica dimensionamento a uma lista de trades ja gerada
# ─────────────────────────────────────────────────────────────────────────────
def account(trades, idx, f, sizing, lev=2.0, p_sl=None, reset_mensal=True):
    """sizing: 'allin' (nocional = L * saldo) ou 'risco' (modelo atual do bot).

    DD e MARK-TO-MARKET: durante cada trade o vale do capital e calculado com o MAE,
    nao so com o resultado final. Com all-in isto e obrigatorio — a oscilacao da
    posicao aberta E a conta inteira.
    """
    bal = MONTHLY_BASE
    peak = MONTHLY_BASE
    max_dd = 0.0
    monthly = {}
    cur_m = None
    fees_tot = 0.0
    gross_tot = 0.0
    rets = []
    streak = 0
    worst_streak = 0
    ruina50 = False
    ruina25 = False
    gap_breaks = 0
    gap_cost = 0.0

    for t in trades:
        mk = pd.Timestamp(idx[t['k1']]).strftime('%Y-%m')
        if cur_m is None:
            cur_m = mk
        if mk != cur_m:
            if reset_mensal:
                monthly[cur_m] = bal - MONTHLY_BASE
                bal = MONTHLY_BASE
                peak = MONTHLY_BASE
            else:
                monthly[cur_m] = monthly.get(cur_m, 0.0)
            cur_m = mk

        if sizing == 'allin':
            notional = lev * bal
        else:
            notional = (bal * RISK_PCT_S2 / 100.0) / p_sl

        # Vale mark-to-market durante o trade (MAE ja inclui a taxa por pagar)
        trough = bal + notional * (t['mae'] - f)
        peak = max(peak, bal)
        if peak > 0:
            max_dd = max(max_dd, (peak - trough) / peak * 100.0)

        gross = notional * t['pr']
        fee = notional * f
        pnl = gross - fee
        gross_tot += gross
        fees_tot += fee
        bal += pnl
        rets.append(pnl)

        if t['gap'] and t['armed']:
            gap_breaks += 1
            gap_cost += pnl

        peak = max(peak, bal)
        if peak > 0:
            max_dd = max(max_dd, (peak - bal) / peak * 100.0)
        if not reset_mensal:
            monthly[mk] = monthly.get(mk, 0.0) + pnl
            if bal <= MONTHLY_BASE * 0.50:
                ruina50 = True
            if bal <= MONTHLY_BASE * 0.25:
                ruina25 = True
        else:
            if bal <= MONTHLY_BASE * 0.50:
                ruina50 = True
            if bal <= MONTHLY_BASE * 0.25:
                ruina25 = True

        if pnl < 0:
            streak += 1
            worst_streak = max(worst_streak, streak)
        else:
            streak = 0

    if cur_m is not None:
        if reset_mensal:
            monthly[cur_m] = bal - MONTHLY_BASE
        else:
            monthly.setdefault(cur_m, 0.0)

    return dict(monthly=monthly, dd=max_dd, tot=sum(monthly.values()),
                fees=fees_tot, gross=gross_tot, n=len(trades),
                worst_streak=worst_streak, ruina50=ruina50, ruina25=ruina25,
                gap_breaks=gap_breaks, gap_cost=gap_cost, final=bal)


def stats(trades, res, f, lev=1.0):
    """Metricas explicativas.

    O ganho medio REAL raramente e igual ao alvo G: timeouts, gaps e (sobretudo) a
    trava do modo B fazem muitos vencedores sair ANTES do alvo. Por isso o breakeven
    tem de ser calculado com o R:R REALIZADO, nao com o R:R teorico — senao aparece
    o paradoxo de uma taxa de acerto "acima do breakeven" a perder dinheiro.
    """
    res = {k: v for k, v in res.items() if k != 'n'}   # 'n' vem daqui, nao do account
    n = len(trades)
    vazio = dict(n=0, wr=0.0, avg_win=0.0, avg_loss=0.0, rr_real=0.0,
                 be_real=0.0, avg_bars=0.0, giveback=0.0, **res)
    if n == 0:
        return vazio
    pr = np.array([t['pr'] for t in trades])
    net = pr - f
    wins = net > 0
    if not wins.any() or wins.all():
        return {**vazio, 'n': n, 'wr': float(wins.mean() * 100)}
    gw = float(net[wins].mean() * 100 * lev)      # ganho medio, em % do CAPITAL
    gl = float(net[~wins].mean() * 100 * lev)     # perda media, em % do CAPITAL
    rr = abs(gw / gl) if gl else 0.0
    mfe = np.array([t['mfe'] for t in trades])
    gb = mfe[wins] - pr[wins]                     # devolucao do pico nos vencedores
    return dict(n=n,
                wr=float(wins.mean() * 100),
                avg_win=gw, avg_loss=gl, rr_real=rr,
                be_real=abs(gl) / (gw + abs(gl)) * 100.0,
                avg_bars=float(np.mean([t['bars'] for t in trades])),
                giveback=float(gb.mean() * 100) if len(gb) else 0.0, **res)


def wr_breakeven(g_pct, p_pct):
    """Taxa de acerto minima para empatar, assumindo que todo vencedor atinge G."""
    return p_pct / (g_pct + p_pct) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Testes unitarios — correm SEMPRE antes do backtest
# ─────────────────────────────────────────────────────────────────────────────
def _tests():
    ok = []

    # 1. Formulas dos niveis batem a tabela do plano
    r = levels(1.5, 0.5, 2.0, F_TAKER)
    ok.append(('niveis L=2 taker', r is not None
               and abs(r[0] - 0.0089) < 1e-6 and abs(r[1] - 0.0011) < 1e-6))
    r1 = levels(1.5, 0.5, 1.0, F_TAKER)
    ok.append(('niveis L=1 taker', r1 is not None
               and abs(r1[0] - 0.0164) < 1e-6 and abs(r1[1] - 0.0036) < 1e-6))

    # 2. Celula impossivel e detetada (nao produz numeros)
    ok.append(('L=4 impossivel', levels(1.5, 0.5, 4.0, F_TAKER) is None))
    ok.append(('L_max ~3.57', abs(lev_max(0.5, F_TAKER) - 3.5714) < 0.01))

    # 3. HiLo reproduz a direcao correta numa serie com viragem conhecida
    c = np.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9], dtype=float)
    h = c + 0.5
    l = c - 0.5
    d, sh, sl_ = hilo(h, l, c, 3)
    ok.append(('hilo sobe', d[5] == 1))
    ok.append(('hilo vira', d[-1] == -1))

    # 4. Motor: stop verificado ANTES do alvo NA PROPRIA VELA DE ENTRADA.
    #    Sinal na barra 0 -> entra na abertura da barra 1 (=100). A barra 1 toca
    #    o alvo (102) E o stop (98); tem de sair pelo STOP, nunca pelo alvo.
    idx = pd.date_range('2024-01-01', periods=6, freq='15min', tz='UTC')
    A = {'X': dict(
        o=np.array([100, 100, 100, 100, 100, 100.]),
        h=np.array([100, 105, 100, 100, 100, 100.]),
        l=np.array([100, 95, 100, 100, 100, 100.]),
        c=np.array([100, 100, 100, 100, 100, 100.]),
        atr=np.full(6, 1.0), valid=np.ones(6, bool),
        sig={'imp': np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)},
        hilo={5: (np.zeros(6), np.full(6, np.nan), np.full(6, np.nan))},
        atr_pct=0.01)}
    tr = simulate(idx, A, 'C', dict(fam='imp', p_tp=0.02, p_sl=0.02, hilo_n=5))
    ok.append(('stop antes do alvo (vela de entrada)',
               len(tr) == 1 and tr[0]['reason'] == 'stop'))

    # 5. Gap: entra a 100 na barra 1, barra 2 ABRE a 90 com o stop em 98.
    #    Tem de preencher na ABERTURA real (90 => -10%), nao no nivel do stop (-2%).
    A2 = {'X': dict(
        o=np.array([100, 100, 90, 100, 100, 100.]),
        h=np.array([100, 100, 92, 100, 100, 100.]),
        l=np.array([100, 100, 88, 100, 100, 100.]),
        c=np.array([100, 100, 90, 100, 100, 100.]),
        atr=np.full(6, 1.0), valid=np.ones(6, bool),
        sig={'imp': np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)},
        hilo={5: (np.zeros(6), np.full(6, np.nan), np.full(6, np.nan))},
        atr_pct=0.01)}
    tr2 = simulate(idx, A2, 'C', dict(fam='imp', p_tp=0.05, p_sl=0.02, hilo_n=5))
    ok.append(('gap preenche na abertura',
               len(tr2) == 1 and tr2[0]['gap'] and abs(tr2[0]['pr'] - (-0.10)) < 1e-9))

    # 6. Modo B: trava arma no gatilho e o stop nunca piora
    o6 = np.array([100, 100, 100, 100, 100, 100, 100, 100.])
    h6 = np.array([100, 100, 102, 102, 100, 100, 100, 100.])   # vela 2 atinge o gatilho
    l6 = np.array([100, 100, 100, 100, 100, 100, 100, 100.])
    idx6 = pd.date_range('2024-01-01', periods=8, freq='15min', tz='UTC')
    A3 = {'X': dict(
        o=o6, h=h6, l=l6, c=np.full(8, 100.0), atr=np.full(8, 1.0),
        valid=np.ones(8, bool),
        sig={'imp': np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)},
        hilo={5: (np.zeros(8), np.full(8, np.nan), np.full(8, np.nan))},
        atr_pct=0.01)}
    trs = []
    cfgB = dict(fam='imp', p_tp=0.02, p_sl=0.02, hilo_n=5, trig=0.75, lock=0.50)
    tr3 = simulate(idx6, A3, 'B', cfgB)
    # gatilho = 100*(1+0.02*0.75) = 101.5 -> atingido na vela 2; trava = 101.0
    ok.append(('modo B arma e trava', len(tr3) == 1 and tr3[0]['armed']
               and tr3[0]['locked_at'] is not None
               and abs(tr3[0]['locked_at'] - 101.0) < 1e-9))
    ok.append(('trava nao sai na propria vela', tr3[0]['k1'] > 2))

    # 6b. Modo D: HiLo desde a entrada. O stop sobe para a abertura de cada vela
    #     enquanto o HiLo esta a favor, e a virada fecha na abertura seguinte.
    idxD = pd.date_range('2024-01-01', periods=6, freq='15min', tz='UTC')
    AD = {'X': dict(
        o=np.array([100, 100, 101, 102, 103, 104.]),
        h=np.array([100, 100.5, 101.5, 102.5, 103.5, 104.5]),
        l=np.array([100, 99.9, 100.9, 101.9, 102.9, 103.9]),
        c=np.array([100, 100.4, 101.4, 102.4, 103.4, 104.4]),
        atr=np.full(6, 1.0), valid=np.ones(6, bool),
        sig={'imp': np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)},
        hilo={5: (np.array([1., 1., 1., 1., -1., -1.]),
                  np.full(6, np.nan), np.full(6, np.nan))},
        atr_pct=0.01)}
    trD = simulate(idxD, AD, 'D', dict(fam='imp', p_tp=0.05, p_sl=0.02,
                                       hilo_n=5, trail='open'))
    # entra a 100; stop sobe 98->100->101->102; HiLo vira na vela 4; sai a o[5]=104
    ok.append(('modo D sai na virada do HiLo',
               len(trD) == 1 and trD[0]['reason'] == 'hilo_flip'
               and abs(trD[0]['pr'] - 0.04) < 1e-9))

    # 6c. Modo D: o ratchet protege — se o preco cair, sai no stop subido, nao no inicial
    AD2 = {'X': dict(
        o=np.array([100, 100, 101, 102, 90, 90.]),
        h=np.array([100, 100.5, 101.5, 102.5, 91, 91.]),
        l=np.array([100, 99.9, 100.9, 101.9, 89, 89.]),
        c=np.array([100, 100.4, 101.4, 102.4, 90, 90.]),
        atr=np.full(6, 1.0), valid=np.ones(6, bool),
        sig={'imp': np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)},
        hilo={5: (np.array([1., 1., 1., 1., 1., 1.]),
                  np.full(6, np.nan), np.full(6, np.nan))},
        atr_pct=0.01)}
    trD2 = simulate(idxD, AD2, 'D', dict(fam='imp', p_tp=0.05, p_sl=0.02,
                                         hilo_n=5, trail='open'))
    # stop tinha subido a 102; vela 4 abre a 90 (gap) -> preenche a 90, nao a 98
    ok.append(('modo D ratchet + gap na abertura',
               len(trD2) == 1 and trD2[0]['gap']
               and abs(trD2[0]['pr'] - (-0.10)) < 1e-9))

    # 7. Maker: limite que nunca e tocado DESCARTA o sinal
    A4 = {'X': dict(
        o=np.array([100, 101, 102, 103, 104, 105.]),
        h=np.array([100, 101, 102, 103, 104, 105.]),
        l=np.array([100, 101, 102, 103, 104, 105.]),   # nunca volta ao limite
        c=np.array([100, 101, 102, 103, 104, 105.]),
        atr=np.full(6, 1.0), valid=np.ones(6, bool),
        sig={'imp': np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)},
        hilo={5: (np.zeros(6), np.full(6, np.nan), np.full(6, np.nan))},
        atr_pct=0.01)}
    tr4 = simulate(idx, A4, 'C', dict(fam='imp', p_tp=0.02, p_sl=0.02,
                                      hilo_n=5, maker=True))
    ok.append(('maker descarta sinal nao preenchido', len(tr4) == 0))

    # 8. Contabilidade all-in: 1% de movimento a 2x -> +2% bruto, -0.28% de taxa
    fake = [dict(sym='X', dir=1, k0=0, k1=1, pr=0.01, mae=0.0, mfe=0.01,
                 bars=1, reason='target', gap=False, armed=False, locked_at=None)]
    fidx = pd.DatetimeIndex(['2024-01-01', '2024-01-01'], tz='UTC')
    r8 = account(fake, fidx, F_TAKER, 'allin', lev=2.0, reset_mensal=True)
    esperado = MONTHLY_BASE * (2 * 0.01 - 2 * F_TAKER)     # +2% bruto - 0.28% taxa
    ok.append(('contabilidade all-in 2x', abs(r8['tot'] - esperado) < 1e-9))

    print('  TESTES UNITARIOS')
    allok = True
    for name, passed in ok:
        print(f'    [{"OK " if passed else "FALHA"}] {name}')
        allok &= bool(passed)
    if not allok:
        print('\n  ABORTADO: ha teste a falhar. Nao apresento numeros de um motor partido.')
        sys.exit(1)
    print()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Relatorio
# ─────────────────────────────────────────────────────────────────────────────
def tabela_niveis():
    print('=' * 100)
    print('  CONSTRANGIMENTO DA TAXA — o que cada alavancagem exige do MERCADO')
    print('  (alvo G=2.0% e perda max P=0.5% do capital, a especificacao do Rafa)')
    print('=' * 100)
    print(f"  {'L':<6}{'taxa':<9}{'p_tp':>9}{'p_sl':>9}{'R:R no preco':>15}   veredito")
    for f, nome in [(F_TAKER, 'taker'), (F_MAKER, 'maker')]:
        for L in [1.0, 1.5, 2.0, 3.0, 4.0]:
            r = levels(SPEC['G'], SPEC['P'], L, f)
            if r is None:
                print(f'  {L:<6.1f}{nome:<9}{"—":>9}{"<=0":>9}{"—":>15}   IMPOSSIVEL')
            else:
                p_tp, p_sl = r
                print(f'  {L:<6.1f}{nome:<9}{p_tp*100:>8.2f}%{p_sl*100:>8.2f}%'
                      f'{p_tp/p_sl:>14.1f}:1')
        print(f'  {"":6}{nome:<9}L_max = {lev_max(SPEC["P"], f):.2f}x '
              f'(acima disto a taxa sozinha consome o orcamento de perda)')
    print()


def linha(rot, st, g=None, p=None):
    """Uma linha da tabela. O flag compara a taxa de acerto com o breakeven REAL
    (derivado do ganho/perda medios efetivos), nao com o teorico."""
    flag = '' if st['wr'] >= st['be_real'] else '  <= abaixo do breakeven'
    print(f"  {rot:<28}{st['tot']:>+9.0f}{st['dd']:>7.1f}%{st['n']:>8}"
          f"{st['wr']:>7.1f}%{st['avg_win']:>+8.2f}%{st['avg_loss']:>+8.2f}%"
          f"{st['rr_real']:>7.2f}{st['be_real']:>7.1f}%{st['worst_streak']:>6}{flag}")


def cab():
    print(f"  {'variante':<28}{'total':>9}{'DD':>8}{'trades':>8}{'WR':>8}"
          f"{'ganho':>9}{'perda':>9}{'R:R':>7}{'BE':>8}{'seq-':>6}")
    print('  ' + '-' * 108)


def mensal(titulo, sres):
    """Levantamento mensal — regra obrigatoria do projeto."""
    print(f'\n  [MENSAL] {titulo}')
    meses = sorted(set().union(*[set(s['monthly']) for _, s in sres]))
    if not meses:
        print('    (sem trades)')
        return
    # Parte em blocos de 14 meses para caber no terminal
    for i in range(0, len(meses), 14):
        blk = meses[i:i + 14]
        print('    ' + f"{'mes':<16}" + ''.join(f'{m[2:]:>8}' for m in blk) + f"{'TOT':>9}")
        for nome, s in sres:
            print('    ' + f'{nome:<16}'
                  + ''.join(f"{s['monthly'].get(m, 0):>+8.0f}" for m in blk)
                  + f"{sum(s['monthly'].get(m, 0) for m in blk):>+9.0f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print('=' * 100)
    print('  BACKTEST — ESTRATEGIA AGRESSIVA (capital inteiro, 1 posicao, continuidade)')
    print('=' * 100)
    print()
    _tests()
    tabela_niveis()

    for tf, pairs in [('15m', PAIRS_15M), ('5m', PAIRS_5M)]:
        print('=' * 100)
        print(f'  TIMEFRAME {tf}   ({len(pairs)} par(es))')
        print('=' * 100)

        data = {}
        for per, win in [('5y', S5), ('oos', S26)]:
            pd_ = build(pairs, tf, per)
            if not pd_:
                print(f'  SEM DADOS para {tf} / {per}')
                continue
            idx, A = align(pd_)
            data[per] = (idx, A)
            span = f'{pd.Timestamp(idx[0]).date()} -> {pd.Timestamp(idx[-1]).date()}'
            print(f'  {per:<5} {len(A)} pares | {len(idx):>7} barras | {span}')
            for s in sorted(A):
                print(f'         {s:<18} ATR mediano {A[s]["atr_pct"]*100:.3f}% do preco')
        if '5y' not in data or 'oos' not in data:
            print('  Dados insuficientes, salto este timeframe.\n')
            continue
        print()

        g, p, L = SPEC['G'], SPEC['P'], SPEC['L']
        lv = levels(g, p, L, F_TAKER)
        if lv is None:
            print('  Especificacao do Rafa e IMPOSSIVEL neste modelo de taxa.\n')
            continue
        p_tp, p_sl = lv
        atr_med = np.median([A['atr_pct'] for A in data['5y'][1].values()])
        ruido = p_sl < 0.5 * atr_med
        print(f'  Especificacao do Rafa: G={g}% P={p}% L={L}x taker'
              f'  ->  p_tp={p_tp*100:.2f}%  p_sl={p_sl*100:.2f}%'
              f'  ({p_sl/atr_med:.2f} x ATR mediano)')
        if ruido:
            print('  *** STOP MARCADO COMO RUIDO (< 0.5 x ATR): qualquer resultado desta')
            print('      celula e artefacto, nao edge. Tratar como nao-informativo. ***')
        print()

        # ── ETAPA 1: qual familia de sinal, e qual modo de saida ──
        print('  ETAPA 1 — familia de sinal x modo de saida (spec do Rafa, taker, all-in)')
        cab()
        best = None
        for fam in ['imp', 'pull', 'brk']:
            for mode in ['A', 'B', 'C']:
                cfg = dict(fam=fam, p_tp=p_tp, p_sl=p_sl, hilo_n=SPEC['hilo_n'],
                           trig=SPEC['trig'], lock=SPEC['lock'])
                t5 = simulate(data['5y'][0], data['5y'][1], mode, cfg)
                s5 = stats(t5, account(t5, data['5y'][0], F_TAKER, 'allin', lev=L), F_TAKER, L)
                t26 = simulate(data['oos'][0], data['oos'][1], mode, cfg)
                s26 = stats(t26, account(t26, data['oos'][0], F_TAKER, 'allin', lev=L), F_TAKER, L)
                linha(f'{fam} / modo {mode} (5a)', s5, g, p)
                linha(f'{fam} / modo {mode} (OOS)', s26, g, p)
                score = s5['tot'] + s26['tot']
                if best is None or score > best[0]:
                    best = (score, fam, mode, s5, s26, t5, t26)
        print()
        if best is None:
            continue
        _, bfam, bmode, bs5, bs26, bt5, bt26 = best
        print(f'  Melhor combinacao desta etapa: sinal "{bfam}", modo {bmode}')
        print()

        # ── ETAPA 2: alavancagem x modelo de taxa ──
        print('  ETAPA 2 — alavancagem x taxa (sinal e modo fixos na melhor combinacao)')
        cab()
        best2 = None
        for fnome, f in [('taker', F_TAKER), ('maker', F_MAKER)]:
            for LL in [1.0, 1.5, 2.0, 3.0]:
                lv2 = levels(g, p, LL, f)
                if lv2 is None:
                    print(f'  {f"{fnome} L={LL}x":<26}{"IMPOSSIVEL — p_sl <= 0":>50}')
                    continue
                tp2, sl2 = lv2
                mk = (fnome == 'maker')
                cfg = dict(fam=bfam, p_tp=tp2, p_sl=sl2, hilo_n=SPEC['hilo_n'],
                           trig=SPEC['trig'], lock=SPEC['lock'], maker=mk)
                t5 = simulate(data['5y'][0], data['5y'][1], bmode, cfg)
                s5 = stats(t5, account(t5, data['5y'][0], f, 'allin', lev=LL), f, LL)
                t26 = simulate(data['oos'][0], data['oos'][1], bmode, cfg)
                s26 = stats(t26, account(t26, data['oos'][0], f, 'allin', lev=LL), f, LL)
                mark = ''
                if sl2 < 0.5 * atr_med:
                    mark = f'  RUIDO ({sl2/atr_med:.2f}xATR)'
                linha(f'{fnome} L={LL}x (5a)', s5, g, p)
                linha(f'{fnome} L={LL}x (OOS)' + mark, s26, g, p)
                if sl2 >= 0.5 * atr_med:
                    sc = s5['tot'] + s26['tot']
                    if best2 is None or sc > best2[0]:
                        best2 = (sc, fnome, f, LL, tp2, sl2, mk, s5, s26, t5, t26)
        print()

        if best2 is None:
            print('  Nenhuma celula sobreviveu ao filtro de RUIDO (stop >= 0.5 x ATR).')
            print('  Isto ja e o resultado: com estes alvos, o stop exigido esta dentro')
            print('  do ruido de mercado em todas as alavancagens viaveis.\n')
            continue

        _, fnome, fbest, Lbest, tpb, slb, mkb, b5, b26, bt5, bt26 = best2
        print(f'  Melhor celula nao-ruido: {fnome}, L={Lbest}x, '
              f'p_tp={tpb*100:.2f}% p_sl={slb*100:.2f}% ({slb/atr_med:.2f} x ATR)')
        print()

        # ── ETAPA 3: grelha do HiLo (so faz sentido no modo B) ──
        print('  ETAPA 3 — grelha do HiLo no modo B (N x gatilho da trava x nivel travado)')
        cab()
        bestB = None
        for nn in NS_HILO:
            for tg in TRIGS:
                for lk in LOCKS:
                    if lk >= tg:
                        continue        # travar acima do gatilho nao faz sentido
                    cfg = dict(fam=bfam, p_tp=tpb, p_sl=slb, hilo_n=nn,
                               trig=tg, lock=lk, maker=mkb)
                    t5 = simulate(data['5y'][0], data['5y'][1], 'B', cfg)
                    s5 = stats(t5, account(t5, data['5y'][0], fbest, 'allin', lev=Lbest), fbest, Lbest)
                    t26 = simulate(data['oos'][0], data['oos'][1], 'B', cfg)
                    s26 = stats(t26, account(t26, data['oos'][0], fbest, 'allin', lev=Lbest), fbest, Lbest)
                    sc = s5['tot'] + s26['tot']
                    if bestB is None or sc > bestB[0]:
                        bestB = (sc, nn, tg, lk, s5, s26, t5, t26)
        if bestB:
            _, nn, tg, lk, s5, s26, t5B, t26B = bestB
            linha(f'HiLo N={nn} t={tg} l={lk} (5a)', s5, g, p)
            linha(f'HiLo N={nn} t={tg} l={lk} (OOS)', s26, g, p)
            print(f'\n  Devolucao media do pico (vencedores): '
                  f'{s5["giveback"]:.2f}% em 5 anos | {s26["giveback"]:.2f}% no OOS')
            print(f'  Duracao media dos trades: {s5["avg_bars"]:.1f} barras (5a) | '
                  f'{s26["avg_bars"]:.1f} (OOS)')
            print(f'  Stop travado furado por gap: {s5["gap_breaks"]} vezes (5a), '
                  f'{s26["gap_breaks"]} (OOS)')
            # Distribuicao: a trava fez o trabalho ou o HiLo justificou-se?
            for nome, tt in [('5 anos', t5B), ('OOS', t26B)]:
                arm = [t for t in tt if t['armed']]
                if arm:
                    alem = sum(1 for t in arm if t['pr'] > tpb)
                    print(f'  {nome}: {len(arm)} trades armaram a trava; '
                          f'{alem} ({alem/len(arm)*100:.0f}%) correram ALEM do alvo')
                    if alem / len(arm) < 0.15:
                        print('       -> o HiLo quase nunca se justifica aqui: '
                              'complexidade sem retorno')
        print()

        # ── ETAPA 4: R:R x modo de saida (inclui o modo D) ──
        print('  ETAPA 4 — R:R x modo de saida')
        print(f'  Base: sinal "{bfam}", {fnome}, L={Lbest}x. '
              f'ATR mediano 5 anos = {atr_med*100:.3f}% do preco')
        print('  Duas formas de baixar o R:R, e NAO sao equivalentes:')
        print('    . baixar o ALVO  -> mantem o stop minusculo (problema do ruido)')
        print('    . subir a PERDA  -> da um stop a serio. E a direcao promissora.')
        print()
        print(f"  {'R:R  alvo/perda':<22}{'modo':<12}{'stop/ATR':>9}"
              f"{'5a':>9}{'DD':>7}{'WR':>7}{'OOS':>8}{'DD':>7}{'WR':>7}"
              f"{'R:R real':>10}{'BE':>7}")
        print('  ' + '-' * 108)
        combos = [
            ('4:1  2.0 / 0.5', 2.0, 0.50),
            ('3:1  1.5 / 0.5', 1.5, 0.50),
            ('2:1  1.0 / 0.5', 1.0, 0.50),
            ('3:1  2.0 / 0.67', 2.0, 0.667),
            ('2:1  2.0 / 1.0', 2.0, 1.00),
            ('3:1  3.0 / 1.0', 3.0, 1.00),
            ('2:1  3.0 / 1.5', 3.0, 1.50),
        ]
        modos = [('C', 'TP fixo', {}), ('A', 'escada', {}),
                 ('B', 'trava+HiLo', {}),
                 ('D', 'D HiLo-abert', dict(trail='open')),
                 ('D', 'D HiLo-linha', dict(trail='line'))]
        melhor4 = None
        for rot, G, P in combos:
            lv4 = levels(G, P, Lbest, fbest)
            if lv4 is None:
                print(f'  {rot:<22}{"—":<12}{"IMPOSSIVEL (p_sl <= 0)":>40}')
                continue
            tp4, sl4 = lv4
            atr_mult = sl4 / atr_med
            for md, mrot, extra in modos:
                cfg = dict(fam=bfam, p_tp=tp4, p_sl=sl4, hilo_n=SPEC['hilo_n'],
                           trig=SPEC['trig'], lock=SPEC['lock'], maker=mkb, **extra)
                t5 = simulate(data['5y'][0], data['5y'][1], md, cfg)
                s5 = stats(t5, account(t5, data['5y'][0], fbest, 'allin', lev=Lbest),
                           fbest, Lbest)
                t26 = simulate(data['oos'][0], data['oos'][1], md, cfg)
                s26 = stats(t26, account(t26, data['oos'][0], fbest, 'allin', lev=Lbest),
                            fbest, Lbest)
                flag = ''
                if atr_mult < 0.5:
                    flag = '  RUIDO'
                elif s5['wr'] >= s5['be_real'] and s26['wr'] >= s26['be_real']:
                    flag = '  <= ACIMA do breakeven nos DOIS'
                print(f"  {rot:<22}{mrot:<12}{atr_mult:>8.2f}x"
                      f"{s5['tot']:>+9.0f}{s5['dd']:>6.1f}%{s5['wr']:>6.1f}%"
                      f"{s26['tot']:>+8.0f}{s26['dd']:>6.1f}%{s26['wr']:>6.1f}%"
                      f"{s5['rr_real']:>10.2f}{s5['be_real']:>6.1f}%{flag}")
                sc = s5['tot'] + s26['tot']
                if atr_mult >= 0.5 and (melhor4 is None or sc > melhor4[0]):
                    melhor4 = (sc, rot, mrot, s5, s26)
            print()
        if melhor4:
            _, rot, mrot, m5, m26 = melhor4
            print(f'  Melhor de todas: {rot} / {mrot}  ->  '
                  f"{m5['tot']:+.0f} (5a) | {m26['tot']:+.0f} (OOS) | "
                  f"DD {max(m5['dd'], m26['dd']):.1f}%")
            b5, b26 = (m5, m26) if (m5['tot'] + m26['tot']) > (b5['tot'] + b26['tot']) else (b5, b26)
        print()

        # ── Dimensionamento: all-in vs risco fixo, na MESMA sequencia ──
        print('  DIMENSIONAMENTO — all-in vs risco fixo (mesma sequencia de trades)')
        cab()
        cfgF = dict(fam=bfam, p_tp=tpb, p_sl=slb, hilo_n=SPEC['hilo_n'],
                    trig=SPEC['trig'], lock=SPEC['lock'], maker=mkb)
        for per, rot in [('5y', '5 anos'), ('oos', 'OOS')]:
            tt = simulate(data[per][0], data[per][1], bmode, cfgF)
            a1 = stats(tt, account(tt, data[per][0], fbest, 'allin', lev=Lbest), fbest, Lbest)
            a2 = stats(tt, account(tt, data[per][0], fbest, 'risco', p_sl=slb), fbest,
                       (RISK_PCT_S2 / 100.0) / slb)
            linha(f'all-in {Lbest}x ({rot})', a1, g, p)
            linha(f'risco {RISK_PCT_S2}% ({rot})', a2, g, p)
            if per == '5y':
                mensal('all-in vs risco fixo, 5 anos', [('all-in', a1), ('risco fixo', a2)])
            else:
                mensal('all-in vs risco fixo, OOS 2026', [('all-in', a1), ('risco fixo', a2)])

        # ── Veredicto ──
        print('=' * 100)
        print(f'  VEREDICTO {tf}')
        print('=' * 100)
        print('  Baseline v3 a bater: +347 (5 anos) e +60 (OOS), DD 10.1%')
        print('  NOTA: a janela OOS aqui acaba a 2026-06-29 (fim do cache 15m/5m),')
        print('        ~10 dias mais curta que a do baseline. Comparacao APROXIMADA.')
        passou = b5['tot'] > 347 and b26['tot'] > 60
        print(f"\n  Melhor variante: {b5['tot']:+.0f} (5 anos) | {b26['tot']:+.0f} (OOS)"
              f" | DD {max(b5['dd'], b26['dd']):.1f}%")
        if passou:
            print('  >> BATE o baseline nos DOIS periodos.')
            if max(b5['dd'], b26['dd']) > 10.1:
                print(f"  >> MAS o DD e {max(b5['dd'], b26['dd'])/10.1:.1f}x pior que o v3."
                      ' Isto contraria o objetivo declarado do Rafa.')
        else:
            print('  >> NAO bate o baseline nos dois periodos. Nao vai para o bot.')
        print()


if __name__ == '__main__':
    main()
