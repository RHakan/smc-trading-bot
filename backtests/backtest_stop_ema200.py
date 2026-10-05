"""
backtest_stop_ema200.py
IDEIA DO RAFA (07/08/2026, a partir do trade ETH/USDC ao vivo): o stop da lateral
fica no lado OPOSTO do range, muito longe (R:R 0.74 — "abrimos em desvantagem").
Ele colocaria o stop junto da EMA200 do 1H, onde o poria à mão.

O QUE JA SABEMOS (medido antes de mexer):
  A lateral quase nunca paga o stop — 18 de 409 trades (4%). A invalidacao apanha
  208 (51%) e corta-os a -0.29R. O R:R EFETIVO e 0.67/0.29 = 2.3:1, nao 0.74.
  Logo, apertar o stop nao ataca uma perda que esteja a acontecer — ataca a
  ESTRUTURA: transforma saidas de invalidacao (-0.29R) em stop-outs (-1R), em troca
  de um alvo que passa a valer muito mais em R (0.86R -> ~4R).
  Menos vitorias, cada uma a valer muito mais. O saldo disso e o que este teste mede.

DESENHO EXPERIMENTAL — separar "EMA200" de "apertado":
  Se um stop em N x ATR der o mesmo que a EMA200, entao a EMA200 nao tem nada de
  especial: o que conta e a distancia. Por isso testam-se os dois eixos.

  range        : lado oposto do range (BASELINE, o bot de hoje)
  ema200       : EMA200(1H) -/+ buffer x ATR; se estiver do lado errado, cai no range
  ema200_perto : o MAIS PERTO entre EMA200 e range (nunca alarga o stop)
  atrK         : close -/+ K x ATR, com K em 1.0/1.5/2.0/3.0 (controlo)

O ALVO NAO MUDA (fica a altura do range projetada, como no print do Rafa) — logo o
R:R sobe sozinho quando o stop aperta.

Aplicado SO a lateral (e onde vive o exemplo dele); bull/bear ficam no baseline,
para o efeito ficar isolado.

Portao: bater +347 (5 anos) E +60 (OOS 2026), sem piorar o DD de 10.1%.
Qualquer vencedor passa pelo TESTE DE PLANALTO (vizinhos) antes de valer alguma coisa.
Sinais e resto do motor IDENTICOS ao bot. 500/mes reset, custos 0.14% RT.
Levantamento mensal no fim. Bot rodando NAO tocado.
Uso: python backtests/backtest_stop_ema200.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# Reaproveita TODO o motor ja validado (reproduz +347/+60 exatamente).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    '_base', str(Path(__file__).parent / 'backtest_v3_trail_alvo.py'))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EMA_STOP = 200          # a media que o Rafa usa no grafico
BUF_ATR = 0.10          # folga abaixo/acima da media, em ATR
ATR_KS = [1.0, 1.5, 2.0, 3.0]

S5 = (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2025, 10, 6, tzinfo=timezone.utc))
S26 = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 9, tzinfo=timezone.utc))


def gen_stop(df, btc_local, modo):
    """Igual a B.gen(), mas o STOP da lateral segue `modo`. Tudo o resto intacto."""
    n = len(df)
    o = df['o'].values; h = df['h'].values; l = df['l'].values; c = df['c'].values
    atr = df['atr'].values; atr_avg = df['atr_avg'].values; adx_a = df['adx'].values
    reg = df['regime'].values; slope = df['slope_d'].values; ema_t = df['ema_trend'].values
    ema_s = df['ema_stop'].values
    side = np.array([None] * n, dtype=object)
    entry = np.full(n, np.nan); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    strat = np.array([None] * n, dtype=object); blevel = np.full(n, np.nan)
    start_i = max(B.B_ATR_AVG + B.BSEL_LB + 5,
                  B.U_SWING_N + B.U_CHOCH_REF + B.U_CHOCH_BARS + 5,
                  B.L_LOOKBACK + 5, EMA_STOP + 10, 210)

    for i in range(start_i, n):
        r = reg[i]; a = atr[i]
        if np.isnan(a) or a <= 0: continue

        if r == 'NEUTRAL':
            if np.isnan(adx_a[i - 1]) or adx_a[i - 1] > B.L_ADX_MAX: continue
            rl = np.min(l[i - B.L_LOOKBACK:i]); rh = np.max(h[i - B.L_LOOKBACK:i])
            if rl <= 0 or rh <= rl: continue
            buf = B.L_BUFFER * a; cl = c[i]; pc = c[i - 1]; height = rh - rl
            em = ema_s[i]

            if cl > rh + buf and pc <= rh and cl > rl:
                stop_range = rl
                stop = _escolhe(modo, 'LONG', cl, stop_range, em, a)
                if stop is None or stop >= cl: continue
                side[i] = 'LONG'; entry[i] = cl; sl[i] = stop
                tp[i] = cl + height           # alvo INALTERADO
                strat[i] = 'lat'; blevel[i] = rh
            elif cl < rl - buf and pc >= rl and rh > cl:
                stop_range = rh
                stop = _escolhe(modo, 'SHORT', cl, stop_range, em, a)
                if stop is None or stop <= cl: continue
                side[i] = 'SHORT'; entry[i] = cl; sl[i] = stop
                tp[i] = cl - height
                strat[i] = 'lat'; blevel[i] = rl

        elif r == 'BULL':
            if not btc_local[i]: continue
            if np.isnan(slope[i]) or slope[i] < B.U_SLOPE_MIN: continue
            if np.isnan(ema_t[i]) or c[i] <= ema_t[i]: continue
            av = atr_avg[i]
            if not np.isnan(av) and a < av: continue
            best = None
            for j in range(i - 1, max(i - B.U_CHOCH_BARS - 1,
                                      B.U_SWING_N + B.U_CHOCH_REF) - 1, -1):
                swing_low = np.min(l[j - B.U_SWING_N:j])
                if not (l[j] < swing_low and c[j] > swing_low): continue
                if B.U_MIN_SWEEP > 0 and (swing_low - l[j]) / swing_low * 100 < B.U_MIN_SWEEP: continue
                ref_high = np.max(h[j - B.U_CHOCH_REF:j])
                if np.any(c[j + 1:i] > ref_high): continue
                if c[i] > ref_high:
                    al = np.min(l[j:i + 1]); risk = c[i] - al
                    if risk > 0 and risk / c[i] <= 0.10: best = (c[i], al); break
            if best:
                e, stop = best; risk = e - stop
                side[i] = 'LONG'; entry[i] = e; sl[i] = stop
                tp[i] = e + B.U_RR * risk; strat[i] = 'bull'

        elif r == 'BEAR':
            cl = c[i]; op = o[i]; av = atr_avg[i]
            if np.isnan(av): continue
            low_n = np.min(l[i - B.BSEL_LB:i])
            if not (cl < low_n and cl < op and a > av): continue
            if np.isnan(adx_a[i - 1]) or adx_a[i - 1] > B.BSEL_ADX: continue
            stop = np.max(h[i - B.BSEL_STRUCT:i + 1]) + 0.1 * a; risk = stop - cl
            if risk > 0:
                side[i] = 'SHORT'; entry[i] = cl; sl[i] = stop
                tp[i] = cl - B.B_RR * risk; strat[i] = 'bear'

    return side, entry, sl, tp, strat, blevel


def _escolhe(modo, lado, close, stop_range, ema, a):
    """Devolve o preço do stop para o modo pedido, ou None se inviável."""
    if modo == 'range':
        return stop_range
    if modo.startswith('atr'):
        k = float(modo[3:])
        return close - k * a if lado == 'LONG' else close + k * a
    # modos baseados na EMA200
    if np.isnan(ema):
        return stop_range
    cand = ema - BUF_ATR * a if lado == 'LONG' else ema + BUF_ATR * a
    # EMA do lado errado (acima da entrada num LONG) => inutil como stop
    if (lado == 'LONG' and cand >= close) or (lado == 'SHORT' and cand <= close):
        return stop_range
    if modo == 'ema200':
        return cand
    if modo == 'ema200_perto':
        # nunca ALARGA o stop face ao range
        return max(cand, stop_range) if lado == 'LONG' else min(cand, stop_range)
    raise ValueError(modo)


def prep_stop(pdata, btcb, modo):
    arrs = {}
    for sym, df in pdata.items():
        d = df.copy()
        d['ema_stop'] = d['c'].ewm(span=EMA_STOP, adjust=False).mean()
        s, e, sl, tp, st, bl = gen_stop(d, btcb.reindex(d.index).fillna(False).values, modo)
        arrs[sym] = d.assign(_s=s, _e=e, _sl=sl, _tp=tp, _st=st, _bl=bl)
    return arrs


def perfil(res):
    """Estatística só da lateral — é onde a mudança atua."""
    t = [x for x in res['trades'] if x['strat'] == 'lat']
    if not t:
        return dict(n=0, wr=0, gw=0, gl=0, rr=0, stop_pct=0, inval_pct=0, alvo_pct=0)
    nr = np.array([x['nr'] for x in t])
    w = nr > 0.03; l = nr < -0.03
    import collections
    rz = collections.Counter(x['reason'] for x in t)
    n = len(t)
    return dict(n=n, wr=w.mean() * 100,
                gw=nr[w].mean() if w.any() else 0.0,
                gl=nr[l].mean() if l.any() else 0.0,
                rr=float(np.mean([x['path'] for x in t])),
                stop_pct=rz['stop'] / n * 100, inval_pct=rz['inval'] / n * 100,
                alvo_pct=rz['alvo'] / n * 100)


def main():
    print('\n' + '=' * 104)
    print('  STOP DA LATERAL — lado oposto do range (hoje) vs EMA200 vs ATR')
    print('=' * 104)
    print('  Alvo NAO muda (altura do range). O R:R sobe sozinho quando o stop aperta.')
    print('  Bull/bear ficam no baseline — o efeito fica isolado na lateral.\n')
    print('  A carregar dados...')
    p5 = B.load(B.fetch_5y); p26 = B.load(B.fetch_26)

    modos = ['range', 'ema200', 'ema200_perto'] + [f'atr{k}' for k in ATR_KS]
    print(f"  {'modo':<14}{'5 anos':>9}{'DD':>7}{'anos+':>7}{'OOS':>8}{'DD':>7}"
          f"{'|':>3}{'n lat':>7}{'R:R':>6}{'WR':>7}{'ganho':>8}{'perda':>8}"
          f"{'%stop':>7}{'%inval':>8}{'%alvo':>7}")
    print('  ' + '-' * 102)

    base = None; best = None
    for modo in modos:
        a5 = prep_stop(*p5, modo); a26 = prep_stop(*p26, modo)
        s5 = B.summ(B.run(a5, *S5)); s26 = B.summ(B.run(a26, *S26))
        pf = perfil(s5)
        if modo == 'range':
            base = (s5, s26)
        mark = ''
        if s5['tot'] > 347 and s26['tot'] > 60:
            mark = '  <= BATE nos 2'
        print(f"  {modo:<14}{s5['tot']:>+9.0f}{s5['dd']:>6.1f}%{s5['anos']:>5}/5"
              f"{s26['tot']:>+8.0f}{s26['dd']:>6.1f}%{'|':>3}{pf['n']:>7}"
              f"{pf['rr']:>6.2f}{pf['wr']:>6.1f}%{pf['gw']:>+8.2f}{pf['gl']:>+8.2f}"
              f"{pf['stop_pct']:>6.0f}%{pf['inval_pct']:>7.0f}%{pf['alvo_pct']:>6.0f}%{mark}")
        sc = s5['tot'] + s26['tot']
        if modo != 'range' and (best is None or sc > best[0]):
            best = (sc, modo, s5, s26)

    print()
    b5, b26 = base
    print(f"  BASELINE (range): {b5['tot']:+.0f} (5 anos) | {b26['tot']:+.0f} (OOS) | DD {b5['dd']:.1f}%")
    if best:
        _, modo, s5, s26 = best
        print(f"  MELHOR alternativa: '{modo}' -> {s5['tot']:+.0f} | {s26['tot']:+.0f} "
              f"| DD {max(s5['dd'], s26['dd']):.1f}%")
        passa = s5['tot'] > 347 and s26['tot'] > 60
        print(f"  {'>> PASSA o portao — ver planalto abaixo.' if passa else '>> NAO passa o portao.'}")
    print()

    # ── Teste de planalto no eixo da distância (buffer da EMA e K do ATR) ──
    print('=' * 104)
    print('  TESTE DE PLANALTO — a vizinhanca confirma, ou e serrilha?')
    print('=' * 104)
    print(f"  {'variante':<18}{'5 anos':>9}{'OOS':>8}{'DD':>7}{'n lat':>8}{'WR':>7}")
    print('  ' + '-' * 60)
    for k in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]:
        a5 = prep_stop(*p5, f'atr{k}'); a26 = prep_stop(*p26, f'atr{k}')
        s5 = B.summ(B.run(a5, *S5)); s26 = B.summ(B.run(a26, *S26))
        pf = perfil(s5)
        print(f"  {'stop ' + str(k) + 'xATR':<18}{s5['tot']:>+9.0f}{s26['tot']:>+8.0f}"
              f"{s5['dd']:>6.1f}%{pf['n']:>8}{pf['wr']:>6.1f}%")
    print()

    # ── Levantamento mensal (regra do projeto) ──
    if best:
        _, modo, s5, s26 = best
        B.mensal(f"baseline vs '{modo}' — 5 anos",
                 [('baseline', b5), (modo, s5)])
        B.mensal(f"baseline vs '{modo}' — OOS 2026",
                 [('baseline', b26), (modo, s26)])


if __name__ == '__main__':
    main()
