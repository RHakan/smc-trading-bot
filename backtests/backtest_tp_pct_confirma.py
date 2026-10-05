"""
backtest_tp_pct_confirma.py
BATERIA DE CONFIRMACAO ao candidato TP 5% / SL 1.67% (R:R 3:1 em % do preco).

DE ONDE VEIO: backtest_tp_fixo_pct.py. Impondo alvo e stop em % do PRECO (em vez de
estrutura), a especificacao do Rafa fica positiva nos DOIS periodos a partir de TP 4%:
    TP 2% / SL 0.67%  ->  acerto 25.0% (precisa 30.2%)  |  -1077 / -48
    TP 3% / SL 1.00%  ->  acerto 26.3% (precisa 28.5%)  |   -247 / -11
    TP 4% / SL 1.33%  ->  acerto 28.7% (precisa 27.6%)  |   +151 / +47
    TP 5% / SL 1.67%  ->  acerto 30.5% (precisa 27.1%)  |   +193 / +35
E o PRIMEIRO candidato genuino de treze testes. NAO bate o baseline (+347/+60) mas e a
estrutura em que o Rafa acredita — e uma estrategia que o operador respeita pode render
mais que uma melhor que ele sabota (ele ja fechou posicoes a mao varias vezes).

QUATRO BLOCOS (tem de sobreviver aos QUATRO)
  1. PLANALTO      — TP de 3.5% a 7% em passos finos. Um pico isolado e serrilha.
  2. SUB-PERIODOS  — 5 anos partidos ao meio. So funcionar numa metade = regime.
  3. ANO A ANO     — o baseline faz 5/5. Quantos faz o candidato?
  4. MES A MES OOS — o total pode vir de um mes sortudo. Aqui ve-se.

Bot rodando NAO tocado.
Uso: python backtests/backtest_tp_pct_confirma.py
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
_p = importlib.util.spec_from_file_location(
    '_pct', str(Path(__file__).parent / 'backtest_tp_fixo_pct.py'))
P = importlib.util.module_from_spec(_p); _p.loader.exec_module(P)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

U = timezone.utc
S5 = (datetime(2021, 1, 1, tzinfo=U), datetime(2025, 10, 6, tzinfo=U))
S26 = (datetime(2026, 1, 1, tzinfo=U), datetime(2026, 7, 9, tzinfo=U))
H1 = (datetime(2021, 1, 1, tzinfo=U), datetime(2023, 6, 30, 23, tzinfo=U))
H2 = (datetime(2023, 7, 1, tzinfo=U), datetime(2025, 10, 6, tzinfo=U))

FAIXA = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
FOCO = [4.0, 4.5, 5.0, 5.5, 6.0]
RATIO = 3.0


def main():
    print('\n' + '=' * 100)
    print('  BATERIA DE CONFIRMACAO — TP em % do preco, SL = TP/3')
    print('=' * 100)
    print('  A carregar...')
    p5 = B.load(B.fetch_5y); p26 = B.load(B.fetch_26)
    a5 = B.prep(*p5); a26 = B.prep(*p26)
    b5 = B.summ(B.run(a5, *S5)); b26 = B.summ(B.run(a26, *S26))
    h1b = B.summ(B.run(a5, *H1)); h2b = B.summ(B.run(a5, *H2))
    print(f"  BASELINE: {b5['tot']:+.0f} / {b26['tot']:+.0f} | DD {b5['dd']:.1f}% | "
          f"{b5['anos']}/5 anos | acerto {b5['wr']:.1f}%")
    if abs(b5['tot'] - 347) > 1:
        print('  ABORTADO: motor nao reproduz o baseline.')
        sys.exit(1)
    print()

    cache = {}

    def cel(tp):
        if tp not in cache:
            x5 = P.prep_pct(*p5, tp, RATIO)
            x26 = P.prep_pct(*p26, tp, RATIO)
            cache[tp] = (x5, x26, B.summ(B.run(x5, *S5)), B.summ(B.run(x26, *S26)))
        return cache[tp]

    # BLOCO 1 -----------------------------------------------------------------
    print('=' * 100)
    print('  BLOCO 1 — PLANALTO (TP em passos de 0.5%). Os vizinhos acompanham?')
    print('=' * 100)
    print(f"  {'TP':>6}{'SL':>7}{'5 anos':>9}{'OOS':>8}{'DD':>7}{'anos+':>7}"
          f"{'acerto':>9}{'precisa':>9}")
    print('  ' + '-' * 62)
    vals = []
    for tp in FAIXA:
        _, _, s5, s26 = cel(tp)
        be = P.be_wr(tp, tp / RATIO)
        vals.append(s5['tot'])
        mk = '  <-- candidato' if tp == 5.0 else ''
        print(f"  {tp:>5.1f}%{tp/RATIO:>6.2f}%{s5['tot']:>+9.0f}{s26['tot']:>+8.0f}"
              f"{s5['dd']:>6.1f}%{s5['anos']:>5}/5{s5['wr']:>8.1f}%{be:>8.1f}%{mk}")
    v = np.array(vals)
    pos = int((v > 0).sum())
    print(f"\n  {pos}/{len(v)} celulas positivas | amplitude {v.min():+.0f} a {v.max():+.0f}")
    if pos >= len(v) - 1:
        print('  >> PLANATO — a regiao inteira funciona, nao depende do valor exato')
    else:
        print('  >> IRREGULAR — o resultado depende do valor exato escolhido')
    print()

    # BLOCO 2 -----------------------------------------------------------------
    print('=' * 100)
    print('  BLOCO 2 — SUB-PERIODOS (5 anos partidos ao meio)')
    print('=' * 100)
    print(f"  {'TP':>6}{'H1 21-23':>11}{'H2 23-25':>11}{'as duas positivas?':>21}")
    print('  ' + '-' * 50)
    print(f"  {'base':>6}{h1b['tot']:>+11.0f}{h2b['tot']:>+11.0f}{'—':>21}")
    for tp in FOCO:
        x5, _, _, _ = cel(tp)
        h1 = B.summ(B.run(x5, *H1))
        h2 = B.summ(B.run(x5, *H2))
        ok = 'SIM' if (h1['tot'] > 0 and h2['tot'] > 0) else 'nao'
        print(f"  {tp:>5.1f}%{h1['tot']:>+11.0f}{h2['tot']:>+11.0f}{ok:>21}")
    print()

    # BLOCO 3 -----------------------------------------------------------------
    print('=' * 100)
    print('  BLOCO 3 — ANO A ANO (consistencia: o criterio declarado do Rafa)')
    print('=' * 100)
    anos = sorted(b5['yr'])
    print(f"  {'TP':>6}" + ''.join(f'{a:>9}' for a in anos) + f"{'anos+':>8}")
    print('  ' + '-' * (6 + 9 * len(anos) + 8))
    print(f"  {'base':>6}" + ''.join(f"{b5['yr'].get(a, 0):>+9.0f}" for a in anos)
          + f"{b5['anos']:>6}/5")
    for tp in FOCO:
        _, _, s5, _ = cel(tp)
        print(f"  {tp:>5.1f}%" + ''.join(f"{s5['yr'].get(a, 0):>+9.0f}" for a in anos)
              + f"{s5['anos']:>6}/5")
    print()

    # BLOCO 4 -----------------------------------------------------------------
    print('=' * 100)
    print('  BLOCO 4 — MES A MES NO OOS 2026 (o total vem de um mes sortudo?)')
    print('=' * 100)
    meses = sorted(b26['monthly'])
    print(f"  {'TP':>6}" + ''.join(f'{m[5:]:>8}' for m in meses)
          + f"{'TOT':>9}{'meses+':>9}")
    print('  ' + '-' * (6 + 8 * len(meses) + 18))
    mb = sum(1 for m in meses if b26['monthly'].get(m, 0) > 0)
    print(f"  {'base':>6}" + ''.join(f"{b26['monthly'].get(m, 0):>+8.0f}" for m in meses)
          + f"{b26['tot']:>+9.0f}{mb:>7}/{len(meses)}")
    for tp in FOCO:
        _, _, _, s26 = cel(tp)
        mp = sum(1 for m in meses if s26['monthly'].get(m, 0) > 0)
        print(f"  {tp:>5.1f}%" + ''.join(f"{s26['monthly'].get(m, 0):>+8.0f}" for m in meses)
              + f"{s26['tot']:>+9.0f}{mp:>7}/{len(meses)}")
    print()

    # VEREDICTO ---------------------------------------------------------------
    print('=' * 100)
    print('  VEREDICTO')
    print('=' * 100)
    aprov = []
    for tp in FOCO:
        x5, _, s5, s26 = cel(tp)
        h1 = B.summ(B.run(x5, *H1))
        h2 = B.summ(B.run(x5, *H2))
        ch = {
            'positivo nos 2 periodos': s5['tot'] > 0 and s26['tot'] > 0,
            'positivo nas 2 metades': h1['tot'] > 0 and h2['tot'] > 0,
            'pelo menos 3/5 anos': s5['anos'] >= 3,
            'acerto acima do breakeven': s5['wr'] > P.be_wr(tp, tp / RATIO),
        }
        falha = [k for k, val in ch.items() if not val]
        estado = 'APROVADO' if not falha else 'reprovado'
        extra = f"  — falha: {'; '.join(falha)}" if falha else ''
        print(f"  TP {tp:.1f}%: {estado}{extra}")
        if not falha:
            aprov.append(tp)
    print()
    if aprov:
        print(f'  >> {len(aprov)} celula(s) sobrevivem aos 4 blocos: TP {aprov}')
        print('     NOTA: nenhuma bate o baseline (+347/+60). Sao uma ALTERNATIVA com')
        print('     R:R 3:1, nao uma melhoria de retorno. A escolha e do Rafa.')
    else:
        print('  >> NENHUMA sobrevive aos 4 blocos.')
    print()


if __name__ == '__main__':
    main()
