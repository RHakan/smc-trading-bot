"""
backtest_stop_atr_confirma.py
BATERIA DE CONFIRMACAO ao stop da lateral em multiplos de ATR (faixa 1.5-2.5).

De onde veio: em backtest_stop_ema200.py, a ideia do Rafa (stop junto da EMA200)
REPROVOU (+166/-14, DD 36%), mas os stops em ATR mostraram um sinal consistente no
FORA-DA-AMOSTRA: +90 a +122 contra os +60 do baseline, em toda a faixa 0.75-3.0xATR.
Ao mesmo tempo o in-sample era ruidoso (1.5x=+282, 1.75x=+365, 2.0x=+338) e a
consistencia anual piorava (5/5 anos no baseline vs 3-4/5 nos ATR).

Sinal real ou artefacto? E o que esta bateria decide. Mesma disciplina que aprovou o
MAX_PER_SIDE=2 e reprovou o step-trail, o R:R forcado e o SuperTrend/HMA.

QUATRO BLOCOS (um candidato tem de sobreviver aos QUATRO):
  1. PLANALTO      — passos finos de 0.05x. Vizinhos tem de acompanhar; um pico
                     isolado entre vizinhos maus e serrilha, nao edge.
  2. SUB-PERIODOS  — 5 anos partidos ao meio. Se so funciona numa metade, e regime,
                     nao vantagem.
  3. ANO A ANO     — quantos anos positivos. O baseline faz 5/5. O Rafa disse
                     explicitamente que prefere acertar mais vezes a acreditar numa
                     teoria: um candidato que ganhe mais mas falhe anos inteiros
                     NAO serve, por melhor que seja o total.
  4. MES A MES OOS — o total do OOS pode vir de 1 mes sortudo. Aqui ve-se.

O alvo NAO muda (altura do range). Bull/bear no baseline — efeito isolado na lateral.
Portao: bater +347 (5 anos) E +60 (OOS 2026) sem piorar o DD de 10.1%.
Bot rodando NAO tocado.
Uso: python backtests/backtest_stop_atr_confirma.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

_s = importlib.util.spec_from_file_location(
    '_ema', str(Path(__file__).parent / 'backtest_stop_ema200.py'))
E = importlib.util.module_from_spec(_s); _s.loader.exec_module(E)
B = E.B                                    # motor v3 validado (reproduz +347/+60)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

U = timezone.utc
S5 = (datetime(2021, 1, 1, tzinfo=U), datetime(2025, 10, 6, tzinfo=U))
S26 = (datetime(2026, 1, 1, tzinfo=U), datetime(2026, 7, 9, tzinfo=U))
H1 = (datetime(2021, 1, 1, tzinfo=U), datetime(2023, 6, 30, 23, tzinfo=U))
H2 = (datetime(2023, 7, 1, tzinfo=U), datetime(2025, 10, 6, tzinfo=U))

FAIXA = [1.50, 1.60, 1.70, 1.75, 1.80, 1.90, 2.00, 2.10, 2.25, 2.40, 2.50]
FOCO = [1.50, 1.75, 2.00, 2.25, 2.50]      # os que passam pelos blocos 2-4

BASE5, BASE26, BASE_DD = 347, 60, 10.1


def corre(a5, a26, janela=None):
    s5 = B.summ(B.run(a5, *(janela or S5)))
    s26 = B.summ(B.run(a26, *S26))
    return s5, s26


def main():
    print('\n' + '=' * 100)
    print('  BATERIA DE CONFIRMACAO — stop da lateral em N x ATR (faixa 1.5-2.5)')
    print('=' * 100)
    print('  A carregar dados...')
    p5 = B.load(B.fetch_5y); p26 = B.load(B.fetch_26)

    # Pre-computa os arrays uma vez por modo (o caro e o gen(), nao o run())
    cache = {}
    def arrs(modo):
        if modo not in cache:
            cache[modo] = (E.prep_stop(*p5, modo), E.prep_stop(*p26, modo))
        return cache[modo]

    a5b, a26b = arrs('range')
    b5 = B.summ(B.run(a5b, *S5)); b26 = B.summ(B.run(a26b, *S26))
    print(f"  BASELINE: {b5['tot']:+.0f} (5 anos) | {b26['tot']:+.0f} (OOS) | "
          f"DD {b5['dd']:.1f}% | {b5['anos']}/5 anos+\n")
    if abs(b5['tot'] - BASE5) > 1:
        print('  ABORTADO: o motor nao reproduz o baseline.'); sys.exit(1)

    # ── BLOCO 1: PLANALTO ────────────────────────────────────────────────────
    print('=' * 100)
    print('  BLOCO 1 — PLANALTO (passos de 0.05-0.15x). Vizinhos acompanham?')
    print('=' * 100)
    print(f"  {'stop':<12}{'5 anos':>9}{'OOS':>8}{'DD':>7}{'anos+':>7}{'WR lat':>9}{'R:R lat':>9}")
    print('  ' + '-' * 62)
    print(f"  {'range (hoje)':<12}{b5['tot']:>+9.0f}{b26['tot']:>+8.0f}{b5['dd']:>6.1f}%"
          f"{b5['anos']:>5}/5{E.perfil(b5)['wr']:>8.1f}%{E.perfil(b5)['rr']:>9.2f}")
    plan = {}
    for k in FAIXA:
        a5, a26 = arrs(f'atr{k}')
        s5, s26 = corre(a5, a26)
        plan[k] = (s5, s26)
        pf = E.perfil(s5)
        m = ''
        if s5['tot'] > BASE5 and s26['tot'] > BASE26:
            m = '  <= bate nos 2'
        print(f"  {str(k) + 'xATR':<12}{s5['tot']:>+9.0f}{s26['tot']:>+8.0f}{s5['dd']:>6.1f}%"
              f"{s5['anos']:>5}/5{pf['wr']:>8.1f}%{pf['rr']:>9.2f}{m}")
    v = np.array([plan[k][0]['tot'] for k in FAIXA])
    trocas = int(np.sum(np.diff(np.sign(v - BASE5)) != 0))
    print(f"\n  amplitude 5 anos na faixa: {v.min():+.0f} a {v.max():+.0f} "
          f"(oscila {v.max() - v.min():.0f})")
    print(f"  vezes que cruza o baseline dentro da faixa: {trocas}")
    print(f"  {'>> PLANALTO' if trocas <= 1 else '>> SERRILHA — o resultado depende do valor exato escolhido'}")
    print()

    # ── BLOCO 2: SUB-PERIODOS ────────────────────────────────────────────────
    print('=' * 100)
    print('  BLOCO 2 — SUB-PERIODOS (5 anos partidos ao meio)')
    print('=' * 100)
    h1b = B.summ(B.run(a5b, *H1)); h2b = B.summ(B.run(a5b, *H2))
    print(f"  {'stop':<12}{'H1 21-23':>11}{'H2 23-25':>11}{'as duas?':>11}")
    print('  ' + '-' * 45)
    print(f"  {'range':<12}{h1b['tot']:>+11.0f}{h2b['tot']:>+11.0f}{'—':>11}")
    for k in FOCO:
        a5, _ = arrs(f'atr{k}')
        h1 = B.summ(B.run(a5, *H1)); h2 = B.summ(B.run(a5, *H2))
        ok = h1['tot'] > h1b['tot'] and h2['tot'] > h2b['tot']
        print(f"  {str(k) + 'xATR':<12}{h1['tot']:>+11.0f}{h2['tot']:>+11.0f}"
              f"{'SIM' if ok else 'nao':>11}")
    print()

    # ── BLOCO 3: ANO A ANO ───────────────────────────────────────────────────
    print('=' * 100)
    print('  BLOCO 3 — ANO A ANO (o criterio declarado do Rafa: consistencia)')
    print('=' * 100)
    anos = sorted(b5['yr'])
    print(f"  {'stop':<12}" + ''.join(f'{a:>9}' for a in anos) + f"{'anos+':>8}")
    print('  ' + '-' * (12 + 9 * len(anos) + 8))
    print(f"  {'range':<12}" + ''.join(f"{b5['yr'].get(a, 0):>+9.0f}" for a in anos)
          + f"{b5['anos']:>6}/5")
    for k in FOCO:
        s5 = plan[k][0]
        neg = [a for a in anos if s5['yr'].get(a, 0) <= 0]
        print(f"  {str(k) + 'xATR':<12}" + ''.join(f"{s5['yr'].get(a, 0):>+9.0f}" for a in anos)
              + f"{s5['anos']:>6}/5" + (f"   negativo em {','.join(neg)}" if neg else ''))
    print()

    # ── BLOCO 4: MES A MES NO OOS ────────────────────────────────────────────
    print('=' * 100)
    print('  BLOCO 4 — MES A MES NO OOS 2026 (o total vem de 1 mes sortudo?)')
    print('=' * 100)
    meses = sorted(b26['monthly'])
    print(f"  {'stop':<12}" + ''.join(f'{m[5:]:>8}' for m in meses)
          + f"{'TOT':>9}{'meses+':>9}")
    print('  ' + '-' * (12 + 8 * len(meses) + 18))
    mb = sum(1 for m in meses if b26['monthly'].get(m, 0) > 0)
    print(f"  {'range':<12}" + ''.join(f"{b26['monthly'].get(m, 0):>+8.0f}" for m in meses)
          + f"{b26['tot']:>+9.0f}{mb:>7}/{len(meses)}")
    for k in FOCO:
        s26 = plan[k][1]
        mp = sum(1 for m in meses if s26['monthly'].get(m, 0) > 0)
        print(f"  {str(k) + 'xATR':<12}" + ''.join(f"{s26['monthly'].get(m, 0):>+8.0f}" for m in meses)
              + f"{s26['tot']:>+9.0f}{mp:>7}/{len(meses)}")
    print()

    # ── VEREDICTO ────────────────────────────────────────────────────────────
    print('=' * 100)
    print('  VEREDICTO')
    print('=' * 100)
    print(f'  Um candidato tem de: bater +{BASE5}/+{BASE26} nos dois periodos, nao piorar')
    print(f'  o DD de {BASE_DD}%, ganhar nas DUAS metades, e manter 5/5 anos positivos.\n')
    aprovados = []
    for k in FOCO:
        s5, s26 = plan[k]
        a5, _ = arrs(f'atr{k}')
        h1 = B.summ(B.run(a5, *H1)); h2 = B.summ(B.run(a5, *H2))
        checks = {
            'bate nos 2 periodos': s5['tot'] > BASE5 and s26['tot'] > BASE26,
            'DD nao piora': max(s5['dd'], s26['dd']) <= BASE_DD + 0.3,
            'ganha nas 2 metades': h1['tot'] > h1b['tot'] and h2['tot'] > h2b['tot'],
            '5/5 anos positivos': s5['anos'] == 5,
        }
        ok = all(checks.values())
        falha = [n for n, v in checks.items() if not v]
        print(f"  {k}xATR: {'APROVADO' if ok else 'reprovado'}"
              + (f"  — falha: {'; '.join(falha)}" if falha else ''))
        if ok:
            aprovados.append(k)
    print()
    if aprovados:
        print(f'  >> {len(aprovados)} candidato(s) sobreviveram aos 4 blocos: {aprovados}')
        print('     Só agora vale a pena discutir implementacao.')
    else:
        print('  >> NENHUM candidato sobrevive aos 4 blocos.')
        print('     O sinal no fora-da-amostra era real mas nao e robusto: o stop no lado')
        print('     oposto do range fica como esta. Nada muda no bot.')
    print()


if __name__ == '__main__':
    main()
