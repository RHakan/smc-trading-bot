"""
backtest_sem_bull.py
DESLIGAR DIRECOES — quanto vale cada uma das 3, e o que acontece sem a bull_smc.

PORQUE (dados que motivaram):
  ao vivo    : bull_smc -545 USDT em 7 trades (14.3% acerto, -0.70R medio) — a pior
               de longe. bear_breakout -292 em 11; lateral -240 em 31.
  backtest   : bull_smc e a MELHOR das tres em 5 anos (+196 vs +93 lat, +69 bear)
  OOS 2026   : bull_smc e a UNICA negativa (-9), com lat +45 e bear +27
  A melhor no historico e a unica negativa fora da amostra e a pior ao vivo. E o
  padrao de uma estrategia sobreajustada. Alem disso paga o stop inteiro em 80% dos
  casos (perda media -0.97R) e nao tem rede de invalidacao, ao contrario da lateral.

  RESSALVA registada: os 7 trades ao vivo sao TODOS da era contaminada (antes do #61).
  Na era limpa a bull_smc nao abriu nenhum. Portanto o julgamento ao vivo nao e limpo —
  quem decide aqui e o backtest + OOS.

O TESTE: desligar cada direcao a vez (o regime correspondente fica FORA do mercado, como
no sistema v1) e medir o que sobra. Depois bateria de confirmacao no melhor candidato.
Sinais e gestao das restantes ficam IDENTICOS. Bot rodando NAO tocado.
Uso: python backtests/backtest_sem_bull.py
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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

U = timezone.utc
S5 = (datetime(2021, 1, 1, tzinfo=U), datetime(2025, 10, 6, tzinfo=U))
S26 = (datetime(2026, 1, 1, tzinfo=U), datetime(2026, 7, 9, tzinfo=U))
H1 = (datetime(2021, 1, 1, tzinfo=U), datetime(2023, 6, 30, 23, tzinfo=U))
H2 = (datetime(2023, 7, 1, tzinfo=U), datetime(2025, 10, 6, tzinfo=U))


def prep_sem(pdata, btcb, desligar):
    """Igual ao v2, mas as estrategias em `desligar` nao emitem sinal."""
    arrs = {}
    for sym, df in pdata.items():
        s, e, sl, tp, st, bl = B.gen(df, btcb.reindex(df.index).fillna(False).values)
        for i in range(len(df)):
            if st[i] in desligar:
                s[i] = None; e[i] = np.nan; sl[i] = np.nan
                tp[i] = np.nan; st[i] = None; bl[i] = np.nan
        arrs[sym] = df.assign(_s=s, _e=e, _sl=sl, _tp=tp, _st=st, _bl=bl)
    return arrs


def linha(rot, s5, s26):
    m = ''
    if s5['tot'] > 347 and s26['tot'] > 60:
        m = '  <= BATE nos 2'
    print(f"  {rot:<28}{s5['tot']:>+9.0f}{s5['dd']:>7.1f}%{s5['anos']:>5}/5"
          f"{s5['meses']:>5}/{s5['nm']}{s26['tot']:>+8.0f}{s26['dd']:>7.1f}%"
          f"{s5['n']:>7}{s5['wr']:>7.1f}%{m}")


def main():
    print('\n' + '=' * 100)
    print('  DESLIGAR DIRECOES — quanto vale cada uma')
    print('=' * 100)
    print('  A carregar...')
    p5 = B.load(B.fetch_5y); p26 = B.load(B.fetch_26)

    print(f"  {'variante':<28}{'5 anos':>9}{'DD':>8}{'anos+':>7}{'meses+':>7}"
          f"{'OOS':>8}{'DD':>8}{'trades':>7}{'acerto':>7}")
    print('  ' + '-' * 90)

    combos = [('as 3 (baseline)', set()),
              ('SEM bull_smc', {'bull'}),
              ('SEM bear_breakout', {'bear'}),
              ('SEM lateral', {'lat'}),
              ('SO lateral', {'bull', 'bear'}),
              ('SO as direcionais', {'lat'})]
    res = {}
    for rot, off in combos:
        if rot == 'SO as direcionais':
            continue
        a5 = prep_sem(*p5, off); a26 = prep_sem(*p26, off)
        s5 = B.summ(B.run(a5, *S5)); s26 = B.summ(B.run(a26, *S26))
        res[rot] = (a5, s5, s26)
        linha(rot, s5, s26)
        if rot == 'as 3 (baseline)' and abs(s5['tot'] - 347) > 1:
            print('  ABORTADO: motor nao reproduz o baseline.'); sys.exit(1)
    print()

    b_a5, b5, b26 = res['as 3 (baseline)']
    print('  CONTRIBUICAO MARGINAL (quanto o total PIORA ao desligar cada uma):')
    for rot in ['SEM bull_smc', 'SEM bear_breakout', 'SEM lateral']:
        _, s5, s26 = res[rot]
        d5 = s5['tot'] - b5['tot']; d26 = s26['tot'] - b26['tot']
        alvo = rot.replace('SEM ', '')
        print(f"    {alvo:<20} 5 anos {d5:>+6.0f}   OOS {d26:>+5.0f}   "
              f"{'-> desligar MELHORA' if (d5 > 0 and d26 > 0) else ''}")
    print()

    # ── Bateria no candidato: SEM bull_smc ──
    a5, s5, s26 = res['SEM bull_smc']
    print('=' * 100)
    print('  BATERIA DE CONFIRMACAO — auto_v2 SEM a bull_smc')
    print('=' * 100)

    h1b = B.summ(B.run(b_a5, *H1)); h2b = B.summ(B.run(b_a5, *H2))
    h1 = B.summ(B.run(a5, *H1)); h2 = B.summ(B.run(a5, *H2))
    print(f"  SUB-PERIODOS      {'H1 21-23':>12}{'H2 23-25':>12}")
    print(f"    baseline        {h1b['tot']:>+12.0f}{h2b['tot']:>+12.0f}")
    print(f"    sem bull_smc    {h1['tot']:>+12.0f}{h2['tot']:>+12.0f}"
          f"   {'melhora nas 2' if (h1['tot'] > h1b['tot'] and h2['tot'] > h2b['tot']) else ''}")
    print()

    anos = sorted(b5['yr'])
    print('  ANO A ANO' + ' ' * 8 + ''.join(f'{a:>9}' for a in anos) + f"{'anos+':>8}")
    print(f"    baseline      " + ''.join(f"{b5['yr'].get(a, 0):>+9.0f}" for a in anos)
          + f"{b5['anos']:>6}/5")
    print(f"    sem bull_smc  " + ''.join(f"{s5['yr'].get(a, 0):>+9.0f}" for a in anos)
          + f"{s5['anos']:>6}/5")
    print()

    meses = sorted(b26['monthly'])
    mb = sum(1 for m in meses if b26['monthly'].get(m, 0) > 0)
    ms = sum(1 for m in meses if s26['monthly'].get(m, 0) > 0)
    print('  MES A MES OOS   ' + ''.join(f'{m[5:]:>8}' for m in meses) + f"{'TOT':>9}{'meses+':>9}")
    print('    baseline      ' + ''.join(f"{b26['monthly'].get(m, 0):>+8.0f}" for m in meses)
          + f"{b26['tot']:>+9.0f}{mb:>7}/{len(meses)}")
    print('    sem bull_smc  ' + ''.join(f"{s26['monthly'].get(m, 0):>+8.0f}" for m in meses)
          + f"{s26['tot']:>+9.0f}{ms:>7}/{len(meses)}")
    print()

    print('=' * 100)
    print('  VEREDICTO')
    print('=' * 100)
    ch = {
        'melhora nos 2 periodos': s5['tot'] > b5['tot'] and s26['tot'] > b26['tot'],
        'melhora nas 2 metades': h1['tot'] > h1b['tot'] and h2['tot'] > h2b['tot'],
        'mantem 5/5 anos': s5['anos'] >= b5['anos'],
        'DD nao piora': max(s5['dd'], s26['dd']) <= max(b5['dd'], b26['dd']) + 0.3,
    }
    falha = [k for k, v in ch.items() if not v]
    print(f"  baseline      {b5['tot']:+.0f} / {b26['tot']:+.0f} | DD {b5['dd']:.1f}% | {b5['anos']}/5 anos")
    print(f"  sem bull_smc  {s5['tot']:+.0f} / {s26['tot']:+.0f} | DD {s5['dd']:.1f}% | {s5['anos']}/5 anos")
    print()
    if not falha:
        print('  >> DESLIGAR a bull_smc sobrevive aos 4 criterios.')
    else:
        print(f"  >> NAO desligar — falha: {'; '.join(falha)}")
    print()


if __name__ == '__main__':
    main()
