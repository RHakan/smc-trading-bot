"""
backtest_tp_fixo_pct.py
ESPECIFICACAO DO RAFA (12/08/2026), em percentagem do PRECO, nao em estrutura:
  "quero que o bot busque um TP de 2% a 5%, que e muito mais alcancavel e mais rapido,
   e que o SL seja 3x menos que isso"

  TP 2%  -> SL 0.67%      TP 4%  -> SL 1.33%
  TP 3%  -> SL 1.00%      TP 5%  -> SL 1.67%

PORQUE E DIFERENTE DO QUE JA SE TESTOU
  Em backtest_rr_realista.py manteve-se o stop ESTRUTURAL (5-6% do preco) e recusaram-se
  os trades sem R:R — resultado: ZERO sinais passam a 3:1, porque a geometria nao fecha.
  Aqui o stop deixa de ser estrutural: e IMPOSTO em % do preco, junto com o alvo. Muda a
  pergunta de "este setup oferece 3:1?" para "e se eu obrigar 3:1 em todos?".

  Os SINAIS (entradas) continuam os do v2: decisor de regime, filtro BTC, mesmas
  condicoes de rompimento. So os niveis mudam. A invalidacao da lateral fica ativa —
  e o que corta perdedores cedo, e o Rafa quer isso.

O QUE MEDIR: com stop de 0.67-1.67% num grafico de 1H onde o ATR e 0.78% do preco, o
stop fica entre 0.9 e 2.1 ATR. Perto do ruido. A pergunta e se a taxa de acerto aguenta.
Cada celula mostra o ACERTO OBTIDO ao lado do ACERTO NECESSARIO (ja com taxas).

Portao: bater +347 (5 anos) E +60 (OOS). Bot rodando NAO tocado.
Uso: python backtests/backtest_tp_fixo_pct.py
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

S5 = (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2025, 10, 6, tzinfo=timezone.utc))
S26 = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 9, tzinfo=timezone.utc))
TPS = [2.0, 3.0, 4.0, 5.0]          # % do preco
RATIOS = [2.0, 2.5, 3.0, 4.0]       # SL = TP / ratio
FEE_PCT = 0.14


def prep_pct(pdata, btcb, tp_pct, ratio):
    """Sinais v2 intactos; niveis impostos em % do preco."""
    sl_pct = tp_pct / ratio
    arrs = {}
    for sym, df in pdata.items():
        s, e, sl, tp, st, bl = B.gen(df, btcb.reindex(df.index).fillna(False).values)
        d = df.assign(_s=s, _e=e, _sl=sl, _tp=tp, _st=st, _bl=bl)
        lado = np.where(d['_s'].values == 'LONG', 1.0,
                        np.where(d['_s'].values == 'SHORT', -1.0, np.nan))
        ev = d['_e'].values
        with np.errstate(invalid='ignore'):
            d['_sl'] = ev * (1 - lado * sl_pct / 100.0)
            d['_tp'] = ev * (1 + lado * tp_pct / 100.0)
        arrs[sym] = d
    return arrs


def be_wr(tp_pct, sl_pct):
    g = tp_pct - FEE_PCT; p = sl_pct + FEE_PCT
    return 100.0 * p / (g + p) if g > 0 else 100.0


def main():
    print('\n' + '=' * 100)
    print('  ALVO E STOP EM % DO PRECO — a especificacao do Rafa (TP 2-5%, SL 3x menor)')
    print('=' * 100)
    print('  A carregar...')
    p5 = B.load(B.fetch_5y); p26 = B.load(B.fetch_26)
    a5 = B.prep(*p5); a26 = B.prep(*p26)
    b5 = B.summ(B.run(a5, *S5)); b26 = B.summ(B.run(a26, *S26))
    print(f"  BASELINE: {b5['tot']:+.0f} | {b26['tot']:+.0f} | DD {b5['dd']:.1f}% | "
          f"{b5['anos']}/5 anos | acerto {b5['wr']:.1f}%")
    if abs(b5['tot'] - 347) > 1:
        print('  ABORTADO.'); sys.exit(1)
    atr = np.median([np.nanmedian(d['atr'].values / d['c'].values) for d in p5[0].values()]) * 100
    print(f'  ATR mediano em 1H = {atr:.3f}% do preco (referencia para ler o stop)\n')

    print(f"  {'TP':>5}{'SL':>7}{'R:R':>6}{'stop/ATR':>10}{'ACERTO':>9}{'precisa':>9}"
          f"{'margem':>8}{'5 anos':>9}{'OOS':>8}{'DD':>7}{'anos+':>7}")
    print('  ' + '-' * 86)
    melhor = None
    for tp in TPS:
        for rt in RATIOS:
            slp = tp / rt
            x5 = prep_pct(*p5, tp, rt); x26 = prep_pct(*p26, tp, rt)
            s5 = B.summ(B.run(x5, *S5)); s26 = B.summ(B.run(x26, *S26))
            be = be_wr(tp, slp); marg = s5['wr'] - be
            m = ''
            if s5['tot'] > 347 and s26['tot'] > 60:
                m = '  <= BATE nos 2'
            elif marg > 0:
                m = '  (acima do breakeven)'
            print(f"  {tp:>4.0f}%{slp:>6.2f}%{rt:>6.1f}{slp/atr:>9.2f}x{s5['wr']:>8.1f}%"
                  f"{be:>8.1f}%{marg:>+8.1f}{s5['tot']:>+9.0f}{s26['tot']:>+8.0f}"
                  f"{s5['dd']:>6.1f}%{s5['anos']:>5}/5{m}")
            sc = s5['tot'] + s26['tot']
            if melhor is None or sc > melhor[0]:
                melhor = (sc, tp, rt, s5, s26, be)
        print()
    print('=' * 100)
    print('  VEREDICTO')
    print('=' * 100)
    print(f"  BASELINE: {b5['tot']:+.0f} / {b26['tot']:+.0f} | DD {b5['dd']:.1f}% | {b5['anos']}/5 anos")
    if melhor:
        _, tp, rt, s5, s26, be = melhor
        print(f"  MELHOR:   TP {tp:.0f}% / SL {tp/rt:.2f}% (R:R {rt:.1f}) -> "
              f"{s5['tot']:+.0f} / {s26['tot']:+.0f} | DD {max(s5['dd'], s26['dd']):.1f}% | "
              f"{s5['anos']}/5 anos | acerto {s5['wr']:.1f}% (precisa {be:.1f}%)")
        ok = s5['tot'] > 347 and s26['tot'] > 60
        print('\n  >> ' + ('PASSA o portao — falta a bateria de confirmacao.'
                           if ok else 'NAO passa o portao.'))
    print()
    # a especificacao EXATA do Rafa, destacada
    print('  A TUA ESPECIFICACAO EXATA (SL = TP/3):')
    for tp in TPS:
        x5 = prep_pct(*p5, tp, 3.0); x26 = prep_pct(*p26, tp, 3.0)
        s5 = B.summ(B.run(x5, *S5)); s26 = B.summ(B.run(x26, *S26))
        be = be_wr(tp, tp / 3)
        print(f"    TP {tp:.0f}% / SL {tp/3:.2f}%  ->  acerto {s5['wr']:.1f}% "
              f"(precisa {be:.1f}%)  |  {s5['tot']:+.0f} / {s26['tot']:+.0f}")
    print()


if __name__ == '__main__':
    main()
