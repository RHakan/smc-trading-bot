"""
backtest_rr_realista.py
REGRA DO RAFA: so abrir o trade se o risco/retorno compensar COM UM ALVO REALISTA.

DIAGNOSTICO QUE ORIGINOU ISTO (5 anos, medido):
  bear_breakout : stop 6.33%  alvo 15.83%  -> alvo pede MAIS QUE UM DIA INTEIRO de
                  amplitude em 82.7% dos trades (a media diaria destes pares e 7.61%)
  bull_smc      : stop 4.89%  alvo 12.22%  -> idem em 80.6%
  lateral       : stop 5.81%  alvo  5.01%  -> 100% dos trades arriscam MAIS do que
                  podem ganhar (R:R 0.86)

  Causa nas direcionais: o alvo e 2.5 x o stop. Stop estrutural largo => alvo absurdo.
  O alvo NAO esta ancorado em nada que o mercado ofereca — e aritmetica sobre o stop.
  Consequencia ja medida: so 12% das direcionais chegam ao alvo.

A REGRA (criterio de RECUSA, nao ajuste de parametro)
  1. alvo_real = min(alvo da estrategia, K x ATR_diario)   <- limita ao que a moeda anda
  2. se alvo_real / distancia_do_stop < RR_MIN  ->  NAO ABRE (o setup nao compensa)
  3. caso contrario abre, com alvo_real
  O stop NAO e mexido: continua estrutural (fundo do range / topo estrutural). So se
  recusa o trade quando a geometria nao paga. E o "nao compensa, nao entro" do Rafa.

ATR diario e SEMPRE do dia anterior (shift 1) — zero lookahead.
Portao: bater +347 (5 anos) E +60 (OOS) sem piorar DD 10.1% nem os 5/5 anos.
Bot rodando NAO tocado.
Uso: python backtests/backtest_rr_realista.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

_s = importlib.util.spec_from_file_location(
    '_base', str(Path(__file__).parent / 'backtest_v3_trail_alvo.py'))
B = importlib.util.module_from_spec(_s); _s.loader.exec_module(B)
_d = importlib.util.spec_from_file_location(
    '_diario', str(Path(__file__).parent / 'backtest_alvo_movimento_diario.py'))
D = importlib.util.module_from_spec(_d); _d.loader.exec_module(D)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

S5 = (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2025, 10, 6, tzinfo=timezone.utc))
S26 = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 9, tzinfo=timezone.utc))
RR_MINS = [1.5, 2.0, 2.5, 3.0]
KS = [0.5, 0.75, 1.0, 1.5]


def prep_rr(pdata, btcb, rr_min, k):
    arrs = {}; aceites = 0; total = 0
    for sym, df in pdata.items():
        s, e, sl, tp, st, bl = B.gen(df, btcb.reindex(df.index).fillna(False).values)
        da = df['datr'].values
        n = len(df)
        for i in range(n):
            if st[i] is None:
                continue
            total += 1
            if any(pd.isna(x) for x in (e[i], sl[i], tp[i])) or np.isnan(da[i]) or da[i] <= 0:
                st[i] = None; s[i] = None; continue
            lado = 1 if s[i] == 'LONG' else -1
            risco = abs(e[i] - sl[i])
            alvo_est = abs(tp[i] - e[i])
            alvo_real = min(alvo_est, k * da[i])          # limita ao que a moeda anda
            if risco <= 0 or alvo_real / risco < rr_min:  # nao compensa -> recusa
                st[i] = None; s[i] = None; e[i] = np.nan
                sl[i] = np.nan; tp[i] = np.nan; bl[i] = np.nan
                continue
            tp[i] = e[i] + lado * alvo_real
            aceites += 1
        arrs[sym] = df.assign(_s=s, _e=e, _sl=sl, _tp=tp, _st=st, _bl=bl)
    return arrs, aceites, total


def main():
    print('\n' + '=' * 100)
    print('  SO ABRE SE O R:R COMPENSAR COM ALVO REALISTA (criterio de recusa)')
    print('=' * 100)
    print('  A carregar...')
    p5 = D.carrega(B.fetch_5y); p26 = D.carrega(B.fetch_26)
    a5 = B.prep(*p5); a26 = B.prep(*p26)
    b5 = B.summ(B.run(a5, *S5)); b26 = B.summ(B.run(a26, *S26))
    print(f"  BASELINE: {b5['tot']:+.0f} | {b26['tot']:+.0f} | DD {b5['dd']:.1f}% | "
          f"{b5['anos']}/5 anos | acerto {b5['wr']:.1f}% | {b5['n']} trades")
    if abs(b5['tot'] - 347) > 1:
        print('  ABORTADO.'); sys.exit(1)
    print()
    print(f"  {'R:R min':<9}{'alvo max':<11}{'5 anos':>9}{'DD':>7}{'anos+':>7}{'OOS':>8}"
          f"{'trades':>8}{'acerto':>8}{'sinais aceites':>16}")
    print('  ' + '-' * 84)
    melhor = None
    for rr in RR_MINS:
        for k in KS:
            x5, ac, tot = prep_rr(*p5, rr, k)
            x26, _, _ = prep_rr(*p26, rr, k)
            s5 = B.summ(B.run(x5, *S5)); s26 = B.summ(B.run(x26, *S26))
            m = '  <= BATE nos 2' if (s5['tot'] > 347 and s26['tot'] > 60) else ''
            print(f"  {rr:<9.1f}{str(k) + 'x dATR':<11}{s5['tot']:>+9.0f}{s5['dd']:>6.1f}%"
                  f"{s5['anos']:>5}/5{s26['tot']:>+8.0f}{s5['n']:>8}{s5['wr']:>7.1f}%"
                  f"{f'{ac}/{tot} ({ac/max(tot,1)*100:.0f}%)':>16}{m}")
            sc = s5['tot'] + s26['tot']
            if melhor is None or sc > melhor[0]:
                melhor = (sc, rr, k, s5, s26, ac, tot)
        print()
    print('=' * 100)
    print('  VEREDICTO')
    print('=' * 100)
    print(f"  BASELINE: {b5['tot']:+.0f} / {b26['tot']:+.0f} | DD {b5['dd']:.1f}% | {b5['anos']}/5 anos")
    if melhor:
        _, rr, k, s5, s26, ac, tot = melhor
        print(f"  MELHOR:   R:R>={rr}, alvo<={k}x dATR -> {s5['tot']:+.0f} / {s26['tot']:+.0f} | "
              f"DD {max(s5['dd'], s26['dd']):.1f}% | {s5['anos']}/5 anos | "
              f"aceita {ac/max(tot,1)*100:.0f}% dos sinais")
        ok = s5['tot'] > 347 and s26['tot'] > 60
        print('\n  >> ' + ('PASSA o portao — falta a bateria de confirmacao.'
                           if ok else 'NAO passa o portao.'))
    print()


if __name__ == '__main__':
    main()
