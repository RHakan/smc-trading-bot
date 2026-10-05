"""
backtest_entrada_limite.py
ENTRADA POR ORDEM LIMITE AO PRECO DO FECHO — sem esperar pelo reteste.

DE ONDE VEIO (backtest_lateral_reteste.py, 5 anos):
  O reteste na BORDA do range entregou o que prometia — R:R subiu de 0.86 para 1.00
  exatamente, DD desceu, taxas desceram — mas o acerto CAIU PARA METADE (34.5% -> 22.1%).
  Motivo: 33% dos sinais nunca retestam, e sao precisamente os rompimentos com forca.
  Esperar pelo reteste e SELECAO ADVERSA: fica-se so com os fracos.

  Mas houve um numero que ficou a piscar: so por trocar a taxa de entrada de taker
  para maker, o mesmo teste passou de +269 para +337. **A taxa maker sozinha vale +25%.**

A IDEIA DESTE TESTE: apanhar o beneficio da taxa maker SEM a seleccao adversa do reteste.
  Em vez de ordem limite la atras na borda do range, ordem limite AO PROPRIO PRECO DO
  FECHO — o mesmo preco a que o bot entraria a mercado. O preco so tem de voltar ao
  ponto onde ja esteve, nao de percorrer o caminho todo de volta.

  Entrada IGUAL a do baseline. Stop IGUAL. Alvo IGUAL. R:R IGUAL.
  So muda: taxa de entrada 0.02% em vez de 0.05%+slippage, e alguns sinais nao preenchem.

  Se a taxa de preenchimento for alta, isto e lucro puro — mesmos trades, menos custo.
  Se for baixa, a seleccao adversa volta, so que mais suave. E o que se vai medir.

MODELACAO HONESTA
  Ordem colocada DEPOIS do fecho da vela do sinal. Preenche se, dentro de N velas, o
  preco tocar o nivel (LONG: minimo <= limite; SHORT: maximo >= limite). Preenche AO
  LIMITE, nunca a um preco melhor (conservador no caso de gap). Se nao tocar em N velas,
  o sinal e DESCARTADO — nunca vira entrada a mercado.

TAXAS
  baseline : entrada taker + saida taker  = 0.14%
  limite   : entrada MAKER + saida taker  = 0.09%   (a saida e sempre a mercado)

Testado so na lateral e nas 3 estrategias. Portao: bater +347 (5 anos) E +60 (OOS),
sem piorar o DD de 10.1% nem os 5/5 anos. Bot rodando NAO tocado.
Uso: python backtests/backtest_entrada_limite.py
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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

S5 = (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2025, 10, 6, tzinfo=timezone.utc))
S26 = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 9, tzinfo=timezone.utc))

RT_TAKER = (0.0005 + 0.0002) * 2          # 0.14%
RT_MISTO = 0.0002 + (0.0005 + 0.0002)     # 0.09% — entrada maker, saida a mercado
JANELAS = [1, 2, 3, 5, 8]


def prep_limite(pdata, btcb, janela, alvo_estrats):
    """Sinais v2 normais, mas os de `alvo_estrats` so valem se a limite preencher."""
    arrs = {}; perd = 0; tot = 0
    for sym, df in pdata.items():
        s, e, sl, tp, st, bl = B.gen(df, btcb.reindex(df.index).fillna(False).values)
        n = len(df); h = df['h'].values; l = df['l'].values
        s2, e2, sl2 = s.copy(), e.copy(), sl.copy()
        tp2, st2, bl2 = tp.copy(), st.copy(), bl.copy()
        for i in range(n):
            if st[i] not in alvo_estrats:
                continue
            tot += 1
            # limpa o sinal original — so reaparece na vela em que preencher
            s2[i] = None; e2[i] = np.nan; sl2[i] = np.nan
            tp2[i] = np.nan; st2[i] = None; bl2[i] = np.nan
            if pd.isna(e[i]):
                perd += 1; continue
            lado = 1 if s[i] == 'LONG' else -1
            lim = e[i]                      # limite ao PROPRIO preco do fecho
            achou = -1
            for j in range(i + 1, min(i + 1 + janela, n)):
                tocou = (l[j] <= lim) if lado > 0 else (h[j] >= lim)
                if tocou:
                    achou = j; break
            if achou < 0 or st2[achou] is not None:
                perd += 1; continue
            # tudo igual ao baseline — so o momento e a taxa mudam
            s2[achou] = s[i]; e2[achou] = e[i]; sl2[achou] = sl[i]
            tp2[achou] = tp[i]; st2[achou] = st[i]; bl2[achou] = bl[i]
        arrs[sym] = df.assign(_s=s2, _e=e2, _sl=sl2, _tp=tp2, _st=st2, _bl=bl2)
    return arrs, perd, tot


def corre(p5, p26, janela, alvo, rt):
    old = B.RT; B.RT = rt
    try:
        a5, perd, tot = prep_limite(*p5, janela, alvo)
        a26, _, _ = prep_limite(*p26, janela, alvo)
        return B.summ(B.run(a5, *S5)), B.summ(B.run(a26, *S26)), perd, tot
    finally:
        B.RT = old


def linha(rot, s5, s26, perd, tot):
    m = ''
    if s5['tot'] > 347 and s26['tot'] > 60:
        m = '  <= BATE nos 2'
    fill = f'{(1-perd/max(tot,1))*100:.0f}%' if tot else '—'
    print(f"  {rot:<30}{s5['tot']:>+9.0f}{s5['dd']:>7.1f}%{s5['anos']:>5}/5"
          f"{s26['tot']:>+8.0f}{s5['n']:>8}{s5['wr']:>7.1f}%{fill:>11}{m}")


def main():
    print('\n' + '=' * 100)
    print('  ENTRADA POR LIMITE AO PRECO DO FECHO (mesma entrada, taxa maker)')
    print('=' * 100)
    print('  A carregar...')
    p5 = B.load(B.fetch_5y); p26 = B.load(B.fetch_26)

    print(f"  {'variante':<30}{'5 anos':>9}{'DD':>8}{'anos+':>7}{'OOS':>8}"
          f"{'trades':>8}{'WR':>7}{'preenche':>11}")
    print('  ' + '-' * 88)

    B.RT = RT_TAKER
    a5 = B.prep(*p5); a26 = B.prep(*p26)
    b5 = B.summ(B.run(a5, *S5)); b26 = B.summ(B.run(a26, *S26))
    linha('BASELINE (mercado, taker)', b5, b26, 0, 0)
    if abs(b5['tot'] - 347) > 1:
        print('  ABORTADO: motor nao reproduz o baseline.'); sys.exit(1)

    # tecto teorico: mesmos trades do baseline, so com a taxa mais baixa
    B.RT = RT_MISTO
    t5 = B.summ(B.run(a5, *S5)); t26 = B.summ(B.run(a26, *S26))
    linha('TECTO (100% preenche, maker)', t5, t26, 0, 1)
    B.RT = RT_TAKER
    print('  ^ so existe se TODOS os sinais preenchessem. E o maximo que a taxa pode dar.')
    print()

    melhor = None
    for nome, alvo in [('so lateral', {'lat'}), ('as 3 estrategias', {'lat', 'bull', 'bear'})]:
        for jan in JANELAS:
            s5, s26, p, t = corre(p5, p26, jan, alvo, RT_MISTO)
            linha(f'limite {jan}v — {nome}', s5, s26, p, t)
            sc = s5['tot'] + s26['tot']
            if melhor is None or sc > melhor[0]:
                melhor = (sc, f'limite {jan}v {nome}', s5, s26, p, t)
        print()

    print('=' * 100)
    print('  VEREDICTO')
    print('=' * 100)
    print(f"  BASELINE:  {b5['tot']:+.0f} / {b26['tot']:+.0f} | DD {b5['dd']:.1f}% | {b5['anos']}/5 anos")
    print(f"  TECTO:     {t5['tot']:+.0f} / {t26['tot']:+.0f}  (se preenchesse sempre)")
    if melhor:
        _, rot, s5, s26, p, t = melhor
        print(f"  MELHOR:    {rot} -> {s5['tot']:+.0f} / {s26['tot']:+.0f} | "
              f"DD {max(s5['dd'], s26['dd']):.1f}% | {s5['anos']}/5 anos | "
              f"preenche {(1-p/max(t,1))*100:.0f}%")
        passa = s5['tot'] > 347 and s26['tot'] > 60
        print()
        print('  >> ' + ('PASSA o portao — falta a bateria de confirmacao.'
                         if passa else 'NAO passa o portao.'))
    print()


if __name__ == '__main__':
    main()
