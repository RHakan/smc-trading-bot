"""
backtest_lateral_reteste.py
ENTRADA NO RETESTE vs FILTRO DE ESTICAO vs BASELINE — so na lateral.

DIAGNOSTICO QUE ORIGINOU ISTO (medido em 745 sinais de 5 anos):
A lateral tem, por desenho, alvo = altura do range e stop = lado oposto. Se a entrada
fosse NA borda, o R:R seria 1.0. Mas a entrada e no FECHO da vela de rompimento, que
costuma esticar-se muito alem da borda. O alvo fica fixo, o stop cresce:

    excesso da entrada (em alturas de range)  ->  R:R medio
      0.00 - 0.10                                   0.95
      0.10 - 0.25                                   0.86
      0.25 - 0.50                                   0.75
      0.50 - 1.00                                   0.61
      > 1.00                                        0.43
    correlacao excesso -> R:R = -0.959

As duas queixas do Rafa ("stops longos demais" e "entradas que deixam movimento para
tras") sao O MESMO problema visto de dois lados: entrar tarde numa vela esticada.

TRES VARIANTES
  baseline — entra no FECHO da vela de rompimento (o bot de hoje).
  filtro   — igual, mas DESCARTA sinais cujo excesso passe um limiar. Evita o mau,
             nao melhora o bom.
  reteste  — em vez de comprar no fecho, deixa ordem LIMITE de volta na BORDA do range.
             Se o preco voltar la dentro de N velas, entra a esse preco; se nao voltar,
             o sinal e DESCARTADO (nao vira entrada a mercado — sem isto seria fantasia).
             Como a entrada passa a ser a propria borda, o R:R volta a 1.0 POR
             CONSTRUCAO. Bonus: ordem limite = taxa MAKER na perna de entrada.

TAXAS (o reteste ganha aqui tambem)
  baseline/filtro : entrada taker + saida taker = 0.14%
  reteste         : entrada MAKER (0.02%, sem slippage) + saida taker = 0.09%

CUSTO HONESTO DO RETESTE: os rompimentos mais violentos nunca retestam, e ha quem
diga que sao justamente os melhores. Esse e o ponto empirico a decidir aqui — por isso
o relatorio mostra sempre quantos sinais foram perdidos por nao preencherem.

Bull/bear ficam INTACTAS nas tres variantes. Portao: bater +347 (5 anos) E +60 (OOS),
sem piorar o DD de 10.1% nem os 5/5 anos positivos.
Bot rodando NAO tocado.
Uso: python backtests/backtest_lateral_reteste.py
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

RT_TAKER = (0.0005 + 0.0002) * 2          # 0.14% — entrada e saida a mercado
RT_MISTO = 0.0002 + (0.0005 + 0.0002)     # 0.09% — entrada maker, saida a mercado

LIMIARES = [0.10, 0.20, 0.30, 0.50]        # filtro: excesso maximo tolerado
JANELAS = [3, 5, 10, 20]                   # reteste: velas de espera pelo preenchimento


def _excesso(e, bl, sl):
    """Quanto a entrada passou da borda rompida, em alturas de range."""
    alt = abs(bl - sl)
    if alt <= 0 or any(pd.isna(x) for x in (e, bl, sl)):
        return np.nan
    return abs(e - bl) / alt


def prep_var(pdata, btcb, modo, param):
    """Sinais v2 normais; so os da LATERAL sao reescritos conforme o modo."""
    arrs = {}
    perdidos = 0; total = 0
    for sym, df in pdata.items():
        s, e, sl, tp, st, bl = B.gen(df, btcb.reindex(df.index).fillna(False).values)
        n = len(df)
        h = df['h'].values; l = df['l'].values

        if modo != 'baseline':
            s2 = s.copy(); e2 = e.copy(); sl2 = sl.copy(); tp2 = tp.copy()
            st2 = st.copy(); bl2 = bl.copy()
            for i in range(n):
                if st[i] != 'lat':
                    continue
                total += 1
                ex = _excesso(e[i], bl[i], sl[i])

                if modo == 'filtro':
                    if not np.isnan(ex) and ex > param:
                        s2[i] = None; e2[i] = np.nan; sl2[i] = np.nan
                        tp2[i] = np.nan; st2[i] = None; bl2[i] = np.nan
                        perdidos += 1
                    continue

                # ── reteste: ordem limite de volta a borda ────────────────────
                # apaga o sinal original; so reaparece se e quando preencher
                s2[i] = None; e2[i] = np.nan; sl2[i] = np.nan
                tp2[i] = np.nan; st2[i] = None; bl2[i] = np.nan
                if any(pd.isna(x) for x in (e[i], bl[i], sl[i])):
                    perdidos += 1; continue
                lado = 1 if s[i] == 'LONG' else -1
                limite = bl[i]                       # a propria borda rompida
                altura = abs(bl[i] - sl[i])
                achou = -1
                for j in range(i + 1, min(i + 1 + param, n)):
                    tocou = (l[j] <= limite) if lado > 0 else (h[j] >= limite)
                    if tocou:
                        achou = j; break
                if achou < 0:
                    perdidos += 1; continue
                if st2[achou] is not None:           # ja ha sinal nessa vela
                    perdidos += 1; continue
                # entrada NA borda -> stop mede exatamente a altura -> R:R = 1.0
                s2[achou] = s[i]
                e2[achou] = limite
                sl2[achou] = sl[i]
                tp2[achou] = limite + lado * altura
                st2[achou] = 'lat'
                bl2[achou] = bl[i]
            s, e, sl, tp, st, bl = s2, e2, sl2, tp2, st2, bl2

        arrs[sym] = df.assign(_s=s, _e=e, _sl=sl, _tp=tp, _st=st, _bl=bl)
    return arrs, perdidos, total


def perfil_lat(res):
    t = [x for x in res['trades'] if x['strat'] == 'lat']
    if not t:
        return dict(n=0, wr=0.0, rr=0.0)
    nr = np.array([x['nr'] for x in t])
    return dict(n=len(t), wr=float((nr > 0.03).mean() * 100),
                rr=float(np.mean([x['path'] for x in t])))


def corre(p5, p26, modo, param, rt):
    old = B.RT
    B.RT = rt
    try:
        a5, perd, tot = prep_var(*p5, modo, param)
        a26, _, _ = prep_var(*p26, modo, param)
        s5 = B.summ(B.run(a5, *S5)); s26 = B.summ(B.run(a26, *S26))
    finally:
        B.RT = old
    return s5, s26, perd, tot


def linha(rot, s5, s26, perd, tot):
    pf = perfil_lat(s5)
    m = ''
    if s5['tot'] > 347 and s26['tot'] > 60:
        m = '  <= BATE nos 2'
    pd_ = f'{perd}/{tot} ({perd/max(tot,1)*100:.0f}%)' if tot else '—'
    print(f"  {rot:<26}{s5['tot']:>+9.0f}{s5['dd']:>7.1f}%{s5['anos']:>5}/5"
          f"{s26['tot']:>+8.0f}{pf['n']:>7}{pf['rr']:>7.2f}{pf['wr']:>7.1f}%{pd_:>15}{m}")


def main():
    print('\n' + '=' * 104)
    print('  LATERAL — entrada no RETESTE vs FILTRO de esticao vs BASELINE')
    print('=' * 104)
    print('  A carregar...')
    p5 = B.load(B.fetch_5y); p26 = B.load(B.fetch_26)

    print(f"  {'variante':<26}{'5 anos':>9}{'DD':>8}{'anos+':>7}{'OOS':>8}"
          f"{'n lat':>7}{'R:R':>7}{'WR lat':>8}{'sinais perdidos':>15}")
    print('  ' + '-' * 96)

    b5, b26, _, _ = corre(p5, p26, 'baseline', 0, RT_TAKER)
    linha('BASELINE (fecho)', b5, b26, 0, 0)
    if abs(b5['tot'] - 347) > 1:
        print('  ABORTADO: motor nao reproduz o baseline.'); sys.exit(1)
    print()

    melhor = None
    for lim in LIMIARES:
        s5, s26, p, t = corre(p5, p26, 'filtro', lim, RT_TAKER)
        linha(f'filtro excesso <= {lim:.2f}', s5, s26, p, t)
        if melhor is None or s5['tot'] + s26['tot'] > melhor[0]:
            melhor = (s5['tot'] + s26['tot'], f'filtro {lim}', s5, s26)
    print()

    for jan in JANELAS:
        s5, s26, p, t = corre(p5, p26, 'reteste', jan, RT_TAKER)
        linha(f'reteste {jan}v (taxa taker)', s5, s26, p, t)
    print()
    for jan in JANELAS:
        s5, s26, p, t = corre(p5, p26, 'reteste', jan, RT_MISTO)
        linha(f'reteste {jan}v (MAKER)', s5, s26, p, t)
        if melhor is None or s5['tot'] + s26['tot'] > melhor[0]:
            melhor = (s5['tot'] + s26['tot'], f'reteste {jan}v maker', s5, s26)
    print()

    print('=' * 104)
    print('  VEREDICTO')
    print('=' * 104)
    print(f"  BASELINE: {b5['tot']:+.0f} / {b26['tot']:+.0f} | DD {b5['dd']:.1f}% | "
          f"{b5['anos']}/5 anos | R:R lat {perfil_lat(b5)['rr']:.2f}")
    if melhor:
        _, rot, s5, s26 = melhor
        pf = perfil_lat(s5)
        print(f"  MELHOR:   {rot} -> {s5['tot']:+.0f} / {s26['tot']:+.0f} | "
              f"DD {max(s5['dd'], s26['dd']):.1f}% | {s5['anos']}/5 anos | R:R lat {pf['rr']:.2f}")
        passa = s5['tot'] > 347 and s26['tot'] > 60
        print()
        print('  >> ' + ('PASSA o portao — falta a bateria de confirmacao.'
                         if passa else 'NAO passa o portao.'))
    print()


if __name__ == '__main__':
    main()
