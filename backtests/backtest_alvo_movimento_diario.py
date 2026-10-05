"""
backtest_alvo_movimento_diario.py
PERGUNTA DO RAFA: e se o v2 deixar de ancorar alvo/stop na GEOMETRIA (altura do range,
topo estrutural) e passar a ancora-los numa FRACAO DO MOVIMENTO DIARIO da moeda?

Os SINAIS ficam IDENTICOS (mesmo decisor, mesmas entradas, mesma invalidacao).
Muda-se SO onde se poem o alvo e o stop:
    hoje  : lateral TP = altura do range | SL = lado oposto
            bull/bear TP = 2.5 x risco   | SL = estrutural
    teste : TP = T x ATR_diario          | SL = S x ATR_diario

PORQUE E QUE ISTO PODE FUNCIONAR (medido antes de correr):
  A amplitude media diaria destes 10 pares e 7.61% (mediana 6.30%), e em 88% dos dias
  o preco afasta-se >=2% da abertura. Com alvos a essa escala, a taxa de 0.14% passa a
  valer ~7% do alvo em vez dos ~26% que vale contra 1 ATR de 15m. O acerto necessario
  para empatar cai de ~62% para ~53%. A barreira das taxas deixa de mandar.

  O ATR diario e SEMPRE do dia ANTERIOR (shift 1) — zero lookahead.

Baseline a bater: +347 (5 anos) E +60 (OOS 2026), DD 10.1%, 5/5 anos positivos.
Um vencedor aqui ainda tem de passar a bateria de confirmacao antes de valer algo.
Bot rodando NAO tocado.
Uso: python backtests/backtest_alvo_movimento_diario.py
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
DATR_P = 14

ALVOS = [0.15, 0.20, 0.30, 0.40, 0.50, 0.75]   # fracao do ATR diario
STOPS = [0.15, 0.20, 0.30, 0.40, 0.50]


def carrega(fetch_fn):
    """Como B.load, mas guarda tambem o ATR DIARIO do dia anterior em cada vela 1H."""
    pdata = {}; btcb = None
    for sym in B.PAIRS:
        dfd = fetch_fn(sym, '1d').copy(); df1 = fetch_fn(sym, '1h').copy()
        if len(df1) < 300: continue
        ema_d = dfd['c'].ewm(span=B.R_EMA, adjust=False).mean()
        slope = (ema_d - ema_d.shift(B.R_SLOPE)) / ema_d.shift(B.R_SLOPE)
        dfd['regime'] = 'NEUTRAL'
        dfd.loc[(dfd['c'] < ema_d) & (slope < -B.R_THRESH), 'regime'] = 'BEAR'
        dfd.loc[(dfd['c'] > ema_d) & (slope > B.R_THRESH), 'regime'] = 'BULL'
        dfd['slope_d'] = slope
        # ATR DIARIO em preco absoluto
        trd = pd.concat([(dfd['h'] - dfd['l']), (dfd['h'] - dfd['c'].shift(1)).abs(),
                         (dfd['l'] - dfd['c'].shift(1)).abs()], axis=1).max(axis=1)
        dfd['datr'] = trd.ewm(com=DATR_P - 1, adjust=False).mean()

        c = df1['c']
        tr = pd.concat([(df1['h'] - df1['l']), (df1['h'] - c.shift(1)).abs(),
                        (df1['l'] - c.shift(1)).abs()], axis=1).max(axis=1)
        df1['atr'] = tr.ewm(com=B.ATR_PERIOD - 1, adjust=False).mean()
        df1['atr_avg'] = df1['atr'].rolling(B.B_ATR_AVG).mean()
        df1['adx'] = B.adx(df1, 14)
        df1['regime'] = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
        df1['slope_d'] = dfd['slope_d'].shift(1).reindex(df1.index, method='ffill')
        # shift(1) = ATR do dia ANTERIOR. Sem isto haveria lookahead.
        df1['datr'] = dfd['datr'].shift(1).reindex(df1.index, method='ffill')
        df1['ema_trend'] = c.ewm(span=B.U_EMA_TREND, adjust=False).mean()
        pdata[sym] = df1
        if sym.startswith('BTC'): btcb = (df1['regime'] == 'BULL')
    return pdata, btcb


def prep_diario(pdata, btcb, T, S):
    """Sinais IDENTICOS ao v2 (B.gen). So os niveis sao reescritos."""
    arrs = {}
    for sym, df in pdata.items():
        s, e, sl, tp, st, bl = B.gen(df, btcb.reindex(df.index).fillna(False).values)
        d = df.assign(_s=s, _e=e, _sl=sl, _tp=tp, _st=st, _bl=bl)
        da = d['datr'].values
        lado = np.where(d['_s'].values == 'LONG', 1.0,
                        np.where(d['_s'].values == 'SHORT', -1.0, np.nan))
        with np.errstate(invalid='ignore'):
            d['_sl'] = d['_e'].values - lado * S * da
            d['_tp'] = d['_e'].values + lado * T * da
        # sem ATR diario valido nao ha trade
        mau = np.isnan(da) | (da <= 0)
        d.loc[mau, ['_s', '_e', '_sl', '_tp', '_st']] = None
        arrs[sym] = d
    return arrs


def main():
    print('\n' + '=' * 100)
    print('  V2 COM ALVO/STOP ANCORADOS NO MOVIMENTO DIARIO (mesmos sinais)')
    print('=' * 100)
    print('  A carregar...')
    p5 = carrega(B.fetch_5y); p26 = carrega(B.fetch_26)

    # baseline geometrico, para referencia
    a5 = B.prep(*p5); a26 = B.prep(*p26)
    b5 = B.summ(B.run(a5, *S5)); b26 = B.summ(B.run(a26, *S26))
    print(f"  BASELINE (geometrico): {b5['tot']:+.0f} | {b26['tot']:+.0f} | "
          f"DD {b5['dd']:.1f}% | {b5['anos']}/5 anos+ | acerto {b5['wr']:.1f}%")
    if abs(b5['tot'] - 347) > 1:
        print('  ABORTADO: motor nao reproduz o baseline.'); sys.exit(1)

    da = np.median([np.nanmedian(d['datr'].values / d['c'].values) for d in p5[0].values()])
    print(f'  ATR diario mediano = {da*100:.2f}% do preco')
    print(f'  taxa 0.14% = {0.0014/da:.3f} do ATR diario\n')

    print(f"  {'alvo':<7}{'stop':<7}{'R:R':>6}{'alvo %':>9}{'stop %':>9}{'acerto':>9}"
          f"{'precisa':>9}{'5 anos':>9}{'OOS':>8}{'DD':>7}{'anos+':>7}{'trades':>8}")
    print('  ' + '-' * 96)
    melhor = None
    for T in ALVOS:
        for S in STOPS:
            a5t = prep_diario(*p5, T, S); a26t = prep_diario(*p26, T, S)
            s5 = B.summ(B.run(a5t, *S5)); s26 = B.summ(B.run(a26t, *S26))
            tp_pct = T * da * 100; sl_pct = S * da * 100
            f = 0.14
            be = 100 * (sl_pct + f) / ((tp_pct - f) + (sl_pct + f)) if tp_pct > f else 100
            m = ''
            if s5['tot'] > 347 and s26['tot'] > 60:
                m = '  <= BATE nos 2'
            print(f"  {T:<7.2f}{S:<7.2f}{T/S:>6.2f}{tp_pct:>8.2f}%{sl_pct:>8.2f}%"
                  f"{s5['wr']:>8.1f}%{be:>8.1f}%{s5['tot']:>+9.0f}{s26['tot']:>+8.0f}"
                  f"{s5['dd']:>6.1f}%{s5['anos']:>5}/5{s5['n']:>8}{m}")
            sc = s5['tot'] + s26['tot']
            if melhor is None or sc > melhor[0]:
                melhor = (sc, T, S, s5, s26)
        print()

    print('=' * 100)
    print('  VEREDICTO')
    print('=' * 100)
    print(f"  Baseline geometrico: {b5['tot']:+.0f} / {b26['tot']:+.0f} | DD {b5['dd']:.1f}%"
          f" | {b5['anos']}/5 anos | acerto {b5['wr']:.1f}%")
    if melhor:
        _, T, S, s5, s26 = melhor
        print(f"  Melhor ancorado no diario: alvo {T} / stop {S} do ATR diario")
        print(f"    {s5['tot']:+.0f} / {s26['tot']:+.0f} | DD {max(s5['dd'], s26['dd']):.1f}%"
              f" | {s5['anos']}/5 anos | acerto {s5['wr']:.1f}% | {s5['n']} trades")
        passa = s5['tot'] > 347 and s26['tot'] > 60
        print()
        if passa:
            print('  >> BATE o baseline nos dois periodos.')
            print('     Ainda tem de passar a bateria de confirmacao (planalto, sub-periodos,')
            print('     ano a ano, mes a mes) antes de se falar em implementar.')
        else:
            print('  >> NAO bate o baseline nos dois periodos.')
            print('     A ancoragem geometrica (altura do range / estrutura) continua melhor:')
            print('     adapta-se ao setup concreto, enquanto uma fracao fixa do movimento')
            print('     diario e a mesma para todos os setups.')
    print()


if __name__ == '__main__':
    main()
