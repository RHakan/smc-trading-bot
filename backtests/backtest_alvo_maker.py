"""
backtest_alvo_maker.py
ALVO POR ORDEM LIMITE EM REPOUSO (maker) em vez de TAKE_PROFIT_MARKET (taker).

DE ONDE VEIO (cadeia de 3 testes desta sessao):
  1. reteste na borda do range   -> R:R 0.86->1.00 mas acerto caiu 34.5%->22.1%
  2. limite ao preco do fecho    -> preenche 81%, +308 vs +347 do baseline
  3. tecto teorico com maker     -> +417 vs +347  =>  **a taxa maker vale +20%**
  Conclusao: o beneficio da taxa e real, mas na ENTRADA e inseparavel da espera, e
  esperar filtra para fora os trades com conviccao (os que ganham).

  NA SAIDA nao ha esse problema. Uma ordem limite de venda pousada NO ALVO, acima do
  mercado, e MAKER — e preenche exatamente quando o preco chega ao alvo, que e quando
  se queria sair de qualquer forma. Zero seleccao adversa: se o preco nao chega la,
  tambem nao se teria saido.

O QUE MUDA
  entrada    : a mercado (taker)  — IGUAL ao de hoje, sem esperas, sem perder sinais
  ALVO       : ordem limite em repouso (MAKER)   <-- a unica mudanca
  stop       : a mercado (taker)  — obrigatorio, nao da para pousar
  invalidacao/timeout : a mercado (taker)

  Logo a taxa deixa de ser constante e passa a depender de COMO o trade sai:
    sai no alvo         -> 0.07% + 0.02% = 0.09%
    sai no stop/inval   -> 0.07% + 0.07% = 0.14%   (igual a hoje)
  Nenhum sinal e perdido, nenhum trade muda de preco. So o custo dos VENCEDORES desce.

BONUS POSSIVEL, E PODE VALER MAIS QUE A TAXA
  Uma ordem limite normal vive no balde BASIC da Binance, nao no balde ALGO das
  condicionais — e foi o balde ALGO que NAO EXECUTOU no trade ETH/USDC #60 (o preco
  passou o alvo duas vezes, `algoStatus: NEW`, posicao presa 40h). Uma limite em
  repouso pode funcionar onde a TAKE_PROFIT_MARKET falhou.

DUAS HIPOTESES DE PREENCHIMENTO (a conservadora e que decide)
  otimista    : preenche se o maximo da vela TOCAR o alvo (mesma regra do motor atual)
  conservador : so preenche se o preco ATRAVESSAR o alvo com margem (fila de ordens —
                tocar no preco nao garante execucao). Se ganhar nesta, ganha a serio.

Portao: bater +347 (5 anos) E +60 (OOS) sem piorar DD nem os 5/5 anos.
Bot rodando NAO tocado.
Uso: python backtests/backtest_alvo_maker.py
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

ENTRADA_TAKER = 0.0005 + 0.0002     # 0.07% — mercado + slippage
SAIDA_TAKER = 0.0005 + 0.0002       # 0.07%
SAIDA_MAKER = 0.0002                # 0.02% — limite em repouso, sem slippage

MARGENS = [0.0, 0.0005, 0.0010, 0.0020]   # quanto o preco tem de atravessar o alvo


def run_maker(arrs, start, end, saida_alvo, margem, max_per_side=B.MAX_PER_SIDE):
    """Motor v3 com taxa de saida DEPENDENTE do motivo da saida.

    `saida_alvo` = taxa cobrada quando o trade sai no alvo (maker ou taker).
    `margem`     = fracao que o preco tem de ATRAVESSAR alem do alvo para a limite
                   preencher. 0 = basta tocar (otimista).
    """
    gest = B.GEST_BASE
    midx = None
    for d in arrs.values():
        midx = d.index if midx is None else midx.union(d.index)
    midx = midx[(midx >= start) & (midx <= end)]
    A = {}
    for sym, d in arrs.items():
        dd = d.reindex(midx)
        A[sym] = {'h': dd['h'].values, 'l': dd['l'].values, 'c': dd['c'].values,
                  'atr': dd['atr'].values, 'side': dd['_s'].values,
                  'entry': dd['_e'].values, 'sl': dd['_sl'].values, 'tp': dd['_tp'].values,
                  'strat': dd['_st'].values, 'bl': dd['_bl'].values}
    bal = B.MONTHLY_BASE; peak = B.MONTHLY_BASE; max_dd = 0.0
    pos = {}; cdu = {s: -1 for s in A}; monthly = {}; cur = None
    trades = []; n_alvo = 0

    def _rec(p, nr, motivo):
        trades.append(dict(nr=nr, strat=p['strat'], motivo=motivo))

    def bruto(p, px):
        return ((px - p['entry']) if p['side'] == 'LONG' else (p['entry'] - px)) / p['risk_px']

    for k in range(len(midx)):
        mk = midx[k].strftime('%Y-%m')
        if cur is None: cur = mk
        if mk != cur:
            for sym in list(pos):
                p = pos[sym]; c = A[sym]['c'][k]
                if np.isnan(c): c = p['entry']
                nr = bruto(p, c) - p['fee_in'] - p['fee_out_taker']
                bal += nr * p['risk_usd']; _rec(p, nr, 'mes'); del pos[sym]
            monthly[cur] = bal - B.MONTHLY_BASE
            bal = B.MONTHLY_BASE; peak = B.MONTHLY_BASE; cur = mk

        for sym in list(pos):
            d = A[sym]; c = d['c'][k]
            if np.isnan(c): continue
            p = pos[sym]; hi = d['h'][k]; lo = d['l'][k]; atr = d['atr'][k]
            a = atr if not np.isnan(atr) else p['risk_px']
            fechou = False; nr = 0.0; motivo = ''

            # stop primeiro (conservador) — sempre a mercado
            if p['side'] == 'LONG':
                if lo <= p['cur']:
                    nr = bruto(p, p['cur']) - p['fee_in'] - p['fee_out_taker']
                    fechou = True; motivo = 'stop'
                elif hi >= p['tp'] * (1 + margem):
                    nr = bruto(p, p['tp']) - p['fee_in'] - p['fee_out_alvo']
                    fechou = True; motivo = 'alvo'
                else:
                    if not p['be'] and (c - p['entry']) / p['risk_px'] >= p['be_t'] * p['path']:
                        p['cur'] = p['entry']; p['be'] = True
                    if p['be'] and p['trail'] > 0:
                        cand = c - p['trail'] * a
                        if cand > p['cur']: p['cur'] = cand
            else:
                if hi >= p['cur']:
                    nr = bruto(p, p['cur']) - p['fee_in'] - p['fee_out_taker']
                    fechou = True; motivo = 'stop'
                elif lo <= p['tp'] * (1 - margem):
                    nr = bruto(p, p['tp']) - p['fee_in'] - p['fee_out_alvo']
                    fechou = True; motivo = 'alvo'
                else:
                    if not p['be'] and (p['entry'] - c) / p['risk_px'] >= p['be_t'] * p['path']:
                        p['cur'] = p['entry']; p['be'] = True
                    if p['be'] and p['trail'] > 0:
                        cand = c + p['trail'] * a
                        if cand < p['cur']: p['cur'] = cand

            if not fechou and p['strat'] == 'lat':
                p['age'] += 1
                if p['age'] <= B.L_INVAL_N and not np.isnan(p['bl']):
                    dentro = (c < p['bl']) if p['side'] == 'LONG' else (c > p['bl'])
                    if dentro:
                        nr = bruto(p, c) - p['fee_in'] - p['fee_out_taker']
                        fechou = True; motivo = 'inval'
                if not fechou and p['age'] >= B.L_TIMEOUT:
                    nr = bruto(p, c) - p['fee_in'] - p['fee_out_taker']
                    fechou = True; motivo = 'timeout'

            if fechou:
                if motivo == 'alvo': n_alvo += 1
                bal += nr * p['risk_usd']; _rec(p, nr, motivo)
                del pos[sym]; cdu[sym] = k + p['cd']

        nl = sum(1 for p in pos.values() if p['side'] == 'LONG')
        nsh = sum(1 for p in pos.values() if p['side'] == 'SHORT')
        for sym in A:
            if sym in pos or k <= cdu[sym]: continue
            d = A[sym]; side = d['side'][k]
            if side is None or (isinstance(side, float) and np.isnan(side)): continue
            e = d['entry'][k]; stop = d['sl'][k]; tp = d['tp'][k]; st = d['strat'][k]
            if np.isnan(e) or np.isnan(stop) or np.isnan(tp) or bal <= 0: continue
            if side == 'LONG' and nl >= max_per_side: continue
            if side == 'SHORT' and nsh >= max_per_side: continue
            rp = abs(e - stop)
            if rp <= 0: continue
            g = gest[st]; path = abs(tp - e) / rp
            if path <= 0: continue
            pos[sym] = {'side': side, 'entry': e, 'cur': stop, 'tp': tp, 'be': False,
                        'age': 0, 'risk_px': rp, 'risk_usd': bal * (B.RISK_PCT / 100.0),
                        'fee_in': e * ENTRADA_TAKER / rp,
                        'fee_out_taker': e * SAIDA_TAKER / rp,
                        'fee_out_alvo': e * saida_alvo / rp,
                        'be_t': g['be'], 'trail': g['trail'], 'cd': g['cd'],
                        'strat': st, 'bl': d['bl'][k], 'path': path}
            if side == 'LONG': nl += 1
            else: nsh += 1
        peak = max(peak, bal)
        max_dd = max(max_dd, (peak - bal) / peak * 100 if peak > 0 else 0)

    for sym in list(pos):
        p = pos[sym]; c = A[sym]['c'][len(midx) - 1]
        if np.isnan(c): c = p['entry']
        nr = bruto(p, c) - p['fee_in'] - p['fee_out_taker']
        bal += nr * p['risk_usd']; _rec(p, nr, 'fim')
    monthly[cur] = bal - B.MONTHLY_BASE

    yr = {}
    for m, v in monthly.items(): yr[m[:4]] = yr.get(m[:4], 0) + v
    nrs = np.array([t['nr'] for t in trades])
    return dict(tot=sum(monthly.values()), dd=max_dd, n=len(nrs),
                wr=float((nrs > 0.03).mean() * 100) if len(nrs) else 0.0,
                anos=sum(1 for v in yr.values() if v > 0), yr=yr,
                n_alvo=n_alvo, monthly=monthly)


def linha(rot, s5, s26, base5=347, base26=60):
    m = ''
    if s5['tot'] > base5 and s26['tot'] > base26:
        m = '  <= BATE nos 2'
    pa = s5['n_alvo'] / max(s5['n'], 1) * 100
    print(f"  {rot:<34}{s5['tot']:>+9.0f}{s5['dd']:>7.1f}%{s5['anos']:>5}/5"
          f"{s26['tot']:>+8.0f}{s5['n']:>8}{s5['wr']:>7.1f}%{pa:>12.0f}%{m}")


def main():
    print('\n' + '=' * 100)
    print('  ALVO POR ORDEM LIMITE (maker) vs ALVO A MERCADO (taker)')
    print('=' * 100)
    print('  A carregar...')
    a5 = B.prep(*B.load(B.fetch_5y)); a26 = B.prep(*B.load(B.fetch_26))

    print(f"  {'variante':<34}{'5 anos':>9}{'DD':>8}{'anos+':>7}{'OOS':>8}"
          f"{'trades':>8}{'WR':>7}{'saiu no alvo':>12}")
    print('  ' + '-' * 92)

    b5 = run_maker(a5, *S5, SAIDA_TAKER, 0.0)
    b26 = run_maker(a26, *S26, SAIDA_TAKER, 0.0)
    linha('BASELINE (alvo a mercado)', b5, b26)
    if abs(b5['tot'] - 347) > 2:
        print(f"  ABORTADO: motor da {b5['tot']:+.0f}, esperado +347."); sys.exit(1)
    print()

    melhor = None
    for mg in MARGENS:
        s5 = run_maker(a5, *S5, SAIDA_MAKER, mg)
        s26 = run_maker(a26, *S26, SAIDA_MAKER, mg)
        rot = ('alvo MAKER (basta tocar)' if mg == 0
               else f'alvo MAKER (atravessa {mg*100:.2f}%)')
        linha(rot, s5, s26)
        if melhor is None or s5['tot'] + s26['tot'] > melhor[0]:
            melhor = (s5['tot'] + s26['tot'], rot, s5, s26, mg)
    print()
    print('  ^ "atravessa X%" = a limite so preenche se o preco passar o alvo com essa')
    print('    margem. E a hipotese conservadora: tocar no preco nao garante execucao')
    print('    quando ha fila de ordens. Se ganhar na margem maior, ganha a serio.')
    print()

    print('=' * 100)
    print('  VEREDICTO')
    print('=' * 100)
    print(f"  BASELINE: {b5['tot']:+.0f} / {b26['tot']:+.0f} | DD {b5['dd']:.1f}% | {b5['anos']}/5 anos")
    if melhor:
        _, rot, s5, s26, mg = melhor
        print(f"  MELHOR:   {rot} -> {s5['tot']:+.0f} / {s26['tot']:+.0f} | "
              f"DD {max(s5['dd'], s26['dd']):.1f}% | {s5['anos']}/5 anos")
        # o teste que conta: a hipotese MAIS conservadora ainda bate?
        cs5 = run_maker(a5, *S5, SAIDA_MAKER, MARGENS[-1])
        cs26 = run_maker(a26, *S26, SAIDA_MAKER, MARGENS[-1])
        ok_c = cs5['tot'] > 347 and cs26['tot'] > 60
        print(f"\n  TESTE DECISIVO — hipotese MAIS conservadora (atravessa {MARGENS[-1]*100:.2f}%):")
        print(f"    {cs5['tot']:+.0f} / {cs26['tot']:+.0f} | DD {max(cs5['dd'], cs26['dd']):.1f}%"
              f" | {cs5['anos']}/5 anos  ->  {'PASSA' if ok_c else 'nao passa'}")
        print()
        print('  >> ' + ('PASSA mesmo no cenario pessimista — falta a bateria de confirmacao.'
                         if ok_c else 'So funciona na hipotese otimista. Nao e de confianca.'))
    print()


if __name__ == '__main__':
    main()
