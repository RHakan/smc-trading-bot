"""
backtest_stepped_stops.py
Compara três sistemas de gestão de stop:

  A) Sistema actual (bear_v12):
     - BE quando progresso >= 50% do alvo
     - Trailing stop 2.0×ATR depois do BE
     - Fecha no alvo (4R)

  B) 3 degraus (33% / 66% / 99%):
     - 33% → stop move para entry (BE)
     - 66% → stop lock 1.33R de lucro
     - 99% → fechar posição

  C) 4 degraus (25% / 50% / 75% / 100%):
     - 25% → stop move para entry (BE)
     - 50% → stop lock 1R de lucro
     - 75% → stop lock 2R de lucro
     - 100% → fechar posição

Entrada idêntica nos três: mesmos sinais bear_v12.
"""
import sys, time, io
from datetime import datetime, timezone, timedelta
from pathlib import Path
import ccxt, numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Captura tudo que é impresso para salvar em arquivo também
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()

_buf = io.StringIO()
sys.stdout = _Tee(sys.__stdout__, _buf)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 24, tzinfo=timezone.utc)

PAIRS = [
    'BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','BNB/USDT:USDT',
    'ADA/USDT:USDT','AVAX/USDT:USDT','DOGE/USDT:USDT','DOT/USDT:USDT',
    'XRP/USDT:USDT','LINK/USDT:USDT',
]

FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2
LOW_LOOKBACK=4; ATR_STOP=1.0; RR_CAP=4.0; COOLDOWN=8
REGIME_EMA=20; SLOPE_BARS=5; SLOPE_THRESH=0.001; ATR_AVG=48
BE_TRIGGER=1.0; TRAIL_ATR=2.0   # parametros actuais v1.2

ex = ccxt.binanceusdm({'enableRateLimit': True})

def fetch(sym, tf, extra):
    since  = int((START - timedelta(days=extra)).timestamp()*1000)
    end_ms = int(END.timestamp()*1000)
    rows   = []
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b: break
        rows.extend(b); since = b[-1][0]+1
        if since >= end_ms or len(b)==0: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=['ts','o','h','l','c','v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()[lambda d: d.index < END]

print('Carregando dados...')
data    = {}
data_1d = {}
for sym in PAIRS:
    dfd = fetch(sym, '1d', 30).copy()
    dfd['ema']   = dfd['c'].ewm(span=REGIME_EMA, adjust=False).mean()
    dfd['slope'] = (dfd['ema'] - dfd['ema'].shift(SLOPE_BARS)) / dfd['ema'].shift(SLOPE_BARS)
    dfd['regime'] = 'NEUTRAL'
    dfd.loc[(dfd['c'] < dfd['ema']) & (dfd['slope'] < -SLOPE_THRESH), 'regime'] = 'BEAR'
    data_1d[sym] = dfd

    df1 = fetch(sym, '1h', 5).copy()
    tr  = pd.concat([
        (df1['h']-df1['l']),
        (df1['h']-df1['c'].shift(1)).abs(),
        (df1['l']-df1['c'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df1['atr']     = tr.ewm(com=13, adjust=False).mean()
    df1['atr_avg'] = df1['atr'].rolling(ATR_AVG).mean()
    df1['low_n']   = df1['l'].shift(1).rolling(LOW_LOOKBACK).min()
    df1['regime']  = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
    data[sym] = df1
    time.sleep(0.1)
print('OK\n')


# ── Simulação: sistema actual (v1.2) ──────────────────────────────────────────

def sim_current(df, ib, entry, sl_price, atr_entry):
    """BE quando progresso >= 50% do caminho até ao alvo (= 2R); trailing 2xATR depois."""
    risk  = sl_price - entry
    if risk <= 0: return 0.0
    reward = risk * RR_CAP
    tp    = entry - reward
    fee_r = entry * RT / risk
    cur   = sl_price; be = False

    for j in range(ib+1, min(ib+400, len(df))):
        bar = df.iloc[j]
        lo=float(bar['l']); hi=float(bar['h']); c=float(bar['c'])
        atr = float(bar['atr']) if not np.isnan(bar['atr']) else atr_entry

        if hi >= cur:
            return (entry - cur) / risk - fee_r
        if lo <= tp:
            return RR_CAP - fee_r

        progress = (entry - c) / reward

        if not be and progress >= 0.5:
            cur = entry; be = True
        if be:
            cand = c + TRAIL_ATR * atr
            if cand < cur:
                cur = cand

    return (entry - float(df.iloc[min(ib+399,len(df)-1)]['c'])) / risk - fee_r


# ── Simulação: stop em degraus ────────────────────────────────────────────────

def sim_stepped(df, ib, entry, sl_price, steps):
    """
    steps: lista de (trigger_pct, lock_pct | None)
      trigger_pct : progresso mínimo para activar este degrau
      lock_pct    : % do reward a garantir como lucro mínimo
                    None = fechar a posição imediatamente

    Lógica do stop para SHORT:
      stop = entry - lock_pct * reward
      (stop abaixo do entry = lucro garantido se preço reverter até lá)
    """
    risk   = sl_price - entry
    if risk <= 0: return 0.0
    reward = risk * RR_CAP
    tp     = entry - reward
    fee_r  = entry * RT / risk
    cur    = sl_price   # stop inicial = entry + risk (acima do entry para SHORT)

    activated = [False] * len(steps)

    for j in range(ib+1, min(ib+400, len(df))):
        bar = df.iloc[j]
        lo=float(bar['l']); hi=float(bar['h']); c=float(bar['c'])

        # Stop atingido (para SHORT: preço sobe até ao stop)
        if hi >= cur:
            return (entry - cur) / risk - fee_r

        # Take profit fixo atingido
        if lo <= tp:
            return RR_CAP - fee_r

        progress = (entry - c) / reward

        # Verifica degraus por ordem
        for idx, (trig, lock) in enumerate(steps):
            if not activated[idx] and progress >= trig:
                activated[idx] = True

                if lock is None:
                    # Fechar posição no preco actual
                    return (entry - c) / risk - fee_r

                # Mover stop para garantir lock_pct × reward de lucro
                # Para SHORT: novo_stop = entry - lock_pct × reward
                # (abaixo do entry = lucro se preço reverter até aqui)
                new_stop = entry - lock * reward

                # Só move se o novo stop for melhor (mais baixo = mais seguro para SHORT)
                if new_stop < cur:
                    cur = new_stop

    return (entry - float(df.iloc[min(ib+399,len(df)-1)]['c'])) / risk - fee_r


# ── Configurações dos sistemas ────────────────────────────────────────────────

STEPS_3 = [
    (0.33, 0.0),    # 33%: stop vai para entry (BE, 0R garantido)
    (0.66, 0.33),   # 66%: stop lock 1.33R (33% do reward de 4R)
    (0.99, None),   # 99%: fechar posição
]

STEPS_4 = [
    (0.25, 0.0),    # 25%: stop vai para entry (BE, 0R garantido)
    (0.50, 0.25),   # 50%: stop lock 1R   (25% do reward de 4R)
    (0.75, 0.50),   # 75%: stop lock 2R   (50% do reward de 4R)
    (1.00, None),   # 100%: fechar posição
]


# ── Coleta trades e simula os 3 sistemas em paralelo ─────────────────────────

results = {'current': [], 'steps3': [], 'steps4': []}
months_current = {}; months_3 = {}; months_4 = {}

for sym, df1 in data.items():
    last_bar = -9999
    start_i  = max(ATR_AVG+10,
                   next((i for i,ts in enumerate(df1.index) if ts>=START), ATR_AVG+10))
    for i in range(start_i, len(df1)-1):
        if i - last_bar < COOLDOWN: continue
        ts_bar = df1.index[i]
        if ts_bar < START: continue
        r = df1.iloc[i]
        if str(r['regime']) != 'BEAR': continue
        cl=float(r['c']); op=float(r['o'])
        atr=float(r['atr']); atr_avg=float(r['atr_avg'])
        low_n=float(r['low_n'])
        if any(np.isnan(v) for v in [cl,atr,atr_avg,low_n]): continue
        if not (cl < low_n and cl < op and atr > atr_avg): continue
        sl_p = cl + ATR_STOP * atr
        if sl_p <= cl: continue

        mk = ts_bar.strftime('%Y-%m')

        nr_c  = sim_current(df1, i, cl, sl_p, atr)
        nr_s3 = sim_stepped(df1, i, cl, sl_p, STEPS_3)
        nr_s4 = sim_stepped(df1, i, cl, sl_p, STEPS_4)

        results['current'].append(nr_c)
        results['steps3'].append(nr_s3)
        results['steps4'].append(nr_s4)

        months_current.setdefault(mk, []).append(nr_c)
        months_3.setdefault(mk, []).append(nr_s3)
        months_4.setdefault(mk, []).append(nr_s4)

        last_bar = i

n = len(results['current'])
print(f'Total de trades (identico nos 3): {n}\n')


# ── Funcao auxiliar ───────────────────────────────────────────────────────────

def compound(nrs, risk=0.01, start=100):
    bal = start
    for r in nrs:
        bal *= (1 + r * risk)
    return bal

def stats(nrs):
    if not nrs: return {}
    wins = [r for r in nrs if r > 0]
    loss = [r for r in nrs if r <= 0]
    avg  = sum(nrs)/len(nrs)
    gw   = sum(wins) if wins else 0
    gl   = abs(sum(loss)) if loss else 0.001
    return {
        'n': len(nrs),
        'wr': len(wins)/len(nrs)*100,
        'avg': avg,
        'pf': gw/gl,
        'final': compound(nrs),
    }


# ── OUTPUT ────────────────────────────────────────────────────────────────────

labels = {
    'current': 'Sistema actual (BE@50%+trail)',
    'steps3':  '3 degraus  (33% / 66% / 99%)',
    'steps4':  '4 degraus  (25% / 50% / 75% / 100%)',
}

print('='*65)
print('RESULTADO GLOBAL')
print('='*65)
for key, lbl in labels.items():
    s = stats(results[key])
    pct = (s['final']/100 - 1) * 100
    print(f"\n  {lbl}")
    print(f"  $100 -> ${s['final']:.2f}  ({pct:+.1f}%)")
    print(f"  WR {s['wr']:.1f}%  |  avg {s['avg']:+.3f}R  |  PF {s['pf']:.2f}")

print('\n' + '='*65)
print('POR MES')
print('='*65)
print(f"\n  {'Mes':<8}  {'Actual':>12}  {'3 degraus':>12}  {'4 degraus':>12}")
print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")

all_months = sorted(set(list(months_current)+list(months_3)+list(months_4)))
month_names = {'01':'Jan','02':'Fev','03':'Mar','04':'Abr','05':'Mai','06':'Jun'}

for mk in all_months:
    mc = months_current.get(mk, [])
    m3 = months_3.get(mk, [])
    m4 = months_4.get(mk, [])
    name = month_names.get(mk.split('-')[1], mk)

    def month_pct(nrs):
        if not nrs: return 0.0
        bal = 100
        for r in nrs:
            bal *= (1 + r * 0.01)
        return (bal - 100)

    pc = month_pct(mc); p3 = month_pct(m3); p4 = month_pct(m4)
    sign_c = '+' if pc>=0 else ''; sign_3 = '+' if p3>=0 else ''; sign_4 = '+' if p4>=0 else ''
    print(f"  {name:<8}  {sign_c}{pc:>10.1f}%  {sign_3}{p3:>10.1f}%  {sign_4}{p4:>10.1f}%")

print('\n' + '='*65)
print('DISTRIBUICAO DE RESULTADOS')
print('='*65)
buckets = [(-99,-2,'< -2R'),(-2,-1,'-2R a -1R'),(-1,-0.5,'-1R a -0.5R'),
           (-0.5,0,'-0.5R a 0'),(0,1,'0 a +1R'),(1,2,'+1R a +2R'),
           (2,3,'+2R a +3R'),(3,99,'> +3R')]

for lo,hi,lbl in buckets:
    cc = sum(1 for r in results['current'] if lo<=r<hi)
    c3 = sum(1 for r in results['steps3']  if lo<=r<hi)
    c4 = sum(1 for r in results['steps4']  if lo<=r<hi)
    print(f"  {lbl:>14}:  actual={cc:3d}  3deg={c3:3d}  4deg={c4:3d}")

print('\n' + '='*65)
print('ANALISE: Marco e Maio (meses negativos)')
print('='*65)
bad_months = ('2026-03','2026-05')
bad_c  = [r for mk in bad_months for r in months_current.get(mk,[])]
bad_s3 = [r for mk in bad_months for r in months_3.get(mk,[])]
bad_s4 = [r for mk in bad_months for r in months_4.get(mk,[])]

for key, nrs, lbl in [
    ('actual', bad_c,  'Sistema actual'),
    ('3deg',   bad_s3, '3 degraus'),
    ('4deg',   bad_s4, '4 degraus'),
]:
    if not nrs: continue
    wins = [r for r in nrs if r > 0]
    avg  = sum(nrs)/len(nrs)
    pct  = month_pct(nrs)
    print(f"\n  {lbl}:  {len(nrs)} trades  WR {len(wins)/len(nrs)*100:.1f}%  avg {avg:+.3f}R  total {pct:+.1f}%")

# ── Salva output em Outputs/ ──────────────────────────────────────────────────
sys.stdout = sys.__stdout__
today = datetime.now().strftime('%Y-%m-%d')
out_path = Path(r'C:\Users\ASUS\OneDrive\Documentos\Claudinho\Outputs') / f'{today}_bot-futures_stepped-stops.txt'
out_path.write_text(_buf.getvalue(), encoding='utf-8')
print(f'\nOutput salvo em: {out_path}')
