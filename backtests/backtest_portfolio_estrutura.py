"""
backtest_portfolio_estrutura.py
Compara DOIS Decisores no mesmo motor de portfolio (saldo unico 5000 USDC):

  A) Decisor EMA      — regime macro via EMA20 diaria + slope (o atual)
  B) Decisor ESTRUTURA — movimento via estrutura de mercado (HH/HL + CHoCH)
                         = a versao objetiva e causal de LTA/LTB

E testa o efeito de um TETO DE EXPOSICAO (max posicoes simultaneas), que ataca
o drawdown de correlacao que destruiu o teste anterior (-75% de DD).

Cenarios corridos:
  1. EMA         — sem limite        (baseline = teste anterior)
  2. ESTRUTURA   — sem limite        (efeito puro da estrutura)
  3. ESTRUTURA   — teto 4 posicoes   (estrutura + controlo de correlacao)

Decisor ESTRUTURA (causal, sem repintura):
  - Swing pivots por fractal: high[p] e topo se for o maximo de [p-L, p+R].
    So e CONHECIDO na barra p+R (lag a direita) -> nunca espreita o futuro.
  - ALTA  = topos e fundos a subir (HH + HL)  -> LTA intacta -> estrategia bull
  - BAIXA = topos e fundos a descer (LH + LL) -> LTB intacta -> estrategia bear
  - CHoCH = close rompe o ultimo pivot -> antecipa a virada (reduz o lag do EMA)
  - INDEF = estrutura ainda nao definida -> fica de fora

⚠️  Bull SMC continua HIPOTETICA. Nada disto vai ao bot sem aprovacao.
Uso: python backtests/backtest_portfolio_estrutura.py
"""
import sys, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ccxt, numpy as np, pandas as pd
from bot.strategies import bear_v13 as BEAR
from bot import regime as regime_mod

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Periodo e saldo ───────────────────────────────────────────────────────────
START = datetime(2025, 6,  1, tzinfo=timezone.utc)
END   = datetime(2026, 6, 29, tzinfo=timezone.utc)
INITIAL_BALANCE = 5000.0
RISK_PCT        = 1.0

PAIRS = [
    'BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT',
    'ADA/USDT:USDT','DOGE/USDT:USDT','LINK/USDT:USDT','SOL/USDT:USDT',
    'AVAX/USDT:USDT','DOT/USDT:USDT',
]
FEE=0.0005; SLIP=0.0002; RT=(FEE+SLIP)*2

# ── Parametros BEAR (fonte unica) ─────────────────────────────────────────────
B_LOOKBACK=BEAR.LOW_LOOKBACK; B_ATR_AVG=BEAR.ATR_AVG_PERIOD; B_ATR_STOP=BEAR.ATR_STOP_MULT
B_COOLDOWN=BEAR.COOLDOWN_BARS; B_RR=BEAR.RR_CAP; B_BE_PCT=BEAR.BE_TRIGGER_PCT
B_TRAIL=BEAR.TRAIL_ATR; ATR_PERIOD=BEAR.ATR_PERIOD
R_EMA=regime_mod.EMA_PERIOD; R_SLOPE=regime_mod.SLOPE_BARS; R_THRESH=regime_mod.SLOPE_THRESH

# ── Parametros BULL SMC (config robusta) ──────────────────────────────────────
U_SWING_N=10; U_CHOCH_BARS=12; U_CHOCH_REF=15; U_MIN_SWEEP=0.05
U_RR=2.5; U_BE_PCT=0.70; U_TRAIL=2.0; U_COOLDOWN=3

# ── Estrutura (Decisor B) ─────────────────────────────────────────────────────
STRUCT_L = 5    # barras a esquerda do pivot
STRUCT_R = 5    # barras a direita (= lag de confirmacao, garante causalidade)

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
ex = ccxt.binanceusdm({'enableRateLimit': True})


def fetch(sym, tf, extra_days):
    safe  = sym.replace('/', '_').replace(':', '_')
    cache = CACHE_DIR / f'{safe}_{tf}_jun25_jun26.csv'
    if cache.exists():
        df = pd.read_csv(cache, index_col='ts', parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    since  = int((START - timedelta(days=extra_days)).timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    rows=[]
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b: break
        rows.extend(b); since=b[-1][0]+1
        if since>=end_ms: break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=['ts','o','h','l','c','v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('ts').sort_index()[lambda d: d.index <= END]
    df.to_csv(cache)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DECISOR B — estrutura de mercado (causal)
# ─────────────────────────────────────────────────────────────────────────────
def detect_structure(df, L=STRUCT_L, R=STRUCT_R):
    """
    Devolve um array de regime ('BULL'/'BEAR'/'NEUTRAL') por barra, derivado da
    estrutura de mercado. Causal: um pivot na barra p so e usado a partir de p+R.
    Mapeamento para reutilizar o mesmo gen_signals:
        ALTA  -> 'BULL'   BAIXA -> 'BEAR'   INDEF -> 'NEUTRAL'
    """
    h=df['h'].values; l=df['l'].values; c=df['c'].values
    n=len(df)
    reg=np.array(['NEUTRAL']*n, dtype=object)
    prev_sh=last_sh=prev_sl=last_sl=None
    state='NEUTRAL'

    for i in range(n):
        p = i - R                       # pivot candidato (R barras atras)
        if p - L >= 0:
            win = slice(p-L, p+R+1)     # janela [p-L, p+R], toda ja conhecida em i
            if h[p] == np.max(h[win]):  # swing high confirmado
                prev_sh, last_sh = last_sh, h[p]
            if l[p] == np.min(l[win]):  # swing low confirmado
                prev_sl, last_sl = last_sl, l[p]

        # Estado por HH/HL vs LH/LL (mantem o ultimo claro se ambiguo)
        if None not in (last_sh, prev_sh, last_sl, prev_sl):
            if last_sh > prev_sh and last_sl > prev_sl:
                state = 'BULL'
            elif last_sh < prev_sh and last_sl < prev_sl:
                state = 'BEAR'

        # CHoCH — antecipa a virada quando o preco rompe o ultimo pivot
        if last_sh is not None and c[i] > last_sh and state != 'BULL':
            state = 'BULL'
        if last_sl is not None and c[i] < last_sl and state != 'BEAR':
            state = 'BEAR'

        reg[i] = state
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# CARGA + INDICADORES
# ─────────────────────────────────────────────────────────────────────────────
print('Backtest PORTFOLIO — Decisor EMA vs Decisor ESTRUTURA')
print(f'  Periodo: {START:%d/%m/%Y} -> {END:%d/%m/%Y}  |  saldo {INITIAL_BALANCE:.0f} USDC\n')
print('A carregar dados (cache se disponivel)...')

pair_data = {}
for sym in PAIRS:
    print(f'  {sym}...', end=' ', flush=True)
    try:
        dfd = fetch(sym, '1d', 40).copy()
        df1 = fetch(sym, '1h', 25).copy()
        if len(df1) < 300:
            print('SEM DADOS'); continue

        ema_d = dfd['c'].ewm(span=R_EMA, adjust=False).mean()
        slope = (ema_d - ema_d.shift(R_SLOPE)) / ema_d.shift(R_SLOPE)
        dfd['regime'] = 'NEUTRAL'
        dfd.loc[(dfd['c'] < ema_d) & (slope < -R_THRESH), 'regime'] = 'BEAR'
        dfd.loc[(dfd['c'] > ema_d) & (slope >  R_THRESH), 'regime'] = 'BULL'

        tr = pd.concat([
            (df1['h']-df1['l']),
            (df1['h']-df1['c'].shift(1)).abs(),
            (df1['l']-df1['c'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df1['atr']     = tr.ewm(com=ATR_PERIOD-1, adjust=False).mean()
        df1['atr_avg'] = df1['atr'].rolling(B_ATR_AVG).mean()
        df1['reg_ema'] = dfd['regime'].shift(1).reindex(df1.index, method='ffill')
        df1['reg_str'] = detect_structure(df1)
        df1['low_n']   = df1['l'].shift(1).rolling(B_LOOKBACK).min()

        pair_data[sym] = df1
        print('OK')
    except Exception as e:
        print(f'ERRO ({e})')
print()


def gen_signals(df, reg_col):
    """Gera (side, entry, sl) por barra usando a coluna de regime indicada."""
    n=len(df)
    o=df['o'].values; h=df['h'].values; l=df['l'].values; c=df['c'].values
    atr=df['atr'].values; atr_avg=df['atr_avg'].values
    low_n=df['low_n'].values; reg=df[reg_col].values

    side=np.array([None]*n, dtype=object)
    entry=np.full(n, np.nan); sl=np.full(n, np.nan)
    start_i=max(B_ATR_AVG+B_LOOKBACK+5, U_SWING_N+U_CHOCH_REF+U_CHOCH_BARS+5, STRUCT_L+STRUCT_R+5)

    for i in range(start_i, n):
        r=reg[i]
        if r=='BEAR':
            cl=c[i]; op=o[i]; a=atr[i]; av=atr_avg[i]; ln=low_n[i]
            if not any(np.isnan(v) for v in (cl,a,av,ln)):
                if cl<ln and cl<op and a>av:
                    sl_p=cl+B_ATR_STOP*a
                    if sl_p>cl: side[i]='SHORT'; entry[i]=cl; sl[i]=sl_p
            continue
        if r=='BULL':
            a=atr[i]; av=atr_avg[i]
            if np.isnan(a): continue
            if not np.isnan(av) and a<av: continue
            best=None
            for j in range(i-1, max(i-U_CHOCH_BARS-1, U_SWING_N+U_CHOCH_REF)-1, -1):
                swing_low=np.min(l[j-U_SWING_N:j])
                if not (l[j]<swing_low and c[j]>swing_low): continue
                if U_MIN_SWEEP>0 and (swing_low-l[j])/swing_low*100 < U_MIN_SWEEP: continue
                ref_high=np.max(h[j-U_CHOCH_REF:j])
                if np.any(c[j+1:i] > ref_high): continue
                if c[i]>ref_high:
                    actual_low=np.min(l[j:i+1]); risk=c[i]-actual_low
                    if risk>0 and risk/c[i]<=0.10:
                        best=(c[i], actual_low); break
            if best:
                side[i]='LONG'; entry[i]=best[0]; sl[i]=best[1]
    return side, entry, sl


# Indice mestre
master_index=None
for sym, df in pair_data.items():
    master_index = df.index if master_index is None else master_index.union(df.index)
master_index = master_index[(master_index>=START)&(master_index<=END)]


def build_arrays(reg_col):
    A={}
    for sym, df in pair_data.items():
        side, entry, sl = gen_signals(df, reg_col)
        d = df.assign(_s=side, _e=entry, _sl=sl).reindex(master_index)
        A[sym]={'h':d['h'].values,'l':d['l'].values,'c':d['c'].values,
                'atr':d['atr'].values,'side':d['_s'].values,
                'entry':d['_e'].values,'sl':d['_sl'].values}
    return A


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────
def update_position(pos, hi, lo, c, atr):
    entry=pos['entry']; risk=pos['risk_px']; fee_r=pos['fee_r']
    rr=pos['rr']; be_pct=pos['be_pct']; trail=pos['trail']
    a=atr if not np.isnan(atr) else risk
    if pos['side']=='SHORT':
        if hi>=pos['cur']:
            m='breakeven' if (pos['be'] and pos['cur']==entry) else ('trailing' if pos['be'] else 'stop')
            return True, (entry-pos['cur'])/risk-fee_r, m
        if lo<=pos['tp']: return True, rr-fee_r, 'take_profit'
        if not pos['be'] and (entry-c)/risk>=be_pct*rr: pos['cur']=entry; pos['be']=True
        if pos['be']:
            cand=c+trail*a
            if cand<pos['cur']: pos['cur']=cand
    else:
        if lo<=pos['cur']:
            m='breakeven' if (pos['be'] and pos['cur']==entry) else ('trailing' if pos['be'] else 'stop')
            return True, (pos['cur']-entry)/risk-fee_r, m
        if hi>=pos['tp']: return True, rr-fee_r, 'take_profit'
        if not pos['be'] and (c-entry)/risk>=be_pct*rr: pos['cur']=entry; pos['be']=True
        if pos['be']:
            cand=c-trail*a
            if cand>pos['cur']: pos['cur']=cand
    return False, 0.0, None


def run_portfolio(A, max_concurrent=None):
    balance=INITIAL_BALANCE; peak=INITIAL_BALANCE; max_dd=0.0
    positions={}; cooldown_until={s:-1 for s in A}
    trades=[]; max_conc=0
    N=len(master_index)
    for k in range(N):
        ts=master_index[k]
        # saidas
        for sym in list(positions.keys()):
            d=A[sym]; c=d['c'][k]
            if np.isnan(c): continue
            pos=positions[sym]
            closed,nr,m = update_position(pos, d['h'][k], d['l'][k], c, d['atr'][k])
            if closed:
                balance += nr*pos['risk_usd']
                trades.append({'ts_out':ts,'sym':sym.replace('/USDT:USDT',''),
                               'strat':pos['strat'],'nr':nr})
                del positions[sym]
                cooldown_until[sym]=k+(B_COOLDOWN if pos['strat']=='bear' else U_COOLDOWN)
        # entradas
        for sym in A:
            if sym in positions: continue
            if max_concurrent is not None and len(positions)>=max_concurrent: break
            if k<=cooldown_until[sym]: continue
            d=A[sym]; side=d['side'][k]
            if side is None or (isinstance(side,float) and np.isnan(side)): continue
            entry=d['entry'][k]; sl=d['sl'][k]
            if np.isnan(entry) or np.isnan(sl) or balance<=0: continue
            risk_usd=balance*(RISK_PCT/100.0)
            if side=='SHORT':
                risk_px=sl-entry; tp=entry-B_RR*risk_px
                rr,be,tr_,st=B_RR,B_BE_PCT,B_TRAIL,'bear'
            else:
                risk_px=entry-sl; tp=entry+U_RR*risk_px
                rr,be,tr_,st=U_RR,U_BE_PCT,U_TRAIL,'bull'
            if risk_px<=0: continue
            positions[sym]={'side':side,'entry':entry,'cur':sl,'tp':tp,'be':False,
                            'risk_px':risk_px,'risk_usd':risk_usd,'fee_r':entry*RT/risk_px,
                            'rr':rr,'be_pct':be,'trail':tr_,'strat':st}
        max_conc=max(max_conc,len(positions))
        peak=max(peak,balance)
        dd=(peak-balance)/peak*100 if peak>0 else 0
        max_dd=max(max_dd,dd)
    return {'balance':balance,'trades':trades,'max_dd':max_dd,'max_conc':max_conc}


def summarize(res, label):
    t=res['trades']; bal=res['balance']
    if not t:
        print(f'  {label:<28}: sem trades'); return
    w=[x for x in t if x['nr']>0]
    gw=sum(x['nr'] for x in w); gl=abs(sum(x['nr'] for x in t if x['nr']<=0))
    ret=(bal/INITIAL_BALANCE-1)*100
    bear=[x for x in t if x['strat']=='bear']; bull=[x for x in t if x['strat']=='bull']
    print(f'  {label:<28}: {bal:>9,.0f} USDC ({ret:>+6.1f}%) | {len(t):>4}t | '
          f'WR {len(w)/len(t)*100:>4.1f}% | PF {gw/gl if gl>0 else 99:>4.2f} | '
          f'DD {res["max_dd"]:>4.1f}% | maxPos {res["max_conc"]:>2} | '
          f'bear {len(bear)} / bull {len(bull)}')
    return res


# ─────────────────────────────────────────────────────────────────────────────
# CENARIOS
# ─────────────────────────────────────────────────────────────────────────────
print('A gerar sinais e correr cenarios...\n')
print('='*128)
print('  COMPARACAO DE DECISORES E CONTROLO DE EXPOSICAO')
print('='*128)

A_ema = build_arrays('reg_ema')
A_str = build_arrays('reg_str')

r1 = summarize(run_portfolio(A_ema, max_concurrent=None), '1) EMA        | sem limite')
r2 = summarize(run_portfolio(A_str, max_concurrent=None), '2) ESTRUTURA  | sem limite')
r3 = summarize(run_portfolio(A_str, max_concurrent=4),    '3) ESTRUTURA  | teto 4 pos')
r4 = summarize(run_portfolio(A_ema, max_concurrent=4),    '4) EMA        | teto 4 pos')

# Detalhe mensal do melhor cenario
cands=[(r1,'EMA sem limite'),(r2,'ESTRUTURA sem limite'),
       (r3,'ESTRUTURA teto 4'),(r4,'EMA teto 4')]
best,best_lbl=max(cands, key=lambda x:x[0]['balance'])

print('\n' + '='*128)
print(f'  MELHOR CENARIO: {best_lbl}  ->  {best["balance"]:,.0f} USDC  '
      f'({(best["balance"]/INITIAL_BALANCE-1)*100:+.1f}%)  |  DD max {best["max_dd"]:.1f}%')
print('='*128)
months={}
for t in best['trades']:
    months.setdefault(t['ts_out'].strftime('%Y-%m'), []).append(t)
ML=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
running=INITIAL_BALANCE
print(f"  {'Mes':<8} | {'Trades':>6} | {'WR':>5} | {'Saldo fim':>14}")
print('  '+'-'*48)
# recalcular saldo mensal a partir dos R (aprox.: usa risco fixo sobre saldo corrente)
# nota: para fidelidade total o saldo vem do motor; aqui mostramos contagem/WR por mes
for mk in sorted(months):
    ts=months[mk]; w=[x for x in ts if x['nr']>0]
    yr,mo=int(mk[:4]),int(mk[5:])
    print(f"  {ML[mo-1]}/{yr%100:02d} | {len(ts):>6} | {len(w)/len(ts)*100:>4.0f}% | "
          f"{'':>14}")

print('\n' + '='*128)
print('  LEITURA')
print('='*128)
print('  • Cenario 1 (EMA sem limite) = o teste anterior. Compara tudo contra ele.')
print('  • 1 vs 2: efeito PURO de trocar o Decisor (EMA -> estrutura), mesmo motor.')
print('  • 2 vs 3: efeito do teto de exposicao (controla a correlacao).')
print('  • Bull SMC continua HIPOTETICA. Custos 0.14% round-trip incluidos.')
