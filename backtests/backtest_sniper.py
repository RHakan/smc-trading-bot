#!/usr/bin/env python3
"""
backtest_sniper.py — Backtest Estratégia Sniper (Sweep & Reclaim, 15m)

Executa matriz de ablação 2×2×2:
  BB%b filter (ON/OFF) × Volume filter (ON/OFF) × Exit mode (Trailing/Fixed)

Métricas por configuração (agregadas em todos os pares):
  trades, win%, avg_r, total_r, profit_factor, max_drawdown_r, t/day, be%, best/worst

Uso:
  python backtest_sniper.py

Resultado salvo em:
  C:\\Users\\ASUS\\OneDrive\\Documentos\\Claudinho\\Outputs\\[data]_sniper_backtest.json
"""

import json
import time
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — ajuste aqui sem mexer no resto do código
# ─────────────────────────────────────────────────────────────────────────────

PAIRS = [
    "BTC/USDT:USDT",  "ETH/USDT:USDT",  "SOL/USDT:USDT",  "BNB/USDT:USDT",
    "XRP/USDT:USDT",  "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT",
    "DOT/USDT:USDT",  "LINK/USDT:USDT",
]

TIMEFRAME   = "15m"
DAYS        = 90        # período do backtest em dias

# Parâmetros da estratégia (idênticos ao strategies.json)
SWING_LB    = 10        # lookback para swing highs/lows (barras de cada lado)
COOLDOWN    = 12        # barras mínimas entre trades por par (12 × 15m = 3h)
RR_MIN      = 2.0       # R:R mínimo obrigatório
STOP_BUF    = 0.25      # buffer ATR para o stop além do extremo do pavio
BB_SHORT    = 0.85      # BB%b mínimo para sinal SHORT
BB_LONG     = 0.15      # BB%b máximo para sinal LONG
VOL_MULT    = 1.2       # mínimo de volume vs SMA para considerar spike

# Parâmetros dos indicadores
ATR_P       = 14
BB_P        = 20
BB_STD      = 2.0
VOL_P       = 20

# Gestão de posição (trailing)
BE_TRIGGER  = 0.50      # % do caminho até TP para mover stop a breakeven
TRAIL_ATR   = 1.5       # multiplicador ATR para trailing stop

# Custos (por lado)
FEE_PCT     = 0.0005    # 0.05% taker Binance Futures
SLIP_PCT    = 0.0002    # 0.02% slippage estimado
ROUND_TRIP  = (FEE_PCT + SLIP_PCT) * 2   # custo total = 0.14% round-trip

# Onde salvar o resultado
OUTPUTS_DIR = Path(r"C:\Users\ASUS\OneDrive\Documentos\Claudinho\Outputs")


# ─────────────────────────────────────────────────────────────────────────────
# EXCHANGE — dados históricos públicos (sem API key)
# ─────────────────────────────────────────────────────────────────────────────

def get_exchange() -> ccxt.Exchange:
    return ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, days: int) -> pd.DataFrame:
    """Busca dados históricos paginando para cobrir o período completo."""
    since_ms    = int((datetime.now(timezone.utc) - timedelta(days=days + 2)).timestamp() * 1000)
    limit_total = days * 96 + 300   # 96 candles 15m/dia + margem

    rows = []
    since = since_ms
    while len(rows) < limit_total:
        batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1500)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1        # próximo lote após o último candle
        if len(batch) < 1500:
            break
        time.sleep(0.05)               # respeita rate limit

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.iloc[-limit_total:]      # mantém apenas o período pedido


# ─────────────────────────────────────────────────────────────────────────────
# INDICADORES — mesma lógica do bot/indicators.py
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ATR (Wilder / EWM)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_P - 1, adjust=False).mean()

    # BB %b baseado em OHLC4 (igual ao bot)
    ohlc4  = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    mid    = ohlc4.rolling(BB_P).mean()
    std    = ohlc4.rolling(BB_P).std()
    upper  = mid + BB_STD * std
    lower  = mid - BB_STD * std
    df["bb_pct_b"] = (ohlc4 - lower) / (upper - lower).replace(0, np.nan)

    # Volume ratio vs SMA
    vol_sma        = df["volume"].rolling(VOL_P).mean()
    df["vol_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SWING DETECTION — mesma lógica do bot/indicators.py
# ─────────────────────────────────────────────────────────────────────────────

def _swing_high_indices(highs: np.ndarray, lb: int) -> np.ndarray:
    result = []
    for i in range(lb, len(highs) - lb):
        if highs[i] == max(highs[i - lb: i + lb + 1]):
            result.append(i)
    return np.array(result, dtype=int)


def _swing_low_indices(lows: np.ndarray, lb: int) -> np.ndarray:
    result = []
    for i in range(lb, len(lows) - lb):
        if lows[i] == min(lows[i - lb: i + lb + 1]):
            result.append(i)
    return np.array(result, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO DE SINAL — mesma lógica do bot/strategies/sniper.py
# ─────────────────────────────────────────────────────────────────────────────

def detect_signal(
    df: pd.DataFrame,
    bar_idx: int,
    use_bb: bool,
    use_vol: bool,
) -> dict | None:
    """
    Analisa a barra bar_idx (já fechada) para sinal Sniper.
    Retorna dict com side/entry/sl/tp/rr/atr ou None se sem sinal.

    Usa apenas barras 0..bar_idx (sem look-ahead bias).
    """
    window = df.iloc[: bar_idx + 1]
    last   = window.iloc[-1]

    atr        = last["atr"]
    bb_pct_b   = last["bb_pct_b"]
    vol_ratio  = last["vol_ratio"]
    close      = float(last["close"])

    # Aguarda indicadores estabilizarem
    if any(np.isnan(v) for v in [atr, bb_pct_b, vol_ratio]):
        return None

    highs_arr = window["high"].values
    lows_arr  = window["low"].values
    sh_idx    = _swing_high_indices(highs_arr, SWING_LB)
    sl_idx    = _swing_low_indices(lows_arr,   SWING_LB)

    # ── SHORT: pavio rompe swing high anterior, corpo fecha abaixo ────────
    if len(sh_idx) > 0:
        sw_h = float(highs_arr[sh_idx[-1]])      # swing high mais recente

        wick_swept     = float(last["high"]) > sw_h
        body_reclaimed = close < sw_h

        if wick_swept and body_reclaimed:
            sl    = float(last["high"]) + STOP_BUF * float(atr)
            entry = close
            risk  = sl - entry

            if risk > 0:
                # Target: swing low abaixo do entry, senão R:R mínimo
                candidates = [float(lows_arr[j]) for j in sl_idx if lows_arr[j] < entry]
                tp = candidates[-1] if candidates else entry - risk * RR_MIN

                rr = (entry - tp) / risk if tp < entry else 0.0

                bb_ok  = (bb_pct_b  > BB_SHORT)  if use_bb  else True
                vol_ok = (vol_ratio >= VOL_MULT)  if use_vol else True

                if bb_ok and vol_ok and rr >= RR_MIN:
                    return {"side": "short", "entry": entry, "sl": sl, "tp": tp,
                            "rr": rr, "atr": float(atr)}

    # ── LONG: pavio rompe swing low anterior, corpo fecha acima ──────────
    if len(sl_idx) > 0:
        sw_l = float(lows_arr[sl_idx[-1]])        # swing low mais recente

        wick_swept     = float(last["low"]) < sw_l
        body_reclaimed = close > sw_l

        if wick_swept and body_reclaimed:
            sl    = float(last["low"]) - STOP_BUF * float(atr)
            entry = close
            risk  = entry - sl

            if risk > 0:
                candidates = [float(highs_arr[j]) for j in sh_idx if highs_arr[j] > entry]
                tp = candidates[-1] if candidates else entry + risk * RR_MIN

                rr = (tp - entry) / risk if tp > entry else 0.0

                bb_ok  = (bb_pct_b  < BB_LONG)   if use_bb  else True
                vol_ok = (vol_ratio >= VOL_MULT)  if use_vol else True

                if bb_ok and vol_ok and rr >= RR_MIN:
                    return {"side": "long", "entry": entry, "sl": sl, "tp": tp,
                            "rr": rr, "atr": float(atr)}

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIMULAÇÃO DO TRADE — replica o bot/position_manager.py
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trade(
    df: pd.DataFrame,
    entry_bar: int,
    sig: dict,
    use_trailing: bool,
) -> dict:
    """
    Simula a evolução do trade a partir da barra entry_bar+1.

    Gestão de posição (trailing mode):
      Fase 1 — a 50% do caminho ao TP → move stop a breakeven
      Fase 2 — trailing 1.5×ATR abaixo/acima do preço atual
      Fase 3 — fecha 50% no TP, bloqueia stop a 2R de lucro

    Fixed mode: sai exatamente no TP ou SL (sem trailing).

    Dentro de uma barra ambígua (SL e TP na mesma barra), assume SL atingido
    primeiro (abordagem conservadora, standard em backtesting).

    Retorna dict com outcome, exit_bar, exit_price, gross_r, net_r, hit_be.
    """
    entry = sig["entry"]
    sl    = sig["sl"]
    tp    = sig["tp"]
    side  = sig["side"]
    atr0  = sig["atr"]         # ATR do candle de entrada (fallback)
    risk  = abs(entry - sl)

    # Custo round-trip em unidades de R (afeta net_r mas não gross_r)
    fee_r = entry * ROUND_TRIP / risk

    current_stop    = sl
    be_hit          = False
    partial_closed  = False

    for j in range(entry_bar + 1, len(df)):
        bar  = df.iloc[j]
        high = float(bar["high"])
        low  = float(bar["low"])
        close= float(bar["close"])
        atr  = float(bar["atr"]) if not np.isnan(bar["atr"]) else atr0

        # ── Verificação de stop (sempre primeiro) ──────────────────────────
        if side == "short":
            stop_hit = high >= current_stop
        else:
            stop_hit = low <= current_stop

        if stop_hit:
            exit_price = current_stop
            gross_r = (entry - exit_price) / risk if side == "short" else (exit_price - entry) / risk
            net_r   = gross_r - fee_r
            outcome = "WIN" if (
                (side == "short" and exit_price < entry) or
                (side == "long"  and exit_price > entry)
            ) else "LOSS"
            return {"outcome": outcome, "exit_bar": j, "exit_price": exit_price,
                    "gross_r": gross_r, "net_r": net_r, "hit_be": be_hit}

        # ── Modo Fixed: saída exata no TP ──────────────────────────────────
        if not use_trailing:
            tp_hit = (low <= tp) if side == "short" else (high >= tp)
            if tp_hit:
                gross_r = (entry - tp) / risk if side == "short" else (tp - entry) / risk
                net_r   = gross_r - fee_r
                return {"outcome": "WIN", "exit_bar": j, "exit_price": tp,
                        "gross_r": gross_r, "net_r": net_r, "hit_be": be_hit}
            continue

        # ── Modo Trailing ──────────────────────────────────────────────────
        if side == "short":
            progress = (entry - close) / abs(entry - tp) if tp < entry else 0.0

            # Fase 3 — fechamento parcial no TP (short: preço desceu até tp)
            if not partial_closed and low <= tp:
                partial_closed   = True
                locked_stop      = entry - 2.0 * risk   # bloqueia 2R de lucro
                if current_stop > locked_stop:
                    current_stop = locked_stop

            # Fase 2 — trailing (aperta stop enquanto preço desce)
            if be_hit and progress >= BE_TRIGGER:
                candidate = close + TRAIL_ATR * atr     # acima do preço (SHORT)
                if candidate < current_stop:
                    current_stop = candidate

            # Fase 1 — breakeven a 50% do caminho
            if not be_hit and progress >= BE_TRIGGER:
                if entry < current_stop:                # move stop para entry
                    current_stop = entry
                be_hit = True

        else:  # LONG
            progress = (close - entry) / abs(tp - entry) if tp > entry else 0.0

            # Fase 3 — fechamento parcial no TP (long: preço subiu até tp)
            if not partial_closed and high >= tp:
                partial_closed   = True
                locked_stop      = entry + 2.0 * risk   # bloqueia 2R de lucro
                if current_stop < locked_stop:
                    current_stop = locked_stop

            # Fase 2 — trailing (aperta stop enquanto preço sobe)
            if be_hit and progress >= BE_TRIGGER:
                candidate = close - TRAIL_ATR * atr     # abaixo do preço (LONG)
                if candidate > current_stop:
                    current_stop = candidate

            # Fase 1 — breakeven
            if not be_hit and progress >= BE_TRIGGER:
                if entry > current_stop:
                    current_stop = entry
                be_hit = True

    # ── Trade ainda aberto no fim dos dados — fecha no último close ────────
    last_close = float(df.iloc[-1]["close"])
    gross_r = (entry - last_close) / risk if side == "short" else (last_close - entry) / risk
    net_r   = gross_r - fee_r
    return {"outcome": "OPEN", "exit_bar": len(df) - 1, "exit_price": last_close,
            "gross_r": gross_r, "net_r": net_r, "hit_be": be_hit}


# ─────────────────────────────────────────────────────────────────────────────
# LOOP DE BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

WARMUP = SWING_LB * 2 + max(ATR_P, BB_P, VOL_P) + 10   # barras de aquecimento


def run_backtest(
    df: pd.DataFrame,
    pair: str,
    use_bb: bool,
    use_vol: bool,
    use_trail: bool,
) -> list[dict]:
    """Percorre o DataFrame barra a barra e simula todos os trades."""
    df = add_indicators(df)
    trades           = []
    last_trade_bar   = -9999

    for i in range(WARMUP, len(df) - 1):
        # Cooldown: evita trades consecutivos no mesmo par
        if i - last_trade_bar < COOLDOWN:
            continue

        sig = detect_signal(df, i, use_bb=use_bb, use_vol=use_vol)
        if sig is None:
            continue

        result = simulate_trade(df, i, sig, use_trailing=use_trail)

        trades.append({
            "pair"      : pair,
            "bar_idx"   : i,
            "timestamp" : str(df.index[i]),
            "side"      : sig["side"],
            "entry"     : round(sig["entry"],   6),
            "sl"        : round(sig["sl"],      6),
            "tp"        : round(sig["tp"],      6),
            "rr_target" : round(sig["rr"],      2),
            "outcome"   : result["outcome"],
            "exit_bar"  : result["exit_bar"],
            "exit_price": round(result["exit_price"], 6),
            "gross_r"   : round(result["gross_r"], 3),
            "net_r"     : round(result["net_r"],    3),
            "hit_be"    : result["hit_be"],
        })
        last_trade_bar = i

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def calc_metrics(trades: list[dict], total_days: int) -> dict:
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "open": 0,
            "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0,
            "profit_factor": 0.0, "max_drawdown_r": 0.0,
            "trades_per_day": 0.0, "pct_be": 0.0,
            "best_r": 0.0, "worst_r": 0.0,
            "longs": 0, "shorts": 0,
        }

    closed = [t for t in trades if t["outcome"] != "OPEN"]
    wins   = [t for t in closed if t["outcome"] == "WIN"]
    losses = [t for t in closed if t["outcome"] == "LOSS"]
    n_open = sum(1 for t in trades if t["outcome"] == "OPEN")

    net_rs = [t["net_r"] for t in closed]
    win_rate     = len(wins) / len(closed) * 100 if closed else 0.0
    avg_r        = float(np.mean(net_rs)) if net_rs else 0.0
    total_r      = float(sum(net_rs))

    gross_wins   = sum(t["net_r"] for t in wins   if t["net_r"] > 0)
    gross_losses = abs(sum(t["net_r"] for t in losses if t["net_r"] < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Max drawdown em R (sequência de perdas acumuladas)
    cum_r  = np.cumsum(net_rs) if net_rs else np.array([0.0])
    peak   = np.maximum.accumulate(cum_r)
    max_dd = float(np.max(peak - cum_r))

    pct_be = sum(1 for t in trades if t.get("hit_be")) / len(trades) * 100

    return {
        "total_trades"   : len(trades),
        "wins"           : len(wins),
        "losses"         : len(losses),
        "open"           : n_open,
        "win_rate"       : round(win_rate, 1),
        "avg_r"          : round(avg_r, 3),
        "total_r"        : round(total_r, 2),
        "profit_factor"  : round(profit_factor, 2),
        "max_drawdown_r" : round(max_dd, 2),
        "trades_per_day" : round(len(trades) / total_days, 2),
        "pct_be"         : round(pct_be, 1),
        "best_r"         : round(max(net_rs) if net_rs else 0.0, 2),
        "worst_r"        : round(min(net_rs) if net_rs else 0.0, 2),
        "longs"          : sum(1 for t in trades if t["side"] == "long"),
        "shorts"         : sum(1 for t in trades if t["side"] == "short"),
    }


def calc_pair_breakdown(trades: list[dict], total_days: int) -> list[dict]:
    """Métricas por par para a configuração default."""
    pairs = sorted(set(t["pair"] for t in trades))
    result = []
    for pair in pairs:
        pt = [t for t in trades if t["pair"] == pair]
        m  = calc_metrics(pt, total_days)
        short_sym = pair.replace("/USDT:USDT", "")
        result.append({"pair": short_sym, **m})
    return sorted(result, key=lambda x: x["total_r"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY — saída formatada no terminal
# ─────────────────────────────────────────────────────────────────────────────

def _cfg_label(use_bb: bool, use_vol: bool, use_trail: bool) -> str:
    bb    = "BB✓"    if use_bb    else "BB✗"
    vol   = "Vol✓"   if use_vol   else "Vol✗"
    trail = "Trail"  if use_trail else "Fixed"
    return f"{bb} {vol} {trail}"


def print_ablation_table(results: list[dict]) -> None:
    hdr = (
        f"  {'Configuração':<18} {'Trades':>6} {'W%':>5} {'AvgR':>6} "
        f"{'TotalR':>7} {'PF':>5} {'MaxDD':>6} {'T/Day':>5} {'BE%':>5}"
    )
    sep = "  " + "─" * (len(hdr) - 2)

    print()
    print("  " + "═" * (len(hdr) - 2))
    print("  SNIPER BACKTEST — Matriz de Ablação 2×2×2")
    print(f"  {DAYS} dias | {len(PAIRS)} pares | fees {ROUND_TRIP*100:.2f}% RT | "
          f"RR≥{RR_MIN} | cooldown {COOLDOWN}×15m")
    print("  " + "═" * (len(hdr) - 2))
    print(hdr)
    print(sep)

    # Ordena por total_r decrescente
    for r in sorted(results, key=lambda x: x["metrics"]["total_r"], reverse=True):
        m    = r["metrics"]
        pf   = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else " ∞"
        flag = " ◄" if r["use_bb"] and r["use_vol"] and r["use_trail"] else ""
        print(
            f"  {r['config']:<18} {m['total_trades']:>6} "
            f"{m['win_rate']:>4.0f}% {m['avg_r']:>+6.2f} "
            f"{m['total_r']:>+7.1f} {pf:>5} "
            f"{m['max_drawdown_r']:>6.1f} {m['trades_per_day']:>5.2f} "
            f"{m['pct_be']:>4.0f}%{flag}"
        )

    print(sep)
    print("  ◄ = configuração padrão do bot")
    print()


def print_pair_table(breakdown: list[dict]) -> None:
    hdr = (
        f"  {'Par':<8} {'Trades':>6} {'W%':>5} {'AvgR':>6} "
        f"{'TotalR':>7} {'Long':>5} {'Short':>5}"
    )
    sep = "  " + "─" * (len(hdr) - 2)

    print("  " + "─" * (len(hdr) - 2))
    print("  Por Par — Configuração Padrão (BB✓ Vol✓ Trail)")
    print(sep)
    print(hdr)
    print(sep)

    for p in breakdown:
        print(
            f"  {p['pair']:<8} {p['total_trades']:>6} "
            f"{p['win_rate']:>4.0f}% {p['avg_r']:>+6.2f} "
            f"{p['total_r']:>+7.1f} {p['longs']:>5} {p['shorts']:>5}"
        )

    print(sep)
    print()


def print_sample_trades(trades: list[dict], n: int = 8) -> None:
    """Mostra os N melhores e N piores trades da configuração padrão."""
    closed = [t for t in trades if t["outcome"] != "OPEN"]
    if not closed:
        return

    best  = sorted(closed, key=lambda x: x["net_r"], reverse=True)[:n]
    worst = sorted(closed, key=lambda x: x["net_r"])[:n]

    def _fmt(t: dict) -> str:
        ts   = t["timestamp"][:16].replace("T", " ")
        side = "▲ LONG " if t["side"] == "long" else "▼ SHORT"
        sym  = t["pair"].replace("/USDT:USDT", "")
        return (f"  {sym:<5} {side}  {ts}  "
                f"entry {t['entry']:.4f}  "
                f"net {t['net_r']:>+.2f}R  ({t['outcome']})")

    print("  Melhores trades (configuração padrão):")
    for t in best:
        print(_fmt(t))
    print()
    print("  Piores trades (configuração padrão):")
    for t in worst:
        print(_fmt(t))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("  ┌─────────────────────────────────────┐")
    print("  │  Backtest Sniper — Sweep & Reclaim  │")
    print("  └─────────────────────────────────────┘")
    print()

    # 1. Fetch de dados históricos ─────────────────────────────────────────
    ex = get_exchange()
    print(f"  Buscando {DAYS} dias de candles 15m para {len(PAIRS)} pares...", end="", flush=True)

    dfs: dict[str, pd.DataFrame] = {}
    for sym in PAIRS:
        try:
            dfs[sym] = fetch_ohlcv(ex, sym, DAYS)
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n  ⚠  Erro ao buscar {sym}: {e}", file=sys.stderr)
    print(" OK\n")

    if not dfs:
        print("  Nenhum dado carregado. Verifica a conexão com a internet.")
        return

    # 2. Matriz de ablação ─────────────────────────────────────────────────
    ablations = [
        (use_bb, use_vol, use_trail)
        for use_bb    in (True, False)
        for use_vol   in (True, False)
        for use_trail in (True, False)
    ]

    results       = []
    default_trades: list[dict] = []

    for idx, (use_bb, use_vol, use_trail) in enumerate(ablations, 1):
        label = _cfg_label(use_bb, use_vol, use_trail)
        print(f"  [{idx}/{len(ablations)}] {label:<22}", end="", flush=True)

        combo_trades: list[dict] = []
        for sym, df in dfs.items():
            combo_trades.extend(run_backtest(df, sym, use_bb, use_vol, use_trail))

        metrics = calc_metrics(combo_trades, DAYS)
        results.append({
            "use_bb"   : use_bb,
            "use_vol"  : use_vol,
            "use_trail": use_trail,
            "config"   : label,
            "metrics"  : metrics,
        })

        # Guarda trades da configuração padrão para breakdown e exemplos
        if use_bb and use_vol and use_trail:
            default_trades = combo_trades

        print(f"→ {metrics['total_trades']:>4} trades | "
              f"W%={metrics['win_rate']:4.0f}% | "
              f"TotalR={metrics['total_r']:>+7.1f}")

    # 3. Output ────────────────────────────────────────────────────────────
    print_ablation_table(results)

    if default_trades:
        breakdown = calc_pair_breakdown(default_trades, DAYS)
        print_pair_table(breakdown)
        print_sample_trades(default_trades, n=5)

    # 4. Salvar JSON ───────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    today    = datetime.now().strftime("%Y-%m-%d")
    out_path = OUTPUTS_DIR / f"{today}_sniper_backtest.json"

    output = {
        "generated_at"     : datetime.now(timezone.utc).isoformat(),
        "days"             : DAYS,
        "pairs"            : PAIRS,
        "fee_roundtrip_pct": round(ROUND_TRIP * 100, 4),
        "params": {
            "rr_min"            : RR_MIN,
            "stop_atr_buffer"   : STOP_BUF,
            "swing_lookback"    : SWING_LB,
            "cooldown_bars"     : COOLDOWN,
            "bb_short_threshold": BB_SHORT,
            "bb_long_threshold" : BB_LONG,
            "volume_spike_mult" : VOL_MULT,
            "be_trigger"        : BE_TRIGGER,
            "trail_atr_mult"    : TRAIL_ATR,
        },
        "ablation_results"    : results,
        "pair_breakdown"      : calc_pair_breakdown(default_trades, DAYS) if default_trades else [],
        "trades_default_config": default_trades,
    }

    out_path.write_text(
        json.dumps(output, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  ✔  Resultados salvos em:")
    print(f"     {out_path}")
    print()


if __name__ == "__main__":
    main()
