"""
Registry central de estrategias.

Mapeia o nome (usado em STRATEGY_MODE, na API e no dashboard) para o modulo que
implementa `analyze(df_entry, df_daily, last_trade_bar)`.

Adicionar uma estrategia nova ao bot e a 2 passos:
  1. Criar o modulo em bot/strategies/<nome>.py com a funcao analyze(...)
  2. Regista-lo aqui no REGISTRY

Depois, para o Decisor (bot/regime.py) passar a usa-la num regime, basta ligar
o regime ao nome em REGIME_STRATEGY. Nada mais precisa mudar no engine.
"""
from . import breakout_short, bear_v12, bear_v13, bull_v1
from . import lateral_breakout, bull_smc, bear_breakout   # sistema v2/v3 (validado em 5 anos)

REGISTRY = {
    "breakout_short": breakout_short,
    "bear_v12":       bear_v12,
    "bear_v13":       bear_v13,
    "bull_v1":        bull_v1,
    # ── sistema v2/v3 — 3 direcoes robustas (breakout de range consolidado) ────
    "lateral_breakout": lateral_breakout,   # NEUTRAL — nucleo robusto (4/5 anos)
    "bull_smc":         bull_smc,            # BULL — complemento
    "bear_breakout":    bear_breakout,       # BEAR — bear reabilitado (seletivo)
}


def get_strategy(name: str):
    """Devolve o modulo da estrategia pelo nome, ou None se nao estiver registada."""
    return REGISTRY.get(name)


def is_valid(name: str) -> bool:
    """True se o nome corresponde a uma estrategia registada."""
    return name in REGISTRY
