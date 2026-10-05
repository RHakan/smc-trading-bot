"""
Metadados descritivos das estratégias — usados no tooltip/card do dropdown do
dashboard (o que cada versão faz, perfil de risco, taxa de acerto, referência
de backtest).

IMPORTANTE: esta é a FONTE ÚNICA da descrição das estratégias. Sempre que uma
estratégia for atualizada (parâmetros, lógica, resultado de backtest), ATUALIZA
o respetivo bloco aqui — o dashboard lê daqui automaticamente via /api/strategies.

Campos:
  label      : nome amigável (aparece no dropdown)
  direction  : "bull" | "bear" | "neutral" | "multi"
  timeframe  : timeframe de análise (ex: "1H")
  style      : "conservadora" | "moderada" | "agressiva"
  summary    : 1–2 frases do que a estratégia faz
  win_rate   : taxa de acerto do backtest em % (None = ainda por confirmar)
  backtest   : string de referência do backtest (retorno/trades/período)
  highlights : lista de pontos-chave (ex: RR, frequência, drawdown)

Os números de risco/exposição (risco por trade, máx. posições, % de capital) NÃO
ficam aqui — são calculados ao vivo pelo endpoint a partir da config (.env), para
refletirem sempre o estado atual do bot.
"""

# Ordem de exibição = ordem do dropdown no dashboard.
STRATEGY_META: dict[str, dict] = {
    "auto_v2": {
        "label":     "🚀 Auto v2 — 3 direções (breakout)",
        "direction": "multi",
        "timeframe": "1H",
        "style":     "conservadora",
        "summary":   "Sistema robusto de 3 direções: segue rompimentos de range "
                     "consolidado em bull, bear e lateral. O núcleo (lateral) foi "
                     "validado em 5 anos de dados.",
        "win_rate":  None,   # a confirmar (backtest 5 anos)
        "backtest":  "Validado 2021–2025 · 4/5 anos positivos · DD máx ~6.6% (núcleo lateral)",
        "highlights": [
            "Combina lateral_breakout (neutral) + bull_smc (bull) + bear_breakout (bear)",
            "Entra só após consolidação (ADX < 20) — filtra ruído/stop-hunt",
            "A opção mais robusta e conservadora do bot",
        ],
    },
    "auto": {
        "label":     "🤖 Auto — Decisor de Regime",
        "direction": "multi",
        "timeframe": "1H",
        "style":     "moderada",
        "summary":   "Decisor que escolhe a estratégia conforme o regime de mercado "
                     "detetado (alta, baixa ou lateral) e opera na direção adequada.",
        "win_rate":  None,
        "backtest":  "Versão anterior ao sistema v2 — ver arquitetura multi-regime",
        "highlights": [
            "Adapta a direção ao regime atual do mercado",
            "Antecessor do Auto v2 (preferir o v2 pela robustez de 5 anos)",
        ],
    },
    "bear_v13": {
        "label":     "Bear Market — v1.3 (1H)",
        "direction": "bear",
        "timeframe": "1H",
        "style":     "agressiva",
        "summary":   "Shorta rompimentos de mínimas em tendência de baixa. Versão de "
                     "alta frequência: entra cedo e reentra rápido em continuações.",
        "win_rate":  None,   # a confirmar
        "backtest":  "1014 trades · +502% (Jan–Jun 2026)",
        "highlights": [
            "Alta frequência — cooldown curto (3H), entra 1 barra mais cedo",
            "RR alvo 2.5 · breakeven a 70% · trailing 2×ATR",
            "Otimizada para Jan–Jun 2026 — menos robusta em janelas longas",
        ],
    },
    "bear_v12": {
        "label":     "Bear Market — v1.2 (1H)",
        "direction": "bear",
        "timeframe": "1H",
        "style":     "agressiva",
        "summary":   "Base da v1.3, com trailing mais largo e alvo mais ambicioso "
                     "(deixa correr mais os movimentos de queda).",
        "win_rate":  None,
        "backtest":  "680 trades · +388% (Jan–Jun 2026)",
        "highlights": [
            "Trailing 2×ATR · RR alvo até 4.0 (mais ambicioso que a v1.3)",
            "Menos trades que a v1.3, deixa correr mais cada um",
        ],
    },
    "breakout_short": {
        "label":     "Bear Market — v1.0 (1H)",
        "direction": "bear",
        "timeframe": "1H",
        "style":     "moderada",
        "summary":   "Versão original: shorta o rompimento do mínimo das últimas 4 "
                     "velas em tendência de baixa, com alvo fixo em RR 3:1.",
        "win_rate":  None,
        "backtest":  "+156% (Jan–Jun 2026)",
        "highlights": [
            "RR 3:1 fixo · trailing após breakeven",
            "A mais simples/direta das versões bear",
        ],
    },
}


def get_meta(name: str) -> dict | None:
    """Metadados de uma estratégia pelo nome (None se não existir)."""
    return STRATEGY_META.get(name)


def all_meta() -> dict[str, dict]:
    """Todos os metadados, na ordem de exibição do dropdown."""
    return STRATEGY_META
