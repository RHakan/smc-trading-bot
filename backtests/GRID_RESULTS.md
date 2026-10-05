# Sistema 4 — Grid / HFT: Diário da Campanha

**Goal (Rafa, jul/2026):** encontrar uma estratégia de market-making / grid / scalping
de alta frequência que persiga **100 USDC/dia**, documentando o melhor resultado.
Cada versão é testada em **períodos de mercado diferentes** (2021 bull, 2022 bear,
2023 recuperação, 2024 bull, 2025 misto, 2026 OOS) para evitar falso resultado.

**Âncora de honestidade:** 100/dia sobre 500 USDC = 20%/dia — impossível de forma
sustentável. Medimos o **%/dia real** e calculamos o **capital necessário** para
100/dia com esse rendimento. O melhor resultado honesto é o que fica documentado.

**Regras da campanha**
- Sistema separado: NADA do v3 (auto_v2) é alterado.
- Custos: maker 0,02%/lado (ordens limit) = 0,04% round-trip; liquidação de stop
  a taker+slip 0,07%. Sem alavancagem por defeito (lev multiplica P&L E DD).
- Todo teste sai com levantamento mensal (regra do projeto).
- Só passa de versão o que for positivo em MÚLTIPLOS períodos, não num só.

---

## Placar (melhor resultado até agora)

| Versão | Estratégia | Melhor config | USDC/dia (500) | %/dia | Capital p/ 100/dia | Períodos + |
|---|---|---|---|---|---|---|
| v1 | Grid long clássico | ETH step 1,2% R10% stop | +0,41 | 0,081% | ~123.000 USDC | 5/6 (só 2026 −52) |
| v2 | Grid two-sided (sizing por equity) | ETH step 0,8% R10% two_sided | +0,39 | 0,078% | ~129.000 USDC | 6/6 — incl. 2022 (+154) e 2026 OOS (+75) |
| **v3** | **v2 + alavancagem** | **ETH two_sided 2x ★ CAMPEÃ** | **+1,06** | **0,212%** | **~47.000 USDC** | **6/6, DD 37,9%, pior mês −201 (dentro dos −250)** |
| v4 | TF 5m + equity-trailing | — nenhuma bateu a v3 | — | — | — | 5m REPROVADO; trail REPROVADO no 15m |

---

## v1 — Grid long clássico (15m, BTC/ETH/SOL)

`backtest_grid_v1.py` — grid estilo Pionex: K níveis de compra abaixo do centro
(step 0,5/0,8/1,2%; half-range 10/20%), cada compra vende 1 step acima; recentra
para cima quando zera o inventário; no fundo do range: `hold` (segura) vs `stop`
(liquida e recentra).

**Resultados (03/07/2026):**
- Melhor por consistência: **ETH step 1,2% R10% stop → +816 em 2005 dias (0,081%/dia), 5/6 períodos+, DD 24,9%**. Mensal: 55% dos meses positivos.
- SOL `hold` parece enorme (+2.344) mas é ilusão: 2021 parabólico paga tudo (+2.430) e 2022/25/26 são negativos — padrão "um ano salva" que já conhecemos.
- **Achado estrutural: `hold` ganha em bull (buy-dips sempre recupera), `stop` protege em bear** — a política ideal do fundo depende do REGIME. Ponte natural com o Decisor que já temos.
- **2026 OOS: TODAS as 36 configs negativas** — bear/transição volátil é hostil ao grid long. Mesmo semestre que castigou o v3.
- **Bug de contabilidade descoberto:** sizing fixo (500/K por nível) não se ajusta à equity → após perdas em série, o grid "alavanca" sozinho (SOL stop 0,5%: DD 242% = conta quebrada). v2 corrige: per_level = equity/K.
- Step maior (1,2%) > menor (0,5%): menos round-trips, mas cada um paga mais vs custo fixo de 0,04%.
- Meta 100/dia: com 0,081%/dia seriam precisos ~123k USDC. v1 fica LONGE da meta.

**Limitações conhecidas da v1 (candidatas a refino na v2):**
- Fills intra-candle otimistas: OHLC de 15m não diz a ordem dos toques; a
  simulação pode superestimar round-trips em candles largos (mitigado: venda só
  de posições de candles anteriores). v2 pode exigir range do candle > 2×step.
- Grid é long-only: em bear prolongado, `hold` prende capital, `stop` realiza
  perdas em série. Grid neutro (shorts acima / longs abaixo) é candidato à v2.
- Sem alavancagem nem liquidação modeladas.

## v2 — Grid two-sided + sizing por equity + modo regime (03/07/2026)

`backtest_grid_v2.py` — 3 modos comparados: `long_v1` (referência), `two_sided`
(shorts acima + longs abaixo, sempre), `regime` (lados ligados pelo regime EMA).

**Resultados:**
- **VENCEDORA: ETH step 0,8% R10% two_sided → +778, 6/6 períodos positivos**,
  DD 22%, ~65% dos meses positivos. Virou 2022 (+154) e **2026 OOS (+75)** —
  o grid é o primeiro sistema da casa positivo no OOS 2026 (o v3 swing deu −28).
- Surpresa: o modo `regime` NÃO bateu o two_sided puro em BTC/ETH (só ajudou no
  SOL, mais tendencioso). Simplicidade venceu de novo — desligar lados perde
  round-trips e o lag do regime custa mais do que protege.
- `long_v1` mostra totais gigantes em BTC/SOL mas é bull-dependente (2022/2026
  negativos, DD 70-82% no SOL) — descartado por inconsistência.
- Rendimento ~igual à v1 (0,078%/dia) mas MUITO mais robusto. Meta 100/dia
  continua exigindo ~129k de capital (ou alavancagem, que multiplica o DD).

**Leitura estratégica:** grid two-sided e o v3 swing são COMPLEMENTARES — o grid
lucra no chop que castiga o swing (2026), o swing captura as tendências que o
grid não pega. Portfólio dos dois sistemas é caminho natural.

## v3 — Multi-par + alavancagem (03/07/2026)

`backtest_grid_v3.py` — config v2 fixa (two-sided 0,8%/R10), 3 portfólios × lev 1/2/3x,
500 TOTAL, DD agregado diário, quebra modelada (equity do par ≤5% = liquidação).

**Resultados:**
- **ETH 3x: +3.920, 6/6 períodos, 0,391%/dia → ~25.600 p/ 100/dia. MAS DD 51,7%**
  e piores meses de −369/−339/−283 sobre 500 — **viola a tolerância de −250/mês
  do Rafa** (no modo acumulativo testado).
- **ETH 2x: +2.130, 6/6, 0,212%/dia, DD 37,9% → ~47k p/ 100/dia.** Piores meses
  ~2/3 dos do 3x (≈−250) — no LIMITE da tolerância. Candidato mais realista.
- **Diversificação NÃO ajudou:** BTC dilui o rendimento (grid rende menos nele)
  na mesma proporção que reduz DD; **SOL QUEBRA a 2x/3x** (1 liquidação) — fora.
- Escala superlinear no retorno (sizing por equity compõe) e sublinear no DD —
  a alavancagem no grid é mais eficiente que o esperado, mas o DD de 50%+ no 3x
  é real e longo.

**Estado da meta:** v1 ~123k → v3 ~25,6k de capital p/ 100/dia (4,8x de melhoria
na campanha). Com os 500 atuais: ~2 USDC/dia no 3x.

## v4 — TF 5m + equity-trailing (03/07/2026)

`backtest_grid_v4.py` — ETH two-sided R10; 15m vs 5m (steps 0,4/0,6/0,8%),
lev 1-2x, equity-trail off/10/20%.

**Resultados — dupla reprovação, v3 confirmada como ótimo:**
- **5m REPROVADO em todas as configs.** Step 0,4% é negativo (custo fixo 0,04%
  come a margem) com DD até 85%; steps 0,6-0,8% ficam ~zero. Mesmo com níveis
  idênticos ao 15m (step 0,8%), o 5m rende +75 vs +778 — a avaliação mais
  frequente das regras de saída/remontagem gera mais liquidações taker em
  momentos ruins. Mais granularidade ≠ mais lucro.
- **Equity-trailing REPROVADO no 15m** (2x: off +2130, 10% +1583, 20% +1675 —
  corta lucro sem reduzir DD). Os saltos no 5m 2x (+1008/+909) são erráticos
  (2024 +1010, 2025 −830) — acaso de onde o lock remonta, não edge. Lição
  repetida do projeto: trailing é ferramenta de MOMENTUM; o grid é reversão
  mecânica — cada família tem a sua gestão.
- **Config campeã final da campanha: ETH 15m two-sided step 0,8% R10 2x, sem
  trail** — +2.130 (0,212%/dia), 6/6 períodos, DD 37,9%, pior mês −201
  (respeita a tolerância de −250/mês do Rafa).

**Estado da meta 100/dia:** platô honesto em ~0,21%/dia → ~47k de capital
(ou ~1 USDC/dia com os 500 atuais). Alavancas de rendimento testadas e esgotadas
(diversificação, 3x, 5m, trail). Restam na fila ideias de ganho provável marginal.

## Teste das 10 moedas do bot (03/07/2026)

`backtest_grid_10moedas.py` — config campeã (two-sided 0,8% R10 2x) em cada uma
das 10 moedas isolada, 500/moeda, 6 períodos. Régua: 5+ períodos+, pior mês ≥ −250,
sem quebra.

**Só 2 de 10 qualificam:**
- **ETH ✓** — 6/6, +2.130, 0,212%/dia, DD 37,9%, pior mês −201.
- **BTC ✓** — 6/6, +677, 0,068%/dia, DD **21,5%** (o menor!), pior mês −117.

**As outras 8 reprovam** — todas com DD 75-95% e/ou quebra (SOL/DOGE liquidam).
As altcoins são voláteis demais para grid alavancado: rompem o range com força e
o two-sided 2x não segura. SUI parece +163 mas é 2023-24 a pagar 2025 (−1064).

**Correção de hipótese:** eu dizia que "BTC dilui". Errado a 2x isolado — o BTC
é o mais SEGURO (DD 21,5% vs 37,9% do ETH), só rende menos (0,068 vs 0,212%/dia).
Ele dilui só quando MISTURADO com ETH no mesmo pote (v3). Isolado, é o par de
menor risco. Perfil claro: **ETH = motor (retorno), BTC = âncora (estabilidade)**.

**ETH+BTC em paralelo (`backtest_grid_eth_btc.py`, DD agregado diário real):**
- ETH250+BTC250 (500 tot): DD 37,9%→**28,0%** (−9,9 pts), %/dia 0,212→0,140,
  pior mês −201→−159. Mas **ret/DD 56,2→50,2: ETH sozinho é MAIS EFICIENTE.**
- O BTC baixa o DD porém corta o retorno numa proporção ainda maior. Combinar
  só compensa para quem prioriza estabilidade sobre eficiência.
- **Decisão em aberto (Rafa):** ETH sozinho já respeita a tolerância (−201 >
  −250), então combinar é OPCIONAL — troca rendimento por suavidade da curva.
  - **ETH 500 sozinho** = máx rendimento (0,212%/dia), DD 37,9%, ret/DD 56.
  - **ETH250+BTC250** = curva mais suave (DD 28%), rende 2/3, ret/DD 50.
- Implementação demo suporta os dois — decisão de perfil, não de código.

## Ideias na fila (probabilidade de ganho: marginal)
- Micro-sweep de step/range no ETH (0,6-1,0% × R8-12%) — refinamento local.
- Recentragem por EMA em vez de trailing por inventário zerado.
- Wyckoff dos materiais: grid só dentro de ranges de acumulação detectados.
- **Implementação demo do grid ETH 2x no bot (sistema separado)** — próximo passo natural.
