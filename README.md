# bot-futures

> **EN —** Self-hosted crypto futures trading bot for Binance USD-M, with a regime-based
> strategy engine, a local web dashboard and a backtesting suite. Testnet and backtest only.
> Educational project. Not financial advice.

Bot de trading automatizado para futuros de criptomoedas na Binance USD-M, com painel web
local, motor de estratégias por regime de mercado e uma suite de backtests.

> ## ⚠️ Aviso importante
>
> Este é um **projeto educativo e experimental**, desenvolvido e testado em **ambiente de
> simulação (testnet)** e em **backtests com dados históricos**.
>
> - **Não é aconselhamento financeiro**, nem recomendação de investimento.
> - **Não há garantia de resultados.** Desempenho passado em backtest não prevê
>   desempenho futuro — a diferença entre os dois costuma ser substancial.
> - Negociar futuros com alavancagem pode **resultar na perda total do capital**, e em
>   alguns casos em perdas superiores ao depositado.
> - Se o usar com uma conta real, **a responsabilidade é inteiramente sua**.
>
> O autor não tem qualquer responsabilidade por perdas decorrentes do uso deste software.

---

## O que faz

O bot varre um conjunto de pares a cada hora, classifica o regime de mercado a partir do
gráfico diário e encaminha a decisão para a estratégia correspondente:

| Regime diário | Estratégia | Lógica |
|---|---|---|
| Lateral | `lateral_breakout` | Rompimento de range consolidado, com saída antecipada se o rompimento falhar |
| Alta | `bull_smc` | Varrimento de liquidez seguido de quebra de estrutura, filtrado pelo regime do BTC |
| Baixa | `bear_breakout` | Perda de suporte consolidado, com stop em estrutura |

Posições abertas são geridas automaticamente: breakeven, trailing por ATR, fecho a mercado
ao atingir o alvo e invalidação de rompimentos falhados.

**Outras funcionalidades:**

- Painel web com gráfico de velas, posições abertas, histórico e log de eventos em tempo real
- Autenticação com sessão em cookie e **2FA opcional** (TOTP, compatível com Google Authenticator)
- Credenciais da exchange **encriptadas na base de dados** (Fernet/AES), nunca em texto simples
- Reconciliação no arranque — readota posições abertas na exchange e repõe ordens de proteção
- Suite de backtests com validação fora da amostra

---

## Stack

- **Python 3.12+**
- **FastAPI** + **Uvicorn** — API REST e WebSocket
- **ccxt** — ligação à Binance USD-M Futures
- **pandas** / **numpy** — indicadores e backtests
- **APScheduler** — varrimento horário e gestão de posições a cada 30 s
- **SQLite** — histórico de trades e credenciais
- **cryptography** — encriptação das credenciais (Fernet) e hash da senha (scrypt)
- **Lightweight Charts** — gráfico do painel

---

## Como executar

### 1. Requisitos

- Python 3.12 ou superior
- Uma conta na [Binance Futures Testnet](https://testnet.binancefuture.com/) (gratuita)

### 2. Instalação

```bash
git clone <url-do-repositorio>
cd bot-futures
pip install -r requirements.txt
```

### 3. Configuração

```bash
cp .env.example .env     # no Windows: copy .env.example .env
```

### 4. Gerar o seu `APP_SECRET` (obrigatório)

O `APP_SECRET` é a chave que encripta as credenciais da exchange na base de dados.
**A aplicação recusa-se a arrancar enquanto este passo não estiver feito.**

Gere um valor próprio:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Cole o resultado no `.env`, na linha `APP_SECRET=`.

> **Porque é obrigatório:** o valor que vem no `.env.example` é público — está neste
> repositório e qualquer pessoa o pode ler. Se o deixasse lá, as credenciais da exchange
> ficariam encriptadas com uma chave conhecida por todos, o que é pior do que não as
> encriptar: dá uma falsa sensação de proteção.
>
> Por isso o arranque é bloqueado se o `APP_SECRET` estiver em falta, for o valor de
> exemplo, ou tiver menos de 32 caracteres. A mensagem de erro diz exatamente o que fazer.

> **Se mudar o `APP_SECRET` mais tarde**, as credenciais já guardadas deixam de poder ser
> decifradas e têm de ser reintroduzidas pelo painel. Guarde-o em lugar seguro.

### 5. Restante configuração

No mesmo `.env`:

- **`DASHBOARD_USER` e `DASHBOARD_PASSWORD`** — usados apenas no primeiro arranque para
  semear o utilizador. Depois disso a senha vive na base de dados em hash scrypt, e pode
  remover as duas linhas do ficheiro.
- **`TRADING_MODE=testnet`** — confirme que está assim antes do primeiro arranque.

### 6. Arranque

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

No Windows existe também o atalho `start.bat`, que valida o ambiente e abre o navegador.

O painel fica em `http://localhost:8080`.

### 7. Ligar à exchange

No painel, abra as definições e introduza a chave e o segredo da API da testnet. Ficam
encriptados na base de dados com o `APP_SECRET`.

> Ao criar a chave na Binance, conceda apenas permissão de **Futuros**. Nunca ative
> levantamentos.

### 8. Activar o 2FA (recomendado)

No cabeçalho do painel há um ícone de cadeado. Ao clicar, aparece um código QR para
adicionar ao autenticador e oito códigos de recuperação — guarde-os.

Se perder o acesso, com o bot parado:

```bash
python -m tools.reset_auth --disable-2fa
python -m tools.reset_auth --set-password NOVA_SENHA
```

---

## Backtests

Os scripts vivem em `backtests/`. Cada um é autónomo:

```bash
python backtests/backtest_sistema_v3.py
```

No primeiro arranque descarregam as velas da Binance e guardam-nas em `backtests/cache/`
(não versionado, cerca de 200 MB quando completo).

**Metodologia usada em todos os testes:**

- Período de desenvolvimento de 5 anos + período de **validação fora da amostra** separado
- Custos de transação e derrapagem modelados em cada operação
- Entradas sempre na abertura da vela seguinte ao sinal — sem antecipação de informação
- Stop verificado antes do alvo dentro da mesma vela, e preenchimento ao preço de abertura
  quando há gap — a leitura conservadora
- Candidatos aprovados passam ainda por uma bateria de confirmação: estabilidade da
  vizinhança de parâmetros, sub-períodos, consistência ano a ano e mês a mês

---

## Limitações conhecidas

- **Só Binance USD-M Futures.** Não há abstração para outras exchanges.
- **Timeframe de entrada fixo em 1 hora**, definido no motor e não configurável por ficheiro.
- `MAX_TRADES_PER_DAY` é lido da configuração mas **ainda não é aplicado**.
- **Sem autenticação multiutilizador.** Foi desenhado para uma única pessoa, na sua própria
  máquina ou rede local.
- **Não exponha o painel à internet** sem colocar HTTPS à frente (proxy reverso) e definir
  `SECURE_COOKIES=true`. Por omissão escuta em `0.0.0.0`, o que o torna acessível a toda a
  rede local.
- Em ambientes de simulação, **as ordens condicionais da exchange nem sempre executam**.
  O bot compensa fechando a mercado ao atingir o alvo, mas convém ter isto presente ao
  interpretar resultados de testnet.
- A reconciliação no arranque depende da exchange responder; se não responder dentro do
  tempo limite, o servidor arranca na mesma e a reconciliação continua em segundo plano.

---

## Estrutura

```
bot/            motor, estratégias, indicadores, ligação à exchange, gestão de risco
api/            FastAPI — endpoints REST, WebSocket, base de dados, autenticação
dashboard/      painel web (HTML, CSS, JS)
backtests/      scripts de backtest, um por hipótese testada
config/         parâmetros de estratégias em JSON
tools/          utilitários de linha de comandos (reposição de credenciais)
```

---

## Licença

MIT — ver `LICENSE`.
