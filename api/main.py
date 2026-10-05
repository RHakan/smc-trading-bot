"""
FastAPI — servidor web do bot.
Fornece REST API + WebSocket para o dashboard.
Autenticação: sessão via cookie HttpOnly (ver api/auth.py).
"""

import asyncio
import json
import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# ── Porta de segurança: valida o APP_SECRET ANTES de qualquer outra coisa ─────
# Corre no import, antes de o servidor abrir a porta e antes de se tocar na base
# de dados. Se o segredo faltar ou for o valor de exemplo (que é público), a
# aplicação não arranca: as credenciais da exchange ficariam encriptadas com uma
# chave que toda a gente conhece, o que é pior que não as encriptar — dá uma
# falsa sensação de proteção.
#
# A mensagem vai para stderr em vez do logger porque o logging ainda não está
# configurado neste ponto, e porque o que interessa é que seja impossível não ver.
from bot.crypto import validate_app_secret as _validate_app_secret

try:
    _validate_app_secret()
except RuntimeError as _erro_segredo:
    print(
        "\n"
        "===============================================================\n"
        "  ARRANQUE CANCELADO — problema de configuração de segurança\n"
        "===============================================================\n"
        f"\n{_erro_segredo}\n\n"
        "A aplicação não arranca enquanto isto não for corrigido.\n",
        file=sys.stderr,
    )
    sys.exit(1)

from api import auth
from api.db import (init_db, get_trades, get_stats, save_signal,
                    save_credentials, get_credentials, has_credentials, delete_credentials)
from api import db as _db
from bot import engine, exchange

def _setup_logging() -> None:
    """Consola + FICHEIRO com rotação.

    Antes só havia consola: quando a janela fechava, o rasto desaparecia. A 06/08/2026
    não existia um único ficheiro de log, e por isso foi impossível determinar o que
    fechou 7 trades em lote a 08/07 e 12/07 — tivemos de inferir pelo padrão temporal.
    Com isto, cada decisão do bot fica gravada e a próxima pergunta destas tem resposta
    em vez de hipótese.

    5 MB por ficheiro, 5 ficheiros = ~25 MB no máximo. UTF-8 explícito (o cmd.exe do
    Windows já nos pregou partidas com acentuação — ver histórico do start.bat).
    """
    from logging.handlers import RotatingFileHandler

    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    fh = RotatingFileHandler(log_dir / "bot.log", maxBytes=5 * 1024 * 1024,
                             backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Silencia o ruído de fundo. O apscheduler regista 2 linhas a cada 5s a dizer
    # que o _broadcast_prices correu bem: ~7 MB/dia, que faria o log rodar de 3 em
    # 3 dias e enterrava os eventos que interessam (fechos, alvo atingido, guarda
    # anti-fantasma). O período de validação dura semanas — o rasto tem de sobreviver.
    # WARNING para cima continua a passar, portanto falhas de jobs aparecem na mesma.
    for ruidoso in ("apscheduler.executors.default", "apscheduler.scheduler"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger(__name__)

# Caminho do .env na raiz do projeto (api/ -> raiz)
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def _persist_env_var(key: str, value: str) -> None:
    """Grava/atualiza uma variável no .env para sobreviver a restarts.

    Edita SÓ a linha da chave (preserva comentários e segredos). Sem isto,
    mudanças feitas pelo dashboard vivem apenas em os.environ e morrem no
    restart — o bot voltava ao default antigo (ex.: bear_v13 reativada 02/07/2026).
    """
    try:
        with open(_ENV_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
        if not found:
            lines.append(f"\n{key}={value}\n")
        with open(_ENV_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info(f"[CONFIG] {key}={value} persistido no .env")
    except Exception as e:
        logger.warning(f"[CONFIG] Falha ao persistir {key} no .env: {e}")


# ---------------------------------------------------------------------------
# WebSocket — lista de conexões ativas
# ---------------------------------------------------------------------------

_ws_clients: list[WebSocket] = []


async def broadcast(event: dict) -> None:
    """Envia evento JSON para todos os clientes WebSocket conectados."""
    data = json.dumps(event, default=str)
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    engine.set_broadcast(broadcast)
    logger.info("Bot Futures — servidor iniciado")

    # Readota posições abertas na Binance para a memória logo no arranque, SEM
    # ligar o bot (não escaneia nem abre trades). Assim, após um restart o
    # dashboard já mostra as posições e permite fechá-las pelo card — sem ter de
    # ir à Binance na mão. As ordens SL/TP na exchange seguem protegendo.
    #
    # CRÍTICO: reconcile_on_boot() é síncrona e faz várias chamadas de rede à
    # Binance (get_positions, get_stop_orders por posição, varredura de órfãs).
    # Chamada direta aqui bloqueava o event loop inteiro — o uvicorn não aceitava
    # NENHUMA ligação (nem servir /login) até a Binance responder, o que já levou
    # ~14s numa rede lenta ("não consegui conectar" no browser). asyncio.to_thread
    # tira-a do event loop; wait_for garante que o boot nunca fica preso além do
    # timeout — se estourar, a reconciliação continua na thread em background e
    # as posições aparecem assim que terminar (ou no próximo tick do scheduler).
    RECONCILE_BOOT_TIMEOUT = 8.0
    try:
        adopted = await asyncio.wait_for(
            asyncio.to_thread(engine.reconcile_on_boot),
            timeout=RECONCILE_BOOT_TIMEOUT,
        )
        if adopted:
            logger.info(f"[BOOT] {adopted} posição(ões) readotada(s) da exchange no arranque")
    except asyncio.TimeoutError:
        logger.warning(
            f"[BOOT] Reconciliação excedeu {RECONCILE_BOOT_TIMEOUT:.0f}s (Binance lenta?) — "
            "servidor sobe já; a reconciliação continua em background."
        )
    except Exception as e:
        logger.warning(f"[BOOT] Reconciliação no arranque falhou: {e}")

    yield
    await engine.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Bot Futures", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoCacheStaticFiles(StaticFiles):
    """
    StaticFiles que força o browser a revalidar CSS/JS a cada carregamento.

    O StaticFiles padrão não envia Cache-Control, então o browser guarda o
    style.css em cache heurística e NÃO o revalida num F5 normal — mudanças de
    estilo só apareciam com hard-refresh (Ctrl+Shift+R). `no-cache` obriga a
    revalidar via ETag/Last-Modified (já suportados): ficheiro igual → 304 leve,
    ficheiro alterado → 200 com a versão nova. Sempre atualizado, custo mínimo.
    """
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


_dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard")
app.mount("/static", NoCacheStaticFiles(directory=_dashboard_path), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LoginInput(BaseModel):
    username: str
    password: str
    totp_code: str | None = None    # obrigatório só se o 2FA estiver ativo


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class TotpEnable(BaseModel):
    code: str


class TotpDisable(BaseModel):
    code: str


class ConfigUpdate(BaseModel):
    mode:      str | None = None
    leverage:  int | None = None
    strategy:  str | None = None
    risk_pct:  float | None = None


class CredentialsInput(BaseModel):
    api_key: str
    secret:  str


class ClosePositionRequest(BaseModel):
    symbol: str
    side:   str


class PartialCloseRequest(BaseModel):
    symbol:   str
    side:     str
    fraction: float   # fração do que RESTA a fechar (0.25, 0.5, 0.75)


class UpdateLevelRequest(BaseModel):
    symbol: str
    side:   str
    kind:   str       # "sl" (stop loss) ou "tp" (take profit)
    price:  float


class BotCommand(BaseModel):
    action:   str
    strategy: str | None = None


# ---------------------------------------------------------------------------
# Rotas públicas (sem autenticação)
# ---------------------------------------------------------------------------

@app.get("/login")
async def login_page(request: Request):
    """Serve a página de login. Se já autenticado, redireciona para o dashboard."""
    session_id = request.cookies.get(auth.COOKIE_NAME)
    if auth.get_session_user(session_id):
        return RedirectResponse(url="/", status_code=302)
    return FileResponse(os.path.join(_dashboard_path, "login.html"))


@app.post("/auth/login")
async def login(body: LoginInput, request: Request):
    """
    Valida credenciais e cria sessão segura.
    Protegido contra brute force (rate limiting por IP) e timing attacks.
    """
    auth.check_rate_limit(request)

    # Credenciais vêm da BD (senha em hash scrypt). verify_login usa comparação
    # em tempo constante nos dois campos.
    creds_ok = _db.verify_login(body.username, body.password)

    if not creds_ok:
        auth.record_failed_attempt(request)
        await asyncio.sleep(0.8)   # delay fixo dificulta enumeração
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Credenciais inválidas.")

    # Segundo fator: se o 2FA estiver ativo, exige um código válido (Authenticator
    # ou código de recuperação). Só corre APÓS a senha passar.
    if _db.totp_enabled():
        if not body.totp_code:
            # senha certa mas falta o código — sinaliza ao frontend para pedir o 2FA
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "2FA_REQUIRED")
        if not _db.verify_totp(body.totp_code):
            auth.record_failed_attempt(request)
            await asyncio.sleep(0.8)
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Código de 2FA inválido.")

    auth.clear_rate_limit(request)
    session_id = auth.create_session(body.username)

    response = JSONResponse({"ok": True})
    response.set_cookie(
        key=auth.COOKIE_NAME,
        value=session_id,
        httponly=True,          # JS não consegue ler — protege contra XSS
        samesite="strict",      # cookie não é enviado em pedidos cross-site — protege contra CSRF
        secure=auth.SECURE_COOKIES,   # definir SECURE_COOKIES=true em produção HTTPS
        max_age=auth.SESSION_MAX_AGE,
        path="/",
    )
    return response


@app.post("/auth/logout")
async def logout(request: Request):
    """Invalida a sessão e limpa o cookie."""
    session_id = request.cookies.get(auth.COOKIE_NAME)
    auth.delete_session(session_id)
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return response


# ---------------------------------------------------------------------------
# Segurança da conta (requer sessão) — mudar senha e gerir 2FA
# ---------------------------------------------------------------------------

@app.get("/api/auth/status", dependencies=[Depends(auth.require_session)])
async def auth_status():
    """Estado da conta para o painel de segurança."""
    return {"username": _db.get_auth_username(), "totp_enabled": _db.totp_enabled()}


@app.post("/api/auth/password", dependencies=[Depends(auth.require_session)])
async def change_password_route(body: PasswordChange, user: str = Depends(auth.require_session)):
    """Muda a senha. Exige a senha atual (defesa se a sessão for sequestrada)."""
    if not _db.verify_login(user, body.current_password):
        await asyncio.sleep(0.5)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Senha atual incorreta.")
    if len(body.new_password) < 12:
        raise HTTPException(400, "A nova senha deve ter pelo menos 12 caracteres.")
    _db.change_password(body.new_password)
    return {"ok": True, "message": "Senha alterada."}


@app.post("/api/auth/2fa/setup", dependencies=[Depends(auth.require_session)])
async def totp_setup_route(user: str = Depends(auth.require_session)):
    """Inicia o enrollment: gera um segredo PENDENTE e devolve QR + chave manual.
    Não ativa nada — só passa a ativo depois de /2fa/enable com um código válido."""
    if _db.totp_enabled():
        raise HTTPException(400, "2FA já está ativo. Desativa primeiro para regenerar.")
    return _db.begin_totp_enrollment(user)


@app.post("/api/auth/2fa/enable", dependencies=[Depends(auth.require_session)])
async def totp_enable_route(body: TotpEnable):
    """Confirma o segredo pendente com um código do Authenticator. Devolve os
    códigos de recuperação UMA vez (guarda-os — não voltam a aparecer)."""
    codes = _db.confirm_totp(body.code)
    if codes is None:
        raise HTTPException(400, "Código inválido — confirma que adicionaste a conta e tenta o código atual.")
    return {"ok": True, "recovery_codes": codes}


@app.post("/api/auth/2fa/disable", dependencies=[Depends(auth.require_session)])
async def totp_disable_route(body: TotpDisable):
    """Desativa o 2FA. Exige um código válido (Authenticator ou recuperação) — prova
    de posse do dispositivo, para uma sessão sequestrada não poder desligar o 2FA."""
    if not _db.totp_enabled():
        return {"ok": True, "message": "2FA já estava desativado."}
    if not _db.verify_totp(body.code):
        await asyncio.sleep(0.5)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Código de 2FA inválido.")
    _db.disable_totp()
    return {"ok": True, "message": "2FA desativado."}


# ---------------------------------------------------------------------------
# Dashboard (requer sessão)
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    session_id = request.cookies.get(auth.COOKIE_NAME)
    if not auth.get_session_user(session_id):
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(os.path.join(_dashboard_path, "index.html"))


# ---------------------------------------------------------------------------
# API (requer sessão)
# ---------------------------------------------------------------------------

@app.get("/api/fng", dependencies=[Depends(auth.require_session)])
async def get_fear_greed():
    """Índice Fear & Greed do mercado cripto (alternative.me, cache de 1h)."""
    from bot.indicators import fetch_fear_greed
    result = await fetch_fear_greed()
    return result or {"value": None, "classification": None, "classification_en": None}


@app.get("/api/strategies", dependencies=[Depends(auth.require_session)])
async def list_strategies():
    """
    Metadados das estratégias para o card do dropdown: descrição estática (META)
    + números de risco/exposição calculados AO VIVO da config (.env), para
    refletirem sempre o estado atual do bot.
    """
    from bot.strategies import meta as strat_meta

    risk_pct     = float(os.getenv("RISK_PER_TRADE_PCT", "2"))
    max_per_side = int(float(os.getenv("MAX_PER_SIDE", "2")))
    portfolio    = float(os.getenv("PORTFOLIO_VALUE_USD", "1000"))

    out = []
    for value, m in strat_meta.all_meta().items():
        # Estratégias de uma só direção usam no máximo MAX_PER_SIDE posições;
        # as multi/neutral podem abrir dos dois lados (≈ o dobro).
        both_sides = m.get("direction") in ("multi", "neutral")
        max_positions = max_per_side * (2 if both_sides else 1)
        max_risk_pct  = round(max_positions * risk_pct, 1)

        out.append({
            "value": value,
            **m,
            "live": {
                "risk_pct":       risk_pct,
                "risk_usdt":      round(portfolio * risk_pct / 100, 2),
                "max_positions":  max_positions,
                "max_risk_pct":   max_risk_pct,
                "max_risk_usdt":  round(portfolio * max_risk_pct / 100, 2),
                "portfolio_usdt": portfolio,
            },
        })
    return {"strategies": out}


@app.get("/api/status", dependencies=[Depends(auth.require_session)])
async def get_status():
    positions = engine._build_positions_snapshot()

    try:
        balance = exchange.get_balance()
    except Exception:
        base = float(os.getenv("PORTFOLIO_VALUE_USD", "1000"))
        stats_data = get_stats()
        current = round(base + (stats_data.get("total_pnl") or 0), 2)
        balance = {"total": current, "free": current, "used": 0}

    return {
        "running":   engine.is_running(),
        "config":    engine.get_config(),
        "positions": positions,
        "stats":     get_stats(),
        "balance":   balance,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/trades", dependencies=[Depends(auth.require_session)])
async def get_trades_route(limit: int = 50, status: str | None = None):
    return {"trades": get_trades(limit=limit, status=status)}


@app.post("/api/bot", dependencies=[Depends(auth.require_session)])
async def control_bot(cmd: BotCommand):
    if cmd.action == "start":
        await engine.start(strategy=cmd.strategy)
        return {"ok": True, "message": "Bot iniciado"}
    elif cmd.action == "stop":
        await engine.stop()
        return {"ok": True, "message": "Bot parado"}
    elif cmd.action == "scan":
        asyncio.create_task(engine.scan_signals(execute_trades=False))
        return {"ok": True, "message": "Scan manual iniciado — sem execução de trades"}
    raise HTTPException(400, "Ação inválida. Use 'start', 'stop' ou 'scan'.")


@app.post("/api/config", dependencies=[Depends(auth.require_session)])
async def update_config(cfg: ConfigUpdate):
    if cfg.mode is not None:
        if cfg.mode not in ("testnet", "live"):
            raise HTTPException(400, "Modo inválido. Use 'testnet' ou 'live'.")
        exchange.set_mode(cfg.mode)
        try:
            balance = exchange.get_balance()
        except Exception:
            portfolio = float(os.getenv("PORTFOLIO_VALUE_USD", "1000"))
            balance = {"total": portfolio, "free": portfolio, "used": 0}
        await broadcast({"type": "mode_changed", "mode": cfg.mode, "balance": balance})

    if cfg.leverage is not None:
        if not 1 <= cfg.leverage <= 20:
            raise HTTPException(400, "Alavancagem deve estar entre 1 e 20.")
        os.environ["DEFAULT_LEVERAGE"] = str(cfg.leverage)
        _persist_env_var("DEFAULT_LEVERAGE", str(cfg.leverage))
        await broadcast({"type": "leverage_changed", "leverage": cfg.leverage})

    if cfg.strategy is not None:
        valid_strategies = ("breakout_short", "bear_v12", "bear_v13", "bull_v1",
                            "lateral_breakout", "bull_smc", "auto", "auto_v2")
        if cfg.strategy not in valid_strategies:
            raise HTTPException(400, f"Estratégia inválida. Use: {', '.join(valid_strategies)}.")
        os.environ["STRATEGY_MODE"] = cfg.strategy
        # Persiste no .env — sem isto a escolha morre no restart e o bot volta ao
        # default antigo (foi o que fez a bear_v13 reativar sozinha em 02/07/2026)
        _persist_env_var("STRATEGY_MODE", cfg.strategy)
        await broadcast({"type": "strategy_changed", "strategy": cfg.strategy})

    if cfg.risk_pct is not None:
        if not 0.1 <= cfg.risk_pct <= 5.0:
            raise HTTPException(400, "Risco por trade deve estar entre 0.1% e 5%.")
        os.environ["RISK_PER_TRADE_PCT"] = str(cfg.risk_pct)
        _persist_env_var("RISK_PER_TRADE_PCT", str(cfg.risk_pct))

    return {"ok": True, "config": engine.get_config()}


@app.get("/api/candles", dependencies=[Depends(auth.require_session)])
async def get_candles(symbol: str = "BTC/USDC:USDC", timeframe: str = "15m",
                      limit: int = 200):
    """Retorna candles OHLCV com indicadores para o gráfico do dashboard."""
    df = exchange.fetch_candles(symbol, timeframe, limit)

    from bot.indicators import add_indicators
    df = add_indicators(df)

    import math

    def _safe(v):
        try:
            return None if (v is None or math.isnan(v) or math.isinf(v)) else float(v)
        except (TypeError, ValueError):
            return None

    records = []
    for ts, row in df.iterrows():
        records.append({
            "t":        ts.isoformat(),
            "o":        float(row["open"]),
            "h":        float(row["high"]),
            "l":        float(row["low"]),
            "c":        float(row["close"]),
            "v":        float(row["volume"]),
            "ema20":    _safe(row.get("ema20")),
            "ema50":    _safe(row.get("ema50")),
            "ema200":   _safe(row.get("ema200")),
            "bb_upper": _safe(row.get("bb_upper")),
            "bb_lower": _safe(row.get("bb_lower")),
        })
    return {"symbol": symbol, "timeframe": timeframe, "candles": records}


def _to_python(obj):
    """Converte tipos numpy para Python nativo (para serialização JSON)."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


@app.get("/api/scan/{symbol}", dependencies=[Depends(auth.require_session)])
async def scan_symbol(symbol: str, strategy: str = "bear_v12"):
    from bot.strategies import breakout_short, bear_v12, bear_v13

    symbol = symbol.replace("-", "/").replace("_", ":")
    df_entry = exchange.fetch_candles(symbol, "1h", limit=200)
    df_daily  = exchange.fetch_candles(symbol, "1d", limit=60)

    if strategy == "bear_v13":
        result = bear_v13.analyze(df_entry, df_daily)
    elif strategy == "breakout_short":
        result = breakout_short.analyze(df_entry, df_daily)
    else:
        result = bear_v12.analyze(df_entry, df_daily)

    result["symbol"]    = symbol
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result = _to_python(result)

    save_signal({**result, "strategy": strategy})
    await broadcast({"type": "signal_update", "strategy": strategy, **result})

    return result


def _resolve_exit_price(symbol: str, order_result: dict) -> float:
    """
    Descobre o preço real de fill de uma ordem a mercado. O demo da Binance
    frequentemente não preenche "average" na resposta imediata, por isso há
    3 níveis de fallback: resposta da ordem → último trade → ticker atual.
    """
    try:
        px = float(order_result.get("average") or order_result.get("price") or 0)
    except (TypeError, ValueError):
        px = 0.0
    if not px:
        try:
            lt = exchange.get_last_trade(symbol)
            if lt:
                px = float(lt.get("price") or lt.get("average") or 0)
        except Exception:
            pass
    if not px:
        try:
            px = float(exchange.get_ticker(symbol).get("last") or 0)
        except Exception:
            px = 0.0
    return px


@app.post("/api/position/close", dependencies=[Depends(auth.require_session)])
async def close_position_route(body: ClosePositionRequest):
    from bot import position_manager
    from bot.position_manager import _calc_close_pnl, calc_pnl
    from api.db import close_trade, get_trades

    result = exchange.close_position(body.symbol, body.side)
    if result is None:
        raise HTTPException(404, "Posição não encontrada na exchange.")
    exchange.cancel_all_orders(body.symbol)

    now_str = datetime.now(timezone.utc).isoformat()
    exit_price = _resolve_exit_price(body.symbol, result)

    # Fecha o registo no banco de dados com P&L calculado
    pos = position_manager.get_positions().get(body.symbol)
    if pos and pos.trade_id:
        try:
            net_r, pnl_usdt = _calc_close_pnl(pos, exit_price) if exit_price else (0.0, 0.0)
            close_trade(
                trade_id=pos.trade_id,
                exit_price=exit_price or pos.entry,
                pnl_usdt=pnl_usdt,
                pnl_pct=net_r,
                closed_at=now_str,
                exit_reason="manual",
            )
        except Exception as e:
            logger.warning(f"[CLOSE] Erro ao fechar trade #{pos.trade_id} no DB: {e}")
    else:
        # Posição não estava em memória (restart?) — recalcula P&L com dados do DB
        for t in get_trades(limit=50, status="open"):
            if t.get("symbol") == body.symbol:
                try:
                    entry    = float(t.get("entry") or 0)
                    sl       = float(t.get("stop_loss") or 0)
                    notional = float(t.get("position_size_usdt") or 0)
                    side_str = (t.get("side") or "").lower()

                    net_r, pnl_usdt = calc_pnl(side_str, entry, sl, notional, exit_price) \
                        if exit_price else (0.0, 0.0)

                    close_trade(
                        trade_id=int(t["id"]),
                        exit_price=exit_price or entry,
                        pnl_usdt=pnl_usdt,
                        pnl_pct=net_r,
                        closed_at=now_str,
                        exit_reason="manual",
                    )
                except Exception as e:
                    logger.warning(f"[CLOSE] Erro ao fechar trade #{t.get('id')} no DB: {e}")
                break

    position_manager.remove_position(body.symbol)

    stats = get_stats()
    try:
        balance = exchange.get_balance()
    except Exception:
        portfolio = float(os.getenv("PORTFOLIO_VALUE_USD", "1000"))
        balance = {"total": portfolio, "free": portfolio, "used": 0}

    await broadcast({"type": "stats_update", "stats": stats, "balance": balance})
    # Atualiza os cards de posição imediatamente (incluindo quando fecha a última posição,
    # onde _broadcast_prices não envia nada por estar vazio)
    await broadcast({"type": "positions_tick", "positions": engine._build_positions_snapshot()})
    await broadcast({
        "type": "log", "level": "warn",
        "message": f"⛔ {body.symbol.replace('/USDC:USDC','')} fechado manualmente",
    })
    return {"ok": True}


@app.post("/api/position/close-partial", dependencies=[Depends(auth.require_session)])
async def close_partial_route(body: PartialCloseRequest):
    from bot import position_manager
    from bot.position_manager import calc_pnl
    from api.db import get_trades, record_partial_close, update_trade_size

    if body.fraction not in (0.25, 0.5, 0.75):
        raise HTTPException(400, "Fração inválida (use 0.25, 0.5 ou 0.75).")

    # Fecha a fração do que RESTA aberto na exchange (não cancela SL/TP — o
    # restante da posição continua protegido; as ordens reduceOnly ajustam-se).
    result = exchange.close_partial(body.symbol, body.side, body.fraction)
    if result is None:
        raise HTTPException(404, "Posição não encontrada na exchange.")

    now_str = datetime.now(timezone.utc).isoformat()
    exit_price = _resolve_exit_price(body.symbol, result)

    parent = next((t for t in get_trades(limit=50, status="open")
                   if t.get("symbol") == body.symbol), None)
    pos = position_manager.get_positions().get(body.symbol)

    # Tamanho/dados atuais: memória tem prioridade; senão o trade-mãe no DB.
    if pos:
        cur_notional, cur_margin = pos.position_size_usdt, pos.margin_usdt
        entry, sl, side_l = pos.entry, pos.stop_loss, pos.side
    elif parent:
        cur_notional = float(parent.get("position_size_usdt") or 0)
        cur_margin   = float(parent.get("margin_usdt") or 0)
        entry  = float(parent.get("entry") or 0)
        sl     = float(parent.get("stop_loss") or 0)
        side_l = (parent.get("side") or "").lower()
    else:
        raise HTTPException(404, "Trade não encontrado para contabilizar a parcial.")

    closed_notional = round(cur_notional * body.fraction, 2)
    closed_margin   = round(cur_margin * body.fraction, 2)
    net_r, pnl_usdt = calc_pnl(side_l, entry, sl, closed_notional, exit_price) \
        if exit_price else (0.0, 0.0)

    # Regista a parcial como linha fechada no histórico e reduz o tamanho do mãe.
    if parent:
        try:
            record_partial_close(parent, body.fraction, closed_notional, closed_margin,
                                  exit_price or entry, pnl_usdt, net_r, now_str)
            update_trade_size(
                int(parent["id"]),
                position_size_usdt=round(cur_notional - closed_notional, 2),
                margin_usdt=round(cur_margin - closed_margin, 2),
            )
        except Exception as e:
            logger.warning(f"[PARTIAL] Erro ao registar parcial no DB: {e}")

    # Reduz o tamanho em memória (no-op se a posição não estiver a ser gerida).
    position_manager.reduce_position(body.symbol, body.fraction)

    stats = get_stats()
    try:
        balance = exchange.get_balance()
    except Exception:
        portfolio = float(os.getenv("PORTFOLIO_VALUE_USD", "1000"))
        balance = {"total": portfolio, "free": portfolio, "used": 0}

    pct = int(round(body.fraction * 100))
    await broadcast({"type": "stats_update", "stats": stats, "balance": balance})
    await broadcast({"type": "positions_tick", "positions": engine._build_positions_snapshot()})
    await broadcast({
        "type": "log", "level": "ok",
        "message": f"✂ {body.symbol.replace('/USDC:USDC','')} — {pct}% fechado "
                   f"({pnl_usdt:+.2f} USDC)",
    })
    return {"ok": True, "pnl_usdt": pnl_usdt, "fraction": body.fraction}


@app.post("/api/position/update-level", dependencies=[Depends(auth.require_session)])
async def update_level_route(body: UpdateLevelRequest):
    """Move o SL ou o TP de uma posição aberta (arraste da linha no gráfico)."""
    from bot import position_manager
    from api.db import get_trades, update_trade_levels

    kind = body.kind.lower()
    if kind not in ("sl", "tp"):
        raise HTTPException(400, "kind inválido (use 'sl' ou 'tp').")
    if body.price <= 0:
        raise HTTPException(400, "Preço inválido.")

    side = body.side.lower()

    pos = position_manager.get_positions().get(body.symbol)
    parent = next((t for t in get_trades(limit=50, status="open")
                   if t.get("symbol") == body.symbol), None)

    # Validação de sanidade contra o PREÇO ATUAL — não contra a entrada.
    #
    # O critério certo é físico: um nível colocado do lado errado do preço
    # dispararia imediatamente. A entrada não tem nada a ver com isso.
    # Validar contra a entrada prendia o stop no breakeven e impedia justamente
    # o que o stop móvel serve para fazer: TRAVAR LUCRO num vencedor, arrastando
    # o stop para além da entrada, em direção ao preço.
    #
    #   LONG : SL abaixo do preço atual, TP acima.
    #   SHORT: SL acima  do preço atual, TP abaixo.
    try:
        price_now = float(exchange.get_ticker(body.symbol)["last"])
    except Exception:
        price_now = None

    if price_now:
        if kind == "sl":
            if side == "long" and body.price >= price_now:
                raise HTTPException(400, f"Stop de LONG tem de ficar ABAIXO do preço atual "
                                         f"({price_now:g}) — senão dispara já.")
            if side == "short" and body.price <= price_now:
                raise HTTPException(400, f"Stop de SHORT tem de ficar ACIMA do preço atual "
                                         f"({price_now:g}) — senão dispara já.")
        else:  # tp
            if side == "long" and body.price <= price_now:
                raise HTTPException(400, f"Alvo de LONG tem de ficar ACIMA do preço atual "
                                         f"({price_now:g}).")
            if side == "short" and body.price >= price_now:
                raise HTTPException(400, f"Alvo de SHORT tem de ficar ABAIXO do preço atual "
                                         f"({price_now:g}).")

    # Aplica na exchange (fonte de verdade). Erros da Binance (ex: gatilho
    # imediato) sobem como 400 com a mensagem original.
    try:
        if kind == "sl":
            exchange.update_stop_loss(body.symbol, side, body.price)
        else:
            exchange.update_take_profit(body.symbol, side, body.price)
    except Exception as e:
        raise HTTPException(400, f"A Binance rejeitou a alteração: {e}")

    # Espelha em memória (linha do gráfico + gestão).
    if pos:
        if kind == "sl":
            # NÃO tocar em pos.stop_loss: é a referência de R (risk = |entry - stop_loss|)
            # que o calc_pnl usa. Sobrescrevê-la com o stop arrastado ZERAVA o risco no
            # breakeven (stop == entry → risk 0 → P&L registado como 0) e falseava o R
            # sempre que o stop passava a entrada. O nível vivo é o current_stop.
            pos.current_stop = body.price
        else:
            pos.take_profit = body.price
    if parent:
        try:
            # Só o TP vai ao DB. O SL arrastado NÃO: a coluna stop_loss guarda o stop
            # ORIGINAL (referência de R). O nível vivo do stop fica na exchange — e a
            # reconciliação no arranque lê-o de lá (exchange = fonte de verdade).
            if kind == "tp":
                update_trade_levels(int(parent["id"]), take_profit=body.price)
        except Exception as e:
            logger.warning(f"[LEVEL] Erro ao atualizar {kind} no DB: {e}")

    await broadcast({"type": "positions_tick", "positions": engine._build_positions_snapshot()})
    label = "Stop Loss" if kind == "sl" else "Take Profit"
    await broadcast({
        "type": "log", "level": "ok",
        "message": f"🎯 {body.symbol.replace('/USDC:USDC','')} — {label} movido para {body.price}",
    })
    return {"ok": True, "kind": kind, "price": body.price}


# ---------------------------------------------------------------------------
# Credenciais API (requer sessão — username vem da sessão, não do body)
# ---------------------------------------------------------------------------

@app.get("/api/credentials/{mode}")
async def check_credentials_route(mode: str, user: str = Depends(auth.require_session)):
    if mode not in ("testnet", "live"):
        raise HTTPException(400, "Modo inválido.")
    return {"mode": mode, "configured": has_credentials(user, mode)}


@app.post("/api/credentials/{mode}")
async def save_credentials_route(mode: str, body: CredentialsInput,
                                 user: str = Depends(auth.require_session)):
    if mode not in ("testnet", "live"):
        raise HTTPException(400, "Modo inválido.")
    if not body.api_key.strip() or not body.secret.strip():
        raise HTTPException(400, "API key e secret são obrigatórios.")
    save_credentials(user, mode, body.api_key.strip(), body.secret.strip())
    exchange.reinit_client(mode)
    return {"ok": True}


@app.delete("/api/credentials/{mode}")
async def delete_credentials_route(mode: str, user: str = Depends(auth.require_session)):
    if mode not in ("testnet", "live"):
        raise HTTPException(400, "Modo inválido.")
    delete_credentials(user, mode)
    return {"ok": True}


# ---------------------------------------------------------------------------
# WebSocket (requer sessão via cookie)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Browsers enviam cookies com pedidos WebSocket — validamos antes de aceitar
    session_id = ws.cookies.get(auth.COOKIE_NAME)
    if not auth.get_session_user(session_id):
        await ws.close(code=4001, reason="Não autenticado")
        return

    await ws.accept()
    _ws_clients.append(ws)
    try:
        await ws.send_text(json.dumps({
            "type":    "connected",
            "running": engine.is_running(),
            "config":  engine.get_config(),
            "stats":   get_stats(),
        }, default=str))

        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Helper (usado internamente pelo engine)
# ---------------------------------------------------------------------------

def position_manager_state() -> list[dict]:
    from bot import position_manager
    result = []
    for sym, pos in position_manager.get_positions().items():
        try:
            ticker = exchange.get_ticker(sym)
            price  = ticker["last"]
        except Exception:
            price = pos.entry

        risk_usdt = round(
            pos.position_size_usdt * abs(pos.entry - pos.stop_loss) / pos.entry, 2
        ) if pos.entry else 0

        # Preço a partir do qual o trade fica LUCRATIVO — fechar exatamente na
        # entrada dá PREJUÍZO, porque a taxa de abertura+fecho já foi paga.
        # É o "ponto zero real": entrada + custo total, projetado no preço.
        # RT vem do position_manager (fonte única) para não haver dois números.
        breakeven = round(
            pos.entry * (1 + position_manager.RT) if pos.side == "long"
            else pos.entry * (1 - position_manager.RT),
            8,
        ) if pos.entry else None

        result.append({
            "symbol":             sym,
            "side":               pos.side,
            "entry":              pos.entry,
            "breakeven":          breakeven,
            "stop_loss":          pos.current_stop,
            "take_profit":        pos.take_profit,
            "leverage":           pos.leverage,
            "margin_usdt":        pos.margin_usdt,
            "position_size_usdt": pos.position_size_usdt,
            "risk_usdt":          risk_usdt,
            "breakeven_hit":      pos.breakeven_hit,
            "partial_closed":     pos.partial_closed,
            "progress_pct":       round(pos.progress(price) * 100, 1),
            "pnl_pct":            pos.unrealized_pnl_pct(price),
        })
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
