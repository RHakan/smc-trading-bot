"""
Autenticação baseada em sessão com cookie HttpOnly.
Sem dependências externas — usa apenas stdlib.

Fluxo:
  POST /auth/login  → valida credenciais → cria sessão → define cookie
  GET  /            → verifica cookie → redireciona para /login se inválido
  POST /auth/logout → apaga sessão → redireciona para /login
  Todas as rotas /api/* → require_session() verifica o cookie
"""

import os
import time
import secrets
import logging
from typing import Optional

from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────

COOKIE_NAME      = "bot_session"
SESSION_MAX_AGE  = int(os.getenv("SESSION_MAX_AGE_HOURS", "24")) * 3600

# Em produção com HTTPS, definir SECURE_COOKIES=true no .env
# para que o browser só envie o cookie por HTTPS.
SECURE_COOKIES   = os.getenv("SECURE_COOKIES", "false").lower() == "true"


# ── Armazenamento de sessões em memória ───────────────────────────────────────
# {session_id: {"username": str, "expires_at": float}}
# Sessões são perdidas ao reiniciar o bot (comportamento intencional — força
# novo login). Para persistência usaríamos BD, mas é desnecessário aqui.
_sessions: dict[str, dict] = {}


def create_session(username: str) -> str:
    """Cria uma sessão segura e devolve o ID para colocar no cookie."""
    # token_urlsafe(32) gera 256 bits de entropia — imune a brute force
    session_id = secrets.token_urlsafe(32)
    _sessions[session_id] = {
        "username":   username,
        "expires_at": time.time() + SESSION_MAX_AGE,
        "created_at": time.time(),
    }
    _cleanup_sessions()
    logger.info("Sessão criada para utilizador '%s'", username)
    return session_id


def get_session_user(session_id: Optional[str]) -> Optional[str]:
    """Devolve o username se a sessão é válida, None caso contrário."""
    if not session_id:
        return None
    session = _sessions.get(session_id)
    if not session:
        return None
    if session["expires_at"] < time.time():
        del _sessions[session_id]
        return None
    return session["username"]


def delete_session(session_id: Optional[str]) -> None:
    if session_id:
        _sessions.pop(session_id, None)


def _cleanup_sessions() -> None:
    """Remove sessões expiradas. Chamado ao criar novas sessões."""
    now = time.time()
    expired = [k for k, v in list(_sessions.items()) if v["expires_at"] < now]
    for k in expired:
        del _sessions[k]


# ── Rate limiting ─────────────────────────────────────────────────────────────
# Previne ataques de força bruta no endpoint de login.
# {ip: [timestamp_da_tentativa, ...]}
_rate_limit: dict[str, list[float]] = {}

MAX_ATTEMPTS   = 5    # tentativas permitidas por janela
WINDOW_SECONDS = 300  # janela de 5 minutos


def _client_ip(request: Request) -> str:
    """IP real do cliente — respeita X-Forwarded-For do Nginx."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def check_rate_limit(request: Request) -> None:
    """Levanta 429 se o IP ultrapassou MAX_ATTEMPTS na janela WINDOW_SECONDS."""
    ip = _client_ip(request)
    now = time.time()

    # Mantém apenas tentativas dentro da janela actual
    attempts = [t for t in _rate_limit.get(ip, []) if now - t < WINDOW_SECONDS]
    _rate_limit[ip] = attempts

    if len(attempts) >= MAX_ATTEMPTS:
        wait = int(WINDOW_SECONDS - (now - attempts[0]))
        logger.warning("Rate limit atingido — IP: %s", ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Demasiadas tentativas falhadas. Aguarda {wait}s antes de tentar novamente.",
        )


def record_failed_attempt(request: Request) -> None:
    ip = _client_ip(request)
    _rate_limit.setdefault(ip, []).append(time.time())
    logger.warning("Login falhado — IP: %s", ip)


def clear_rate_limit(request: Request) -> None:
    """Limpa o contador após login bem-sucedido."""
    _rate_limit.pop(_client_ip(request), None)


# ── Dependência FastAPI ───────────────────────────────────────────────────────

def require_session(request: Request) -> str:
    """
    Dependência para rotas protegidas.
    Valida o cookie de sessão e devolve o username autenticado.
    Levanta 401 se a sessão não existir ou estiver expirada.
    """
    session_id = request.cookies.get(COOKIE_NAME)
    username   = get_session_user(session_id)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessão inválida ou expirada — faça login novamente.",
        )
    return username
