"""
SQLite helper — inicialização do banco e funções de acesso.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "trades.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Cria as tabelas se não existirem."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT    NOT NULL,
                side          TEXT    NOT NULL,
                entry         REAL    NOT NULL,
                stop_loss     REAL,
                take_profit   REAL,
                exit_price    REAL,
                rr            REAL,
                leverage      INTEGER,
                margin_usdt   REAL,
                position_size_usdt REAL,
                order_id      TEXT,
                mode          TEXT,
                exchange_mode TEXT,
                strategy      TEXT,
                status        TEXT    DEFAULT 'open',
                pnl_usdt      REAL,
                pnl_pct       REAL,
                timestamp     TEXT    NOT NULL,
                closed_at     TEXT
            );

            CREATE TABLE IF NOT EXISTS api_credentials (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user       TEXT NOT NULL,
                mode       TEXT NOT NULL,
                api_key    TEXT NOT NULL,
                secret     TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user, mode)
            );

            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT    NOT NULL,
                strategy      TEXT,
                signal        TEXT,
                conditions    TEXT,
                entry         REAL,
                stop_loss     REAL,
                take_profit   REAL,
                rr            REAL,
                timestamp     TEXT    NOT NULL
            );

            -- Credenciais do dashboard (linha única = o admin). A senha é HASH
            -- (scrypt, irreversível); o utilizador e o segredo TOTP são encriptados
            -- (Fernet). Substitui DASHBOARD_USER/PASSWORD do .env.
            CREATE TABLE IF NOT EXISTS dashboard_auth (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                username_enc     TEXT    NOT NULL,
                pass_hash        TEXT    NOT NULL,
                totp_secret_enc  TEXT,
                totp_enabled     INTEGER DEFAULT 0,
                totp_pending_enc TEXT,
                recovery_hashes  TEXT,
                created_at       TEXT,
                updated_at       TEXT
            );
        """)

        # Migração: coluna que marca uma linha do histórico como fecho PARCIAL
        # (ex: 25.0). NULL = trade normal. SQLite não tem ADD COLUMN IF NOT EXISTS,
        # por isso verificamos o schema atual antes de alterar.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(trades)")}
        if "partial_pct" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN partial_pct REAL")
        # Migração: marca trades de estratégias antigas (fora do v3) como arquivados.
        # 1 = arquivado (não conta no placar do dashboard). NULL/0 = ativo.
        if "archived" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN archived INTEGER DEFAULT 0")
        # Migração (06/08/2026): MOTIVO do fecho. Sem isto o histórico diz o QUÊ
        # mas nunca o PORQUÊ — e foi por isso que não conseguimos explicar 7 trades
        # fechados em lote a 08/07 e 12/07. Valores usados:
        #   'stop' | 'alvo' | 'breakeven' | 'trailing' | 'invalidacao' | 'parcial'
        #   'manual' | 'exchange' | 'reconciliacao' | 'fantasma_evitado'
        # NULL = trade fechado antes desta migração (não sabemos o motivo).
        if "exit_reason" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN exit_reason TEXT")

    # Semeia a auth do dashboard a partir do .env, uma única vez. Mantém o login
    # atual a funcionar após a migração (mesmo user/senha, agora hasheados na BD).
    # A partir daí o .env deixa de ser fonte da senha — pode ser removido.
    if not auth_is_seeded():
        import os
        user = os.getenv("DASHBOARD_USER", "admin")
        pw = os.getenv("DASHBOARD_PASSWORD", "changeme")
        seed_auth(user, pw)


def save_trade(trade: dict) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (symbol, side, entry, stop_loss, take_profit, rr, leverage,
                margin_usdt, position_size_usdt, order_id, mode, exchange_mode,
                strategy, status, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade["symbol"], trade["side"], trade["entry"],
                trade["stop_loss"], trade["take_profit"], trade["rr"],
                trade["leverage"], trade["margin_usdt"],
                trade["position_size_usdt"], trade["order_id"],
                trade["mode"], trade["exchange_mode"], trade["strategy"],
                trade["status"], trade["timestamp"],
            ),
        )
        return cur.lastrowid


def close_trade(trade_id: int, exit_price: float, pnl_usdt: float,
                pnl_pct: float, closed_at: str, exit_reason: str | None = None) -> None:
    """Fecha um trade no histórico.

    `exit_reason` é opcional por compatibilidade, mas TODOS os pontos de fecho do
    bot devem passá-lo — é a única forma de responder "porque é que este trade
    fechou?" sem adivinhar a partir de timestamps.
    """
    with _connect() as conn:
        conn.execute(
            """UPDATE trades
               SET exit_price=?, pnl_usdt=?, pnl_pct=?, status='closed', closed_at=?,
                   exit_reason=COALESCE(?, exit_reason)
               WHERE id=?""",
            (exit_price, pnl_usdt, pnl_pct, closed_at, exit_reason, trade_id),
        )


def update_trade_size(trade_id: int, position_size_usdt: float,
                      margin_usdt: float) -> None:
    """Atualiza o tamanho restante de um trade ainda aberto após um fecho parcial."""
    with _connect() as conn:
        conn.execute(
            "UPDATE trades SET position_size_usdt=?, margin_usdt=? WHERE id=?",
            (position_size_usdt, margin_usdt, trade_id),
        )


def update_trade_levels(trade_id: int, stop_loss: float | None = None,
                        take_profit: float | None = None) -> None:
    """
    Atualiza SL e/ou TP de um trade aberto (ajuste manual pelo dashboard).
    Só altera os campos passados. Mantém o DB coerente com a exchange, para que
    a reconciliação no boot recoloque os níveis mais recentes após um restart.
    """
    sets, params = [], []
    if stop_loss is not None:
        sets.append("stop_loss=?"); params.append(stop_loss)
    if take_profit is not None:
        sets.append("take_profit=?"); params.append(take_profit)
    if not sets:
        return
    params.append(trade_id)
    with _connect() as conn:
        conn.execute(f"UPDATE trades SET {', '.join(sets)} WHERE id=?", params)


def record_partial_close(parent: dict, fraction: float, closed_notional: float,
                         closed_margin: float, exit_price: float,
                         pnl_usdt: float, pnl_pct: float, closed_at: str) -> int:
    """
    Insere no histórico uma linha JÁ FECHADA representando o fecho parcial de um
    trade. Herda os dados do trade-mãe (`parent`) e regista o tamanho/P&L da fatia.
    `partial_pct` marca a linha como parcial (ex: 25.0) para o dashboard destacá-la.
    """
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (symbol, side, entry, stop_loss, take_profit, rr, leverage,
                margin_usdt, position_size_usdt, order_id, mode, exchange_mode,
                strategy, status, pnl_usdt, pnl_pct, timestamp, closed_at,
                exit_price, partial_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed', ?, ?, ?, ?, ?, ?)""",
            (
                parent.get("symbol"), parent.get("side"), parent.get("entry"),
                parent.get("stop_loss"), parent.get("take_profit"), parent.get("rr"),
                parent.get("leverage"), round(closed_margin, 2),
                round(closed_notional, 2), parent.get("order_id"),
                parent.get("mode"), parent.get("exchange_mode"),
                parent.get("strategy"), round(pnl_usdt, 2), pnl_pct,
                closed_at, closed_at, exit_price, round(fraction * 100, 1),
            ),
        )
        return cur.lastrowid


def save_signal(signal: dict) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO signals
               (symbol, strategy, signal, conditions, entry, stop_loss,
                take_profit, rr, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal["symbol"], signal.get("strategy"), signal["signal"],
                json.dumps(signal.get("conditions", []), default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                signal.get("entry"), signal.get("stop_loss"),
                signal.get("take_profit"), signal.get("rr"),
                signal["timestamp"],
            ),
        )


def get_trades(limit: int = 100, status: str | None = None) -> list[dict]:
    with _connect() as conn:
        # Trades ABERTOS nunca são filtrados por 'archived'. O bot depende destas
        # buscas para reconciliar, fechar e mover níveis (5 pontos no engine/API):
        # arquivar um trade aberto deixaria a posição sem registo e partiria o fecho.
        # Arquivar é uma operação de HISTÓRICO — não pode tocar em estado operacional.
        if status == "open":
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        # Esconde os arquivados (testes antigos) da lista do dashboard.
        elif status:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status=? "
                "AND (archived IS NULL OR archived = 0) ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE (archived IS NULL OR archived = 0) "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def save_credentials(user: str, mode: str, api_key: str, api_secret: str) -> None:
    """Guarda as credenciais encriptadas na BD. Substitui se já existirem."""
    from bot.crypto import encrypt
    with _connect() as conn:
        conn.execute(
            """INSERT INTO api_credentials (user, mode, api_key, secret, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user, mode) DO UPDATE SET
                   api_key    = excluded.api_key,
                   secret     = excluded.secret,
                   created_at = excluded.created_at""",
            (user, mode, encrypt(api_key), encrypt(api_secret),
             datetime.now(timezone.utc).isoformat()),
        )


def get_credentials(user: str, mode: str) -> dict | None:
    """Retorna as credenciais desencriptadas ou None se não existirem."""
    from bot.crypto import decrypt
    with _connect() as conn:
        row = conn.execute(
            "SELECT api_key, secret FROM api_credentials WHERE user=? AND mode=?",
            (user, mode),
        ).fetchone()
    if not row:
        return None
    return {"api_key": decrypt(row["api_key"]), "secret": decrypt(row["secret"])}


def has_credentials(user: str, mode: str) -> bool:
    """Verifica se existem credenciais para este user e modo."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM api_credentials WHERE user=? AND mode=?",
            (user, mode),
        ).fetchone()
    return row is not None


def delete_credentials(user: str, mode: str) -> None:
    """Remove as credenciais do user para o modo indicado."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM api_credentials WHERE user=? AND mode=?",
            (user, mode),
        )


def get_stats() -> dict:
    """Retorna estatísticas gerais (win rate, total R, P&L)."""
    with _connect() as conn:
        # Só trades ATIVOS (não arquivados) entram no placar — os testes antigos
        # (bear_v12/v13) ficam no banco mas fora das estatísticas.
        rows = conn.execute(
            "SELECT pnl_usdt, rr FROM trades "
            "WHERE status='closed' AND (archived IS NULL OR archived = 0)"
        ).fetchall()

    if not rows:
        return {"total_trades": 0, "win_rate": 0, "total_pnl": 0, "total_r": 0}

    total = len(rows)
    wins = sum(1 for r in rows if (r["pnl_usdt"] or 0) > 0)
    total_pnl = sum((r["pnl_usdt"] or 0) for r in rows)

    return {
        "total_trades": total,
        "win_rate": round(wins / total * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "wins": wins,
        "losses": total - wins,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Autenticação do dashboard (BD) — senha HASH (scrypt), user/TOTP encriptados.
# Linha única (o admin). Todas as funções operam sobre a 1ª linha.
# ═══════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auth_row() -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM dashboard_auth ORDER BY id LIMIT 1").fetchone()
    return dict(r) if r else None


def auth_is_seeded() -> bool:
    return _auth_row() is not None


def seed_auth(username: str, password: str) -> None:
    """Cria a linha de auth (só se não existir). Semeia a partir do .env no boot."""
    if auth_is_seeded():
        return
    from bot.crypto import encrypt, hash_password
    with _connect() as conn:
        conn.execute(
            "INSERT INTO dashboard_auth (username_enc, pass_hash, totp_enabled, created_at, updated_at) "
            "VALUES (?, ?, 0, ?, ?)",
            (encrypt(username), hash_password(password), _now(), _now()),
        )


def get_auth_username() -> str | None:
    row = _auth_row()
    if not row:
        return None
    from bot.crypto import decrypt
    try:
        return decrypt(row["username_enc"])
    except Exception:
        return None


def verify_login(username: str, password: str) -> bool:
    """Confirma user + senha (tempo constante nas duas comparações)."""
    import hmac
    row = _auth_row()
    if not row:
        return False
    from bot.crypto import decrypt, verify_password
    try:
        stored_user = decrypt(row["username_enc"])
    except Exception:
        return False
    user_ok = hmac.compare_digest(stored_user.encode(), (username or "").encode())
    pass_ok = verify_password(password or "", row["pass_hash"])
    return user_ok and pass_ok


def change_password(new_password: str) -> None:
    from bot.crypto import hash_password
    row = _auth_row()
    if not row:
        raise RuntimeError("Auth não inicializada.")
    with _connect() as conn:
        conn.execute("UPDATE dashboard_auth SET pass_hash=?, updated_at=? WHERE id=?",
                     (hash_password(new_password), _now(), row["id"]))


# ── 2FA (TOTP) ───────────────────────────────────────────────────────────────

def totp_enabled() -> bool:
    row = _auth_row()
    return bool(row and row["totp_enabled"])


def begin_totp_enrollment(account: str) -> dict:
    """Gera um segredo PENDENTE (ainda não ativo) e devolve o URI + QR + chave.
    Só passa a ativo depois de confirm_totp() com um código válido."""
    from bot.crypto import encrypt
    from bot import totp as totp_mod
    row = _auth_row()
    if not row:
        raise RuntimeError("Auth não inicializada.")
    secret = totp_mod.generate_secret()
    with _connect() as conn:
        conn.execute("UPDATE dashboard_auth SET totp_pending_enc=?, updated_at=? WHERE id=?",
                     (encrypt(secret), _now(), row["id"]))
    uri = totp_mod.provisioning_uri(secret, account)
    return {"secret": secret, "uri": uri, "qr_svg": totp_mod.qr_svg(uri)}


def confirm_totp(code: str) -> list[str] | None:
    """Confirma o segredo pendente com um código. Se válido, ativa o 2FA, gera e
    devolve os códigos de recuperação (uma única vez). Devolve None se o código falhar."""
    from bot.crypto import decrypt, encrypt, hash_password
    from bot import totp as totp_mod
    row = _auth_row()
    if not row or not row["totp_pending_enc"]:
        return None
    secret = decrypt(row["totp_pending_enc"])
    if not totp_mod.verify(secret, code):
        return None
    import secrets as _s
    codes = [f"{_s.token_hex(5)}" for _ in range(8)]     # 8 códigos de 10 hex
    hashes = json.dumps([hash_password(c) for c in codes])
    with _connect() as conn:
        conn.execute(
            "UPDATE dashboard_auth SET totp_secret_enc=?, totp_pending_enc=NULL, "
            "totp_enabled=1, recovery_hashes=?, updated_at=? WHERE id=?",
            (encrypt(secret), hashes, _now(), row["id"]),
        )
    return codes


def verify_totp(code: str) -> bool:
    """Valida um código do Authenticator OU um código de recuperação (uso único)."""
    from bot.crypto import decrypt, verify_password
    from bot import totp as totp_mod
    row = _auth_row()
    if not row or not row["totp_enabled"] or not row["totp_secret_enc"]:
        return False
    secret = decrypt(row["totp_secret_enc"])
    if totp_mod.verify(secret, code):
        return True
    # fallback: código de recuperação (consome-o)
    if row["recovery_hashes"]:
        hashes = json.loads(row["recovery_hashes"])
        for h in hashes:
            if verify_password((code or "").strip(), h):
                hashes.remove(h)
                with _connect() as conn:
                    conn.execute("UPDATE dashboard_auth SET recovery_hashes=?, updated_at=? WHERE id=?",
                                 (json.dumps(hashes), _now(), row["id"]))
                return True
    return False


def disable_totp() -> None:
    row = _auth_row()
    if not row:
        return
    with _connect() as conn:
        conn.execute(
            "UPDATE dashboard_auth SET totp_enabled=0, totp_secret_enc=NULL, "
            "totp_pending_enc=NULL, recovery_hashes=NULL, updated_at=? WHERE id=?",
            (_now(), row["id"]),
        )
