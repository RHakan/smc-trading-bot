"""
Encriptação simétrica para credenciais sensíveis (API keys Binance).
Usa Fernet (AES-128-CBC + HMAC-SHA256) com chave derivada do APP_SECRET.
O APP_SECRET nunca é a API key — é apenas a chave mestre da aplicação.
"""

import base64
import hashlib
import hmac
import os
import secrets

from cryptography.fernet import Fernet

# ── Hash de senha (scrypt, stdlib) ────────────────────────────────────────────
# A senha do dashboard é HASHEADA, não encriptada: é irreversível, por isso
# sobrevive mesmo que a BD E o APP_SECRET vazem juntos (ao contrário da
# encriptação, que é reversível com a chave). O login não precisa de saber a
# senha — só de confirmar que a introduzida gera o mesmo hash.
_SCRYPT_N = 2 ** 14   # custo CPU/memória (16384) — forte e rápido o suficiente
_SCRYPT_R = 8
_SCRYPT_P = 1


def hash_password(password: str) -> str:
    """Devolve 'scrypt$N$r$p$salt_b64$hash_b64' — salt aleatório por senha."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt,
                        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return (f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}$"
            f"{base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}")


def verify_password(password: str, stored: str) -> bool:
    """Confirma a senha contra o hash guardado, em tempo constante."""
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.scrypt(password.encode(), salt=salt,
                            n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# ── Validação do APP_SECRET ───────────────────────────────────────────────────
# O APP_SECRET é a chave mestra que encripta as credenciais da exchange na base
# de dados. Se ficar no valor de exemplo — que está publicado no .env.example e
# é portanto conhecido por qualquer pessoa — as credenciais guardadas deixam de
# estar protegidas: basta ter o ficheiro da BD para as decifrar. Por isso a
# aplicação RECUSA ARRANCAR nesse estado, em vez de deixar passar em silêncio.

_COMANDO_GERAR = 'python -c "import secrets; print(secrets.token_hex(32))"'

# Valores de exemplo que já circularam neste projeto. Listados explicitamente
# para apanhar instalações antigas que os tenham copiado.
_PLACEHOLDERS_CONHECIDOS = {
    "cole_aqui_um_valor_gerado_com_secrets_token_hex_32",
    "gere_o_seu_com_secrets_token_hex_32",
    "changeme",
    "change_me",
}

# Palavras que só aparecem em texto de instrução, nunca num segredo real.
_MARCAS_DE_EXEMPLO = ("cole_aqui", "gere_o_seu", "token_hex", "placeholder",
                      "exemplo", "example", "changeme", "change_me", "your_")

# Um token_hex(32) tem 64 caracteres. Aceitam-se outras formas (uma frase-passe
# longa, por exemplo), mas abaixo disto não há entropia que chegue.
_COMPRIMENTO_MINIMO = 32


def validate_app_secret(secret: str | None = None) -> str:
    """Valida o APP_SECRET e devolve-o. Levanta RuntimeError se não servir.

    Chamada no arranque (api/main.py) para falhar cedo e com uma mensagem útil,
    e também por _get_cipher(), para que nenhum caminho de código consiga
    encriptar com uma chave inválida — mesmo que o arranque seja contornado.
    """
    if secret is None:
        secret = os.getenv("APP_SECRET", "")
    secret = (secret or "").strip()

    if not secret:
        raise RuntimeError(
            "APP_SECRET não está definido."
            "\n  Defina-o no ficheiro .env, na raiz do projeto."
            f"\n  Gere um valor com:  {_COMANDO_GERAR}"
        )

    baixo = secret.lower()
    if baixo in _PLACEHOLDERS_CONHECIDOS or any(m in baixo for m in _MARCAS_DE_EXEMPLO):
        raise RuntimeError(
            "APP_SECRET ainda está com o valor de EXEMPLO."
            "\n  Esse valor é público (vem no .env.example), por isso as credenciais"
            "\n  da exchange guardadas na base de dados NÃO ficariam protegidas."
            f"\n  Gere um valor próprio com:  {_COMANDO_GERAR}"
            "\n  Depois cole-o no .env, na linha APP_SECRET="
        )

    if len(secret) < _COMPRIMENTO_MINIMO:
        raise RuntimeError(
            f"APP_SECRET é demasiado curto ({len(secret)} caracteres; "
            f"mínimo {_COMPRIMENTO_MINIMO})."
            "\n  Um segredo curto é adivinhável por força bruta."
            f"\n  Gere um valor adequado com:  {_COMANDO_GERAR}"
        )

    return secret


def _get_cipher() -> Fernet:
    secret = validate_app_secret()
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt(value: str) -> str:
    """Encripta um valor e retorna string base64."""
    return _get_cipher().encrypt(value.encode()).decode()


def decrypt(value: str) -> str:
    """Desencripta um valor previamente encriptado por encrypt()."""
    return _get_cipher().decrypt(value.encode()).decode()
