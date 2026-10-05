"""
TOTP (RFC 6238) compatível com Google Authenticator — stdlib pura (hmac/hashlib).
Parâmetros do Google Authenticator: SHA1, 6 dígitos, passo de 30s, segredo base32.

Também gera o URI otpauth:// (para enrollment) e um QR em SVG (renderizado à mão a
partir da matriz do `qrcode`, sem depender de Pillow/lxml).

Nada aqui guarda estado — o segredo é fornecido por quem chama (a BD guarda-o
encriptado com Fernet, ver bot/crypto + api/db).
"""
import base64
import hmac
import hashlib
import secrets
import struct
import time
from urllib.parse import quote

STEP = 30      # segundos por código
DIGITS = 6
_ALG = hashlib.sha1


def generate_secret(length: int = 20) -> str:
    """Segredo aleatório em base32 (sem padding), como o Google Authenticator espera.
    20 bytes = 160 bits = o recomendado pelo RFC 4226."""
    return base64.b32encode(secrets.token_bytes(length)).decode("ascii").rstrip("=")


def _code_at(secret_b32: str, counter: int) -> str:
    # base32 precisa de padding para múltiplo de 8; o Authenticator omite-o.
    pad = "=" * (-len(secret_b32) % 8)
    key = base64.b32decode(secret_b32.upper() + pad)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, _ALG).digest()
    offset = digest[-1] & 0x0F
    binary = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(binary % (10 ** DIGITS)).zfill(DIGITS)


def now_code(secret_b32: str, at: float | None = None) -> str:
    t = int(at if at is not None else time.time())
    return _code_at(secret_b32, t // STEP)


def verify(secret_b32: str, code: str, at: float | None = None, window: int = 1) -> bool:
    """Valida um código, tolerando ±`window` passos (relógios ligeiramente dessincronizados).
    Comparação em tempo constante para não vazar informação por timing."""
    if not code or not code.strip().isdigit():
        return False
    code = code.strip()
    t = int(at if at is not None else time.time())
    counter = t // STEP
    for delta in range(-window, window + 1):
        if hmac.compare_digest(_code_at(secret_b32, counter + delta), code):
            return True
    return False


def provisioning_uri(secret_b32: str, account: str, issuer: str = "BotFutures") -> str:
    """URI otpauth:// para o Google Authenticator (via QR ou 'introduzir chave')."""
    label = quote(f"{issuer}:{account}")
    return (f"otpauth://totp/{label}?secret={secret_b32}"
            f"&issuer={quote(issuer)}&algorithm=SHA1&digits={DIGITS}&period={STEP}")


def qr_svg(data: str, box: int = 8, border: int = 2) -> str:
    """QR em SVG puro, renderizado a partir da matriz do `qrcode` (sem Pillow/lxml)."""
    import qrcode
    qr = qrcode.QRCode(border=border, box_size=1)
    qr.add_data(data)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    n = len(matrix)
    size = n * box
    rects = []
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell:
                rects.append(f'<rect x="{x*box}" y="{y*box}" width="{box}" height="{box}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" shape-rendering="crispEdges">'
        f'<rect width="{size}" height="{size}" fill="#fff"/>'
        f'<g fill="#000">{"".join(rects)}</g></svg>'
    )
