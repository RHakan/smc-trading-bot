"""
Reset de emergência da autenticação do dashboard — REDE DE SEGURANÇA.

Corre na PRÓPRIA máquina (exige acesso físico ao ficheiro da BD e ao .env), por
isso é a saída se te trancares fora: perdeste a senha, ou o telemóvel com o Google
Authenticator, ou o 2FA ficou dessincronizado.

Uso (a partir da pasta do projeto, com o bot PARADO para evitar locks):

  python -m tools.reset_auth --status               # ver estado (user, 2FA on/off)
  python -m tools.reset_auth --set-password NOVA     # define nova senha
  python -m tools.reset_auth --disable-2fa           # desliga o 2FA
  python -m tools.reset_auth --set-user NOME         # muda o utilizador

Não imprime nem precisa da senha antiga — o acesso ao ficheiro É a autorização.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from api import db
from bot.crypto import encrypt, hash_password


def main():
    ap = argparse.ArgumentParser(description="Reset da auth do dashboard (rede de segurança local).")
    ap.add_argument("--status", action="store_true", help="mostra o estado atual")
    ap.add_argument("--set-password", metavar="NOVA", help="define uma nova senha")
    ap.add_argument("--disable-2fa", action="store_true", help="desliga o 2FA")
    ap.add_argument("--set-user", metavar="NOME", help="muda o nome de utilizador")
    args = ap.parse_args()

    db.init_db()
    row = db._auth_row()
    if not row:
        print("Nenhuma conta na BD ainda — arranca o bot uma vez para semear do .env.")
        return

    if args.status or not any([args.set_password, args.disable_2fa, args.set_user]):
        print(f"  utilizador : {db.get_auth_username()}")
        print(f"  2FA ativo  : {'SIM' if db.totp_enabled() else 'não'}")
        if not any([args.set_password, args.disable_2fa, args.set_user]):
            return

    if args.set_password:
        if len(args.set_password) < 12:
            print("ERRO: a senha deve ter pelo menos 12 caracteres."); return
        db.change_password(args.set_password)
        print("  -> senha redefinida.")

    if args.set_user:
        with db._connect() as conn:
            conn.execute("UPDATE dashboard_auth SET username_enc=?, updated_at=? WHERE id=?",
                         (encrypt(args.set_user), db._now(), row["id"]))
        print(f"  -> utilizador alterado para '{args.set_user}'.")

    if args.disable_2fa:
        db.disable_totp()
        print("  -> 2FA desligado.")

    print("\nFeito. Reinicia o bot se estava a correr.")


if __name__ == "__main__":
    main()
