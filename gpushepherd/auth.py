"""Password hashing (scrypt, standard library) and first-run admin password."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from pathlib import Path

from .config import Store

N, R, P = 2 ** 14, 8, 1
# No look-alikes (0/O, 1/l/I) and no punctuation: the first password is often read off a terminal
# and typed in by hand on another machine.
ALPHABET = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def generate_password(length: int = 16) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=N, r=R, p=P, dklen=32)
    return "scrypt${}${}${}${}${}".format(N, R, P, base64.b64encode(salt).decode(), base64.b64encode(digest).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, digest = stored.split("$")
        if algo != "scrypt":
            return False
        calc = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                              dklen=len(base64.b64decode(digest)))
        return hmac.compare_digest(calc, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


def initial_password_file(store: Store) -> Path:
    return store.root / "INITIAL_PASSWORD.txt"


def _write_initial(store: Store, user: str, password: str) -> None:
    path = initial_password_file(store)
    path.write_text(f"GPU Shepherd first login\nuser: {user}\npassword: {password}\n"
                    "Change it under Settings; this file is deleted when you do.\n", encoding="utf-8")
    os.chmod(path, 0o600)


def ensure_admin(store: Store) -> str | None:
    """Make sure an admin password exists. Returns the password if one was just generated."""
    settings = store.settings()
    if settings.password_hash:
        return None
    password = os.environ.get("SHEPHERD_ADMIN_PASSWORD") or generate_password()
    settings.password_hash = hash_password(password)
    store.save_settings(settings)
    if os.environ.get("SHEPHERD_ADMIN_PASSWORD"):
        initial_password_file(store).unlink(missing_ok=True)   # left from an earlier run: no longer true
        return None
    _write_initial(store, settings.admin_user, password)
    return password


def reset_admin(store: Store) -> tuple[str, str]:
    """Give the admin a new generated password (for a lost one). Returns (user, password)."""
    settings = store.settings()
    password = generate_password()
    settings.password_hash = hash_password(password)
    store.save_settings(settings)
    _write_initial(store, settings.admin_user, password)
    return settings.admin_user, password


# --------------------------------------------------------------------------- API token
# The JSON API (api.py) and the `gpu-shepherd` command authenticate with one token, kept in the data
# folder readable only by its owner: whoever can read it can already change everything there.

def api_token_file(store: Store) -> Path:
    return store.root / "api-token"


def api_token(store: Store) -> str:
    """The API token, made on first use."""
    path = api_token_file(store)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    return token or new_api_token(store)


def new_api_token(store: Store) -> str:
    """Replace the API token; the old one stops working at once."""
    token = "gs-" + secrets.token_urlsafe(32)
    path = api_token_file(store)
    tmp = path.with_name(".api-token.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    tmp.replace(path)
    return token


def check_api_token(store: Store, presented: str) -> bool:
    return bool(presented) and hmac.compare_digest(presented.encode(), api_token(store).encode())
