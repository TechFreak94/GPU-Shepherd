"""Password hashing (scrypt, standard library), the first-run admin password, what keeps a login
valid (session_stamp) and the limit on wrong logins (LoginLimiter)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import threading
import time
from pathlib import Path

from .config import Settings, Store

N, R, P = 2 ** 14, 8, 1
MIN_LENGTH = 10           # for a password the admin picks (Settings, or --reset-password)
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


def same(a: str, b: str) -> bool:
    """Compare two strings without telling how much of them matched (any characters, not only ASCII)."""
    return hmac.compare_digest(a.encode(), b.encode())


def session_stamp(settings: Settings) -> str:
    """What a login cookie must carry to still count. It changes when the password does (changed
    under Settings, or reset with --reset-password) and at every logout (session_epoch), so a
    cookie copied before that stops working although its signature is still good."""
    text = f"{settings.session_epoch}:{settings.password_hash}".encode()
    return hmac.new(settings.session_secret.encode(), text, hashlib.sha256).hexdigest()[:32]


class LoginLimiter:
    """Slows down password guessing: after MAX wrong logins from one address within WINDOW
    seconds, that address waits until the oldest of them is WINDOW old. A second, higher limit
    counts every address together, for a guesser who has many. Kept in memory: a restart clears
    it, and --reset-password never goes through it."""
    MAX, MAX_ALL, WINDOW = 5, 50, 300

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._failed: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, now: float) -> None:
        for client, times in list(self._failed.items()):
            times[:] = [t for t in times if now - t < self.WINDOW]
            if not times:
                del self._failed[client]

    def wait(self, client: str) -> int:
        """Seconds this address has to wait before its next try (0 = it may try now)."""
        with self._lock:
            now = self.clock()
            self._recent(now)
            own = self._failed.get(client, [])
            everyone = sorted(t for times in self._failed.values() for t in times)
            if len(own) >= self.MAX:
                return max(int(self.WINDOW - (now - own[-self.MAX])) + 1, 1)
            if len(everyone) >= self.MAX_ALL:
                return max(int(self.WINDOW - (now - everyone[-self.MAX_ALL])) + 1, 1)
            return 0

    def failed(self, client: str) -> None:
        with self._lock:
            self._failed.setdefault(client, []).append(self.clock())

    def passed(self, client: str) -> None:
        with self._lock:
            self._failed.pop(client, None)


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


def reset_admin(store: Store, password: str = "") -> tuple[str, str]:
    """Give the admin a new password (for a lost one): the one given, or a generated one when none
    is. A generated one is also kept in INITIAL_PASSWORD.txt; a chosen one is not written anywhere,
    and an old INITIAL_PASSWORD.txt is deleted, as it no longer holds the password. Returns
    (user, password)."""
    if password and len(password) < MIN_LENGTH:
        raise ValueError(f"Use at least {MIN_LENGTH} characters.")
    settings = store.settings()
    chosen = bool(password)
    password = password or generate_password()
    settings.password_hash = hash_password(password)
    store.save_settings(settings)
    if chosen:
        initial_password_file(store).unlink(missing_ok=True)
    else:
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
