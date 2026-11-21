"""Open WebUI connector: changing its settings through its own API (the picture settings).

GPU Shepherd talks to Open WebUI's own API as its admin. It doesn't need an API key from the
user: a short script run inside the Open WebUI container (TOKEN_SCRIPT) signs a login token for the
first admin with Open WebUI's own secret, the same kind of token its login page hands out.
"""
from __future__ import annotations

import httpx

CONTAINER_PORT = 8080

# run with python3 in the Open WebUI container; prints {"token": ...} or {"error": ...}
TOKEN_SCRIPT = """
import json, os, sqlite3, time, uuid
try:
    import jwt
    key = os.environ.get("WEBUI_SECRET_KEY") or os.environ.get("WEBUI_JWT_SECRET_KEY")
    if not key:                       # start.sh keeps a made-up key in a file beside it
        f = os.environ.get("WEBUI_SECRET_KEY_FILE") or ".webui_secret_key"
        f = f if os.path.isabs(f) else os.path.join("/app/backend", f)
        key = open(f).read().strip() if os.path.exists(f) else "t0p-s3cr3t"   # older versions' default
    db = sqlite3.connect(os.path.join(os.environ.get("DATA_DIR", "/app/backend/data"), "webui.db"))
    row = db.execute("select id from user where role = 'admin' order by created_at limit 1").fetchone()
    if not row:
        print(json.dumps({"error": "Open WebUI has no admin yet: make the first account in Open WebUI"}))
    else:
        now = int(time.time())
        claims = {"id": row[0], "typ": "session", "auth_method": "password", "auth_time": now, "iat": now,
                  "exp": now + 12 * 3600, "jti": str(uuid.uuid4()), "mfa_verified": True}
        try:                          # newer versions: the token must carry the account's session stamp
            stamp = db.execute("select session_stamp from auth where id = ?", (row[0],)).fetchone()
            if stamp:
                claims["session_stamp"] = stamp[0]
        except sqlite3.OperationalError:
            pass
        print(json.dumps({"token": jwt.encode(claims, key, algorithm="HS256")}))
except Exception as e:
    print(json.dumps({"error": f"{e.__class__.__name__}: {e}"}))
"""


class OpenWebUIError(Exception):
    pass


class OpenWebUI:
    def __init__(self, base_url: str, token: str, timeout: float = 60.0):
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        self.timeout = timeout

    def _call(self, method: str, path: str, ok404: bool = False, **kw):
        try:
            r = httpx.request(method, self.base + path, headers=self.headers,
                              timeout=kw.pop("timeout", self.timeout), **kw)
        except httpx.HTTPError as err:
            raise OpenWebUIError(f"couldn't reach Open WebUI: {err.__class__.__name__}: {err}") from err
        if ok404 and r.status_code in (404, 405):
            return None
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail") or r.text
            except ValueError:
                detail = r.text
            raise OpenWebUIError(f"Open WebUI answered {r.status_code}: {str(detail)[:300]}")
        try:
            return r.json()
        except ValueError:
            return None

    # ------------------------------------------------------------------ settings
    def images(self) -> dict:
        """Open WebUI's image settings (making and editing pictures)."""
        data = self._call("GET", "/api/v1/images/config", ok404=True)
        return data if isinstance(data, dict) else {}

    def use_images(self, values: dict) -> None:
        """Change some image settings, keeping the rest (Open WebUI wants them all in one go)."""
        current = self.images()
        if not current:
            raise OpenWebUIError("this Open WebUI has no image settings to change: update it to a newer version")
        self._call("POST", "/api/v1/images/config/update", json=current | values)

