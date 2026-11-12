"""Open WebUI connector: putting documents into its knowledge bases (and taking them out again).

GPU Shepherd talks to Open WebUI's own API as its admin. It doesn't need an API key from the
user: a short script run inside the Open WebUI container (TOKEN_SCRIPT) signs a login token for the
first admin with Open WebUI's own secret, the same kind of token its login page hands out.

A document goes in the way Open WebUI's own "upload to knowledge" does it: upload the file (Open WebUI
splits it into chunks and embeds them, with the embeddings model it is set to use), wait until it
is processed, then add it to the knowledge base. Text that is already in the knowledge base is
reported by Open WebUI as a duplicate, which counts as done.

Answers differ a little between Open WebUI versions (lists or pages, processing in the background
or not); the client accepts both.
"""
from __future__ import annotations

import time

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
    def __init__(self, base_url: str, token: str, timeout: float = 60.0, wait: float = 3600.0,
                 sleep=time.sleep):
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        self.timeout = timeout
        self.wait = wait                  # the longest one book may take to be processed
        self.sleep = sleep

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

    # ------------------------------------------------------------------ knowledge bases
    def knowledge_bases(self) -> dict[str, str]:
        """name -> id (newer versions answer a page at a time)"""
        found: dict[str, str] = {}
        for page in range(1, 1000):
            data = self._call("GET", f"/api/v1/knowledge/?page={page}")
            items = data.get("items", []) if isinstance(data, dict) else data or []
            for k in items:
                if isinstance(k, dict) and k.get("id"):
                    found.setdefault(k.get("name", ""), k["id"])
            if not isinstance(data, dict) or not items or len(found) >= int(data.get("total") or 0):
                break
        return found

    def knowledge_list(self) -> list[dict]:
        """Every knowledge base: {"id", "name", "description", "files"} (files: how many, None when
        this version doesn't say in the list; see file_count)."""
        out: list[dict] = []
        seen: set[str] = set()
        for page in range(1, 1000):
            data = self._call("GET", f"/api/v1/knowledge/?page={page}")
            items = data.get("items", []) if isinstance(data, dict) else data or []
            for k in items:
                if isinstance(k, dict) and k.get("id") and k["id"] not in seen:
                    seen.add(k["id"])
                    files = k.get("files")
                    if files is None and isinstance(k.get("data"), dict):
                        files = k["data"].get("file_ids")
                    out.append({"id": k["id"], "name": k.get("name", ""), "description": k.get("description") or "",
                                "files": len(files) if isinstance(files, list) else None})
            if not isinstance(data, dict) or not items or len(seen) >= int(data.get("total") or 0):
                break
        return out

    def file_count(self, kid: str) -> int | None:
        """How many files a knowledge base has (None when it can't tell). Older versions list them
        with the knowledge base, newer ones a page at a time."""
        data = self._call("GET", f"/api/v1/knowledge/{kid}", ok404=True)
        if isinstance(data, dict):
            files = data.get("files")
            if files is None and isinstance(data.get("data"), dict):
                files = data["data"].get("file_ids")
            if isinstance(files, list):
                return len(files)
        page = self._call("GET", f"/api/v1/knowledge/{kid}/files?page=1", ok404=True)
        if isinstance(page, dict) and page.get("total") is not None:
            return int(page["total"])
        if isinstance(page, list):
            return len(page)
        return None

    def knowledge_id(self, name: str, description: str = "") -> str:
        """The id of the knowledge base called `name`, made if there isn't one."""
        found = self.knowledge_bases().get(name)
        if found:
            return found
        data = self._call("POST", "/api/v1/knowledge/create", json={"name": name, "description": description})
        if not isinstance(data, dict) or not data.get("id"):
            raise OpenWebUIError("Open WebUI didn't make the knowledge base")
        return data["id"]

    # ------------------------------------------------------------------ files
    def upload(self, file_name: str, text: str) -> str:
        # processed before it answers (older versions always do), so it can go straight into the knowledge base
        data = self._call("POST", "/api/v1/files/?process=true&process_in_background=false", timeout=self.wait,
                          files={"file": (file_name, text.encode("utf-8"), "text/markdown")})
        if not isinstance(data, dict) or not data.get("id"):
            raise OpenWebUIError("Open WebUI didn't take the file")
        fid = data["id"]
        self._wait_processed(fid)
        return fid

    def _wait_processed(self, fid: str) -> None:
        """Newer versions process an upload in the background: wait for it. Older ones are done
        when the upload answers (and have no status to ask)."""
        start = time.monotonic()
        while True:
            st = self._call("GET", f"/api/v1/files/{fid}/process/status", ok404=True)
            status = (st or {}).get("status", "completed") if isinstance(st, dict) else "completed"
            if status == "completed":
                return
            if status == "failed":
                raise OpenWebUIError("Open WebUI couldn't process it: " + str((st or {}).get("error", ""))[:300])
            if time.monotonic() - start > self.wait:
                raise OpenWebUIError("Open WebUI took too long to process it")
            self.sleep(2)

    def add(self, kid: str, fid: str) -> None:
        try:
            self._call("POST", f"/api/v1/knowledge/{kid}/file/add", timeout=self.wait, json={"file_id": fid})
        except OpenWebUIError as err:
            if "duplicate" not in str(err).lower():
                raise

    def remove(self, kid: str, fid: str) -> None:
        self._call("POST", f"/api/v1/knowledge/{kid}/file/remove", ok404=True, json={"file_id": fid})

    def delete_file(self, fid: str) -> None:
        """Delete an uploaded file (and its search index) from Open WebUI."""
        self._call("DELETE", f"/api/v1/files/{fid}", ok404=True)

    def knowledge_empty(self, kid: str) -> bool:
        """Whether a knowledge base has no files left (False when it can't tell)."""
        data = self._call("GET", f"/api/v1/knowledge/{kid}", ok404=True)
        if not isinstance(data, dict):
            return False
        files = data.get("files")
        if files is None:
            files = (data.get("data") or {}).get("file_ids") if isinstance(data.get("data"), dict) else None
        return isinstance(files, list) and not files

    def delete_knowledge(self, kid: str) -> None:
        self._call("DELETE", f"/api/v1/knowledge/{kid}/delete", ok404=True)

    # ------------------------------------------------------------------ settings
    def embedding(self) -> dict:
        """Which embeddings model Open WebUI uses: engine, model and (for OpenAI) the address."""
        data = self._call("GET", "/api/v1/retrieval/embedding", ok404=True) or {}
        if not isinstance(data, dict):
            data = {}
        url = (data.get("openai_config") or {}).get("url", "")
        return {"engine": data.get("RAG_EMBEDDING_ENGINE", data.get("embedding_engine", "")) or "",
                "model": data.get("RAG_EMBEDDING_MODEL", data.get("embedding_model", "")) or "", "url": url or ""}

    def use_embedding(self, url: str, key: str, model: str, query_prefix: str = "", batch: int = 16) -> None:
        """Embed with a model server (OpenAI-style API). Field names differ between versions: send both."""
        body = {"RAG_EMBEDDING_ENGINE": "openai", "RAG_EMBEDDING_MODEL": model, "RAG_EMBEDDING_BATCH_SIZE": batch,
                "embedding_engine": "openai", "embedding_model": model, "embedding_batch_size": batch,
                "openai_config": {"url": url, "key": key}}
        if query_prefix:
            body["RAG_EMBEDDING_QUERY_PREFIX"] = query_prefix
        self._call("POST", "/api/v1/retrieval/embedding/update", json=body)

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

