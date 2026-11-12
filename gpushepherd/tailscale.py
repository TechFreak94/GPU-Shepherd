"""Tailscale on the host: reach the dashboard and every app from your other devices, without opening ports.

Tailscale runs on the host, not in a container: the dashboard and the apps already publish their
ports there, so once the machine is in a tailnet they answer on its Tailscale name and address, and
the dashboard's app links follow whatever address the browser used. GPU Shepherd can't install or
log in Tailscale from inside its container, so update.sh's service on the host does it (dashboard
updates must be on), through the same request file as updates (updates.request("tailscale", ACTION)).
The host writes its answers to <data dir>/tailscale/:

  state.json   written by update.sh after every request: installed, wsl, port (the dashboard's port
               on the host), state (idle / busy / failed), message, at (ISO time in UTC)
  status.json  the raw `tailscale status --json` (BackendState, AuthURL, Self.DNSName, TailscaleIPs,
               CertDomains...)
  serve.json   the raw `tailscale serve status --json` (TCP/Web: what `tailscale serve` shares)
  log.txt      the output of the last request

install.sh --tailscale does the same at install time.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import updates

ACTIONS = ("refresh", "install", "up", "down", "https-on", "https-off", "logout")
STUCK = 900             # seconds: a request that reported nothing for this long has died
REFRESH_AFTER = 300     # seconds: the Settings card asks for a fresh status when it is older than this
URL = "https://tailscale.com"
ADMIN_DNS = "https://login.tailscale.com/admin/dns"


@dataclass
class TailscaleStatus:
    checked: bool = False           # the host has answered at least once
    installed: bool = False
    wsl: bool = False               # WSL2: Tailscale belongs on Windows instead
    port: int = 8090                # the dashboard's port on the host
    run: str = ""                   # idle / busy / failed: the last request
    message: str = ""
    at: str = ""
    age: float = 0.0
    backend: str = ""               # Running / NeedsLogin / NeedsMachineAuth / Stopped / Starting / NoState
    auth_url: str = ""              # the login link while logging in
    name: str = ""                  # this machine's full Tailscale name: box.tailnet-name.ts.net
    ips: list[str] = field(default_factory=list)
    tailnet: str = ""
    cert_domains: list[str] = field(default_factory=list)   # empty: HTTPS certificates are off in the tailnet
    https: bool = False             # the dashboard is served at https://<name>
    service: bool = False           # update.sh's service is there to do the work
    image: bool = False             # the published image: no update.sh at all
    command: str = ""               # update.sh, for the hints
    pending: str = ""               # a Tailscale request the host hasn't picked up yet
    pending_stale: bool = False
    stuck: bool = False

    @property
    def busy(self) -> bool:
        return (self.run == "busy" and not self.stuck) or (bool(self.pending) and not self.pending_stale)

    @property
    def connected(self) -> bool:
        return self.backend == "Running"

    @property
    def needs_login(self) -> bool:
        return self.backend in ("NeedsLogin", "NoState", "")

    @property
    def ip(self) -> str:
        return next((a for a in self.ips if "." in a), self.ips[0] if self.ips else "")

    @property
    def dashboard_urls(self) -> list[str]:
        """Where your other devices open the dashboard, best first."""
        if not self.connected:
            return []
        out = [f"https://{self.name}"] if self.https and self.name else []
        out += [f"http://{h}:{self.port}" for h in (self.name, self.ip) if h]
        return out


def folder(root: Path) -> Path:
    return Path(root) / "tailscale"


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def status(root: Path, now: float | None = None) -> TailscaleStatus:
    u = updates.status(root, now)
    st = TailscaleStatus(service=u.dashboard_updates and not u.image, image=u.image, command=u.command)
    d = folder(root)
    raw = _json(d / "state.json")
    if isinstance(raw, dict):
        st.checked = True
        st.installed = raw.get("installed") is True
        st.wsl = raw.get("wsl") is True
        try:
            st.port = int(raw.get("port") or 8090)
        except (TypeError, ValueError):
            pass
        st.run, st.message, st.at = (str(raw.get(k) or "") for k in ("state", "message", "at"))
        st.age = updates.seconds_since(st.at, now)
        st.stuck = st.run == "busy" and st.age > STUCK
    ts = _json(d / "status.json") if st.installed else None
    if isinstance(ts, dict):
        st.backend = str(ts.get("BackendState") or "")
        st.auth_url = str(ts.get("AuthURL") or "") if st.backend != "Running" else ""
        if not st.auth_url.startswith("https://"):
            st.auth_url = ""
        me = ts.get("Self") if isinstance(ts.get("Self"), dict) else {}
        st.name = str(me.get("DNSName") or "").rstrip(".")
        st.ips = [str(a) for a in ts.get("TailscaleIPs") or me.get("TailscaleIPs") or []]
        st.cert_domains = [str(c) for c in ts.get("CertDomains") or []]
        tn = ts.get("CurrentTailnet") if isinstance(ts.get("CurrentTailnet"), dict) else {}
        st.tailnet = str(tn.get("Name") or "")
    serve = _json(d / "serve.json") if st.installed else None
    if isinstance(serve, dict) and st.name:
        web = serve.get("Web") if isinstance(serve.get("Web"), dict) else {}
        st.https = any(str(k).split(":")[0] == st.name for k in web)
    req = updates.folder(root) / "request"
    try:
        lines = req.read_text().split()
        if lines and lines[0] == "tailscale":
            st.pending = lines[1] if len(lines) > 1 else "refresh"
            st.pending_stale = (now or time.time()) - req.stat().st_mtime > updates.STALE_REQUEST
    except OSError:
        pass
    return st


def needs_refresh(st: TailscaleStatus) -> bool:
    """The Settings card asks the host for a fresh status when it has none or an old one."""
    return st.service and not st.busy and (not st.checked or st.age < 0 or st.age > REFRESH_AFTER)


def request(root: Path, action: str) -> None:
    if action not in ACTIONS:
        raise ValueError(f"unknown Tailscale action {action!r}")
    updates.request(root, "tailscale", action)


def log_text(root: Path) -> str:
    try:
        return (folder(root) / "log.txt").read_text(errors="replace")[-20000:]
    except OSError:
        return ""
