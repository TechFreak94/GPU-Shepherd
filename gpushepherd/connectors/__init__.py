"""Connectors for model servers that GPU Shepherd can manage beyond start/stop."""
from __future__ import annotations

import os

from ..config import Instance


def instance_url(inst: Instance, container_port: int) -> str:
    """Inside a container on the shared network: talk to the instance by its alias.
    Otherwise (running GPU Shepherd directly on the host): use its published port."""
    mode = os.environ.get("SHEPHERD_CONNECT", "auto")
    if mode == "auto":
        mode = "network" if os.path.exists("/.dockerenv") else "host"
    if mode == "network":
        return f"http://{inst.id}:{container_port}"
    port = next((p.host for p in inst.ports if p.container == container_port), None)
    return f"http://127.0.0.1:{port or container_port}"
