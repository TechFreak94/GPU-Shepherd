"""Let running work finish before a container is stopped.

A GPU swap, a restart or Stop used to stop containers at once, cutting off a chat answer or an
image half way. Now the engine first asks each container that is about to stop whether it is
busy (see DrainPolicy in config.py), and waits until it is idle, the timeout passes, or someone
presses "Stop now" or "Cancel" on the dashboard.

GPU Shepherd isn't in the path of the requests, so it can't turn new work away while it waits:
work that arrives in the meantime (a new chat message, another queued prompt) is waited for too,
up to the same timeout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import httpx

from .config import Catalog, Instance
from .connectors import instance_url, llamaswap
from .jobs import Cancelled

DEFAULT_PORTS = {"llama-swap": llamaswap.CONTAINER_PORT, "llama-server": 8080, "comfyui": 8188}


@dataclass
class Activity:
    busy: int | None        # pieces of work still running or queued; None = could not tell
    detail: str = ""        # e.g. "2 requests (coder)", or why it could not tell


def check_for(inst: Instance) -> str:
    """The check an instance uses, with "auto" worked out from its kind and image."""
    check = inst.drain.check
    if check != "auto":
        return check
    image = inst.image.lower()
    if inst.kind == "llama-swap" or "llama-swap" in image:
        return "llama-swap"
    if "comfy" in image:
        return "comfyui"
    if "llama.cpp" in image or "llama-cpp" in image or "llama-server" in image:
        return "llama-server"
    return "none"


def _port(inst: Instance, check: str) -> int:
    if inst.drain.port:
        return inst.drain.port
    default = DEFAULT_PORTS.get(check)
    ports = [p.container for p in inst.ports]
    if default in ports or not ports:
        return default or 80
    return ports[0]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _amount(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    if isinstance(value, (list, dict)):
        return len(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError(f"cannot read an amount of work from {value!r}")


def probe(inst: Instance, catalog: Catalog | None = None, timeout: float = 5.0) -> Activity:
    """Ask a running instance how much work it has. Never raises."""
    check = check_for(inst)
    if check == "none":
        return Activity(0)
    base = instance_url(inst, _port(inst, check))
    try:
        if check == "llama-swap":
            catalog = catalog or Catalog(instance_id=inst.id)
            client = llamaswap.LlamaSwapClient(base, catalog.api_key, timeout=timeout)
            got = client.in_flight({m.id for m in catalog.models})
            if got is None:
                return Activity(None, "llama-swap did not say which requests it is answering")
            total, per_model = got
            models = ", ".join(sorted(m for m in per_model if m != "?"))
            return Activity(total, _plural(total, "request") + (f" ({models})" if models else ""))
        if check == "llama-server":
            busy = llamaswap.busy_slots(_get(base + "/slots", timeout))
            return Activity(busy, _plural(busy, "request"))
        if check == "comfyui":
            q = _get(base + "/queue", timeout)
            running, queued = len(q.get("queue_running") or []), len(q.get("queue_pending") or [])
            detail = _plural(running, "prompt") + " running" + (f", {queued} queued" if queued else "")
            return Activity(running + queued, detail)
        data = _get(base + inst.drain.path, timeout)                      # http
        for part in [k for k in inst.drain.key.split(".") if k]:
            data = data[int(part)] if isinstance(data, list) else data[part]
        busy = _amount(data)
        return Activity(busy, f"{busy} busy")
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError) as err:
        return Activity(None, f"{check} check failed: {err.__class__.__name__}: {err}"[:200])


def _get(url: str, timeout: float):
    r = httpx.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _clock(secs: float) -> str:
    secs = max(int(secs), 0)
    return f"{secs // 60}:{secs % 60:02d}"


def wait_until_idle(instances: dict[str, Instance], iids: list[str], log, job=None,
                    ask: Callable[[Instance], Activity] = probe, interval: float = 2.0) -> None:
    """Block until every instance in `iids` has no running work (or its timeout passes).

    `job` (jobs.Job) shows the wait on the dashboard and carries the "Stop now" and "Cancel"
    buttons. Raises Cancelled when the change should not go ahead."""
    names = {i: instances[i].name for i in iids}
    pending = {i: time.monotonic() + instances[i].drain.timeout
               for i in iids if instances[i].drain.timeout > 0 and check_for(instances[i]) != "none"}
    started, said = time.monotonic(), set()
    try:
        while pending:
            if job is not None and job.cancel.is_set():
                raise Cancelled("cancelled while waiting for running work to finish; nothing was changed")
            if job is not None and job.skip.is_set():
                log("Stop now: not waiting any longer")
                return
            now = time.monotonic()
            for iid in list(pending):
                inst = instances[iid]
                act = ask(inst)
                if act.busy is None:
                    log(f"Could not tell whether {names[iid]} is busy ({act.detail}); not waiting for it")
                    del pending[iid]
                elif act.busy == 0:
                    if iid in said:
                        log(f"{names[iid]} finished its work after {_clock(now - started)}")
                    del pending[iid]
                elif now >= pending[iid]:
                    waited = _clock(inst.drain.timeout)
                    if inst.drain.on_timeout == "cancel":
                        raise Cancelled(f"{names[iid]} was still busy ({act.detail}) after {waited}; "
                                        "nothing was changed")
                    log(f"{names[iid]} is still busy ({act.detail}) after {waited}; stopping it anyway")
                    del pending[iid]
                else:
                    if iid not in said:
                        log(f"Waiting for {names[iid]} to finish: {act.detail}")
                        said.add(iid)
                    if job is not None:
                        job.waiting[iid] = {"name": names[iid], "detail": act.detail,
                                            "left": _clock(pending[iid] - now),
                                            "then": inst.drain.on_timeout}
                if iid not in pending and job is not None:
                    job.waiting.pop(iid, None)
            if pending:
                if job is not None:
                    job.skip.wait(interval)
                else:
                    time.sleep(interval)
    finally:
        if job is not None:
            job.waiting.clear()
