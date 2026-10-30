"""Learning how a model server's models really run, and the adjustments offered from it.

Measurements (<data>/state/models.yaml): for each server and model, and each number of GPUs it
ran on, the most GPU memory it used and how fast it answered (tokens a second, averaged over
recent answers). The service measures running servers in the background (Shepherd.measure).

Suggestions are worked out from the measurements and the catalogue (suggest(), pure). Nothing
changes until the user applies one; the reasons say what changes and why. Every decision is
kept in <data>/state/tuning.yaml with the value it replaced, so an applied suggestion can be
undone and a dismissed one applied later. With Catalog.auto_tune on, new suggestions are
applied as they come, and stay listed so they can be undone the same way.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from . import placement
from .config import Catalog

GB = 1e9
FULL = 0.95            # a model using more than this share of its GPUs' memory may run out
SPEED_GAIN = 1.10      # a GPU count must answer this much faster to be suggested
MIN_SAMPLES = 3        # answers measured before speeds are compared
MAX_SAMPLES = 20       # the average follows the most recent answers


@dataclass
class Suggestion:
    key: str             # stable: the same advice gets the same key, so a dismissal sticks
    server: str
    model: str           # "" for a setting of the whole server
    field: str           # "gpus" (CatalogModel) or "fit_gpus" (Catalog)
    value: object
    old: object
    title: str
    reason: str
    status: str = "new"  # new | applied | dismissed
    at: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class _YamlState:
    """A small YAML file under <data>/state, written atomically."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            try:
                raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
                self._data = raw if isinstance(raw, dict) else {}
            except (OSError, yaml.YAMLError):
                self._data = {}
        return self._data

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump(self._data, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass


class Measurements(_YamlState):
    """server -> model -> GPUs it ran on -> {"vram": most bytes used, "tps": tokens/s, "n": answers}."""

    def __init__(self, data_dir: Path):
        super().__init__(Path(data_dir) / "state" / "models.yaml")

    def get(self, server: str) -> dict[str, dict[int, dict]]:
        with self._lock:
            raw = self._load().get(server) or {}
            return {m: {int(k): dict(v) for k, v in per.items()} for m, per in raw.items()}

    def record(self, server: str, model: str, gpus: int, vram: int = 0, tps: float = 0.0) -> None:
        with self._lock:
            entry = self._load().setdefault(server, {}).setdefault(model, {}).setdefault(gpus, {})
            changed = False
            if vram > entry.get("vram", 0) * 1.02:
                entry["vram"], changed = int(vram), True
            if tps > 0:
                n = min(entry.get("n", 0), MAX_SAMPLES - 1)
                entry["tps"] = round((entry.get("tps", 0.0) * n + tps) / (n + 1), 2)
                entry["n"], changed = n + 1, True
            if changed:
                self._save()

    def forget(self, server: str, model: str | None = None) -> None:
        with self._lock:
            data = self._load()
            if model is None:
                data.pop(server, None)
            else:
                (data.get(server) or {}).pop(model, None)
            self._save()


class Decisions(_YamlState):
    """Suggestion key -> the suggestion as decided (status applied or dismissed, the old value)."""

    def __init__(self, data_dir: Path):
        super().__init__(Path(data_dir) / "state" / "tuning.yaml")

    def for_server(self, server: str) -> dict[str, Suggestion]:
        with self._lock:
            return {k: Suggestion(**v) for k, v in self._load().items() if v.get("server") == server}

    def put(self, s: Suggestion, status: str) -> Suggestion:
        s = Suggestion(**(s.as_dict() | {"status": status, "at": time.strftime("%Y-%m-%d %H:%M")}))
        with self._lock:
            self._load()[s.key] = s.as_dict()
            self._save()
        return s


def _gb(n: float) -> str:
    return f"{n / GB:.1f} GB"


def _gpus(n: int) -> str:
    return "one GPU" if n == 1 else f"{n} GPUs"


def suggest(cat: Catalog, measured: dict[str, dict[int, dict]], needs: dict[str, int],
            gpu_mem: list[int]) -> list[Suggestion]:
    """What could run better on a server with these GPUs (`gpu_mem`, bytes each, all it may get)."""
    out, n, iid = [], len(gpu_mem), cat.instance_id
    if not n:
        return out
    if cat.fit_gpus is None:
        on = cat.model_copy(update={"fit_gpus": True})
        placed = placement.place(on, needs, gpu_mem)
        moves = [f"{mid} on {_gpus(len(pos))}" for mid, pos in placed.items()]
        wants = placement.wants(on, needs, gpu_mem)
        reason = ("Each model would run on the fewest GPUs that hold it" +
                  (f" ({', '.join(moves)})" if moves else "") + "." +
                  (f" The server would take {wants} of the {n} GPUs and leave the rest free for other apps."
                   if wants and wants < n else "") +
                  " A model on fewer GPUs answers about as fast, as no work passes between GPUs.")
        out.append(Suggestion(f"{iid}::fit_gpus", iid, "", "fit_gpus", True, None,
                              "Run each model on only the GPUs it needs", reason))
    placed = placement.place(cat, needs, gpu_mem)
    for m in cat.models:
        if placement.on_cpu(m, n):
            continue
        k = len(placed.get(m.id, [])) or n
        runs = measured.get(m.id, {})
        now = runs.get(k, {})
        usable = sum(max(x - placement.PER_GPU, 0) for x in gpu_mem[:k])
        advice = None
        if now.get("vram") and usable and now["vram"] > FULL * usable and k < n:
            advice = (k + 1, f"Give {m.id} another GPU",
                      f"It used up to {_gb(now['vram'])} of the {_gb(usable)} its {_gpus(k)} can hold, so a long "
                      "chat may run it out of memory.")
        speeds = {g: r["tps"] for g, r in runs.items() if r.get("n", 0) >= MIN_SAMPLES and r.get("tps")}
        if advice is None and k in speeds:
            best = max(speeds, key=speeds.get)
            if best != k and best <= n and speeds[best] >= speeds[k] * SPEED_GAIN:
                advice = (best, f"Run {m.id} on {_gpus(best)}",
                          f"It answered at {speeds[best]:.0f} tokens a second on {_gpus(best)} and "
                          f"{speeds[k]:.0f} on {_gpus(k)}.")
        if advice is None and now.get("vram") and k > 1:
            fewer = placement.fits_on(int(now["vram"] * 1.05), gpu_mem[:k])
            if 0 < fewer < k:
                advice = (fewer, f"Run {m.id} on {_gpus(fewer)}",
                          f"It used at most {_gb(now['vram'])} on {_gpus(k)}, which fits on {_gpus(fewer)}. "
                          "Fewer GPUs answer about as fast and leave the others free for more models.")
        if advice:
            value, title, reason = advice
            out.append(Suggestion(f"{iid}:{m.id}:gpus={value}", iid, m.id, "gpus", value, m.gpus, title, reason))
    return out


def listing(current: list[Suggestion], decided: dict[str, Suggestion]) -> list[Suggestion]:
    """Current suggestions with their decisions, then applied ones that no longer come up."""
    out = []
    for s in current:
        d = decided.get(s.key)
        out.append(d if d and d.status != "applied" else s)
    shown = {s.key for s in out}
    out += [d for k, d in decided.items() if k not in shown and d.status == "applied"]
    out += [d for k, d in decided.items() if k not in shown and d.status == "dismissed"
            and d.field == "fit_gpus"]
    order = {"new": 0, "applied": 1, "dismissed": 2}
    return sorted(out, key=lambda s: order.get(s.status, 3))


def metrics_speeds(items: list[dict], since: float) -> tuple[list[tuple[str, float]], float]:
    """(model, tokens/s) of llama-swap's answers newer than `since`, and the newest time seen.
    Its metrics have model, tokens_per_second (or output_tokens and duration_ms) and a timestamp."""
    out, newest = [], since
    for it in items:
        if not isinstance(it, dict):
            continue
        ts = _time(it.get("timestamp"))
        if ts <= since:
            continue
        newest = max(newest, ts)
        tps = it.get("tokens_per_second")
        if not tps and it.get("output_tokens") and it.get("duration_ms"):
            tps = it["output_tokens"] / (it["duration_ms"] / 1000)
        if isinstance(it.get("model"), str) and isinstance(tps, (int, float)) and tps > 0 \
                and (it.get("output_tokens") or 0) >= 16:          # very short answers say little
            out.append((it["model"], float(tps)))
    return out, newest


def _time(v) -> float:
    if isinstance(v, (int, float)):
        return float(v) / (1000 if v > 1e11 else 1)
    if isinstance(v, str):
        try:
            from datetime import datetime
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0
