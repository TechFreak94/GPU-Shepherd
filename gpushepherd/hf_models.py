"""Suggested models from Hugging Face: the GGUF models trending there now, from publishers that
quantize carefully, offered beside the built-in picks (model_library.py) when they fit this machine.

fetch() asks Hugging Face (the network). Library keeps its answer in hf-models.json in the data
folder and is refreshed about once a day in the background (Shepherd.watch_hf_models), so a page
never waits on it; without the network the list is simply empty and the built-in picks remain.
picks() is pure: for this machine's GPUs and RAM it chooses each model's quantization (a 4-bit one,
or a smaller one for a big model that would not fit otherwise), keeps what runs well, and drops what
duplicates a built-in pick. A model someone added or downloaded is remembered with the file it got,
so it stays on the list after it stops trending.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from . import model_library as ml

log = logging.getLogger(__name__)

API = "https://huggingface.co/api/models"
# publishers whose GGUF files are made with care and named the same way every time
PUBLISHERS = ("unsloth", "ggml-org", "bartowski", "lmstudio-community")
PER_PUBLISHER = 30           # trending models asked for from each publisher
CANDIDATES = 40              # models whose files are looked at (each one is a request)
SHOWN = 10                   # at most this many are offered on a machine
REFRESH_HOURS = 24
RETRY_HOURS = 1              # after a failed refresh
BIG = 30e9                   # parameters from which a 3- or 2-bit file is offered when 4-bit won't fit

# quantizations by preference: 4-bit first; the smaller ones only for big models
Q4 = ("UD-Q4_K_XL", "Q4_K_M", "Q4_K_L", "Q4_K_S", "IQ4_XS", "IQ4_NL", "MXFP4_MOE", "MXFP4")
Q3 = ("UD-Q3_K_XL", "Q3_K_XL", "Q3_K_M", "UD-IQ3_XXS", "IQ3_M", "IQ3_XXS")
Q2 = ("UD-Q2_K_XL", "Q2_K_L", "Q2_K", "UD-IQ2_M", "IQ2_M")
SMALL = 4.5e9                # below this, an 8-bit file: it is still small and noticeably better
Q8 = ("Q8_0", "UD-Q8_K_XL")

CHAT_PIPELINES = {"text-generation", "image-text-to-text", "", None}
SKIP = re.compile(r"abliterat|uncensor|heretic|nsfw|embed|rerank|whisper|tts|draft|"
                  r"flux|stable-diffusion|sdxl|wan2|hunyuanvideo|qwen-image", re.I)
MOE_ARCHS = re.compile(r"moe|gpt-?oss|deepseek2|deepseek3|llama4|qwen3next|minimax|dots1|grok|arctic|dbrx|"
                       r"mixtral|jamba|bailing|ernie4_5-moe|glm4moe|hunyuan-moe|kimi", re.I)
ACTIVE = ml.ACTIVE
SAFE_PATH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+/-]*\.gguf$", re.I)   # plain file names only
QUANT = re.compile(r"((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|MXFP4(?:_MOE)?|BF16|F16|F32))"
                   r"(?:-\d{5}-of-\d{5})?\.gguf$", re.I)


# --------------------------------------------------------------------------- asking Hugging Face

def _get(url: str, params=None):
    import httpx
    r = httpx.get(url, params=params, timeout=20, follow_redirects=True,
                  headers={"User-Agent": "gpu-shepherd (model suggestions)"})
    r.raise_for_status()
    return r.json()


def _wanted(row: dict) -> bool:
    g = row.get("gguf") or {}
    return (row.get("pipeline_tag") in CHAT_PIPELINES and bool(g.get("total")) and bool(g.get("context_length"))
            and not SKIP.search(row.get("id", "")))


def _license(tags: list) -> str:
    return next((t.split(":", 1)[1] for t in tags or [] if isinstance(t, str) and t.startswith("license:")), "")


def fetch(get=_get, publishers: tuple[str, ...] = PUBLISHERS) -> list[dict]:
    """The models to choose from, most trending first, each with its .gguf files and their sizes.
    One model per base model: the first publisher listed wins."""
    rows, seen = [], set()
    expand = [("expand[]", x) for x in ("gguf", "downloads", "likes", "trendingScore", "pipeline_tag", "tags",
                                         "lastModified")]
    for author in publishers:
        params = [("author", author), ("filter", "gguf"), ("sort", "trendingScore"), ("direction", "-1"),
                  ("limit", str(PER_PUBLISHER)), *expand]
        try:
            found = get(API, params)
        except Exception as err:  # noqa: BLE001 - one publisher failing leaves the others
            log.info("Hugging Face: could not list %s's models: %s", author, err)
            continue
        for row in found if isinstance(found, list) else []:
            base = base_name(row.get("id", ""))
            if not _wanted(row) or base in seen:
                continue
            seen.add(base)
            g = row["gguf"]
            rows.append({"repo": row["id"], "params": int(g["total"]), "arch": str(g.get("architecture", "")),
                         "ctx": int(g["context_length"]), "downloads": int(row.get("downloads") or 0),
                         "likes": int(row.get("likes") or 0), "trending": float(row.get("trendingScore") or 0),
                         "license": _license(row.get("tags")), "vision": row.get("pipeline_tag") == "image-text-to-text",
                         "modified": str(row.get("lastModified", ""))})
    rows.sort(key=lambda r: -r["trending"])
    out = []
    for r in rows[:CANDIDATES]:
        try:
            tree = get(f"{API}/{r['repo']}/tree/main", {"recursive": "true"})
        except Exception as err:  # noqa: BLE001
            log.info("Hugging Face: could not list the files of %s: %s", r["repo"], err)
            continue
        files = [{"path": f["path"], "size": int((f.get("lfs") or {}).get("size") or f.get("size") or 0)}
                 for f in tree if isinstance(f, dict) and f.get("type") == "file"
                 and SAFE_PATH.match(str(f.get("path", ""))) and ".." not in str(f["path"])]
        if files:
            out.append({**r, "files": files})
    return out


def base_name(repo: str) -> str:
    """unsloth/Qwen3.8-27B-GGUF and bartowski/Qwen_Qwen3.8-27B-GGUF -> qwen3.8-27b"""
    name = repo.rsplit("/", 1)[-1]
    name = re.sub(r"[-_.]gguf$", "", name, flags=re.I)
    name = re.sub(r"^[A-Za-z0-9-]+_(?=[A-Za-z])", "", name)       # bartowski's "Org_" prefix
    return name.lower()


# --------------------------------------------------------------------------- choosing files

def quant(path: str) -> str:
    m = QUANT.search(path.rsplit("/", 1)[-1])
    return m.group(1).upper() if m else ""


def _groups(cand: dict) -> dict[str, list[dict]]:
    """Quantization -> its model files (several when split), first part first."""
    out: dict[str, list[dict]] = {}
    for f in cand["files"]:
        name = f["path"].rsplit("/", 1)[-1].lower()
        if "mmproj" in name or name.startswith(("mtp-", "imatrix")):
            continue
        q = quant(f["path"])
        if q:
            out.setdefault(q, []).append(f)
    for parts in out.values():
        parts.sort(key=lambda f: f["path"])
    return {q: p for q, p in out.items() if _complete(p)}


def _complete(parts: list[dict]) -> bool:
    m = ml.SPLIT.search(parts[0]["path"])
    if not m:
        return len(parts) == 1
    n = int(re.search(r"of-(\d{5})", parts[0]["path"]).group(1))
    return len(parts) == n and "-00001-of-" in parts[0]["path"]


def _mmproj(cand: dict) -> dict | None:
    def order(f):
        n = f["path"].lower()
        return (0 if "f16" in n and "bf16" not in n else 1 if "bf16" in n else 3 if "f32" in n else 2, n)
    found = sorted((f for f in cand["files"] if "mmproj" in f["path"].lower()), key=order)
    return found[0] if found else None


def _gb(size: int) -> float:
    return round(size / 1e9, 2)


def _file(repo: str, f: dict) -> ml.File:
    return ml.File(name=f["path"].rsplit("/", 1)[-1], url=f"{ml.HF}/{repo}/resolve/main/{f['path']}", gb=_gb(f["size"]))


def moe(cand: dict) -> bool:
    return bool(MOE_ARCHS.search(cand.get("arch", "")) or ACTIVE.search(cand["repo"].rsplit("/", 1)[-1]))


def _sampling(cand: dict) -> str:
    arch, name = cand.get("arch", "").lower(), cand["repo"].lower()
    if arch.startswith("gemma"):
        return ml.GEMMA
    if arch.startswith("qwen") and "coder" in name:
        return ml.QWEN_CODER
    return ""                     # llama.cpp takes the model's own defaults from its file


def _billions(n: int) -> str:
    b = n / 1e9
    return f"{b:.0f}B" if b >= 10 else f"{b:.1f}B".replace(".0B", "B")


def key_for(cand: dict) -> str:
    return "hf-" + ml._slug(base_name(cand["repo"]))


def suggestion(cand: dict, q: str) -> ml.Suggestion | None:
    """The suggestion for one model in one quantization (None when it has no such file)."""
    parts = _groups(cand).get(q)
    if not parts:
        return None
    repo, name = cand["repo"], cand["repo"].rsplit("/", 1)[-1]
    first, rest = _file(repo, parts[0]), tuple(_file(repo, f) for f in parts[1:])
    proj = _mmproj(cand)
    coder = bool(re.search(r"coder|devstral|codestral|code", name, re.I))
    author = repo.split("/", 1)[0]
    blurb = (f"Trending on Hugging Face ({author}): {_billions(cand['params'])} parameters, "
             f"{cand['ctx'] // 1024}K context, {cand['downloads']:,} downloads last month. "
             f"{q} file{', reads images' if proj else ''}; llama.cpp's settings.")
    title = re.sub(r"[-_.]GGUF$", "", name, flags=re.I)
    title = re.sub(r"^[A-Za-z0-9-]+_(?=[A-Za-z])", "", title)
    return ml.Suggestion(
        key=key_for(cand), name=title, blurb=blurb, uses=("chat", "coder") if coder else ("chat",),
        model=first, shards=rest, page=f"{ml.HF}/{repo}", license=cand.get("license", ""), moe=moe(cand),
        mmproj=_file(repo, proj) if proj else None, sampling=_sampling(cand),
        ctx_size=min(32768, cand["ctx"]) if cand["ctx"] >= 4096 else cand["ctx"])


def _tiers(cand: dict) -> tuple[str, ...]:
    if cand["params"] < SMALL:
        return Q8 + Q4
    return Q4 + (Q3 + Q2 if cand["params"] >= BIG else ())


def choose(cand: dict, gpu_mem_gb: list[float], ram_gb: float) -> tuple[ml.Suggestion, ml.Fit] | None:
    """The best file of a model that runs well here (None if none does)."""
    for q in _tiers(cand):
        s = suggestion(cand, q)
        if not s:
            continue
        f = ml.fit(s, gpu_mem_gb, ram_gb)
        if not gpu_mem_gb and ram_gb and ml._need(s) > ram_gb * ml.RAM_HEADROOM:
            continue                                  # no GPUs known: it must at least fit in RAM
        if f.ok and not f.slow:
            return s, f
    return None


def picks(cands: list[dict], kept: dict[str, dict], gpu_mem_gb: list[float], ram_gb: float,
          known: tuple[ml.Suggestion, ...] = ml.SUGGESTIONS, shown: int = SHOWN) -> tuple[ml.Suggestion, ...]:
    """The Hugging Face models to offer on this machine: the ones kept (added or downloaded before,
    in the file they got), then the trending ones that run well here and aren't a built-in pick."""
    taken_pages = {s.page.lower() for s in known}
    taken_names = {base_name(s.page) for s in known if s.page.startswith(ml.HF)}
    out: list[ml.Suggestion] = []
    keys = {s.key for s in known}
    for k, x in kept.items():
        s = suggestion(x["cand"], x["quant"])
        if s and s.key == k and k not in keys:
            out.append(s)
            keys.add(k)
    fresh = 0
    for c in cands:
        if fresh >= shown:
            break
        if key_for(c) in keys or f"{ml.HF}/{c['repo']}".lower() in taken_pages or base_name(c["repo"]) in taken_names:
            continue
        got = choose(c, gpu_mem_gb, ram_gb)
        if got:
            out.append(got[0])
            keys.add(got[0].key)
            fresh += 1
    return tuple(out)


# --------------------------------------------------------------------------- the list on disk

class Library:
    """hf-models.json: the last answer from Hugging Face, when it came, and the models kept."""

    def __init__(self, root: Path, fetcher=fetch):
        self.path = Path(root) / "hf-models.json"
        self.fetcher = fetcher
        self._lock = threading.Lock()
        self._data: dict | None = None

    def _read(self) -> dict:
        if self._data is None:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError
            except (OSError, ValueError):
                data = {}
            self._data = {"checked": float(data.get("checked") or 0), "tried": float(data.get("tried") or 0),
                          "models": list(data.get("models") or []), "kept": dict(data.get("kept") or {})}
        return self._data

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def checked(self) -> float:
        with self._lock:
            return self._read()["checked"]

    def state(self) -> tuple[list[dict], dict[str, dict]]:
        with self._lock:
            d = self._read()
            return list(d["models"]), dict(d["kept"])

    def due(self, now: float | None = None) -> bool:
        now = now or time.time()
        with self._lock:
            d = self._read()
            return now - d["checked"] >= REFRESH_HOURS * 3600 and now - d["tried"] >= RETRY_HOURS * 3600

    def refresh(self) -> int:
        """Ask Hugging Face again; the last list stays when that fails. Returns how many it found."""
        with self._lock:
            self._read()["tried"] = time.time()
        try:
            models = self.fetcher()
        except Exception as err:  # noqa: BLE001 - offline is normal
            log.info("Hugging Face model list not refreshed: %s", err)
            models = []
        with self._lock:
            d = self._read()
            if models:
                d["models"], d["checked"] = models, time.time()
            self._write()
        return len(models)

    def keep(self, s: ml.Suggestion) -> None:
        """Remember a model from the list in the file it got, so it stays offered (it was added or downloaded)."""
        with self._lock:
            d = self._read()
            if s.key in d["kept"]:
                return
            cand = next((c for c in d["models"] if key_for(c) == s.key), None)
            if cand:
                d["kept"][s.key] = {"cand": cand, "quant": quant(s.model.name)}
                self._write()
