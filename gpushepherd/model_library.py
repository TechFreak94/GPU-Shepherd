"""Suggested models: a short list of GGUF models that work well with the LLM server, each with
its download links and the settings its makers recommend.

Picking one (in the install form or on a model server's Models page) downloads it into the
models folder and adds it to the catalogue already set up: sampling, thinking, vision projector,
MTP and how it is split between GPUs and system RAM. Every model here is a public download that
runs on any NVIDIA machine; whether it fits is worked out from the machine's GPUs and RAM.

Each model lives in its own subfolder of the models folder, so files that share a name
(mmproj-F16.gguf) don't overwrite each other.

This module is pure: callers pass in the GPUs' memory and the RAM size.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from .config import CatalogModel, OffloadRule

HF = "https://huggingface.co"
# how much bigger than its files a model is once loaded (context, buffers); as instance_templates
LOAD_OVERHEAD = 1.2
RAM_HEADROOM = 0.75          # share of the RAM a model may use; the rest is for everything else
GPU_SHARE = 0.75             # share of the GPUs' memory a MoE model's layers may fill (the rest: context)
DENSE_ON_GPU = 0.8           # a dense model that keeps this share of its layers on the GPUs is still quick

# the makers' recommended sampling, by model family and mode
QWEN_THINKING = "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0"
QWEN_INSTRUCT = "--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5"
QWEN_CODER = "--temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05"
QWEN_CODER_NEXT = "--temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01"
GEMMA = "--temp 1.0 --top-p 0.95 --top-k 64"


@dataclass(frozen=True)
class File:
    name: str                # file name, in the suggestion's own subfolder
    url: str
    gb: float                # download size


@dataclass(frozen=True)
class Suggestion:
    key: str                 # also the model's ID and its subfolder
    name: str
    blurb: str
    uses: tuple[str, ...]    # install-form slots it suits: chat, coder, embed
    model: File
    page: str                # the model's download page
    license: str = ""        # the model's license, from its model card
    moe: bool = False        # mixture of experts: expert layers can live in system RAM at good speed
    layers: int = 0          # its layer count: MoE expert layers that can go to RAM; dense, set it to
                             # let a few layers go to RAM when the GPUs are just too small
    mmproj: File | None = None     # vision projector: lets it read images
    draft: File | None = None      # MTP draft file; with it, mtp is turned on
    mtp: int = 0
    sampling: str = ""
    thinking: str = "auto"
    ctx_size: int = 32768
    kv_cache: str = "q8_0"
    role: str = "chat"
    keep_loaded: bool = False
    preload: bool = False          # loaded as soon as the server starts
    gpu_layers: int | None = None  # set for models that should stay off the GPUs (embeddings: 0)
    extra_args: str = ""

    @property
    def files(self) -> tuple[File, ...]:
        return tuple(f for f in (self.model, self.mmproj, self.draft) if f)

    @property
    def download_gb(self) -> float:
        return round(sum(f.gb for f in self.files), 1)

    def path(self, f: File) -> str:
        """Where a file goes, relative to the models folder."""
        return f"{self.key}/{f.name}"

    @property
    def vision(self) -> bool:
        return self.mmproj is not None


def _hf(repo: str, name: str, gb: float) -> File:
    return File(name=name.rsplit("/", 1)[-1], url=f"{HF}/{repo}/resolve/main/{name}", gb=gb)


SUGGESTIONS: tuple[Suggestion, ...] = (
    Suggestion(
        key="qwen3.5-9b", name="Qwen3.5 9B",
        blurb="All-round chat that reads images and calls tools; 256K context. Fits one 8 GB GPU.",
        uses=("chat", "coder"), license="Apache-2.0", page=f"{HF}/unsloth/Qwen3.5-9B-GGUF",
        model=_hf("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-UD-Q4_K_XL.gguf", 5.97),
        mmproj=_hf("unsloth/Qwen3.5-9B-GGUF", "mmproj-F16.gguf", 0.92),
        sampling=QWEN_INSTRUCT, thinking="off"),
    Suggestion(
        key="gemma-4-e4b", name="Gemma 4 E4B",
        blurb="Light, quick chat from Google that reads images and audio. Fits one 8 GB GPU with room for context.",
        uses=("chat",), license="Apache-2.0", page=f"{HF}/unsloth/gemma-4-E4B-it-GGUF",
        model=_hf("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-UD-Q4_K_XL.gguf", 5.13),
        mmproj=_hf("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf", 0.99),
        draft=_hf("unsloth/gemma-4-E4B-it-GGUF", "mtp-gemma-4-E4B-it.gguf", 0.1), mtp=2,
        sampling=GEMMA, thinking="off"),
    Suggestion(
        key="gemma-4-12b", name="Gemma 4 12B",
        blurb="Stronger Gemma chat with images and audio; 256K context. For a 12 GB GPU or two 8 GB GPUs.",
        uses=("chat",), license="Apache-2.0", page=f"{HF}/unsloth/gemma-4-12b-it-GGUF",
        model=_hf("unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-UD-Q4_K_XL.gguf", 7.37),
        mmproj=_hf("unsloth/gemma-4-12b-it-GGUF", "mmproj-F16.gguf", 0.18),
        draft=_hf("unsloth/gemma-4-12b-it-GGUF", "mtp-gemma-4-12b-it.gguf", 0.47), mtp=2,
        sampling=GEMMA, thinking="off"),
    Suggestion(
        key="qwen3-coder-30b-a3b", name="Qwen3 Coder 30B-A3B",
        blurb="Coding model for agents and editors. A mixture of experts: on one 8 GB GPU its experts sit in "
              "about 13 GB of RAM and it still answers at reading speed.",
        uses=("coder",), license="Apache-2.0", page=f"{HF}/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", moe=True, layers=48,
        model=_hf("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf", 17.67),
        sampling=QWEN_CODER, ctx_size=65536),
    Suggestion(
        key="qwen3.8-27b", name="Qwen3.8 27B",
        blurb="Newest dense Qwen (August 2026): the best answers here for chat and coding, with images. "
              "Needs two 8 GB GPUs or one 16 GB GPU; slower than the A3B models.",
        uses=("chat", "coder"), license="Apache-2.0", page=f"{HF}/unsloth/Qwen3.8-27B-GGUF",
        model=_hf("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q3_K_XL.gguf", 13.15),
        mmproj=_hf("unsloth/Qwen3.8-27B-GGUF", "mmproj-F16.gguf", 0.93),
        sampling=QWEN_THINKING),
    Suggestion(
        key="muse-glimmer-30b", name="Muse Glimmer 30B",
        blurb="Dense agent and coding model that reads images (screenshots, charts); 128K context. Needs two "
              "8 GB GPUs or one 16 GB GPU (a 3-bit build, so it fits whole: about 30 tokens/s on two RTX 3070s).",
        uses=("chat", "coder"), license="Apache-2.0", page=f"{HF}/unsloth/Muse-Glimmer-30B-GGUF", layers=52,
        model=_hf("unsloth/Muse-Glimmer-30B-GGUF", "Muse-Glimmer-30B-UD-IQ3_XXS.gguf", 13.13),
        mmproj=_hf("unsloth/Muse-Glimmer-30B-GGUF", "mmproj-Muse-Glimmer-30B-Q8_0.gguf", 2.05),
        sampling=GEMMA),
    Suggestion(
        key="qwen3.6-35b-a3b", name="Qwen3.6 35B-A3B",
        blurb="Fast mixture-of-experts chat and coding with images. Fits two 8 GB GPUs, or one GPU with "
              "its experts in RAM.",
        uses=("chat", "coder"), license="Apache-2.0", page=f"{HF}/unsloth/Qwen3.6-35B-A3B-GGUF", moe=True, layers=40,
        model=_hf("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf", 16.85),
        mmproj=_hf("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf", 0.9),
        sampling=QWEN_THINKING),
    Suggestion(
        key="qwen3-coder-next", name="Qwen3 Coder Next 80B-A3B",
        blurb="The strongest coding model here, for long agent sessions. Its experts live in system RAM: "
              "needs about 40 GB of free RAM beside two 8 GB GPUs (64 GB machines).",
        uses=("coder",), license="Apache-2.0", page=f"{HF}/unsloth/Qwen3-Coder-Next-GGUF", moe=True, layers=48,
        model=_hf("unsloth/Qwen3-Coder-Next-GGUF", "Qwen3-Coder-Next-UD-Q4_K_XL.gguf", 49.61),
        sampling=QWEN_CODER_NEXT, ctx_size=65536),
    Suggestion(
        key="qwen3-embedding-0.6b", name="Qwen3 Embedding 0.6B",
        blurb="Turns documents into vectors for chat-with-your-files (RAG) in Open WebUI, AnythingLLM and "
              "LibreChat. Runs on the CPU and stays loaded, so it never swaps the chat model out.",
        uses=("embed",), license="Apache-2.0", page=f"{HF}/Qwen/Qwen3-Embedding-0.6B-GGUF",
        model=_hf("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf", 0.64),
        role="embed", keep_loaded=True, preload=True, gpu_layers=0, ctx_size=8192, kv_cache="f16",
        extra_args="--pooling last --ubatch-size 8192"),
)


def get(key: str, extra: tuple[Suggestion, ...] = ()) -> Suggestion | None:
    """A suggestion by key; `extra` are the user's own, added from a link."""
    return next((s for s in SUGGESTIONS + tuple(extra) if s.key == key), None)


def for_file(path: str, models_root: str = "/models", extra: tuple[Suggestion, ...] = ()) -> Suggestion | None:
    """The suggestion a model file (path inside the container) was downloaded for, if any."""
    return next((s for s in SUGGESTIONS + tuple(extra) if path == f"{models_root}/{s.path(s.model)}"), None)


# --------------------------------------------------------------------------- models added from a link
LINKED = "Added from a link."           # the blurb of a model added from a link
SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)


def gguf_link(url: str) -> str:
    """A download link for a .gguf file, from what people copy: a Hugging Face file page (/blob/)
    becomes its download link (/resolve/), and ?download=true and the like are dropped."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("https", "http") or not parts.netloc:
        raise ValueError("The link must start with https://")
    path = parts.path
    if parts.netloc in ("huggingface.co", "www.huggingface.co"):
        path = re.sub(r"^(/[^/]+/[^/]+)/blob/", r"\1/resolve/", path)
        if "/resolve/" not in path:
            raise ValueError("Use the link of the .gguf file itself (on the model page: Files, then the file), "
                             "not the model page.")
    if not path.lower().endswith(".gguf"):
        raise ValueError("The link must point at a .gguf file.")
    if SPLIT.search(path):
        raise ValueError("That model is split into several .gguf files; models in one file are supported.")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9.]+", "-", text.lower())).strip("-.")[:60] or "model"


def from_link(url: str, name: str = "", gb: float = 0.0, mmproj_url: str = "", mmproj_gb: float = 0.0,
              key: str = "") -> Suggestion:
    """A model the user added by its download link, set up like a suggested one: its own folder,
    downloaded the same way. Its settings are llama.cpp's defaults, to tune on the Models page."""
    url = gguf_link(url)
    file = url.rsplit("/", 1)[1]
    stem = file[:-5]
    parts = urlsplit(url)
    m = re.match(r"^(/[^/]+/[^/]+)/resolve/", parts.path)
    page = urlunsplit((parts.scheme, parts.netloc, m.group(1), "", "")) if m else url.rsplit("/", 1)[0]
    mmproj = None
    if mmproj_url.strip():
        mmproj_url = gguf_link(mmproj_url)
        mmproj = File(name=mmproj_url.rsplit("/", 1)[1], url=mmproj_url, gb=round(mmproj_gb, 2))
        if mmproj.name == file:
            raise ValueError("The vision projector must be a different file from the model.")
    return Suggestion(key=key or _slug(stem), name=name.strip() or stem, blurb=LINKED, uses=("chat", "coder"),
                      model=File(name=file, url=url, gb=round(gb, 2)), page=page, mmproj=mmproj)


@dataclass(frozen=True)
class Fit:
    ok: bool                 # runs on this machine
    gpus: int                # GPUs it needs (or uses, with RAM) - 0 when the machine has none
    ram_gb: float            # system RAM it needs beside them
    text: str                # what to tell the user

    @property
    def rank(self) -> tuple:
        """Sort key: what runs best on this machine first."""
        return (not self.ok, self.ram_gb > 0, self.gpus)


def _need(s: Suggestion) -> float:
    """GPU memory to keep the whole model on the GPUs."""
    return s.model.gb * LOAD_OVERHEAD + (s.draft.gb if s.draft else 0)


def _in_ram(s: Suggestion, vram: float) -> float:
    """GB of a MoE model's expert layers that go to system RAM beside `vram` GB of GPUs. A quarter
    of the GPUs' memory is kept for the context and llama.cpp's buffers."""
    return max(s.model.gb + (s.draft.gb if s.draft else 0) - vram * GPU_SHARE + 1, 0)


def _dense_layers(s: Suggestion, vram: float) -> int:
    """How many layers of a dense model fit in `vram` GB of GPUs, keeping room for the context."""
    return min(s.layers, math.floor(s.layers * vram / _need(s)))


def fit(s: Suggestion, gpu_mem_gb: list[float], ram_gb: float) -> Fit:
    """Whether a suggestion runs here, and how: on the fewest GPUs it fits (the smallest ones, as the
    planner may hand out any), or for a mixture-of-experts model, on every GPU with its experts
    partly in system RAM. A dense model split that way is too slow to recommend, unless only a few
    of its layers (a fifth at most) have to go to RAM."""
    need = _need(s)
    if not s.model.gb:                       # added from a link whose size wasn't known
        return Fit(True, 0, 0, "Size not known")
    if s.gpu_layers == 0:                    # runs on the CPU
        return Fit(True, 0, round(need, 1), f"Runs on the CPU, about {max(round(need), 1)} GB of RAM")
    if not gpu_mem_gb:
        return Fit(True, 0, 0, f"Needs about {round(need)} GB of GPU memory")
    smallest = sorted(gpu_mem_gb)
    for n in range(1, len(smallest) + 1):
        if sum(smallest[:n]) >= need:
            return Fit(True, n, 0, f"Fits on {_gpus(n)}")
    if not s.moe:
        n, on_gpu = len(smallest), _dense_layers(s, sum(smallest)) if s.layers else 0
        if not s.layers or on_gpu < s.layers * DENSE_ON_GPU:
            return Fit(False, n, 0, f"Too big for your GPUs (needs about {round(need)} GB)")
        ram = max(round(s.model.gb * (s.layers - on_gpu) / s.layers), 1)
        if ram_gb and ram > ram_gb * RAM_HEADROOM:
            return Fit(False, n, ram, f"Too big: needs about {ram} GB of RAM beside your GPUs")
        text = f"{_gpus(n)} with {on_gpu} of its {s.layers} layers, the rest (about {ram} GB) in RAM"
        return Fit(True, n, ram, text[0].upper() + text[1:])
    n = len(smallest)
    ram = round(_in_ram(s, sum(smallest)))
    ok = not ram_gb or ram <= ram_gb * RAM_HEADROOM
    if not ok:
        return Fit(False, n, ram, f"Too big: needs about {ram} GB of RAM beside your GPUs")
    text = f"{_gpus(n)} + about {ram} GB of RAM"
    if n > 1:
        text += f" (one GPU: about {round(_in_ram(s, smallest[0]))} GB)"
    return Fit(True, n, ram, text[0].upper() + text[1:])


def _gpus(n: int) -> str:
    return "one GPU" if n == 1 else f"{n} GPUs"


def offload(s: Suggestion, gpu_mem_gb: list[float]) -> list[OffloadRule]:
    """GPU / RAM rules for each number of GPUs the server may get. A MoE model gets none: llama.cpp
    fits it to the GPUs' free memory itself, keeping just enough expert tensors in RAM and spreading
    the rest evenly over the GPUs. (--n-cpu-moe turns that off and moves the experts of the first
    layers, which all sit on the first GPU, so the last GPU overflowed.)"""
    if s.gpu_layers is not None:
        return [OffloadRule(min_gpus=1, gpu_layers=s.gpu_layers)]
    if not s.layers or not gpu_mem_gb or s.moe:
        return []
    smallest = sorted(gpu_mem_gb)            # dense: the layers that don't fit stay in RAM
    rules = [OffloadRule(min_gpus=n, gpu_layers=None if sum(smallest[:n]) >= _need(s)
                         else _dense_layers(s, sum(smallest[:n]))) for n in range(1, len(smallest) + 1)]
    return [] if all(r.gpu_layers is None for r in rules) else rules


def catalog_model(s: Suggestion, gpu_mem_gb: list[float], models_root: str = "/models",
                  aliases: list[str] | None = None, ctx_size: int | None = None) -> CatalogModel:
    """The catalogue entry for a suggestion, with its makers' settings."""
    return CatalogModel(
        id=s.key, name=s.name, description=s.blurb.split(". ")[0].rstrip(".") + ".",
        file=f"{models_root}/{s.path(s.model)}", ctx_size=ctx_size or s.ctx_size, kv_cache=s.kv_cache,
        sampling=s.sampling, extra_args=s.extra_args, thinking=s.thinking, role=s.role,
        mmproj=f"{models_root}/{s.path(s.mmproj)}" if s.mmproj else "",
        mtp=s.mtp, mtp_draft=f"{models_root}/{s.path(s.draft)}" if s.draft else "",
        keep_loaded=s.keep_loaded, preload=s.preload, aliases=list(aliases or []), offload=offload(s, gpu_mem_gb))


def for_slot(slot: str, gpu_mem_gb: list[float], ram_gb: float) -> list[tuple[Suggestion, Fit]]:
    """Suggestions for an install-form slot, what runs best here first."""
    out = [(s, fit(s, gpu_mem_gb, ram_gb)) for s in SUGGESTIONS if slot in s.uses]
    return sorted(out, key=lambda x: x[1].rank)
