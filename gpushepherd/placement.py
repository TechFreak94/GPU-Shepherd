"""Which of a model server's GPUs each model runs on, and how many GPUs the server can use at all.

Pure functions, no I/O (budget.model_memory measures the models). Splitting a model over more
GPUs than it needs makes it no faster: each GPU gets its own buffers and passes its work to the
next one. So with "Use only the GPUs it needs" on (Catalog.fit_gpus):

- A model runs on the fewest of the server's GPUs that hold its weights and context
  (CatalogModel.gpus sets the number by hand). Models that don't fit anywhere, models with a
  GPU / RAM split for this many GPUs, and models whose size isn't known use every GPU, as before.
- Models swapped in and out start at the server's first GPU; models kept loaded (an embedding
  model) take the last GPUs, so the two run side by side instead of sharing one GPU.
- The server only grows into as many GPUs as its models can use (wants()); the planner leaves the
  rest free for other instances.
"""
from __future__ import annotations

from .config import Catalog, CatalogModel

GB = 1e9
PER_GPU = 1.2 * GB         # what llama.cpp keeps on every GPU it uses: CUDA context and compute buffers
USABLE = 0.95              # share of the rest a model may fill


def fits_on(need: int, gpu_mem: list[int]) -> int:
    """The fewest of these GPUs (in order) that hold `need` bytes; 0 if all of them together don't."""
    total = 0.0
    for n, mem in enumerate(gpu_mem, 1):
        total += max(mem - PER_GPU, 0) * USABLE
        if total >= need:
            return n
    return 0


def on_cpu(m: CatalogModel, gpu_count: int) -> bool:
    rule = m.offload_for(max(gpu_count, 1))
    return bool(rule and rule.gpu_layers == 0)


def model_gpus(m: CatalogModel, need: int, gpu_mem: list[int], fit: bool = True) -> int:
    """How many of the server's GPUs (`gpu_mem`, bytes each) a model uses. `need` is its weights
    plus context in bytes (0 = not known). 0 = it runs on the CPU."""
    n = len(gpu_mem)
    if not n or on_cpu(m, n):
        return 0
    if m.gpus:
        return min(m.gpus, n)
    rule = m.offload_for(n)
    if not fit or not need or (rule and (rule.cpu_moe or rule.gpu_layers is not None)):
        return n
    return fits_on(need, gpu_mem) or n


def place(cat: Catalog, needs: dict[str, int], gpu_mem: list[int]) -> dict[str, list[int]]:
    """Model id -> the positions (0 = the server's first GPU) of the GPUs it runs on, for the
    models that use fewer than all of them. Models not listed see every GPU the server has."""
    n = len(gpu_mem)
    out: dict[str, list[int]] = {}
    end = n                                    # kept-loaded models fill from the last GPU down
    for m in cat.models:
        if not m.keep_loaded:
            continue
        k = model_gpus(m, needs.get(m.id, 0), gpu_mem, cat.fit_gpus)
        if 0 < k < n:
            if end - k < 0:
                end = n                        # no GPU left to itself: share from the last one again
            out[m.id] = list(range(end - k, end))
            end -= k
    for m in cat.models:
        if m.keep_loaded:
            continue
        k = model_gpus(m, needs.get(m.id, 0), gpu_mem, cat.fit_gpus)
        if 0 < k < n:
            out[m.id] = list(range(k))
    return out


def wants(cat: Catalog, needs: dict[str, int], gpu_mem: list[int]) -> int | None:
    """How many of the machine's GPUs (`gpu_mem`) the server can put to use: the most any swapped
    model needs plus what the kept-loaded ones need, and at least what an auto alias asks for.
    None = no limit (the setting is off, or a model would use every GPU)."""
    n = len(gpu_mem)
    if not cat.fit_gpus or not n or not cat.models:
        return None
    swapped, kept = 0, 0
    for m in cat.models:
        k = model_gpus(m, needs.get(m.id, 0), gpu_mem, True)
        if k >= n and not m.gpus:
            return None
        if m.keep_loaded:
            kept += k
        else:
            swapped = max(swapped, k)
    auto = max((r.min_gpus for a in cat.auto_aliases for r in a.rules), default=0)
    return min(max(swapped + kept, auto, 1), n)
