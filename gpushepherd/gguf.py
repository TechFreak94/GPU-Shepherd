"""What a GGUF model file's header says about the memory it needs on top of its weights.

Only the metadata at the start of the file is read (no tensors). The context's KV cache is what
matters: it grows with the context size and can be as big as a small model's weights.
"""
from __future__ import annotations

import os
import struct
import threading
from dataclasses import dataclass

MAGIC = b"GGUF"
# value types: size in bytes of the fixed-size ones (8 = string, 9 = array)
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FORMAT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
KV_BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q4_0": 18 / 32}   # per value
FALLBACK_PER_TOKEN = 160_000      # KV bytes per token at f16 when a file can't be read (an 8B model's)


@dataclass(frozen=True)
class Shape:
    kv_layers: float          # layers that keep a KV cache (hybrid models: only the attention ones)
    kv_width: int             # K + V values per token and layer, over all KV heads

    def kv_bytes(self, ctx: int, cache_type: str = "f16") -> int:
        return int(ctx * self.kv_layers * self.kv_width * KV_BYTES.get(cache_type, 2.0))


_cache: dict[tuple[str, int, int], Shape | None] = {}
_lock = threading.Lock()


def shape(path: str) -> Shape | None:
    """The model's KV cache shape, or None if the file can't be read or isn't a GGUF model."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    with _lock:
        if key in _cache:
            return _cache[key]
    try:
        with open(path, "rb") as fh:
            result = _shape(metadata(fh))
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        result = None
    with _lock:
        _cache[key] = result
    return result


def kv_bytes(path: str, ctx: int, cache_type: str = "f16") -> int:
    """KV cache for `ctx` tokens; a guess for an average model when the header can't be read."""
    s = shape(path)
    if s is None:
        return int(ctx * FALLBACK_PER_TOKEN * KV_BYTES.get(cache_type, 2.0) / 2)
    return s.kv_bytes(ctx, cache_type)


def metadata(fh, wanted: tuple[str, ...] = ("general.architecture",), keep_suffixes: tuple[str, ...] = (
        ".block_count", ".attention.head_count", ".attention.head_count_kv", ".attention.key_length",
        ".attention.value_length", ".embedding_length", ".full_attention_interval")) -> dict:
    """The header's keys that describe the model's layers (big arrays like the vocabulary are skipped)."""
    if fh.read(4) != MAGIC:
        raise ValueError("not a GGUF file")
    version, = struct.unpack("<I", fh.read(4))
    if version < 2:
        raise ValueError("GGUF version 1 is not supported")
    _tensors, count = struct.unpack("<QQ", fh.read(16))
    out = {}
    for _ in range(count):
        key = _string(fh)
        vtype, = struct.unpack("<I", fh.read(4))
        keep = key in wanted or key.endswith(keep_suffixes)
        value = _value(fh, vtype, keep)
        if keep:
            out[key] = value
    return out


def _string(fh) -> str:
    n, = struct.unpack("<Q", fh.read(8))
    if n > 1 << 24:
        raise ValueError("string too long")
    return fh.read(n).decode("utf-8")


def _value(fh, vtype: int, keep: bool):
    if vtype in _FIXED:
        return struct.unpack(_FORMAT[vtype], fh.read(_FIXED[vtype]))[0]
    if vtype == 8:
        return _string(fh)
    if vtype == 9:
        itype, n = struct.unpack("<IQ", fh.read(12))
        if itype in _FIXED and not keep:
            fh.seek(_FIXED[itype] * n, os.SEEK_CUR)
            return None
        items = [_value(fh, itype, keep) for _ in range(n)]
        return items if keep else None
    raise ValueError(f"unknown GGUF value type {vtype}")


def _shape(meta: dict) -> Shape | None:
    arch = meta.get("general.architecture")
    layers = meta.get(f"{arch}.block_count")
    if not arch or not layers:
        return None
    heads = meta.get(f"{arch}.attention.head_count") or 0
    kv_heads = meta.get(f"{arch}.attention.head_count_kv", heads)
    embed = meta.get(f"{arch}.embedding_length") or 0
    if isinstance(heads, list):
        heads = max(heads, default=0)
    head_dim = embed // heads if heads else 0
    k_len = meta.get(f"{arch}.attention.key_length") or head_dim
    v_len = meta.get(f"{arch}.attention.value_length") or head_dim
    if isinstance(kv_heads, list):        # per layer; 0 for layers without attention
        kv_layers, kv_heads = float(sum(1 for h in kv_heads if h)), max(kv_heads, default=0)
    else:
        kv_layers = float(layers)
    interval = meta.get(f"{arch}.full_attention_interval")
    if isinstance(interval, int) and interval > 1:     # hybrid models: one attention layer in `interval`
        kv_layers = min(kv_layers, -(-layers // interval))
    if not kv_heads or not (k_len + v_len):
        return None
    return Shape(kv_layers=kv_layers, kv_width=int(kv_heads) * int(k_len + v_len))
