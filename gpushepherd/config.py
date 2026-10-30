"""Configuration: pydantic models stored as human-readable YAML files.

Layout of the data directory (SHEPHERD_DATA_DIR):

    settings.yaml              app settings (login, network, defaults)
    instances/<id>.yaml        one file per instance (a container the app manages)
    catalogs/<id>.yaml         model catalogue for each model-server instance
    linked-models.yaml         models added from a link on a Models page (listed with the suggested ones)
    generated/<id>/...         files the app renders for containers (do not edit)

Everything the UI saves lands in these files, so they can be backed up, diffed and kept in
git. They can also be edited by hand; the app re-reads them on every request.
"""
from __future__ import annotations

import os
import re
import secrets
import shlex
import tempfile
import threading
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, field_validator, model_validator

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")


class ConfigError(ValueError):
    """A user-facing configuration problem."""


# libyaml's parser when PyYAML was built with it: the same results, many times faster
_YamlLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load_yaml(text: str):
    """yaml.safe_load, with the fast parser where there is one."""
    return yaml.load(text, Loader=_YamlLoader)  # noqa: S506 - a safe loader


# --------------------------------------------------------------------------- instances

class PortMap(BaseModel):
    host: int = Field(ge=1, le=65535)
    container: int = Field(ge=1, le=65535)
    protocol: Literal["tcp", "udp"] = "tcp"


class VolumeMap(BaseModel):
    host: str
    container: str
    read_only: bool = False

    @field_validator("host", "container")
    @classmethod
    def _absolute(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("/"):
            raise ValueError(f"'{v}' must be an absolute path")
        return v


class GpuPolicy(BaseModel):
    """How an instance uses GPUs.

    none     no GPU
    fixed    always exactly the GPUs in `gpus` (e.g. ComfyUI on GPU 0)
    elastic  at least `min_gpus`, and grows into every free GPU up to `max_gpus` (0 = no limit).
             It shrinks again when something else needs a GPU (e.g. an LLM server).
    priority decides who keeps their GPUs when there are not enough to go around; the instance
    you are starting always wins, anything that can no longer fit is stopped (after you confirm).
    """
    mode: Literal["none", "fixed", "elastic"] = "none"
    gpus: list[int] = []
    min_gpus: int = Field(default=1, ge=1, le=64)
    max_gpus: int = Field(default=0, ge=0, le=64)
    priority: int = Field(default=50, ge=0, le=100)

    @model_validator(mode="after")
    def _check(self) -> "GpuPolicy":
        if self.mode == "fixed" and not self.gpus:
            raise ValueError("a fixed GPU policy needs at least one GPU")
        if self.mode == "elastic" and self.max_gpus and self.max_gpus < self.min_gpus:
            raise ValueError("max GPUs must be 0 (no limit) or at least min GPUs")
        if len(set(self.gpus)) != len(self.gpus):
            raise ValueError("a GPU is listed twice")
        return self


class DrainPolicy(BaseModel):
    """Finish running work before the container is stopped (by a GPU swap, a restart or Stop).

    check    how GPU Shepherd tells whether the instance is busy:
             auto          picked from the instance (llama-swap, ComfyUI or llama.cpp server images)
             llama-swap    requests llama-swap is still answering
             llama-server  llama.cpp server slots that are processing (/slots)
             comfyui       running and queued prompts (/queue)
             http          GET `path`; the number, list or true/false at JSON `key` is the work left
             none          never wait
    timeout  seconds to wait at most (0 = do not wait); then `on_timeout` decides:
             stop = stop it anyway, cancel = leave everything as it was
    """
    check: Literal["auto", "none", "llama-swap", "llama-server", "comfyui", "http"] = "auto"
    port: int | None = Field(default=None, ge=1, le=65535)   # container port to ask (default: by check)
    path: str = ""
    key: str = ""
    timeout: int = Field(default=600, ge=0, le=86400)
    on_timeout: Literal["stop", "cancel"] = "stop"

    @model_validator(mode="after")
    def _check(self) -> "DrainPolicy":
        self.path, self.key = self.path.strip(), self.key.strip()
        if self.check == "http" and not self.path.startswith("/"):
            raise ValueError("the busy check needs a path that starts with /, e.g. /queue")
        return self


class Instance(BaseModel):
    id: str
    name: str
    description: str = ""
    kind: Literal["container", "llama-swap"] = "container"
    image: str
    command: list[str] = []
    env: dict[str, str] = {}
    ports: list[PortMap] = []
    volumes: list[VolumeMap] = []
    gpu: GpuPolicy = GpuPolicy()
    autostart: bool = False
    open_port: int | None = None      # host port the "Open" button uses (default: first port)
    open_path: str = "/"
    shm_size: str | None = None       # e.g. "1g"
    user: str | None = None           # e.g. "1000:1000"
    memory_limit: str | None = None   # e.g. "4g"
    drain: DrainPolicy = DrainPolicy()  # wait for running work before stopping
    app: str = ""                     # the App Store app it was installed from, if any
    requires: list[str] = []          # instances that start with it (e.g. its database)
    skipped_updates: list[str] = []   # app updates the user chose not to apply (see app_updates.py)
    # the name in its file when the app shows its App Store name instead (older "Chat UI" names);
    # restart checks use it, so the better name alone never asks for a restart
    _file_name: str | None = PrivateAttr(default=None)

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not SLUG.match(v):
            raise ValueError("use lowercase letters, digits and dashes (max 40 characters)")
        return v

    @field_validator("image")
    @classmethod
    def _image(cls, v: str) -> str:
        v = v.strip()
        if not v or " " in v:
            raise ValueError("enter an image name like ghcr.io/org/app:tag")
        return v

    @field_validator("env")
    @classmethod
    def _env(cls, v: dict[str, str]) -> dict[str, str]:
        for key in v:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise ValueError(f"'{key}' is not a valid environment variable name")
        return v

    @model_validator(mode="after")
    def _ports_unique(self) -> "Instance":
        hosts = [p.host for p in self.ports]
        if len(hosts) != len(set(hosts)):
            raise ValueError("a host port is used twice")
        return self

    @field_validator("requires")
    @classmethod
    def _requires(cls, v: list[str]) -> list[str]:
        v = list(dict.fromkeys(x.strip() for x in v if x.strip()))
        for x in v:
            if not SLUG.match(x):
                raise ValueError(f"'{x}' is not an instance ID")
        return v

    @property
    def open_url_port(self) -> int | None:
        if self.open_port:
            return self.open_port
        return self.ports[0].host if self.ports else None


# --------------------------------------------------------------------------- llama-swap catalogue

class OffloadRule(BaseModel):
    """How much of a model stays on the GPUs when the server has `min_gpus` or more GPUs.
    The rest is kept in system RAM, so a model larger than the GPUs' memory still runs."""
    min_gpus: int = Field(ge=1, le=64)
    gpu_layers: int | None = Field(default=None, ge=0, le=999)   # --n-gpu-layers; None = llama.cpp default
    cpu_moe: int = Field(default=0, ge=0, le=999)                # --n-cpu-moe: MoE expert layers kept in RAM


class CatalogModel(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    file: str                          # path INSIDE the container, e.g. /models/x.gguf
    ctx_size: int = Field(default=32768, ge=512, le=4_194_304)
    parallel: int = Field(default=1, ge=1, le=64)
    sampling: str = ""                 # e.g. "--temp 0.6 --top-p 0.95 --top-k 20"
    extra_args: str = ""               # anything else for llama-server
    kv_cache: Literal["f16", "bf16", "q8_0", "q5_1", "q4_0"] = "f16"
    mmproj: str = ""                   # vision projector path inside the container (optional)
    aliases: list[str] = []
    offload: list[OffloadRule] = []    # GPU / system RAM split, by how many GPUs the server has
    # what the model is for: chat (and coding), or answering /v1/embeddings or /v1/rerank for RAG apps
    role: Literal["chat", "embed", "rerank"] = "chat"
    thinking: Literal["auto", "on", "off"] = "auto"   # reasoning before answering; auto = the model's default
    mtp: int = Field(default=0, ge=0, le=6)  # multi-token prediction: tokens guessed ahead per step, 0 = off
    mtp_draft: str = ""                # separate mtp-*.gguf file, for models that ship it beside the model
    split_mode: Literal["layer", "row", "tensor"] = "layer"   # how a model is split over 2+ GPUs
    tensor_split: str = ""             # share of the model per GPU with 2+ GPUs, e.g. "8,12"; "" = by memory
    gpus: int = Field(default=0, ge=0, le=64)   # GPUs it runs on; 0 = the fewest it fits (see placement.py)
    ttl: int = Field(default=0, ge=0, le=604_800)   # unload after this many idle seconds; 0 = stay loaded
    keep_loaded: bool = False          # stays loaded beside other models instead of swapping them out
    preload: bool = False              # loaded as soon as the server starts

    def offload_for(self, gpu_count: int) -> OffloadRule | None:
        """The rule with the highest min_gpus that gpu_count reaches, like AutoAlias.resolve()."""
        best = None
        for rule in sorted(self.offload, key=lambda r: r.min_gpus):
            if gpu_count >= rule.min_gpus:
                best = rule
        return best

    @field_validator("offload")
    @classmethod
    def _offload(cls, v: list[OffloadRule]) -> list[OffloadRule]:
        counts = [r.min_gpus for r in v]
        if len(counts) != len(set(counts)):
            raise ValueError("two offload settings are for the same number of GPUs")
        return sorted(v, key=lambda r: r.min_gpus)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not MODEL_ID.match(v):
            raise ValueError("model IDs may use letters, digits, '.', '_', ':' and '-'")
        return v

    @field_validator("sampling", "extra_args")
    @classmethod
    def _args(cls, v: str) -> str:
        try:
            shlex.split(v)
        except ValueError as err:
            raise ValueError(f"cannot parse arguments: {err}") from err
        return v.strip()

    @field_validator("file")
    @classmethod
    def _file(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("/"):
            raise ValueError("use the path inside the container, e.g. /models/model.gguf")
        return v

    @field_validator("mtp_draft")
    @classmethod
    def _draft(cls, v: str) -> str:
        v = v.strip()
        if v and not v.startswith("/"):
            raise ValueError("MTP draft file: use the path inside the container, e.g. /models/mtp-model.gguf")
        return v

    @field_validator("tensor_split")
    @classmethod
    def _split(cls, v: str) -> str:
        v = v.replace(" ", "").strip(",")
        if v and not re.fullmatch(r"\d+(\.\d+)?(,\d+(\.\d+)?)+", v):
            raise ValueError("GPU split: give a share per GPU separated by commas, e.g. 8,12")
        return v


class AutoRule(BaseModel):
    min_gpus: int = Field(ge=1, le=64)
    model: str


class AutoAlias(BaseModel):
    """A model name that points at different models depending on how many GPUs the
    model server currently has. Example: "coder" = qwen3-coder-next with 2+ GPUs, else qwen3.6."""
    alias: str
    default_model: str
    rules: list[AutoRule] = []

    def resolve(self, gpu_count: int) -> str:
        best = None
        for rule in sorted(self.rules, key=lambda r: r.min_gpus):
            if gpu_count >= rule.min_gpus:
                best = rule.model
        return best or self.default_model


class Catalog(BaseModel):
    instance_id: str
    api_key: str = Field(default_factory=lambda: "sk-" + secrets.token_hex(20))
    threads: int = Field(default=0, ge=0, le=512)        # 0 = physical cores
    llama_server: str = ""                               # path inside the image; "" = detect
    health_timeout: int = Field(default=900, ge=15, le=7200)
    # models use only the GPUs they need (placement.py); None = not chosen yet: off, and offered
    # as a suggestion (tuning.py). New servers start with it on.
    fit_gpus: bool | None = None
    auto_tune: bool = False                              # apply tuning suggestions without asking
    models: list[CatalogModel] = []
    auto_aliases: list[AutoAlias] = []

    @model_validator(mode="after")
    def _consistent(self) -> "Catalog":
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("two models have the same ID")
        names = set(ids)
        seen_alias: set[str] = set()
        for m in self.models:
            for a in m.aliases:
                if a in names or a in seen_alias:
                    raise ValueError(f"alias '{a}' is used twice")
                seen_alias.add(a)
        for auto in self.auto_aliases:
            if auto.alias in names or auto.alias in seen_alias:
                raise ValueError(f"auto alias '{auto.alias}' clashes with a model or alias")
            seen_alias.add(auto.alias)
            for target in [auto.default_model] + [r.model for r in auto.rules]:
                if target not in names:
                    raise ValueError(f"auto alias '{auto.alias}' points at unknown model '{target}'")
        return self


# --------------------------------------------------------------------------- settings

class Settings(BaseModel):
    admin_user: str = "admin"
    password_hash: str = ""
    session_secret: str = Field(default_factory=lambda: secrets.token_hex(32))
    network: str = "gpushepherd"          # Docker network every instance joins
    bind_address: str = "0.0.0.0"         # host address published ports bind to
    container_prefix: str = "shepherd-"
    trash_days: int = Field(default=30, ge=0, le=3650)   # empty trash entries older than this (0 = never)
    auto_mount_drives: bool = True        # mount a newly attached drive that has data on it (see drives.py)


# --------------------------------------------------------------------------- store

def _atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prev = path.with_suffix(path.suffix + ".prev")
        prev.write_bytes(path.read_bytes())
        os.chmod(prev, mode)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _dump(model: BaseModel, header: str) -> str:
    data = model.model_dump(mode="json", exclude_defaults=False)
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


# Parsed files, kept while the file on disk is unchanged (same inode, size and change time), so
# a dashboard refresh doesn't parse settings.yaml for every card. Saves replace the file
# (_atomic_write), and hand edits change its time, so the next read always sees them.
_parsed: dict[Path, tuple[tuple[int, int, int], object]] = {}
_parsed_lock = threading.Lock()


def _read_yaml(path: Path):
    st = path.stat()
    key = (st.st_ino, st.st_size, st.st_mtime_ns)
    with _parsed_lock:
        hit = _parsed.get(path)
    if hit and hit[0] == key:
        return hit[1]
    raw = load_yaml(path.read_text(encoding="utf-8")) or {}
    with _parsed_lock:
        _parsed[path] = (key, raw)
    return raw


def _load(path: Path, cls):
    try:
        # validating builds new objects (no "before" validators touch the input), so the cached
        # raw data is never changed by a caller
        raw = _read_yaml(path)
    except yaml.YAMLError as err:
        raise ConfigError(f"{path.name}: not valid YAML ({err})") from err
    try:
        return cls.model_validate(raw)
    except ValidationError as err:
        raise ConfigError(f"{path.name}: {friendly_errors(err)}") from err


def friendly_errors(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"] if x != "__root__")
        msg = e["msg"].removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def _named(inst: Instance) -> Instance:
    """Show an App Store app's real name where its file only says what kind it is ("Chat UI")."""
    from .instance_templates import app_name   # it builds on this module
    name = app_name(inst)
    if name != inst.name:
        inst._file_name, inst.name = inst.name, name
    return inst


class Store:
    """Reads and writes the YAML files. Stateless: every call goes to disk."""

    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        for sub in ("instances", "catalogs", "generated"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # settings
    @property
    def settings_path(self) -> Path:
        return self.root / "settings.yaml"

    def settings(self) -> Settings:
        if not self.settings_path.exists():
            s = Settings()
            self.save_settings(s)
            return s
        return _load(self.settings_path, Settings)

    def recover_settings(self) -> str:
        """Run at startup. When settings.yaml can't be read (a bad hand edit, a half-written disk),
        put back the copy kept before its last save, so GPU Shepherd still starts instead of
        crashing on every restart. The broken file is kept as settings.yaml.broken. Returns what
        was done, for the dashboard, or "" when the file was fine."""
        try:
            self.settings()
            return ""
        except ConfigError as err:
            prev = self.settings_path.with_name(self.settings_path.name + ".prev")
            try:
                _load(prev, Settings)
            except (ConfigError, OSError):
                raise ConfigError(f"{err}. There is no readable {prev.name} to go back to: fix the file, or put "
                                  "back a backup with update.sh --restore-settings") from err
            broken = self.settings_path.with_name(self.settings_path.name + ".broken")
            os.replace(self.settings_path, broken)
            self.settings_path.write_bytes(prev.read_bytes())
            os.chmod(self.settings_path, 0o600)
            return (f"{err}. GPU Shepherd went back to the settings from before their last change; "
                    f"the unreadable file is kept as {broken.name}.")

    def save_settings(self, s: Settings) -> None:
        _atomic_write(self.settings_path, _dump(s, "# GPU Shepherd settings (contains secrets)\n"), 0o600)

    # instances
    def instance_path(self, iid: str) -> Path:
        if not SLUG.match(iid):
            raise ConfigError(f"invalid instance id '{iid}'")
        return self.root / "instances" / f"{iid}.yaml"

    def instances(self) -> tuple[dict[str, Instance], list[str]]:
        """All valid instances, plus a list of problems with files that could not be loaded."""
        found: dict[str, Instance] = {}
        problems: list[str] = []
        for path in sorted((self.root / "instances").glob("*.yaml")):
            try:
                inst = _load(path, Instance)
            except ConfigError as err:
                problems.append(str(err))
                continue
            if inst.id != path.stem:
                problems.append(f"{path.name}: id '{inst.id}' does not match the file name")
                continue
            found[inst.id] = _named(inst)
        return found, problems

    def instance(self, iid: str) -> Instance:
        path = self.instance_path(iid)
        if not path.exists():
            raise ConfigError(f"no instance '{iid}'")
        return _named(_load(path, Instance))

    def save_instance(self, inst: Instance) -> None:
        others, _ = self.instances()
        used = {p.host: o.id for o in others.values() if o.id != inst.id for p in o.ports}
        for p in inst.ports:
            if p.host in used:
                raise ConfigError(f"host port {p.host} is already used by '{used[p.host]}'")
        _atomic_write(self.instance_path(inst.id), _dump(inst, f"# Instance '{inst.id}' - managed by GPU Shepherd\n"))

    def delete_instance(self, iid: str) -> None:
        for path in (self.instance_path(iid), self.catalog_path(iid)):
            if path.exists():
                path.unlink()

    # catalogues
    def catalog_path(self, iid: str) -> Path:
        if not SLUG.match(iid):
            raise ConfigError(f"invalid instance id '{iid}'")
        return self.root / "catalogs" / f"{iid}.yaml"

    def catalog(self, iid: str) -> Catalog:
        path = self.catalog_path(iid)
        if not path.exists():
            cat = Catalog(instance_id=iid, fit_gpus=True)
            self.save_catalog(cat)
            return cat
        return _load(path, Catalog)

    def save_catalog(self, cat: Catalog) -> None:
        _atomic_write(self.catalog_path(cat.instance_id),
                      _dump(cat, f"# Model catalogue for '{cat.instance_id}' (contains its API key)\n"), 0o600)

    # models added from a link (shown with the suggested models)
    @property
    def linked_models_path(self) -> Path:
        return self.root / "linked-models.yaml"

    def linked_models(self) -> list[dict]:
        """[{key, name, url, gb, mmproj_url, mmproj_gb}]; a bad file reads as none."""
        try:
            raw = load_yaml(self.linked_models_path.read_text(encoding="utf-8")) or []
        except (OSError, yaml.YAMLError):
            return []
        return [x for x in raw if isinstance(x, dict) and x.get("url")] if isinstance(raw, list) else []

    def save_linked_models(self, rows: list[dict]) -> None:
        _atomic_write(self.linked_models_path, "# Models added from a link on a model server's Models page\n"
                      + yaml.safe_dump(rows, sort_keys=False, allow_unicode=True, width=200))

    def generated_dir(self, iid: str) -> Path:
        path = self.root / "generated" / iid
        path.mkdir(parents=True, exist_ok=True)
        return path


# --------------------------------------------------------------------------- form helpers
# The UI edits lists with docker-style one-line-per-entry text boxes.

def parse_ports(text: str) -> list[PortMap]:
    """'8188:8188' or '8080:80/udp', one per line."""
    out = []
    for n, line in enumerate(_lines(text), 1):
        proto = "tcp"
        if "/" in line:
            line, proto = line.rsplit("/", 1)
        parts = line.split(":")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            raise ConfigError(f"ports line {n}: use HOST:CONTAINER, e.g. 8188:8188")
        out.append(PortMap(host=int(parts[0]), container=int(parts[1]), protocol=proto.strip() or "tcp"))
    return out


def parse_volumes(text: str) -> list[VolumeMap]:
    """'/host/path:/container/path[:ro]', one per line."""
    out = []
    for n, line in enumerate(_lines(text), 1):
        parts = line.split(":")
        ro = False
        if len(parts) == 3 and parts[2] in ("ro", "rw"):
            ro = parts[2] == "ro"
            parts = parts[:2]
        if len(parts) != 2:
            raise ConfigError(f"volumes line {n}: use /host/path:/container/path or add :ro")
        out.append(VolumeMap(host=parts[0], container=parts[1], read_only=ro))
    return out


def parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for n, line in enumerate(_lines(text), 1):
        if "=" not in line:
            raise ConfigError(f"environment line {n}: use NAME=value")
        key, value = line.split("=", 1)
        out[key.strip()] = value
    return out


def parse_command(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError as err:
        raise ConfigError(f"command: {err}") from err


def format_ports(ports: list[PortMap]) -> str:
    return "\n".join(f"{p.host}:{p.container}" + ("" if p.protocol == "tcp" else f"/{p.protocol}") for p in ports)


def format_volumes(vols: list[VolumeMap]) -> str:
    return "\n".join(f"{v.host}:{v.container}" + (":ro" if v.read_only else "") for v in vols)


def format_env(env: dict[str, str]) -> str:
    return "\n".join(f"{k}={v}" for k, v in env.items())


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]
