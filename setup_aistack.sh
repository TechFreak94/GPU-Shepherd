#!/usr/bin/env bash
# =============================================================================
# setup_aistack.sh  (v2: web control panel + automatic GPU handling)
#
# Ubuntu Server 26.04.1  ->  Docker-based AI stack you run from a web page
# Target box: i7-9800X, 128 GB DDR4, 2 x RTX 3070 (8 GB each)
#
# Instances (docker compose project "aistack", everything lives in ~/aistack):
#   panel    AI Stack control panel: start/stop each instance, GPU use, logs,
#            load/unload LLM models                                             :8090
#   comfyui  ComfyUI + Z-Image Turbo coloring-book setup, one instance, one GPU   :8188
#   llm      llama-swap + llama.cpp (CUDA), OpenAI-compatible API. Models:
#              qwen3.6-35b-a3b  Qwen3.6 35B-A3B (MoE, fast on one 8 GB GPU + RAM)
#              muse-glimmer     Meta Muse Glimmer 30B (dense, best with both GPUs)
#              auto             follows the GPUs: Qwen while ComfyUI runs, Muse when it is stopped
#   hermes   Hermes Agent for homelab + Home Assistant tasks (uses "auto")       :9119
#   webui    Open WebUI chat for the models and for Hermes                        :3000
#
# GPU handling is automatic - no modes or commands to switch:
#   ComfyUI running -> it keeps GPU 0; the LLM uses GPU 1 + system RAM
#   ComfyUI stopped -> the LLM uses both GPUs
#   (Starting/stopping ComfyUI restarts the LLM container to move it; the panel warns first.)
#
# What the script does (idempotent - safe to re-run, also upgrades a v1 install):
#   1. apt prerequisites, NVIDIA driver check (reboot + re-run if it had to install one)
#   2. Docker Engine + Compose, NVIDIA Container Toolkit, GPU-in-container test
#   3. Disables the earlier systemd services (comfyui@N, comfyui, llama-server). Nothing deleted.
#   4. Writes ~/aistack: compose.yaml, .env, image build files, model catalogue, README, CLI
#   5. Downloads Muse Glimmer, Qwen3.6-35B-A3B and the Z-Image files (reusing ~/comfyui models)
#   6. Builds/pulls images, starts the stack, points Hermes at "auto", runs a smoke test
#
# Usage:
#   chmod +x setup_aistack.sh
#   ./setup_aistack.sh                                       # normal user with sudo
#   HASS_URL=http://192.168.1.50:8123 HASS_TOKEN=... ./setup_aistack.sh   # connect Hermes to Home Assistant
#   ENABLE_WEBUI=0 ./setup_aistack.sh                        # leave a service out (COMFYUI/HERMES/WEBUI/PANEL)
#
# Settings below can be overridden from the environment on the FIRST run.
# After that they live in ~/aistack/.env (edit it, then restart the instance from the panel).
# =============================================================================
set -Eeuo pipefail
# NOTE: with pipefail, `cmd | grep -q x` can report failure even when x was found (grep exits
# early, cmd gets SIGPIPE). Checks below either drop -q or capture output in a variable first.

# ----------------------------- configuration ---------------------------------
STACK_DIR="${STACK_DIR:-$HOME/aistack}"
LISTEN_ADDR="${LISTEN_ADDR:-0.0.0.0}"            # 127.0.0.1 = reachable from this machine only
PANEL_PORT="${PANEL_PORT:-8090}"
COMFY_PORT="${COMFY_PORT:-8188}"
LLM_PORT="${LLM_PORT:-8080}"
WEBUI_PORT="${WEBUI_PORT:-3000}"
HERMES_DASH_PORT="${HERMES_DASH_PORT:-9119}"
HERMES_API_PORT="${HERMES_API_PORT:-8642}"       # bound to 127.0.0.1 only
COMFY_GPU="${COMFY_GPU:-0}"                      # the one GPU ComfyUI uses

MUSE_REPO="${MUSE_REPO:-unsloth/Muse-Glimmer-30B-GGUF}"
MUSE_QUANT="${MUSE_QUANT:-UD-Q4_K_XL}"
MUSE_VISION="${MUSE_VISION:-0}"                  # 1 = also download the vision projector
QWEN_REPO="${QWEN_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
QWEN_QUANT="${QWEN_QUANT:-UD-Q4_K_XL}"
LLM_CTX="${LLM_CTX:-131072}"                     # total context, shared by LLM_PARALLEL slots
LLM_PARALLEL="${LLM_PARALLEL:-2}"                # simultaneous requests (Hermes + you)

ENABLE_PANEL="${ENABLE_PANEL:-1}"
ENABLE_COMFYUI="${ENABLE_COMFYUI:-1}"
ENABLE_HERMES="${ENABLE_HERMES:-1}"
ENABLE_WEBUI="${ENABLE_WEBUI:-1}"

ZIMAGE_VARIANT="${ZIMAGE_VARIANT:-fp8}"          # fp8 (fits 8 GB) | bf16 | both
INSTALL_LORAS="${INSTALL_LORAS:-1}"
INSTALL_UPSCALER="${INSTALL_UPSCALER:-1}"
COMFY_MODELS_DIR="${COMFY_MODELS_DIR:-auto}"     # auto = reuse ~/comfyui/ComfyUI/models if present
LEGACY_LLM_DIR="${LEGACY_LLM_DIR:-$HOME/llm/models}"   # earlier Qwen3-Coder-Next install, if any

HASS_URL="${HASS_URL:-}"
HASS_TOKEN="${HASS_TOKEN:-}"
MIGRATE_OLD="${MIGRATE_OLD:-1}"
LAN_CIDR="${LAN_CIDR:-}"
HF_TOKEN="${HF_TOKEN:-}"

IMG_LLAMA_SWAP="${IMG_LLAMA_SWAP:-ghcr.io/mostlygeek/llama-swap:cuda}"
IMG_HERMES="${IMG_HERMES:-nousresearch/hermes-agent:latest}"
IMG_WEBUI="${IMG_WEBUI:-ghcr.io/open-webui/open-webui:main}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

case "$ZIMAGE_VARIANT" in fp8|bf16|both) ;; *) echo "ZIMAGE_VARIANT must be fp8, bf16 or both" >&2; exit 1 ;; esac

ENV_FILE="$STACK_DIR/.env"
LOG="$HOME/aistack-setup.log"
FAILED=()

# ------------------------------- helpers -------------------------------------
log()  { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

exec > >(tee -a "$LOG") 2>&1
trap 'printf "\n[error] failed near line %s (exit %s). Log: %s\n" "$LINENO" "$?" "$LOG" >&2' ERR

# dl_try <dest> <url> [fallback-url ...] - resumable; returns 1 if every URL failed
dl_try() {
  local dest="$1"; shift
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then log "already have $(basename "$dest")"; return 0; fi
  local url
  for url in "$@"; do
    local auth=()
    if [ -n "$HF_TOKEN" ] && [[ "$url" == https://huggingface.co/* ]]; then
      auth=(-H "Authorization: Bearer $HF_TOKEN")
    fi
    log "downloading $(basename "$dest")"
    if curl -g -fL --retry 5 --retry-delay 5 --retry-connrefused -C - --progress-bar \
         "${auth[@]}" -o "$dest.part" "$url"; then
      mv -f "$dest.part" "$dest"
      return 0
    fi
    warn "failed: $url"
  done
  return 1
}
dl() { dl_try "$@" || FAILED+=("$(basename "$1")"); return 0; }

# hf_list <repo> - every file path in a Hugging Face repo, one per line (empty on failure)
hf_list() {
  local auth=()
  [ -n "$HF_TOKEN" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  curl -fsSL "${auth[@]}" "https://huggingface.co/api/models/$1" 2>/dev/null | python3 -c '
import sys, json
try:
    for s in json.load(sys.stdin).get("siblings", []):
        print(s["rfilename"])
except Exception:
    pass
' || true
}

# fetch_gguf <repo> <quant> <conventional-file-name> - downloads the quant (all shards) into
# $STACK_DIR/models and prints the file name to load (first shard)
fetch_gguf() {
  local repo="$1" quant="$2" fallback="$3" listing paths=() p
  listing="$(hf_list "$repo")"
  mapfile -t paths < <(printf '%s\n' "$listing" | grep -i -- "$quant" | grep -i '\.gguf$' \
                         | grep -vi 'mmproj' | grep -vi 'dflash' | grep -vi 'mtp' | sort)
  if [ "${#paths[@]}" -eq 0 ]; then
    warn "Could not list $repo; trying the conventional file name $fallback" >&2
    paths=("$fallback")
  fi
  for p in "${paths[@]}"; do
    dl_try "$STACK_DIR/models/$(basename "$p")" "https://huggingface.co/$repo/resolve/main/$p" >&2 \
      || die "Download failed: $repo/$p. If it says 401/403, accept the license on Hugging Face and re-run with HF_TOKEN=hf_xxx."
  done
  basename "${paths[0]}"
}

# set_env_key KEY VALUE - create or replace KEY="VALUE" in .env
set_env_key() {
  if grep -s "^$1=" "$ENV_FILE" >/dev/null; then
    sed -i "s|^$1=.*|$1=\"$2\"|" "$ENV_FILE"
  else
    printf '%s="%s"\n' "$1" "$2" >> "$ENV_FILE"
  fi
}
# set_env_default KEY VALUE - add KEY only if .env does not have it yet
set_env_default() { grep -s "^$1=" "$ENV_FILE" >/dev/null || printf '%s="%s"\n' "$1" "$2" >> "$ENV_FILE"; }

gen_secret() { openssl rand -hex "${1:-24}"; }

# install_managed <dest> [mode] < content
# Writes a generated file, but never clobbers your edits. Setup remembers a fingerprint of
# every file it writes (in .managed-hashes):
#   file unchanged since setup wrote it -> updated to the new version
#   file edited by you                  -> kept; the new version is saved as <file>.new
#   file from an install that predates this tracking and differs -> backed up to <file>.bak, then updated
HASHES_FILE="$STACK_DIR/.managed-hashes"
install_managed() {
  local dest="$1" mode="${2:-644}" rel tmp new_sum cur_sum rec_sum
  rel="${dest#"$STACK_DIR"/}"
  tmp="$(mktemp)"; cat > "$tmp"
  new_sum="$(sha256sum "$tmp" | cut -d' ' -f1)"
  rec_sum="$(awk -v f="$rel" '$2 == f {print $1}' "$HASHES_FILE" 2>/dev/null || true)"
  if [ -f "$dest" ]; then
    cur_sum="$(sha256sum "$dest" | cut -d' ' -f1)"
    if [ "$cur_sum" = "$new_sum" ]; then
      rm -f "$tmp"; _record_hash "$rel" "$new_sum"; return 0
    fi
    if [ -n "$rec_sum" ] && [ "$cur_sum" != "$rec_sum" ]; then
      mv -f "$tmp" "$dest.new"; chmod "$mode" "$dest.new"
      warn "Kept your edited $rel. The new version from this setup run is $rel.new - merge anything you want from it."
      return 0
    fi
    if [ -z "$rec_sum" ]; then cp -p "$dest" "$dest.bak"; log "Backed up $rel to $rel.bak before updating it"; fi
  fi
  mkdir -p "$(dirname "$dest")"
  cat "$tmp" > "$dest"; rm -f "$tmp"     # in place: keeps bind mounts and ownership intact
  chmod "$mode" "$dest"
  _record_hash "$rel" "$new_sum"
}
_record_hash() {
  local rel="$1" sum="$2" tmp
  tmp="$(mktemp)"
  awk -v f="$rel" '$2 != f' "$HASHES_FILE" 2>/dev/null > "$tmp" || true
  printf '%s %s\n' "$sum" "$rel" >> "$tmp"
  cat "$tmp" > "$HASHES_FILE"; rm -f "$tmp"
}

# --------------------------- 0. preflight ------------------------------------
log "Preflight checks"
[ "$(id -u)" -ne 0 ] || die "Run as a regular user with sudo rights, not as root."

# shellcheck disable=SC1091
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || warn "Not Ubuntu (${PRETTY_NAME:-unknown}); this script is written for Ubuntu Server 26.04."
case "${VERSION_ID:-}" in 26.04*) ;; *) warn "Written for Ubuntu 26.04.x; detected ${VERSION_ID:-unknown}. Continuing." ;; esac

sudo -v
( while true; do sudo -n true 2>/dev/null; sleep 45; kill -0 "$$" 2>/dev/null || exit 0; done ) &
SUDO_KEEPALIVE_PID=$!
trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true' EXIT

ram_gb="$(awk '/^MemTotal:/ {printf "%d", $2 / 1048576}' /proc/meminfo)"
[ "$ram_gb" -ge 48 ] || warn "Only ${ram_gb} GB RAM. Both LLMs offload part of themselves to RAM; 64 GB+ is recommended."

if [ "$COMFY_MODELS_DIR" = "auto" ]; then
  if [ -d "$HOME/comfyui/ComfyUI/models" ]; then
    COMFY_MODELS_DIR="$HOME/comfyui/ComfyUI/models"
  else
    COMFY_MODELS_DIR="$STACK_DIR/data/comfyui/models"
  fi
fi

need_gb=70                                              # images (~20 GB) + Muse (~19) + Qwen (~23) + headroom
[ "$ENABLE_COMFYUI" = 1 ] && need_gb=$((need_gb + 15))  # ComfyUI image build (PyTorch)
if [ "$ENABLE_COMFYUI" = 1 ] && [ ! -d "$HOME/comfyui/ComfyUI/models/diffusion_models" ]; then
  need_gb=$((need_gb + 12))
fi
mkdir -p "$STACK_DIR/models"
have_gb=0
for f in "$STACK_DIR"/models/*.gguf; do [ -f "$f" ] && have_gb=$(( have_gb + $(stat -c %s "$f") / 1073741824 )); done
need_gb=$(( need_gb - have_gb ))
avail_kb="$(df -Pk "$STACK_DIR" | awk 'NR==2 {print $4}')"
if [ "$avail_kb" -lt $((need_gb * 1024 * 1024)) ]; then
  die "Need roughly ${need_gb} GB free under $STACK_DIR; only $((avail_kb / 1024 / 1024)) GB available."
fi

curl -fsSL -o /dev/null --max-time 20 https://huggingface.co || die "Cannot reach huggingface.co - check internet/DNS."
curl -fsSL -o /dev/null --max-time 20 https://github.com      || die "Cannot reach github.com - check internet/DNS."

# ---------------------- 1. system packages + NVIDIA driver -------------------
log "Installing system packages"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl wget git gnupg openssl pciutils python3 ubuntu-drivers-common iproute2
for pkg in htop nvtop tmux; do
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg" || warn "optional package '$pkg' not installed"
done

lspci | grep -i nvidia || die "No NVIDIA GPU detected by lspci."

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  log "No working NVIDIA driver - installing the recommended compute driver"
  if command -v mokutil >/dev/null 2>&1 && [[ "$(mokutil --sb-state 2>/dev/null || true)" == *[Ee]nabled* ]]; then
    warn "Secure Boot is enabled: you may be asked to set a MOK password and enroll it on next boot."
  fi
  sudo ubuntu-drivers install --gpgpu || sudo ubuntu-drivers install
  log "Driver installed. REBOOT NOW (sudo reboot), then run this script again to continue."
  exit 0
fi

driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p)"
[ "${driver_ver%%.*}" -ge 580 ] || die "Driver $driver_ver is too old (need 580+ for the CUDA 13 images). Run: sudo ubuntu-drivers install --gpgpu, reboot, re-run."
log "NVIDIA driver $driver_ver"
nvidia-smi -L
GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU' || true)"
[ "$GPU_COUNT" -ge 1 ] || die "No GPUs found."
[ "$COMFY_GPU" -lt "$GPU_COUNT" ] || die "COMFY_GPU=$COMFY_GPU but only $GPU_COUNT GPU(s) found."

LLM_GPUS_ALL=""
for ((i = 0; i < GPU_COUNT; i++)); do LLM_GPUS_ALL="${LLM_GPUS_ALL:+$LLM_GPUS_ALL,}$i"; done
LLM_GPUS_SHARED=""
for ((i = 0; i < GPU_COUNT; i++)); do
  [ "$i" -eq "$COMFY_GPU" ] && continue
  LLM_GPUS_SHARED="${LLM_GPUS_SHARED:+$LLM_GPUS_SHARED,}$i"
done
[ -n "$LLM_GPUS_SHARED" ] || LLM_GPUS_SHARED="$LLM_GPUS_ALL"   # single GPU: ComfyUI and the LLM take turns

phys_cores="$(lscpu -p=CORE,SOCKET 2>/dev/null | awk -F, '!/^#/ {seen[$0] = 1} END {n = 0; for (k in seen) n++; print n}')"
if [ -z "$phys_cores" ] || [ "$phys_cores" -lt 1 ]; then phys_cores=$(( $(nproc) / 2 )); fi
[ "$phys_cores" -ge 1 ] || phys_cores=1

# -------------------------- 2. Docker + GPU runtime --------------------------
if ! command -v docker >/dev/null 2>&1 || ! sudo docker compose version >/dev/null 2>&1; then
  log "Installing Docker Engine + Compose (Ubuntu packages)"
  if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2; then
    warn "Ubuntu's Docker packages were not available - trying Docker's own repository"
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
      | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
      || die "Could not install Docker. Install Docker Engine + the compose plugin manually, then re-run."
  fi
fi
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker-buildx >/dev/null 2>&1 || true
sudo systemctl enable --now docker
sudo docker compose version || die "'docker compose' is not working."
if [[ " $(id -nG) " != *" docker "* ]]; then
  sudo usermod -aG docker "$(id -un)"
  warn "Added $(id -un) to the docker group. Until you log out and back in, 'aistack' will use sudo."
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  log "Installing the NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
fi
docker_info="$(sudo docker info 2>/dev/null || true)"
if [[ "$docker_info" != *nvidia* ]]; then
  log "Registering the NVIDIA runtime with Docker"
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
log "Testing GPU access from a container"
sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L \
  || die "Containers cannot see the GPUs. Check: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"

# ---------------------- 3. retire the earlier services -----------------------
if [ "$MIGRATE_OLD" = 1 ]; then
  old_units=()
  if [ -f /etc/systemd/system/comfyui@.service ]; then
    for ((i = 0; i < GPU_COUNT; i++)); do old_units+=("comfyui@$i"); done
  fi
  [ -f /etc/systemd/system/comfyui.service ] && old_units+=("comfyui")
  [ -f /etc/systemd/system/llama-server.service ] && old_units+=("llama-server")
  if [ "${#old_units[@]}" -gt 0 ]; then
    log "Disabling the earlier services (kept on disk): ${old_units[*]}"
    for u in "${old_units[@]}"; do sudo systemctl disable --now "$u" >/dev/null 2>&1 || true; done
  fi
fi
for p in "$PANEL_PORT" "$COMFY_PORT" "$LLM_PORT" "$WEBUI_PORT" "$HERMES_DASH_PORT" "$HERMES_API_PORT"; do
  busy="$(ss -ltnH "sport = :$p" 2>/dev/null || true)"
  if [ -n "$busy" ] && [ -z "$(sudo docker ps -q --filter "publish=$p" 2>/dev/null || true)" ]; then
    warn "Port $p is already in use by something outside Docker - that instance will fail to start. ($busy)"
  fi
done

# ------------------------------ 4. stack files -------------------------------
log "Writing the stack in $STACK_DIR"
mkdir -p "$STACK_DIR"/{build/comfyui,build/panel,llama-swap,models} \
         "$STACK_DIR"/data/comfyui/{output,input,user,custom_nodes,cache} \
         "$STACK_DIR"/data/{hermes,open-webui} \
         "$COMFY_MODELS_DIR"
chmod 700 "$STACK_DIR/data/hermes"
rm -f "$STACK_DIR/.mode"                     # v1 GPU modes are gone

LEGACY_FILE=""
if [ -d "$LEGACY_LLM_DIR" ]; then
  for f in "$LEGACY_LLM_DIR"/Qwen3-Coder-Next*.gguf; do [ -f "$f" ] && { LEGACY_FILE="$(basename "$f")"; break; }; done
fi
if [ -n "$LEGACY_FILE" ]; then
  LEGACY_MOUNT="$LEGACY_LLM_DIR"
else
  LEGACY_MOUNT="$STACK_DIR/models/.no-legacy"; mkdir -p "$LEGACY_MOUNT"
fi

comfy_tag="$(git ls-remote --tags --refs https://github.com/comfyanonymous/ComfyUI.git 'v*' 2>/dev/null \
             | awk -F/ '{print $3}' | sort -V | tail -n 1 || true)"
comfy_tag="${comfy_tag:-master}"

wanted=""
[ "$ENABLE_PANEL" = 1 ]   && wanted="$wanted panel"
[ "$ENABLE_COMFYUI" = 1 ] && wanted="$wanted comfyui"
wanted="$wanted llm"
[ "$ENABLE_HERMES" = 1 ]  && wanted="$wanted hermes"
[ "$ENABLE_WEBUI" = 1 ]   && wanted="$wanted webui"

if [ ! -f "$ENV_FILE" ]; then
  ( umask 077; cat > "$ENV_FILE" <<ENVEOF
# aistack settings - read by docker compose, the aistack CLI and the control panel.
# Edit, then restart the affected instance (panel button, or: aistack restart <name>).
# Secrets below were generated on first install. Keep this file private (mode 600).

ENABLED_SERVICES="${wanted# }"
LISTEN_ADDR="$LISTEN_ADDR"
PANEL_PORT="$PANEL_PORT"
COMFY_PORT="$COMFY_PORT"
LLM_PORT="$LLM_PORT"
WEBUI_PORT="$WEBUI_PORT"
HERMES_DASH_PORT="$HERMES_DASH_PORT"
HERMES_API_PORT="$HERMES_API_PORT"

# ComfyUI (one instance, one GPU)
COMFY_GPU="$COMFY_GPU"
COMFY_REF="$comfy_tag"
COMFY_MODELS_DIR="$COMFY_MODELS_DIR"
COMFY_EXTRA_ARGS=""

# LLM (llama-swap). Change, then restart the LLM.
LLM_CTX="$LLM_CTX"
LLM_PARALLEL="$LLM_PARALLEL"
MUSE_QUANT="$MUSE_QUANT"
MUSE_VISION="$MUSE_VISION"
QWEN_QUANT="$QWEN_QUANT"
# What the model name "auto" (used by Hermes) points to
AUTO_MODEL_SPLIT="qwen3.6-35b-a3b"   # while ComfyUI is running (LLM has one GPU)
AUTO_MODEL_FULL="muse-glimmer"       # while ComfyUI is stopped (LLM has all GPUs)

# Images
IMG_LLAMA_SWAP="$IMG_LLAMA_SWAP"
IMG_HERMES="$IMG_HERMES"
IMG_WEBUI="$IMG_WEBUI"

# Secrets
PANEL_USER="admin"
PANEL_PASS="$(gen_secret 12)"
LLM_API_KEY="sk-local-$(gen_secret 24)"
HERMES_API_KEY="$(gen_secret 32)"
HERMES_DASH_USER="admin"
HERMES_DASH_PASS="$(gen_secret 12)"
HERMES_DASH_SECRET="$(gen_secret 32)"
WEBUI_SECRET_KEY="$(gen_secret 32)"

# --- Detected by setup_aistack.sh (overwritten on every run) ---
ENVEOF
  )
  log "Created $ENV_FILE (settings + generated secrets)"
else
  log "Keeping existing $ENV_FILE (adding any new v2 settings)"
  set_env_default PANEL_PORT "$PANEL_PORT"
  set_env_default PANEL_USER "admin"
  set_env_default PANEL_PASS "$(gen_secret 12)"
  set_env_default QWEN_QUANT "$QWEN_QUANT"
  set_env_default AUTO_MODEL_SPLIT "qwen3.6-35b-a3b"
  set_env_default AUTO_MODEL_FULL "muse-glimmer"
  sed -i '/^DEFAULT_MODE=/d' "$ENV_FILE"
  current="$(sed -n 's/^ENABLED_SERVICES="\(.*\)"$/\1/p' "$ENV_FILE")"
  if [ "$ENABLE_PANEL" = 1 ] && [[ " $current " != *" panel "* ]]; then
    set_env_key ENABLED_SERVICES "panel $current"
  fi
fi
set_env_key STACK_DIR "$STACK_DIR"
set_env_key HOST_UID "$(id -u)"
set_env_key HOST_GID "$(id -g)"
set_env_key GPU_COUNT "$GPU_COUNT"
set_env_key LLM_GPUS_ALL "$LLM_GPUS_ALL"
set_env_key LLM_GPUS_SHARED "$LLM_GPUS_SHARED"
set_env_key LLM_THREADS "$phys_cores"
set_env_key LEGACY_LLM_DIR "$LEGACY_MOUNT"
chmod 600 "$ENV_FILE"

# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

# ---- compose.yaml (written by setup; your edits are kept - see install_managed)
install_managed "$STACK_DIR/compose.yaml" 644 <<'COMPOSEEOF'
# Generated by setup_aistack.sh. If you edit it, re-running setup keeps your version and writes
# the new one as compose.yaml.new. Small additions are easier in compose.override.yaml (merged automatically).
# Values like ${LLM_PORT} come from .env in this folder.
name: aistack

services:
  panel:
    build:
      context: ./build/panel
    image: aistack/panel:local
    container_name: aistack-panel
    restart: unless-stopped
    ports:
      - "${LISTEN_ADDR}:${PANEL_PORT}:8090"
    environment:
      AISTACK_ROOT: ${STACK_DIR}
      AISTACK_DOCKER: docker
      PANEL_USER: ${PANEL_USER}
      PANEL_PASS: ${PANEL_PASS}
      NVIDIA_DRIVER_CAPABILITIES: utility
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ${STACK_DIR}:${STACK_DIR}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, utility]

  comfyui:
    build:
      context: ./build/comfyui
      args:
        HOST_UID: ${HOST_UID}
        HOST_GID: ${HOST_GID}
        COMFY_REF: ${COMFY_REF}
    image: aistack/comfyui:local
    container_name: aistack-comfyui
    restart: unless-stopped
    ports:
      - "${LISTEN_ADDR}:${COMFY_PORT}:8188"
    environment:
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      COMFY_EXTRA_ARGS: ${COMFY_EXTRA_ARGS:-}
    volumes:
      - ${COMFY_MODELS_DIR}:/opt/ComfyUI/models
      - ./data/comfyui/output:/opt/ComfyUI/output
      - ./data/comfyui/input:/opt/ComfyUI/input
      - ./data/comfyui/user:/opt/ComfyUI/user
      - ./data/comfyui/custom_nodes:/opt/ComfyUI/custom_nodes
      - ./data/comfyui/cache:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${COMFY_GPU}"]
              capabilities: [gpu]

  llm:
    image: ${IMG_LLAMA_SWAP}
    container_name: aistack-llm
    restart: unless-stopped
    ports:
      - "${LISTEN_ADDR}:${LLM_PORT}:8080"
    environment:
      LLM_API_KEY: ${LLM_API_KEY}
    volumes:
      # mounted at both the older and the newer llama-swap image's config path
      - ./llama-swap/config.yaml:/app/config.yaml:ro
      - ./llama-swap/config.yaml:/etc/llama-swap/config/config.yaml:ro
      - ./models:/models:ro
      - ${LEGACY_LLM_DIR}:/models-legacy:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  hermes:
    image: ${IMG_HERMES}
    container_name: aistack-hermes
    restart: unless-stopped
    command: gateway run
    shm_size: "1g"
    ports:
      - "${LISTEN_ADDR}:${HERMES_DASH_PORT}:9119"
      - "127.0.0.1:${HERMES_API_PORT}:8642"
    environment:
      HERMES_UID: ${HOST_UID}
      HERMES_GID: ${HOST_GID}
      HERMES_DASHBOARD: "1"
      HERMES_DASHBOARD_BASIC_AUTH_USERNAME: ${HERMES_DASH_USER}
      HERMES_DASHBOARD_BASIC_AUTH_PASSWORD: ${HERMES_DASH_PASS}
      HERMES_DASHBOARD_BASIC_AUTH_SECRET: ${HERMES_DASH_SECRET}
      API_SERVER_ENABLED: "true"
      API_SERVER_HOST: "0.0.0.0"
      API_SERVER_KEY: ${HERMES_API_KEY}
    volumes:
      - ./data/hermes:/opt/data
    deploy:
      resources:
        limits:
          memory: 4g
          cpus: "2.0"

  webui:
    image: ${IMG_WEBUI}
    container_name: aistack-webui
    restart: unless-stopped
    ports:
      - "${LISTEN_ADDR}:${WEBUI_PORT}:8080"
    environment:
      OPENAI_API_BASE_URLS: "http://llm:8080/v1;http://hermes:8642/v1"
      OPENAI_API_KEYS: "${LLM_API_KEY};${HERMES_API_KEY}"
      ENABLE_OLLAMA_API: "false"
      WEBUI_SECRET_KEY: ${WEBUI_SECRET_KEY}
    volumes:
      - ./data/open-webui:/app/backend/data
COMPOSEEOF

# ---- ComfyUI image build files (written by setup; your edits are kept - see install_managed)
install_managed "$STACK_DIR/build/comfyui/Dockerfile" 644 <<'DOCKEREOF'
# ComfyUI image for aistack. Built locally by setup_aistack.sh and `aistack update`.
FROM ubuntu:24.04
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG COMFY_REF=master
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_CACHE_DIR=/cache/uv \
    TORCH_INDEX=${TORCH_INDEX} \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates ffmpeg libgl1 libglib2.0-0t64 build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN uv python install 3.13 && uv venv --python 3.13 /opt/venv

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
 && git -C /opt/ComfyUI checkout "${COMFY_REF}"

RUN export UV_NO_CACHE=1 \
 && uv pip install --python /opt/venv/bin/python --extra-index-url "${TORCH_INDEX}" --index-strategy unsafe-best-match \
      torch torchvision torchaudio \
 && uv pip install --python /opt/venv/bin/python --extra-index-url "${TORCH_INDEX}" --index-strategy unsafe-best-match \
      -r /opt/ComfyUI/requirements.txt \
 && if [ -f /opt/ComfyUI/manager_requirements.txt ]; then \
      uv pip install --python /opt/venv/bin/python -r /opt/ComfyUI/manager_requirements.txt; fi \
 && uv pip install --python /opt/venv/bin/python img2pdf

# Run as your host user so files in the mounted folders stay yours
RUN (userdel -r ubuntu 2>/dev/null || true) \
 && (getent group "${HOST_GID}" >/dev/null || groupadd -g "${HOST_GID}" comfy) \
 && (getent passwd "${HOST_UID}" >/dev/null || useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash comfy) \
 && mkdir -p /cache/uv \
 && chown -R "${HOST_UID}:${HOST_GID}" /opt/venv /opt/ComfyUI /cache
COPY entrypoint.sh /usr/local/bin/comfy-entrypoint
RUN chmod 755 /usr/local/bin/comfy-entrypoint
USER ${HOST_UID}:${HOST_GID}
ENV HOME=/tmp
WORKDIR /opt/ComfyUI
EXPOSE 8188
ENTRYPOINT ["/usr/local/bin/comfy-entrypoint"]
DOCKEREOF

install_managed "$STACK_DIR/build/comfyui/entrypoint.sh" 755 <<'ENTRYEOF'
#!/usr/bin/env bash
# Starts ComfyUI. Installs Python dependencies of custom nodes (e.g. ones added
# with the Manager) once per requirements.txt version, since the venv lives in the image.
set -uo pipefail
cd /opt/ComfyUI
for req in custom_nodes/*/requirements.txt; do
  [ -f "$req" ] || continue
  name="$(basename "$(dirname "$req")")"
  sum="$(sha256sum "$req" | cut -d' ' -f1)"
  marker="/opt/venv/.custom-node-${name}-${sum}"
  [ -f "$marker" ] && continue
  echo "[comfy-entrypoint] installing requirements for custom node: $name"
  if uv pip install --python /opt/venv/bin/python --extra-index-url "${TORCH_INDEX}" \
       --index-strategy unsafe-best-match -r "$req"; then
    touch "$marker"
  else
    echo "[comfy-entrypoint] WARNING: requirements for $name failed to install" >&2
  fi
done
help_out="$(python main.py --help 2>/dev/null || true)"
manager=()
if [[ "$help_out" == *--enable-manager* ]]; then manager=(--enable-manager); fi
read -r -a extra <<< "${COMFY_EXTRA_ARGS:-}"
exec python main.py --listen 0.0.0.0 --port 8188 "${manager[@]}" "${extra[@]}"
ENTRYEOF

# ---- control panel image (written by setup; your edits are kept - see install_managed)
install_managed "$STACK_DIR/build/panel/Dockerfile" 644 <<'PANELDOCKEREOF'
# aistack control panel. Built locally by setup_aistack.sh and `aistack update`.
FROM docker:cli AS dockercli
FROM python:3.13-slim
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
# docker compose plugin, straight from its GitHub releases
RUN set -e; arch="$(uname -m)"; case "$arch" in aarch64|arm64) arch=aarch64 ;; *) arch=x86_64 ;; esac; \
    mkdir -p /usr/local/libexec/docker/cli-plugins; \
    python3 -c "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$arch" \
      /usr/local/libexec/docker/cli-plugins/docker-compose; \
    chmod 755 /usr/local/libexec/docker/cli-plugins/docker-compose; \
    docker compose version --short >/dev/null 2>&1 || docker --version
COPY panel.py /app/panel.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8090
CMD ["python3", "/app/panel.py"]
PANELDOCKEREOF

install_managed "$STACK_DIR/build/panel/panel.py" 644 <<'PANELEOF'
#!/usr/bin/env python3
"""aistack control panel.

Start/stop the AI instances, watch GPU use, read logs, load/unload LLM models.
Runs in its own container with the Docker socket and the stack folder mounted at
the same path as on the host. Every start/stop goes through the `aistack` CLI, so
the panel and the command line follow exactly the same GPU rules.
"""
import base64
import hmac
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.environ.get("AISTACK_ROOT", "/stack")
CLI = ["bash", os.path.join(ROOT, "aistack")]
USER = os.environ.get("PANEL_USER", "admin")
PASS = os.environ.get("PANEL_PASS", "")
LLM_URL = os.environ.get("PANEL_LLM_URL", "http://llm:8080").rstrip("/")
PORT = int(os.environ.get("PANEL_LISTEN_PORT", "8090"))
DOCKER = os.environ.get("PANEL_DOCKER", "docker").split()

SERVICES = {
    "comfyui": {"title": "ComfyUI", "subtitle": "Image generation (Z-Image coloring pages)",
                "port_key": "COMFY_PORT", "path": "/"},
    "llm": {"title": "LLM server", "subtitle": "llama-swap: loads whichever model a client asks for",
            "port_key": "LLM_PORT", "path": "/ui"},
    "hermes": {"title": "Hermes", "subtitle": "Agent for homelab and Home Assistant tasks",
               "port_key": "HERMES_DASH_PORT", "path": "/"},
    "webui": {"title": "Chat UI", "subtitle": "Open WebUI for the models and Hermes",
              "port_key": "WEBUI_PORT", "path": "/"},
}
ACTIONS = ("start", "stop", "restart")

action_lock = threading.Lock()
loading = {}          # model id -> time the load started
events = deque(maxlen=20)


def read_env():
    env = {}
    try:
        with open(os.path.join(ROOT, ".env"), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                try:
                    parts = shlex.split(value)
                except ValueError:
                    parts = [value]
                env[key.strip()] = parts[0] if parts else ""
    except OSError:
        pass
    return env


def run(args, timeout=120):
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except OSError as err:
        return 127, str(err)


def container_state(svc):
    code, out = run(DOCKER + ["inspect", "--format", "{{json .State}}", f"aistack-{svc}"], 15)
    if code != 0 or not out:
        return {"exists": False, "running": False, "status": "not created"}
    try:
        st = json.loads(out.splitlines()[-1])
    except ValueError:
        return {"exists": True, "running": False, "status": "unknown"}
    return {"exists": True, "running": bool(st.get("Running")), "status": st.get("Status", ""),
            "health": (st.get("Health") or {}).get("Status", ""), "started": st.get("StartedAt", "")}


def gpus():
    code, out = run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"], 10)
    result = []
    if code != 0:
        return result
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            result.append({"index": parts[0], "name": parts[1], "used": float(parts[2]),
                           "total": float(parts[3]), "util": float(parts[4]) if parts[4] not in ("", "[N/A]") else 0})
        except ValueError:
            continue
    return result


def llm_request(env, path, body=None, timeout=5, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(LLM_URL + path, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Authorization": f"Bearer {env.get('LLM_API_KEY', '')}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def models(env):
    out = {"available": [], "running": {}, "error": ""}
    try:
        _, body = llm_request(env, "/v1/models")
        for m in json.loads(body).get("data", []):
            if m.get("id"):
                out["available"].append({"id": m["id"], "name": m.get("name") or m["id"]})
    except Exception as err:  # noqa: BLE001 - shown to the user
        out["error"] = f"model list unavailable ({err})"
        return out
    ids = {m["id"] for m in out["available"]}
    try:
        _, body = llm_request(env, "/running")
        running = json.loads(body)
    except Exception:  # noqa: BLE001
        running = None

    def walk(node):
        if isinstance(node, dict):
            mid = node.get("model") or node.get("id") or node.get("name")
            if isinstance(mid, str) and mid in ids:
                out["running"][mid] = str(node.get("state", "ready"))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(running)
    return out


def layout(env, comfy_running):
    gpu_count = int(env.get("GPU_COUNT", "1") or 1)
    split = gpu_count >= 2 and comfy_running
    llm_gpus = env.get("LLM_GPUS_SHARED" if split else "LLM_GPUS_ALL", "")
    if gpu_count >= 2 and not comfy_running:
        auto = env.get("AUTO_MODEL_FULL", "muse-glimmer")
    else:
        auto = env.get("AUTO_MODEL_SPLIT", "qwen3.6-35b-a3b")
    return {"name": "split" if split else "full", "llm_gpus": llm_gpus, "auto": auto,
            "gpu_count": gpu_count, "comfy_gpu": env.get("COMFY_GPU", "0")}


def state():
    env = read_env()
    enabled = env.get("ENABLED_SERVICES", "").split()
    services = []
    for key, meta in SERVICES.items():
        st = container_state(key)
        services.append({"key": key, "title": meta["title"], "subtitle": meta["subtitle"],
                         "port": env.get(meta["port_key"], ""), "path": meta["path"],
                         "enabled": key in enabled, **st})
    by_key = {s["key"]: s for s in services}
    comfy_running = by_key["comfyui"]["running"]
    llm_running = by_key["llm"]["running"]
    lay = layout(env, comfy_running)
    gpu_list = gpus()
    for g in gpu_list:
        users = []
        if comfy_running and g["index"] == lay["comfy_gpu"]:
            users.append("ComfyUI")
        if llm_running and g["index"] in lay["llm_gpus"].split(","):
            users.append("LLM")
        g["users"] = users
    now = time.time()
    for mid in [m for m, t in loading.items() if now - t > 1800]:
        loading.pop(mid, None)
    mdl = models(env) if llm_running else {"available": [], "running": {}, "error": ""}
    return {"services": services, "gpus": gpu_list, "layout": lay, "models": mdl,
            "loading": sorted(loading), "llm_loaded": bool(mdl["running"]),
            "busy": action_lock.locked(), "events": list(events)}


def do_action(svc, action):
    if svc not in SERVICES or action not in ACTIONS:
        return 400, {"error": "unknown service or action"}
    if not action_lock.acquire(blocking=False):
        return 409, {"error": "another start/stop is still running - try again in a moment"}
    try:
        code, out = run(CLI + [action, svc], 900)
    finally:
        action_lock.release()
    events.appendleft({"time": time.strftime("%H:%M:%S"), "what": f"{action} {svc}", "ok": code == 0, "output": out[-2000:]})
    return (200 if code == 0 else 500), {"output": out or f"{action} {svc}: done", "error": "" if code == 0 else out}


def model_action(model_id, action):
    env = read_env()
    try:
        ids = {m["id"] for m in models(env)["available"]}
    except Exception:  # noqa: BLE001
        ids = set()
    if action == "unload":
        last = ""
        for path, method in (("/api/models/unload", "POST"), ("/unload", "GET"), ("/unload", "POST")):
            try:
                llm_request(env, path, timeout=60, method=method)
                return 200, {"message": "Model unloaded - its GPU memory is free now."}
            except Exception as err:  # noqa: BLE001
                last = str(err)
        return 502, {"error": f"unload failed: {last}"}
    if action != "load" or model_id not in ids:
        return 400, {"error": "unknown model or action"}
    if model_id in loading:
        return 200, {"message": f"{model_id} is already loading."}

    def worker():
        loading[model_id] = time.time()
        try:
            llm_request(env, "/v1/chat/completions", timeout=1800,
                        body={"model": model_id, "max_tokens": 1,
                              "messages": [{"role": "user", "content": "hi"}]})
        except Exception:  # noqa: BLE001 - the state view shows whether it loaded
            pass
        finally:
            loading.pop(model_id, None)
    threading.Thread(target=worker, daemon=True).start()
    return 200, {"message": f"Loading {model_id} - this can take a few minutes from disk."}


def logs(svc):
    if svc not in SERVICES:
        return 400, "unknown service"
    _, out = run(DOCKER + ["logs", "--tail", "300", f"aistack-{svc}"], 20)
    return 200, out or "(no output)"


class Handler(BaseHTTPRequestHandler):
    server_version = "aistack-panel"

    def log_message(self, fmt, *args):  # quiet access log
        pass

    def _authorized(self):
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            except Exception:  # noqa: BLE001
                user = password = ""
            if hmac.compare_digest(user.encode(), USER.encode()) and hmac.compare_digest(password.encode(), PASS.encode()):
                return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="AI Stack", charset="UTF-8"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _send(self, code, payload, ctype="application/json; charset=utf-8"):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if not self._authorized():
            return
        url = urllib.parse.urlparse(self.path)
        if url.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif url.path == "/api/state":
            self._send(200, state())
        elif url.path == "/api/logs":
            svc = urllib.parse.parse_qs(url.query).get("svc", [""])[0]
            code, text = logs(svc)
            self._send(code, text, "text/plain; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if not self._authorized():
            return
        if self.headers.get("X-Panel") != "1":   # blocks cross-site form posts
            self._send(403, {"error": "missing X-Panel header"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 10000)
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"error": "bad JSON"})
            return
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/action":
            code, payload = do_action(str(body.get("svc", "")), str(body.get("action", "")))
        elif path == "/api/model":
            code, payload = model_action(str(body.get("id", "")), str(body.get("action", "")))
        else:
            code, payload = 404, {"error": "not found"}
        self._send(code, payload)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Stack</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--fg:#1b1e23;--muted:#626a74;--line:#e1e4e9;--accent:#2f6fde;--ok:#1d8048;--warn:#a86d12;--bad:#c22f2f;--bar:#e7eaef}
@media (prefers-color-scheme:dark){:root{--bg:#111316;--card:#1a1d22;--fg:#e6e8eb;--muted:#99a0aa;--line:#2b2f37;--accent:#6d9dff;--ok:#4cc38a;--warn:#e0a84a;--bad:#f07272;--bar:#292d35}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:0 16px}
header{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;padding:20px 0 6px}
h1{font-size:20px;margin:0;letter-spacing:-.01em}
.muted{color:var(--muted)}.small{font-size:13px}
.gpus{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:12px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.bar{height:8px;background:var(--bar);border-radius:4px;overflow:hidden;margin:8px 0 4px}
.bar>i{display:block;height:100%;background:var(--accent);transition:width .4s}
.layout{margin:4px 0 14px;padding:10px 14px;border-left:3px solid var(--accent);background:var(--card);border-radius:6px;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px;margin-bottom:24px}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px}
h2{font-size:16px;margin:0}
.sub{font-size:13px;color:var(--muted);margin:3px 0 12px}
.pill{font-size:12px;padding:1px 9px;border-radius:999px;border:1px solid currentColor;white-space:nowrap}
.on{color:var(--ok)}.off{color:var(--muted)}.busy{color:var(--warn)}.bad{color:var(--bad)}
.btns{display:flex;gap:6px;flex-wrap:wrap}
button,a.btn{font:inherit;font-size:13px;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer;text-decoration:none;line-height:1.3}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.danger{color:var(--bad)}
button:disabled{opacity:.45;cursor:default}
.models{margin-top:12px;border-top:1px solid var(--line);padding-top:8px}
.model{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:6px 0}
.model .name{font-size:14px}
.tag{font-size:11px;padding:1px 6px;border-radius:4px;background:var(--bar);margin-left:6px;color:var(--muted)}
.tag.auto{color:var(--accent)}
dialog{width:min(920px,94vw);border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg);padding:0}
dialog::backdrop{background:rgba(0,0,0,.4)}
dialog pre{margin:0;padding:14px;max-height:70vh;overflow:auto;font-size:12px;white-space:pre-wrap;word-break:break-all}
.dlgtop{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line)}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--fg);color:var(--bg);padding:8px 14px;border-radius:8px;font-size:13px;display:none;max-width:92vw;white-space:pre-wrap}
</style></head><body>
<div class="wrap">
<header><h1>AI Stack</h1><span class="muted small" id="updated">loading…</span></header>
<div class="gpus" id="gpus"></div>
<div class="layout" id="layout"></div>
<div class="grid" id="services"></div>
</div>
<dialog id="logdlg"><div class="dlgtop"><strong id="logtitle"></strong><span class="btns"><button data-do="relog">Refresh</button><button data-do="closelog">Close</button></span></div><pre id="logtext"></pre></dialog>
<div id="toast"></div>
<script>
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let S = null, pending = {}, logSvc = null;
function toast(msg){const t=document.getElementById("toast");t.textContent=msg;t.style.display="block";clearTimeout(t._h);t._h=setTimeout(()=>t.style.display="none",7000);}
async function api(path, body){
  const opt = body ? {method:"POST",headers:{"Content-Type":"application/json","X-Panel":"1"},body:JSON.stringify(body)} : {cache:"no-store"};
  const r = await fetch(path, opt); const text = await r.text(); let data;
  try { data = JSON.parse(text) } catch { data = {output:text} }
  if (!r.ok) throw new Error(data.error || data.output || ("HTTP " + r.status));
  return data;
}
async function refresh(){
  try { S = await api("/api/state"); draw(); document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString(); }
  catch(e){ document.getElementById("updated").textContent = "panel not reachable: " + e.message; }
}
function openUrl(s){ return `http://${location.hostname}:${s.port}${s.path}`; }
function draw(){
  const lay = S.layout;
  document.getElementById("gpus").innerHTML = S.gpus.length ? S.gpus.map(g => {
    const pct = Math.round(100 * g.used / Math.max(g.total, 1));
    return `<div class="card"><div class="top"><strong>GPU ${esc(g.index)}</strong><span class="small ${g.users.length?"on":"muted"}">${esc(g.users.join(" + ") || "idle")}</span></div>
      <div class="muted small">${esc(g.name)}</div><div class="bar"><i style="width:${pct}%"></i></div>
      <div class="muted small">${(g.used/1024).toFixed(1)} / ${(g.total/1024).toFixed(1)} GB · ${Math.round(g.util)}% busy</div></div>`;
  }).join("") : `<div class="card muted small">GPU stats unavailable</div>`;
  const comfyOn = S.services.find(s => s.key === "comfyui")?.running;
  let msg;
  if (lay.gpu_count < 2) msg = `One GPU: ComfyUI and the LLM take turns - starting one stops the other. Hermes' model <b>auto</b> = ${esc(lay.auto)}.`;
  else if (comfyOn) msg = `ComfyUI is running on GPU ${esc(lay.comfy_gpu)}, so the LLM uses GPU ${esc(lay.llm_gpus)} + system RAM. Hermes' model <b>auto</b> = ${esc(lay.auto)}. Stop ComfyUI to give the LLM every GPU.`;
  else msg = `ComfyUI is stopped, so the LLM uses GPUs ${esc(lay.llm_gpus)}. Hermes' model <b>auto</b> = ${esc(lay.auto)}. Starting ComfyUI moves the LLM back to one GPU.`;
  document.getElementById("layout").innerHTML = msg;
  document.getElementById("services").innerHTML = S.services.map(card).join("");
}
function card(s){
  const p = pending[s.key];
  let pill;
  if (p) pill = `<span class="pill busy">${esc(p)}…</span>`;
  else if (!s.exists) pill = `<span class="pill off">not created</span>`;
  else if (s.running) pill = `<span class="pill on">running</span>`;
  else pill = `<span class="pill off">stopped</span>`;
  const dis = (p || S.busy) ? "disabled" : "";
  let btns = "";
  if (s.running) {
    btns += `<button data-act="stop" data-svc="${s.key}" class="danger" ${dis}>Stop</button>`;
    btns += `<button data-act="restart" data-svc="${s.key}" ${dis}>Restart</button>`;
    btns += `<a class="btn" href="${esc(openUrl(s))}" target="_blank" rel="noopener">Open ↗</a>`;
  } else {
    btns += `<button data-act="start" data-svc="${s.key}" class="primary" ${dis}>Start</button>`;
  }
  if (s.exists) btns += `<button data-log="${s.key}">Logs</button>`;
  let extra = "";
  if (s.key === "llm" && s.running) extra = modelsBlock();
  const note = s.enabled ? "" : ` <span class="tag">not in auto-start</span>`;
  return `<div class="card"><div class="top"><h2>${esc(s.title)}${note}</h2>${pill}</div>
    <div class="sub">${esc(s.subtitle)}</div><div class="btns">${btns}</div>${extra}</div>`;
}
function modelsBlock(){
  const m = S.models, lay = S.layout;
  if (m.error) return `<div class="models muted small">${esc(m.error)} - it may still be starting.</div>`;
  const rows = m.available.map(x => {
    const loaded = x.id in m.running, isLoading = S.loading.includes(x.id);
    const status = isLoading ? `<span class="pill busy">loading…</span>` : loaded ? `<span class="pill on">loaded</span>` : "";
    const btn = loaded ? `<button data-model="${esc(x.id)}" data-mact="unload">Unload</button>`
                       : `<button data-model="${esc(x.id)}" data-mact="load" ${isLoading?"disabled":""}>Load</button>`;
    const auto = x.id === lay.auto ? `<span class="tag auto">auto</span>` : "";
    return `<div class="model"><div><span class="name">${esc(x.name)}</span>${auto}<div class="muted small">${esc(x.id)}</div></div>
      <div class="btns">${status}${btn}</div></div>`;
  }).join("");
  return `<div class="models"><div class="muted small">Models load on demand when a client asks for them; one at a time.</div>${rows}</div>`;
}
async function act(svc, action){
  const llm = S.services.find(s => s.key === "llm");
  if (svc === "comfyui" && action !== "restart" && S.layout.gpu_count >= 2 && llm?.running &&
      !confirm("This restarts the LLM server to move it between GPUs. A loaded model is unloaded and any running request is cut off. Continue?")) return;
  if (svc === "comfyui" && action === "start" && S.layout.gpu_count < 2 && llm?.running &&
      !confirm("This machine has one GPU: starting ComfyUI stops the LLM server. Continue?")) return;
  if (svc === "llm" && action === "start" && S.layout.gpu_count < 2 && S.services.find(s => s.key === "comfyui")?.running &&
      !confirm("This machine has one GPU: starting the LLM stops ComfyUI. Continue?")) return;
  pending[svc] = action === "stop" ? "stopping" : action === "start" ? "starting" : "restarting"; draw();
  try { const r = await api("/api/action", {svc, action}); toast(r.output); }
  catch(e){ toast("Failed: " + e.message); }
  delete pending[svc]; refresh();
}
async function model(id, action){
  try { const r = await api("/api/model", {id, action}); toast(r.message); } catch(e){ toast("Failed: " + e.message); }
  refresh();
}
async function showLogs(svc){
  logSvc = svc;
  document.getElementById("logtitle").textContent = "Logs: " + svc;
  const pre = document.getElementById("logtext"); pre.textContent = "loading…";
  const dlg = document.getElementById("logdlg"); if (!dlg.open) dlg.showModal();
  try { const r = await fetch("/api/logs?svc=" + encodeURIComponent(svc), {cache:"no-store"}); pre.textContent = await r.text(); pre.scrollTop = pre.scrollHeight; }
  catch(e){ pre.textContent = "Could not load logs: " + e.message; }
}
document.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.act) act(b.dataset.svc, b.dataset.act);
  else if (b.dataset.model) model(b.dataset.model, b.dataset.mact);
  else if (b.dataset.log) showLogs(b.dataset.log);
  else if (b.dataset.do === "relog" && logSvc) showLogs(logSvc);
  else if (b.dataset.do === "closelog") document.getElementById("logdlg").close();
});
refresh(); setInterval(refresh, 5000);
</script></body></html>"""


def main():
    if not PASS:
        sys.exit("PANEL_PASS is not set - refusing to start without a password.")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"aistack panel listening on :{PORT} (stack: {ROOT})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
PANELEOF

# ---- llama-swap model catalogue (YOUR file to edit; upgraded once from v1 with a backup)
TEMPLATE="$STACK_DIR/llama-swap/config.template.yaml"
if [ -f "$TEMPLATE" ] && [[ "$(head -n 1 "$TEMPLATE")" != *"aistack-template v2"* ]]; then
  mv -f "$TEMPLATE" "$TEMPLATE.v1.bak"
  warn "Saved the v1 model catalogue as $TEMPLATE.v1.bak and wrote a v2 one (copy any models you added)."
fi
if [ ! -f "$TEMPLATE" ]; then
  cat > "$TEMPLATE" <<'TPLEOF'
# aistack-template v2 - llama-swap model catalogue.
# Edit THIS file, then restart the LLM server (panel, or: aistack restart llm).
# config.yaml is generated from it. The @NAME@ placeholders are filled in from ~/aistack/.env
# and the GPU layout (which follows whether ComfyUI is running):
#   @LLM_GPUS@     GPUs the LLM may use right now
#   @AUTO_FOR_x@   becomes the alias "auto" on model x when x is the current pick, else removed
#
# Add a model: put its .gguf in ~/aistack/models (seen here as /models), copy a block below,
# then change the key, name, aliases, --model path and sampling flags.
# One model is loaded at a time; asking for another one swaps it in.

healthCheckTimeout: 900
logLevel: info
sendLoadingState: true
globalTTL: 0

apiKeys:
  - "${env.LLM_API_KEY}"

macros:
  "llama-server": "@LLAMA_SERVER@ --host 127.0.0.1 --port ${PORT} --jinja --threads @THREADS@"
  "gpus": "CUDA_VISIBLE_DEVICES=@LLM_GPUS@"

models:
  "qwen3.6-35b-a3b":
    name: "Qwen3.6 35B-A3B (@QWEN_QUANT@) - fast on one GPU"
    description: "Mixture-of-experts, 3B active. Default while ComfyUI is running."
    aliases:
      - "qwen"
@AUTO_FOR_qwen3.6-35b-a3b@
    env:
      - "CUDA_DEVICE_ORDER=PCI_BUS_ID"
      - "${gpus}"
    cmd: |
      ${llama-server}
      --model /models/@QWEN_FILE@
      --ctx-size @LLM_CTX@
      --parallel @LLM_PARALLEL@
      --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0

  "muse-glimmer":
    name: "Muse Glimmer 30B (@MUSE_QUANT@) - best with both GPUs"
    description: "Meta's open agentic/coding model. Default while ComfyUI is stopped."
    aliases:
      - "muse"
@AUTO_FOR_muse-glimmer@
    env:
      - "CUDA_DEVICE_ORDER=PCI_BUS_ID"
      - "${gpus}"
    cmd: |
      ${llama-server}
      --model /models/@MUSE_FILE@
      @MUSE_VISION_ARGS@
      --ctx-size @LLM_CTX@
      --parallel @LLM_PARALLEL@
      --cache-type-k q8_0
      --cache-type-v q8_0
      --temp 1.0 --top-p 0.95 --top-k 64
    # Reasoning-effort variants: request "muse-glimmer:low" or "muse-glimmer:high".
    # (Whether Muse's chat template reads this exact parameter is unverified.)
    filters:
      setParamsByID:
        "${MODEL_ID}:low":
          chat_template_kwargs:
            reasoning_effort: low
        "${MODEL_ID}:high":
          chat_template_kwargs:
            reasoning_effort: high
TPLEOF
  if [ -n "$LEGACY_FILE" ]; then
    cat >> "$TEMPLATE" <<TPLEOF2

  # From the earlier setup_local_llm.sh install (80B mixture-of-experts, 3B active)
  "qwen3-coder-next":
    name: "Qwen3-Coder-Next 80B-A3B"
    env:
      - "CUDA_DEVICE_ORDER=PCI_BUS_ID"
      - "\${gpus}"
    cmd: |
      \${llama-server}
      --model /models-legacy/$LEGACY_FILE
      --ctx-size @LLM_CTX@
      --parallel @LLM_PARALLEL@
      --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01
TPLEOF2
  fi
  log "Created the model catalogue: $TEMPLATE"
else
  log "Keeping your model catalogue: $TEMPLATE"
fi

# ---- the aistack CLI (written by setup; your edits are kept - see install_managed); the panel calls it too
sed "s|@ROOT@|$STACK_DIR|g" <<'CLIEOF' | install_managed "$STACK_DIR/aistack" 755
#!/usr/bin/env bash
# aistack - manage the AI stack in @ROOT@  (the web panel uses these same commands)
#
#   aistack status                      what is running, GPU use, URLs
#   aistack start|stop|restart <name>   name: comfyui | llm | hermes | webui | panel | all
#   aistack logs <name> [-f]
#   aistack models                      models the LLM server offers, and what is loaded
#   aistack render                      regenerate llama-swap/config.yaml from the template
#   aistack enable|disable <name>       include/exclude an instance from "start all"
#   aistack hermes <args>               Hermes CLI inside its container (e.g. aistack hermes model)
#   aistack update                      pull new images, rebuild ComfyUI + panel, keep on/off states
#   aistack urls
#
# GPU rule, applied automatically when ComfyUI or the LLM starts/stops:
#   ComfyUI running -> it keeps its GPU; the LLM uses the other GPU(s) + RAM ("auto" = AUTO_MODEL_SPLIT)
#   ComfyUI stopped -> the LLM uses every GPU ("auto" = AUTO_MODEL_FULL)
#   One GPU only    -> ComfyUI and the LLM take turns: starting one stops the other
set -uo pipefail
ROOT="@ROOT@"
ENV_FILE="$ROOT/.env"
SERVICES="panel comfyui llm hermes webui"
die() { echo "aistack: $*" >&2; exit 1; }
[ -f "$ENV_FILE" ] || die "missing $ENV_FILE"
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
if [ -z "${AISTACK_DOCKER:-}" ]; then
  if docker info >/dev/null 2>&1; then AISTACK_DOCKER="docker"; else AISTACK_DOCKER="sudo docker"; fi
fi
read -r -a DOCKER <<< "$AISTACK_DOCKER"
dk() { "${DOCKER[@]}" "$@"; }
dc() { "${DOCKER[@]}" compose --project-directory "$ROOT" "$@"; }
ctr() { echo "aistack-$1"; }
exists()  { dk inspect "$(ctr "$1")" >/dev/null 2>&1; }
running() { [ "$(dk inspect -f '{{.State.Running}}' "$(ctr "$1")" 2>/dev/null)" = "true" ]; }
multi_gpu() { [ "${GPU_COUNT:-1}" -ge 2 ]; }
is_enabled() { [[ " ${ENABLED_SERVICES} " == *" $1 "* ]]; }
valid_svc() { [[ " $SERVICES " == *" $1 "* ]] || die "unknown name '$1' (use: $SERVICES)"; }

current_layout() { if multi_gpu && running comfyui; then echo split; else echo full; fi; }

render() {
  local layout="${1:-$(current_layout)}" gpus auto vision="" tpl out tmp
  if [ "$layout" = split ]; then gpus="$LLM_GPUS_SHARED"; else gpus="$LLM_GPUS_ALL"; fi
  if [ "$layout" = full ] && multi_gpu; then auto="${AUTO_MODEL_FULL:-muse-glimmer}"; else auto="${AUTO_MODEL_SPLIT:-qwen3.6-35b-a3b}"; fi
  if [ "${MUSE_VISION:-0}" = 1 ] && [ -n "${MUSE_MMPROJ_FILE:-}" ]; then
    vision="--mmproj /models/${MUSE_MMPROJ_FILE} --no-mmproj-offload"
  fi
  tpl="$ROOT/llama-swap/config.template.yaml"; out="$ROOT/llama-swap/config.yaml"; tmp="$out.tmp"
  [ -f "$tpl" ] || die "missing $tpl"
  sed -e "s|^ *@AUTO_FOR_${auto}@ *\$|      - \"auto\"|" -e '/@AUTO_FOR_[^@]*@/d' \
      -e "s|@LLM_GPUS@|$gpus|g" -e "s|@LLM_CTX@|$LLM_CTX|g" -e "s|@LLM_PARALLEL@|$LLM_PARALLEL|g" \
      -e "s|@THREADS@|$LLM_THREADS|g" -e "s|@LLAMA_SERVER@|$LLAMA_SERVER_BIN|g" \
      -e "s|@MUSE_FILE@|${MUSE_FILE:-missing.gguf}|g" -e "s|@MUSE_QUANT@|${MUSE_QUANT:-}|g" \
      -e "s|@QWEN_FILE@|${QWEN_FILE:-missing.gguf}|g" -e "s|@QWEN_QUANT@|${QWEN_QUANT:-}|g" \
      -e "s|@MUSE_VISION_ARGS@|$vision|g" "$tpl" > "$tmp"
  # write in place: the file is bind-mounted, and replacing its inode would hide the change
  cat "$tmp" > "$out"; rm -f "$tmp"
}

_up() {   # start an existing container, or create + start it
  if exists "$1"; then dk start "$(ctr "$1")" >/dev/null; else dc up -d --no-deps "$1"; fi
}

start_svc() {
  local s="$1"
  if running "$s"; then echo "$s is already running"; return 0; fi
  case "$s" in
    comfyui)
      if ! multi_gpu; then
        if running llm; then echo "one GPU: stopping the LLM server so ComfyUI can use the GPU"; dk stop "$(ctr llm)" >/dev/null; fi
      else
        render split
        if running llm; then echo "moving the LLM server to GPU(s) $LLM_GPUS_SHARED (restarting it)"; dk restart "$(ctr llm)" >/dev/null; fi
      fi
      _up comfyui ;;
    llm)
      if ! multi_gpu && running comfyui; then echo "one GPU: stopping ComfyUI so the LLM can use the GPU"; dk stop "$(ctr comfyui)" >/dev/null; fi
      render
      _up llm ;;
    *) _up "$s" ;;
  esac
  echo "started $s"
}

stop_svc() {
  local s="$1"
  if ! exists "$s"; then echo "$s is not created"; return 0; fi
  if ! running "$s"; then echo "$s is already stopped"; return 0; fi
  dk stop "$(ctr "$s")" >/dev/null
  echo "stopped $s"
  if [ "$s" = comfyui ] && multi_gpu; then
    render full
    if running llm; then echo "giving the LLM server every GPU ($LLM_GPUS_ALL; restarting it)"; dk restart "$(ctr llm)" >/dev/null; fi
  fi
}

restart_svc() {
  local s="$1"
  if ! running "$s"; then start_svc "$s"; return; fi
  [ "$s" = llm ] && render
  dk restart "$(ctr "$s")" >/dev/null
  echo "restarted $s"
}

server_ip() {
  local ip
  ip="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") print $(i + 1)}' | sed -n 1p)"
  echo "${ip:-<server-ip>}"
}
urls() {
  local ip; ip="$(server_ip)"
  echo "  Control panel    http://$ip:$PANEL_PORT   (user $PANEL_USER, password PANEL_PASS in $ENV_FILE)"
  echo "  ComfyUI          http://$ip:$COMFY_PORT"
  echo "  LLM API          http://$ip:$LLM_PORT/v1   (key LLM_API_KEY; models: auto, qwen3.6-35b-a3b, muse-glimmer)"
  echo "  Chat UI          http://$ip:$WEBUI_PORT   (first account you create becomes admin)"
  echo "  Hermes dashboard http://$ip:$HERMES_DASH_PORT   (user $HERMES_DASH_USER, password HERMES_DASH_PASS in $ENV_FILE)"
}

each() {   # each <start|stop|restart> - in a GPU-safe order
  local action="$1" s order
  case "$action" in
    start)   order="panel comfyui llm hermes webui" ;;
    stop)    order="webui hermes llm comfyui panel" ;;
    restart) order="comfyui llm hermes webui panel" ;;
  esac
  for s in $order; do
    case "$action" in
      start)
        is_enabled "$s" || continue
        if [ "$s" = comfyui ] && ! multi_gpu && is_enabled llm; then echo "one GPU: not auto-starting ComfyUI (the LLM gets the GPU)"; continue; fi
        start_svc "$s" ;;
      stop) stop_svc "$s" ;;
      restart) running "$s" && restart_svc "$s" ;;
    esac
  done
  return 0
}

cmd="${1:-status}"; shift || true
case "$cmd" in
  status)
    echo "GPU layout: $(current_layout)   ($(if [ "$(current_layout)" = split ]; then echo "ComfyUI on GPU $COMFY_GPU, LLM on GPU(s) $LLM_GPUS_SHARED + RAM"; else echo "LLM may use GPU(s) $LLM_GPUS_ALL"; fi))"
    for s in $SERVICES; do
      st="$(dk inspect -f '{{.State.Status}}' "$(ctr "$s")" 2>/dev/null || echo "not created")"
      printf '  %-8s %-12s %s\n' "$s" "$st" "$(is_enabled "$s" && echo "" || echo "(not in auto-start)")"
    done
    echo
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    echo; urls ;;
  start|stop|restart)
    name="${1:-}"; [ -n "$name" ] || die "usage: aistack $cmd <name|all>"
    if [ "$name" = all ]; then each "$cmd"; else valid_svc "$name"; "${cmd}_svc" "$name"; fi ;;
  mode)
    die "GPU modes were replaced: stop ComfyUI and the LLM gets every GPU; start it and the LLM moves to one GPU." ;;
  up)   each start ;;
  down) each stop ;;
  logs)
    [ -n "${1:-}" ] || die "usage: aistack logs <name> [-f]"
    valid_svc "$1"; s="$1"; shift; dk logs --tail 150 "$@" "$(ctr "$s")" ;;
  enable|disable)
    svc="${1:-}"; valid_svc "$svc"
    new=""
    for s in $SERVICES; do
      if [ "$s" = "$svc" ]; then [ "$cmd" = enable ] && new="$new $s"
      elif is_enabled "$s"; then new="$new $s"; fi
    done
    sed -i "s|^ENABLED_SERVICES=.*|ENABLED_SERVICES=\"${new# }\"|" "$ENV_FILE"
    ENABLED_SERVICES="${new# }"
    if [ "$cmd" = disable ]; then stop_svc "$svc"; else start_svc "$svc"; fi ;;
  render)
    render; echo "wrote $ROOT/llama-swap/config.yaml (layout $(current_layout)); restart the LLM to apply" ;;
  models)
    curl -fsS -H "Authorization: Bearer $LLM_API_KEY" "http://127.0.0.1:$LLM_PORT/v1/models" \
      | python3 -c 'import json, sys
for m in json.load(sys.stdin).get("data", []):
    print(" ", m.get("id"), "-", m.get("name", ""))' || die "the LLM server is not answering (aistack logs llm)"
    echo "loaded now:"; curl -fsS -H "Authorization: Bearer $LLM_API_KEY" "http://127.0.0.1:$LLM_PORT/running" || true; echo ;;
  hermes)
    dk exec -it aistack-hermes hermes "$@" ;;
  update)
    was=""; for s in $SERVICES; do running "$s" && was="$was $s"; done
    tag="$(git ls-remote --tags --refs https://github.com/comfyanonymous/ComfyUI.git 'v*' 2>/dev/null \
           | awk -F/ '{print $3}' | sort -V | tail -n 1 || true)"
    if [ -n "$tag" ]; then sed -i "s|^COMFY_REF=.*|COMFY_REF=\"$tag\"|" "$ENV_FILE"; COMFY_REF="$tag"; fi
    render
    dc pull --ignore-buildable
    build=""; for s in comfyui panel; do is_enabled "$s" && build="$build $s"; done
    # shellcheck disable=SC2086
    [ -n "$build" ] && dc build --pull $build
    # recreate containers with the new images, leaving them stopped, then restore what was running
    # shellcheck disable=SC2086
    dc up --no-start $ENABLED_SERVICES
    for s in panel comfyui llm hermes webui; do [[ " $was " == *" $s "* ]] && start_svc "$s"; done
    echo "update done" ;;
  urls)
    urls ;;
  -h|--help|help)
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//' ;;
  *)
    die "unknown command '$cmd' (try: aistack help)" ;;
esac
CLIEOF
sudo ln -sf "$STACK_DIR/aistack" /usr/local/bin/aistack

# ------------------------------- 5. models -----------------------------------
log "Qwen3.6-35B-A3B ($QWEN_REPO, $QWEN_QUANT)"
QWEN_FILE="$(fetch_gguf "$QWEN_REPO" "$QWEN_QUANT" "Qwen3.6-35B-A3B-${QWEN_QUANT}.gguf")"
set_env_key QWEN_FILE "$QWEN_FILE"

log "Muse Glimmer ($MUSE_REPO, $MUSE_QUANT)"
MUSE_FILE="$(fetch_gguf "$MUSE_REPO" "$MUSE_QUANT" "Muse-Glimmer-30B-${MUSE_QUANT}.gguf")"
set_env_key MUSE_FILE "$MUSE_FILE"
mmproj_path="$(hf_list "$MUSE_REPO" | grep -i 'mmproj' | grep -i 'bf16' | sed -n 1p || true)"
[ -n "$mmproj_path" ] || mmproj_path="mmproj-BF16.gguf"
set_env_key MUSE_MMPROJ_FILE "$(basename "$mmproj_path")"
if [ "${MUSE_VISION:-0}" = 1 ]; then
  dl "$STACK_DIR/models/$(basename "$mmproj_path")" "https://huggingface.co/$MUSE_REPO/resolve/main/$mmproj_path"
fi

if [[ " $ENABLED_SERVICES " == *" comfyui "* ]]; then
  log "Z-Image Turbo (into $COMFY_MODELS_DIR)"
  M="$COMFY_MODELS_DIR"
  ZI_REPO="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files"
  want_bf16=0; want_bf16_te=0
  case "$ZIMAGE_VARIANT" in
    fp8|both)
      dl_try "$M/diffusion_models/z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors" \
        "https://huggingface.co/Kijai/Z-Image_comfy_fp8_scaled/resolve/main/z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors" \
        || { warn "fp8 model download failed; falling back to the official bf16 build"; want_bf16=1; }
      dl_try "$M/text_encoders/qwen_3_4b_fp8_mixed.safetensors" "$ZI_REPO/text_encoders/qwen_3_4b_fp8_mixed.safetensors" \
        || { warn "fp8 text encoder download failed; using the bf16 text encoder"; want_bf16_te=1; } ;;
  esac
  case "$ZIMAGE_VARIANT" in bf16|both) want_bf16=1; want_bf16_te=1 ;; esac
  [ "$want_bf16" = 1 ] && dl "$M/diffusion_models/z_image_turbo_bf16.safetensors" "$ZI_REPO/diffusion_models/z_image_turbo_bf16.safetensors"
  if [ "$want_bf16_te" = 1 ] || [ "$want_bf16" = 1 ]; then
    dl "$M/text_encoders/qwen_3_4b.safetensors" "$ZI_REPO/text_encoders/qwen_3_4b.safetensors"
  fi
  dl "$M/vae/ae.safetensors" "$ZI_REPO/vae/ae.safetensors"
  if [ "$INSTALL_LORAS" = 1 ]; then
    dl "$M/loras/zimage/ColoringBook_Redmond_ZImage.safetensors" \
       "https://huggingface.co/artificialguybr/COLORINGBOOK-REDMOND-ZIMAGE/resolve/main/%5BZImage.Turbo%5DColoringBook_Redmond.safetensors"
  fi
  if [ "$INSTALL_UPSCALER" = 1 ]; then
    dl "$M/upscale_models/RealESRGAN_x4plus_anime_6B.pth" \
       "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
  fi
fi

# --------------------------- 6. images + start -------------------------------
DC=(sudo docker compose --project-directory "$STACK_DIR")
log "Pulling container images"
"${DC[@]}" pull --ignore-buildable $ENABLED_SERVICES

log "Locating llama-server inside the llama-swap image"
LLAMA_SERVER_BIN="$(sudo docker run --rm --entrypoint sh "$IMG_LLAMA_SWAP" -c \
  'command -v llama-server || { [ -x /app/llama-server ] && echo /app/llama-server; } || find / -name llama-server -type f 2>/dev/null | head -n 1' \
  2>/dev/null | tail -n 1 || true)"
if [ -z "$LLAMA_SERVER_BIN" ]; then
  LLAMA_SERVER_BIN="/app/llama-server"
  warn "Could not find llama-server in $IMG_LLAMA_SWAP; assuming $LLAMA_SERVER_BIN"
fi
log "llama-server: $LLAMA_SERVER_BIN"
set_env_key LLAMA_SERVER_BIN "$LLAMA_SERVER_BIN"

export AISTACK_DOCKER="sudo docker"
"$STACK_DIR/aistack" render >/dev/null      # config.yaml must exist before the llm container is created

build=""
for s in panel comfyui; do [[ " $ENABLED_SERVICES " == *" $s "* ]] && build="$build $s"; done
if [ -n "$build" ]; then
  log "Building local images:$build (the ComfyUI build downloads PyTorch, ~10-20 minutes the first time)"
  # shellcheck disable=SC2086
  "${DC[@]}" build $build
fi

log "Creating the containers"
# shellcheck disable=SC2086
"${DC[@]}" up --no-start $ENABLED_SERVICES
log "Starting the stack"
"$STACK_DIR/aistack" start all

# --------------------------- 7. configure Hermes -----------------------------
HERMES_DIR="$STACK_DIR/data/hermes"
if [[ " $ENABLED_SERVICES " == *" hermes "* ]]; then
  log "Waiting for Hermes to initialise its data folder"
  for _ in $(seq 1 60); do [ -f "$HERMES_DIR/config.yaml" ] && break; sleep 3; done
  if [ ! -f "$HERMES_DIR/config.yaml" ]; then
    warn "Hermes did not create config.yaml yet - check: aistack logs hermes"
  elif [ ! -f "$STACK_DIR/.hermes_configured_v2" ]; then
    log "Pointing Hermes at the local model \"auto\" (follows the GPU layout)"
    slot_ctx=$(( LLM_CTX / LLM_PARALLEL ))
    hermes_ok=1
    hx() { sudo docker exec aistack-hermes hermes "$@" || hermes_ok=0; }
    hx config set model.provider custom
    hx config set model.base_url "http://llm:8080/v1"
    hx config set model.default "auto"
    hx config set model.api_key "$LLM_API_KEY"
    hx config set model.context_length "$slot_ctx"
    hx config set terminal.backend local
    if [ "$hermes_ok" = 1 ]; then
      touch "$STACK_DIR/.hermes_configured_v2"
    else
      warn "Some Hermes settings did not apply. Set them by hand with: aistack hermes model  (Custom endpoint, base URL http://llm:8080/v1, model auto, key LLM_API_KEY from .env)"
    fi
    [ "$slot_ctx" -ge 65536 ] || warn "Hermes gets ${slot_ctx} tokens of context per request; it wants ~64K. Raise LLM_CTX or lower LLM_PARALLEL in .env."
  else
    log "Hermes was configured on an earlier run (delete $STACK_DIR/.hermes_configured_v2 to redo it)"
  fi

  if [ -n "$HASS_URL" ] && [ -n "$HASS_TOKEN" ] && [ -f "$HERMES_DIR/.env" ]; then
    if grep -s '^HASS_TOKEN=' "$HERMES_DIR/.env" >/dev/null; then
      sed -i -e "s|^HASS_TOKEN=.*|HASS_TOKEN=$HASS_TOKEN|" -e "s|^HASS_URL=.*|HASS_URL=$HASS_URL|" "$HERMES_DIR/.env"
    else
      printf '\n# Home Assistant (added by setup_aistack.sh)\nHASS_URL=%s\nHASS_TOKEN=%s\n' "$HASS_URL" "$HASS_TOKEN" >> "$HERMES_DIR/.env"
    fi
    log "Home Assistant connection saved for Hermes ($HASS_URL)"
  fi
  "$STACK_DIR/aistack" restart hermes >/dev/null
fi

if [ -n "$LAN_CIDR" ] && command -v ufw >/dev/null 2>&1 && [[ "$(sudo ufw status 2>/dev/null)" == *"Status: active"* ]]; then
  for p in "$PANEL_PORT" "$COMFY_PORT" "$LLM_PORT" "$WEBUI_PORT" "$HERMES_DASH_PORT"; do
    sudo ufw allow from "$LAN_CIDR" to any port "$p" proto tcp >/dev/null
  done
  log "ufw: opened the stack's ports to $LAN_CIDR"
fi

# ------------------------------ 8. smoke tests -------------------------------
wait_http() {   # wait_http <name> <url> <tries> [curl args...]
  local name="$1" url="$2" tries="$3"; shift 3
  for _ in $(seq 1 "$tries"); do
    if curl -fs -o /dev/null "$@" "$url"; then echo "  $name: up"; return 0; fi
    sleep 5
  done
  warn "$name is not answering yet at $url"; return 1
}
log "Checking the instances"
[[ " $ENABLED_SERVICES " == *" panel "* ]] && { wait_http "Control panel" "http://127.0.0.1:$PANEL_PORT/api/state" 24 -u "$PANEL_USER:$PANEL_PASS" || true; }
if [[ " $ENABLED_SERVICES " == *" llm "* ]]; then
  wait_http "LLM server" "http://127.0.0.1:$LLM_PORT/v1/models" 24 -H "Authorization: Bearer $LLM_API_KEY" || true
  log "First request to the \"auto\" model (loads it from disk; can take a few minutes)"
  reply="$(curl -fsS --max-time 1500 "http://127.0.0.1:$LLM_PORT/v1/chat/completions" \
    -H "Authorization: Bearer $LLM_API_KEY" -H "Content-Type: application/json" \
    -d '{"model":"auto","max_tokens":60,"messages":[{"role":"user","content":"Reply with one short sentence confirming you are ready."}]}' 2>&1 || true)"
  answer="$(printf '%s' "$reply" | python3 -c 'import json, sys
try:
    print(json.load(sys.stdin)["choices"][0]["message"].get("content", "").strip())
except Exception:
    pass' 2>/dev/null || true)"
  if [ -n "$answer" ]; then echo "  auto model: $answer"
  elif [[ "$reply" == *choices* ]]; then echo "  auto model answered (the reply was all reasoning; that is fine)."
  else warn "No answer from the LLM yet. Check the LLM server logs in the panel (or: aistack logs llm)"; fi
fi
[[ " $ENABLED_SERVICES " == *" comfyui "* ]] && { wait_http "ComfyUI" "http://127.0.0.1:$COMFY_PORT/system_stats" 60 || true; }
[[ " $ENABLED_SERVICES " == *" webui "* ]] && { wait_http "Chat UI" "http://127.0.0.1:$WEBUI_PORT/health" 60 || true; }
[[ " $ENABLED_SERVICES " == *" hermes "* ]] && { wait_http "Hermes API" "http://127.0.0.1:$HERMES_API_PORT/health" 24 || true; }

# ------------------------------- 9. README -----------------------------------
SERVER_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") print $(i + 1)}' | sed -n 1p)"
SERVER_IP="${SERVER_IP:-<server-ip>}"
cat > "$STACK_DIR/README.md" <<MDEOF
# aistack

Run everything from the **control panel: http://$SERVER_IP:$PANEL_PORT** (user $PANEL_USER, password
PANEL_PASS in \`$STACK_DIR/.env\`). Each instance has Start / Stop / Restart / Open / Logs, the GPU cards
show who is using which GPU, and the LLM card loads/unloads models.

| Instance | URL | Notes |
|---|---|---|
| ComfyUI (Z-Image) | http://$SERVER_IP:$COMFY_PORT | GPU $COMFY_GPU; models in \`$COMFY_MODELS_DIR\`; images in \`data/comfyui/output\` |
| LLM server (llama-swap) | http://$SERVER_IP:$LLM_PORT/v1 | OpenAI-compatible; key = LLM_API_KEY; its own swap/log page at /ui |
| Chat UI (Open WebUI) | http://$SERVER_IP:$WEBUI_PORT | first account created becomes admin; lists the models and \`hermes-agent\` |
| Hermes dashboard | http://$SERVER_IP:$HERMES_DASH_PORT | user $HERMES_DASH_USER, password HERMES_DASH_PASS in .env |

## How the GPUs are shared (automatic)
- ComfyUI running: it keeps GPU $COMFY_GPU; the LLM uses GPU(s) $LLM_GPUS_SHARED + system RAM.
- ComfyUI stopped: the LLM uses GPUs $LLM_GPUS_ALL.
- Starting or stopping ComfyUI restarts the LLM server to move it (the panel asks first).
- The model name **auto** (Hermes uses it) follows along: \`$AUTO_MODEL_SPLIT\` while ComfyUI runs,
  \`$AUTO_MODEL_FULL\` while it is stopped. Change the picks in .env (AUTO_MODEL_SPLIT / AUTO_MODEL_FULL).

## Models
Clients choose a model by name: \`auto\`, \`qwen3.6-35b-a3b\` (alias \`qwen\`), \`muse-glimmer\` (alias \`muse\`).
llama-swap loads the requested one and unloads the previous one. To add a model, edit
\`llama-swap/config.template.yaml\`, put the .gguf in \`models/\`, then restart the LLM server.

## Coding from your workstation
Any OpenAI-compatible tool (VS Code: Cline, Roo Code, Continue; terminal: aider, OpenCode):
base URL \`http://$SERVER_IP:$LLM_PORT/v1\`, API key LLM_API_KEY, model \`auto\` (or a fixed one).

## Hermes (homelab + Home Assistant)
- Home Assistant: in HA create a long-lived access token (preferably on a dedicated non-admin user), add
  \`HASS_URL=http://<ha-ip>:8123\` and \`HASS_TOKEN=...\` to \`data/hermes/.env\`, then restart Hermes.
- TrueNAS: create a Read-Only Administrator service account + user-linked API key on the NAS, review a
  TrueNAS MCP server, then add it under \`mcp_servers:\` in \`data/hermes/config.yaml\`.
- Commands Hermes runs execute inside its own container (no Docker socket, only \`data/hermes\` mounted),
  with dangerous-command approval on. It can still reach your LAN, so give it read-only credentials.

## Customizing safely
You can edit \`aistack\`, \`compose.yaml\`, \`build/panel/panel.py\` and the files in \`build/comfyui/\`.
Re-running setup keeps any file you changed and writes its new version next to it as \`<file>.new\`.
Files you never touched are updated normally. To go back to the stock version of a file, delete it
and re-run setup. After editing panel.py or a Dockerfile, rebuild with: \`aistack update\`

## Command line (optional - the panel does the same)
    aistack status | start|stop|restart <name|all> | logs <name> -f | models | update

## Security
The control panel can start and stop containers through the Docker socket, which is equivalent to root
on this server. It has a password, but keep it (and ComfyUI, which has none) on your LAN only.

## Removing
    aistack stop all
    sudo docker rm -f aistack-panel aistack-comfyui aistack-llm aistack-hermes aistack-webui
    sudo docker image rm aistack/comfyui:local aistack/panel:local $IMG_LLAMA_SWAP $IMG_HERMES $IMG_WEBUI
    sudo rm -f /usr/local/bin/aistack
    rm -rf $STACK_DIR          # includes both LLMs, Hermes memory, chat history
(ComfyUI models in $COMFY_MODELS_DIR are not removed by this.)

## Earlier installs
The old systemd services (comfyui@N / comfyui / llama-server) were disabled, not deleted, and the old
helpers ai-mode, llm-ask and llm-code no longer apply. To remove them:
    sudo rm -f /etc/systemd/system/comfyui@.service /etc/systemd/system/comfyui.service \\
               /etc/systemd/system/llama-server.service /usr/local/bin/{ai-mode,llm-ask,llm-code}
    sudo systemctl daemon-reload
MDEOF

# ------------------------------- 10. summary ---------------------------------
log "Done"
"$STACK_DIR/aistack" urls
cat <<EOF

  Control panel login:  $PANEL_USER / $PANEL_PASS
  LLM API key:          $LLM_API_KEY
  Hermes login:         $HERMES_DASH_USER / $HERMES_DASH_PASS
  Docs:                 $STACK_DIR/README.md      Setup log: $LOG

  GPUs: stop ComfyUI in the panel and the LLM gets both GPUs; start it and the LLM moves to one.

  SECURITY: the panel controls Docker (root-equivalent) and ComfyUI has no login. Keep the
  stack on your LAN (LISTEN_ADDR=$LISTEN_ADDR); do not port-forward it.
EOF
if [ -z "$HASS_TOKEN" ] && [[ " $ENABLED_SERVICES " == *" hermes "* ]]; then
  echo "  Home Assistant:  not connected yet - see 'Hermes' in $STACK_DIR/README.md"
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  warn "These downloads failed: ${FAILED[*]}"
  warn "If the error was 401/403: accept the model license on Hugging Face and re-run with HF_TOKEN=hf_xxx"
fi
