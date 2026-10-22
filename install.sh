#!/usr/bin/env bash
# GPU Shepherd installer for Ubuntu (written for 26.04 LTS; 24.04 works too).
#
#   sudo ./install.sh                         # install what's missing, then start GPU Shepherd
#   sudo ./install.sh --models-dir /srv/models --port 8090
#
# What it does, skipping anything already in place:
#   1. checks the NVIDIA driver (installs the recommended one with --install-driver)
#   2. installs Docker Engine and the Compose plugin from Ubuntu's own packages
#   3. installs the NVIDIA Container Toolkit and connects it to Docker
#   4. checks that a container can see the GPUs
#   5. writes compose.yaml (if there isn't one yet) and starts GPU Shepherd
#
# Run it again at any time: it only does what is still missing. To stop GPU Shepherd later:
#   sudo docker compose down        (in this folder; your data folder is kept)
set -euo pipefail

DATA_DIR=/opt/gpu-shepherd
MODELS_DIR=""
PORT=8090
INSTALL_DRIVER=0
SKIP_DEPS=0
NO_START=0
GPU_TEST_IMAGE=ubuntu:24.04
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")

usage() {
    cat <<EOF
Usage: sudo ./install.sh [options]

  --data-dir DIR      where GPU Shepherd keeps its settings (default: $DATA_DIR)
  --models-dir DIR    a folder of .gguf model files to pick from in the UI (mounted read-only)
  --port N            web UI port (default: $PORT)
  --install-driver    install the recommended NVIDIA driver if none is working (needs a reboot)
  --skip-deps         don't install anything; only write compose.yaml and start GPU Shepherd
  --no-start          prepare everything but don't start GPU Shepherd
  -h, --help          show this help
EOF
}

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)       DATA_DIR="${2:?--data-dir needs a folder}"; shift 2 ;;
        --models-dir)     MODELS_DIR="${2:?--models-dir needs a folder}"; shift 2 ;;
        --port)           PORT="${2:?--port needs a number}"; shift 2 ;;
        --install-driver) INSTALL_DRIVER=1; shift ;;
        --skip-deps)      SKIP_DEPS=1; shift ;;
        --no-start)       NO_START=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                usage >&2; die "unknown option: $1" ;;
    esac
done

case "$PORT" in ''|*[!0-9]*) die "--port must be a number" ;; esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || die "--port must be between 1 and 65535"
case "$DATA_DIR" in /*) ;; *) die "--data-dir must be an absolute path" ;; esac
if [ -n "$MODELS_DIR" ]; then
    case "$MODELS_DIR" in /*) ;; *) die "--models-dir must be an absolute path" ;; esac
    [ -d "$MODELS_DIR" ] || die "--models-dir $MODELS_DIR does not exist"
fi
for d in "$DATA_DIR" "$MODELS_DIR"; do
    case "$d" in *[[:space:]:\#\|\&\\]*) die "folder paths can't contain spaces or ':#|&\\' ($d)" ;; esac
done

if [ "$(id -u)" -ne 0 ]; then
    have sudo || die "run this as root"
    say "Re-running with sudo"
    exec sudo -- "$HERE/$(basename "$0")" "${ARGS[@]}"
fi

# --------------------------------------------------------------------------- checks

if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [ "${ID:-}" != "ubuntu" ]; then
        warn "this script is written for Ubuntu; found ${PRETTY_NAME:-unknown}. Continuing anyway."
    else
        say "Ubuntu ${VERSION_ID:-?} detected"
    fi
fi
[ "$(uname -m)" = "x86_64" ] || [ "$(uname -m)" = "aarch64" ] || warn "untested CPU architecture $(uname -m)"

ram_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1048576 + 0.5}' /proc/meminfo)
say "System RAM: ${ram_gb} GB"
if [ "$ram_gb" -lt 32 ]; then
    warn "under 32 GB of RAM: keep models small enough to fit on the GPUs (no RAM offload)."
elif [ "$ram_gb" -lt 60 ]; then
    warn "64 GB of RAM is recommended to run large models with part of them in system RAM."
fi

apt_updated=0
apt_install() {
    if [ "$apt_updated" -eq 0 ]; then
        DEBIAN_FRONTEND=noninteractive apt-get update -q
        apt_updated=1
    fi
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$@"
}

if [ "$SKIP_DEPS" -eq 0 ]; then
    # ----------------------------------------------------------------------- NVIDIA driver
    if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
        say "NVIDIA driver OK:"
        nvidia-smi -L | sed 's/^/      /'
    elif [ "$INSTALL_DRIVER" -eq 1 ]; then
        say "Installing the recommended NVIDIA driver"
        apt_install ubuntu-drivers-common
        ubuntu-drivers install
        say "Driver installed. Reboot, then run this script again:  sudo reboot"
        exit 0
    else
        die "no working NVIDIA driver (nvidia-smi failed). Install one with
       sudo ./install.sh --install-driver
   or  sudo ubuntu-drivers install
then reboot and run this script again."
    fi

    # ----------------------------------------------------------------------- Docker
    if ! have docker; then
        say "Installing Docker Engine and Compose (Ubuntu packages)"
        apt_install docker.io docker-compose-v2 \
            || die "could not install Docker from Ubuntu's packages (is the 'universe' repository enabled?
       sudo add-apt-repository universe). Or install Docker Engine yourself and run this again."
    fi
    if ! docker compose version >/dev/null 2>&1; then
        say "Installing the Docker Compose plugin"
        if dpkg -s docker-ce >/dev/null 2>&1; then
            apt_install docker-compose-plugin       # Docker's own repository
        else
            apt_install docker-compose-v2
        fi
    fi
    systemctl enable --now docker >/dev/null 2>&1 || true
    docker info >/dev/null 2>&1 || die "Docker is installed but not running (try: sudo systemctl status docker)"
    say "Docker OK: $(docker --version)"

    # ----------------------------------------------------------------------- NVIDIA Container Toolkit
    if ! have nvidia-ctk; then
        say "Installing the NVIDIA Container Toolkit"
        apt_install ca-certificates curl gpg
        keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o "$keyring"
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
            > /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt_updated=0
        apt_install nvidia-container-toolkit
    fi
    if ! docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
        say "Connecting the NVIDIA runtime to Docker"
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
    fi

    say "Checking that containers can see the GPUs"
    if ! docker run --rm --gpus all "$GPU_TEST_IMAGE" nvidia-smi -L; then
        die "a test container could not see the GPUs. Check the driver (nvidia-smi) and
       'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'."
    fi
else
    have docker || die "--skip-deps given, but Docker is not installed"
fi

# --------------------------------------------------------------------------- GPU Shepherd

mkdir -p "$DATA_DIR"
cd "$HERE"
[ -f compose.example.yaml ] || die "compose.example.yaml not found next to install.sh - run this from the repository"

if [ -f compose.yaml ]; then
    say "Keeping your existing compose.yaml (delete it to have it written again)"
else
    say "Writing compose.yaml (data: $DATA_DIR, port: $PORT${MODELS_DIR:+, models: $MODELS_DIR})"
    sed -e "s#/opt/gpu-shepherd#$DATA_DIR#g" \
        -e "s#\"8090:8090\"#\"$PORT:8090\"#" \
        compose.example.yaml > compose.yaml
    if [ -n "$MODELS_DIR" ]; then
        sed -i "s|# - /srv/models:/srv/models:ro|- $MODELS_DIR:$MODELS_DIR:ro|" compose.yaml
        grep -q -- "- $MODELS_DIR:$MODELS_DIR:ro" compose.yaml \
            || warn "could not add the models folder to compose.yaml; add it under volumes by hand"
    fi
fi

if [ "$NO_START" -eq 1 ]; then
    say "Ready. Start GPU Shepherd with:  sudo docker compose up -d --build"
    exit 0
fi

say "Building and starting GPU Shepherd"
docker compose up -d --build

url="http://127.0.0.1:$PORT"
say "Waiting for the web UI"
for _ in $(seq 1 60); do
    if curl -fsS "$url/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
curl -fsS "$url/healthz" >/dev/null 2>&1 || die "GPU Shepherd did not answer on $url. See: sudo docker compose logs gpu-shepherd"

host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
say "GPU Shepherd is running: http://${host_ip:-<this-machine>}:$PORT"
if [ -f "$DATA_DIR/INITIAL_PASSWORD.txt" ]; then
    echo "    First login (also saved in $DATA_DIR/INITIAL_PASSWORD.txt):"
    sed -n 's/^\(user\|password\): /    \1: /p' "$DATA_DIR/INITIAL_PASSWORD.txt"
    echo "    Change the password under Settings; that file is deleted when you do."
fi
echo "    Keep it on a trusted network: it controls Docker, which is root-equivalent."
