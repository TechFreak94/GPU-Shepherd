#!/usr/bin/env bash
# GPU Shepherd installer for Ubuntu (written for 26.04 LTS; 24.04 works too).
#
#   sudo ./install.sh              # guided: looks at the machine, asks a few questions, then installs
#   sudo ./install.sh --yes        # no questions: defaults plus whatever flags you give
#   sudo ./install.sh --help       # every flag (for scripted or advanced installs)
#
# What it does, skipping anything already in place:
#   1. checks the NVIDIA driver (offers to install the recommended one)
#   2. installs Docker Engine and the Compose plugin from Ubuntu's own packages
#   3. installs the NVIDIA Container Toolkit and connects it to Docker
#   4. checks that a container can see the GPUs
#   5. adds you to the docker group, so docker works without sudo
#   6. on the first run, offers a preset: a ready-made set of instances to start from
#   7. writes compose.yaml (if there isn't one yet) and starts GPU Shepherd
#   8. with a preset, offers to download its models
#
# Nothing is installed or changed until you confirm the plan. A flag answers its question in
# advance, so the guided mode only asks what the flags left open. Run it again at any time: it
# only does what is still missing. Exit status: 0 done, 2 you declined the plan (nothing changed),
# 3 the NVIDIA driver was installed and the machine must reboot first, 1 an error. To stop it later:
#   sudo docker compose down        (in this folder; your data folder is kept)
set -euo pipefail

DEFAULT_DATA_DIR=/opt/gpu-shepherd
DEFAULT_PORT=8090
DATA_DIR=""
MODELS_DIR=""
MODELS_SET=0            # 1 once --models-dir (or the user's answer) decided it, even if empty
PORT=""
INSTALL_DRIVER=0
SKIP_DEPS=0
NO_START=0
ASSUME_YES=0
LIST_PRESETS=0
PRESET=""               # preset name or folder, or "none", once decided by a flag or an answer
DOWNLOAD=""             # yes / no: download the preset's models at the end
DOCKER_GROUP=1
GPU_TEST_IMAGE=ubuntu:24.04
# Below this much installed RAM, models must fit entirely in GPU memory. Keeping part of a model in
# system RAM (MoE expert offload) needs room for those layers plus the OS, Docker and the other
# containers: a 35B-A3B model at Q4 is ~22 GB on its own, so 32 GB is the smallest size where offload
# is practical. 64 GB fits a large MoE coding model such as Qwen3 Coder Next (~49 GB at Q4).
MIN_RAM_GB=32
RECOMMENDED_RAM_GB=64
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")

usage() {
    cat <<EOF
Usage: sudo ./install.sh [options]

Run it without options for a guided install. Every option below answers one of its questions
in advance, for scripted or advanced installs.

  --models-dir DIR    a folder of .gguf model files to pick from in the UI (mounted read-only);
                      pass "" for none
  --port N            web UI port (default: $DEFAULT_PORT)
  --data-dir DIR      where GPU Shepherd keeps its settings (default: $DEFAULT_DATA_DIR)
  --install-driver    install the recommended NVIDIA driver if none is working (needs a reboot)
  --skip-deps         don't install anything; only write compose.yaml and start GPU Shepherd
  --preset NAME|DIR   on the first run, create the instances of a preset: one of the folders in
                      gpushepherd/presets (see --list-presets) or a preset folder of your own;
                      with --yes nothing is created unless this is given
  --no-preset         don't offer a preset
  --list-presets      list the presets and exit
  --download-models   with a preset, download its models at the end (resumable);
                      with --yes they are only downloaded when this is given
  --no-download       don't offer to download the models
  --no-docker-group   don't add you to the docker group (docker then needs sudo)
  --no-start          prepare everything but don't start GPU Shepherd
  -y, --yes           don't ask anything: use the defaults for whatever no option sets
  -h, --help          show this help
EOF
}

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
item() { printf '    %b %s\n' "$1" "$2"; }
OK='\033[1;32m✓\033[0m'
NO='\033[1;33m•\033[0m'

while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)       DATA_DIR="${2:?--data-dir needs a folder}"; shift 2 ;;
        --models-dir)     [ $# -ge 2 ] || die "--models-dir needs a folder (or \"\" for none)"
                          MODELS_DIR="$2"; MODELS_SET=1; shift 2 ;;
        --port)           PORT="${2:?--port needs a number}"; shift 2 ;;
        --install-driver) INSTALL_DRIVER=1; shift ;;
        --skip-deps)      SKIP_DEPS=1; shift ;;
        --no-start)       NO_START=1; shift ;;
        --preset)         PRESET="${2:?--preset needs a preset name or folder}"; shift 2 ;;
        --no-preset)      PRESET=none; shift ;;
        --list-presets)   LIST_PRESETS=1; shift ;;
        --download-models) DOWNLOAD=yes; shift ;;
        --no-download)    DOWNLOAD=no; shift ;;
        --no-docker-group) DOCKER_GROUP=0; shift ;;
        -y|--yes)         ASSUME_YES=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                usage >&2; die "unknown option: $1" ;;
    esac
done

# Answers are checked the same way whether they come from a flag or a question.
port_problem() {
    case "$1" in ''|*[!0-9]*) echo "the port must be a number"; return ;; esac
    if [ "$1" -lt 1 ] || [ "$1" -gt 65535 ]; then echo "the port must be between 1 and 65535"; fi
}
dir_problem() {   # $1 = folder, $2 = what it is for
    case "$1" in /*) ;; *) echo "the $2 folder must be an absolute path (starting with /)"; return ;; esac
    case "$1" in *[[:space:]:\#\|\&\\]*) printf '%s\n' "the $2 folder can't contain spaces or ':#|&\\'" ;; esac
}
port_in_use() {
    have ss && [ -n "$(ss -ltnH "sport = :$1" 2>/dev/null)" ]
}
free_port_from() {
    local p=$1
    while port_in_use "$p" && [ "$p" -lt 65535 ]; do p=$((p + 1)); done
    echo "$p"
}

if [ -n "$PORT" ]; then p=$(port_problem "$PORT"); [ -z "$p" ] || die "--port: $p"; fi
if [ -n "$DATA_DIR" ]; then p=$(dir_problem "$DATA_DIR" settings); [ -z "$p" ] || die "--data-dir: $p"; fi
if [ -n "$MODELS_DIR" ]; then
    p=$(dir_problem "$MODELS_DIR" models); [ -z "$p" ] || die "--models-dir: $p"
    [ -d "$MODELS_DIR" ] || die "--models-dir: $MODELS_DIR does not exist"
fi

# Presets: folders with preset.yaml (name, description, needs, download_gb), instances/, and
# optionally catalogs/, appdata/, downloads.txt and NOTES.txt. See gpushepherd/presets/README.md.
PRESETS_DIR="$HERE/gpushepherd/presets"
preset_field() { sed -n "s/^$2: *//p" "$1/preset.yaml" 2>/dev/null | head -n 1; }
preset_list() {
    local d
    for d in "$PRESETS_DIR"/*/; do [ -f "$d/preset.yaml" ] && basename "$d"; done
}
preset_dir() {   # name or folder -> folder (empty if it isn't a preset)
    local d=$1
    case "$d" in */*) ;; *) d="$PRESETS_DIR/$d" ;; esac
    if [ -d "$d/instances" ]; then (cd "$d" && pwd); fi
}
if [ "$LIST_PRESETS" -eq 1 ]; then
    for p in $(preset_list); do
        printf '%s\n    %s: %s\n    needs %s\n' "$p" "$(preset_field "$PRESETS_DIR/$p" name)" \
            "$(preset_field "$PRESETS_DIR/$p" description)" "$(preset_field "$PRESETS_DIR/$p" needs)"
    done
    exit 0
fi
if [ -n "$PRESET" ] && [ "$PRESET" != none ]; then
    [ -n "$(preset_dir "$PRESET")" ] || die "--preset: '$PRESET' is not a preset (see --list-presets, or give a folder with instances/ in it)"
    PRESET=$(preset_dir "$PRESET")
fi

if [ "$(id -u)" -ne 0 ]; then
    have sudo || die "run this as root"
    say "Re-running with sudo"
    exec sudo -- "$HERE/$(basename "$0")" "${ARGS[@]}"
fi

# Questions are read from the terminal, so they also work when the script's input is a pipe.
INTERACTIVE=0
if [ "$ASSUME_YES" -eq 0 ] && { : </dev/tty; } 2>/dev/null; then
    INTERACTIVE=1
fi

ask() {   # ask VAR "question" "default"  - Enter keeps the default
    local answer
    if [ -n "$3" ]; then
        read -r -p "$(printf '\033[1m?\033[0m %s [%s]: ' "$2" "$3")" answer </dev/tty || answer=""
    else
        read -r -p "$(printf '\033[1m?\033[0m %s: ' "$2")" answer </dev/tty || answer=""
    fi
    printf -v "$1" '%s' "${answer:-$3}"
}
confirm() {   # confirm "question" Y|N  - returns 0 for yes
    local answer hint="[Y/n]"
    [ "$2" = N ] && hint="[y/N]"
    while :; do
        read -r -p "$(printf '\033[1m?\033[0m %s %s ' "$1" "$hint")" answer </dev/tty || answer=""
        case "${answer:-$2}" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo])     return 1 ;;
        esac
        echo "    Please answer y or n."
    done
}

if [ "$INTERACTIVE" -eq 1 ]; then
    echo
    printf '\033[1mGPU Shepherd installer\033[0m\n'
    echo "Press Enter to accept the answer in [brackets]. Nothing is installed or changed until you"
    echo "confirm the plan at the end of the questions. Ctrl+C quits at any time."
    echo
fi

# --------------------------------------------------------------------------- look at the machine

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

# The kernel reports a little less than what is installed (firmware and the GPU driver reserve
# some), e.g. 31.3 GiB on a 32 GB machine. Rounding up to the next 4 GB gives the installed size
# back, so a 32 GB machine meets the 32 GB limit.
mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
usable_gb=$(( (mem_kb + 524288) / 1048576 ))
# shellcheck disable=SC2017  # dividing first is the point: it rounds up to a multiple of 4
ram_gb=$(( (mem_kb + 4 * 1048576 - 1) / (4 * 1048576) * 4 ))
say "System RAM: ${ram_gb} GB installed (${usable_gb} GB usable)"
if [ "$ram_gb" -lt "$MIN_RAM_GB" ]; then
    warn "under ${MIN_RAM_GB} GB of RAM. Models must fit in GPU memory: keeping part of a model in system
         RAM needs room for it plus the OS and your other containers."
elif [ "$ram_gb" -lt "$RECOMMENDED_RAM_GB" ]; then
    warn "${RECOMMENDED_RAM_GB} GB of RAM is recommended for the largest models (e.g. Qwen3 Coder Next, ~49 GB),
         which keep most of their layers in system RAM. Mid-size models will work."
fi

HAS_DRIVER=0; HAS_DOCKER=0; HAS_COMPOSE=0; HAS_CTK=0; HAS_RUNTIME=0; HAS_CURL=0; HAS_COMPOSE_FILE=0
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then HAS_DRIVER=1; fi
if have docker; then HAS_DOCKER=1; fi
if [ "$HAS_DOCKER" -eq 1 ] && docker compose version >/dev/null 2>&1; then HAS_COMPOSE=1; fi
if have nvidia-ctk; then HAS_CTK=1; fi
if [ "$HAS_DOCKER" -eq 1 ] && docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then HAS_RUNTIME=1; fi
if have curl; then HAS_CURL=1; fi
if [ -f "$HERE/compose.yaml" ]; then HAS_COMPOSE_FILE=1; fi

say "This machine has:"
if [ "$HAS_DRIVER" -eq 1 ]; then
    item "$OK" "NVIDIA driver, with:"
    nvidia-smi -L | sed 's/ (UUID.*//; s/^/        /'
else
    item "$NO" "no working NVIDIA driver"
fi
status() { if [ "$1" -eq 1 ]; then item "$OK" "$2"; else item "$NO" "no $2"; fi; }
status "$HAS_DOCKER" "Docker"
status "$HAS_COMPOSE" "Docker Compose"
status "$HAS_CTK" "NVIDIA Container Toolkit"
[ "$HAS_COMPOSE_FILE" -eq 1 ] && item "$OK" "a compose.yaml from an earlier run (it will be kept)"
echo

# --------------------------------------------------------------------------- NVIDIA driver

apt_updated=0
apt_install() {
    if [ "$apt_updated" -eq 0 ]; then
        DEBIAN_FRONTEND=noninteractive apt-get update -q
        apt_updated=1
    fi
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$@"
}

no_driver_help="Install one with
       sudo ./install.sh --install-driver
   or  sudo ubuntu-drivers install
then reboot and run this script again."

if [ "$SKIP_DEPS" -eq 0 ] && [ "$HAS_DRIVER" -eq 0 ]; then
    if [ "$INSTALL_DRIVER" -eq 0 ] && [ "$INTERACTIVE" -eq 1 ]; then
        echo "GPU Shepherd needs the NVIDIA driver. Ubuntu can install the recommended one for your GPU;"
        echo "the machine then has to reboot before the rest of the install can continue."
        if confirm "Install the recommended NVIDIA driver now?" Y; then
            INSTALL_DRIVER=1
        else
            die "no working NVIDIA driver. $no_driver_help"
        fi
    fi
    [ "$INSTALL_DRIVER" -eq 1 ] || die "no working NVIDIA driver (nvidia-smi failed). $no_driver_help"
    say "Installing the recommended NVIDIA driver"
    apt_install ubuntu-drivers-common
    ubuntu-drivers install
    say "Driver installed. After the reboot, run this script again to finish: sudo ./install.sh"
    if [ "$INTERACTIVE" -eq 1 ] && confirm "Reboot now?" N; then
        reboot
    fi
    exit 3
fi

# --------------------------------------------------------------------------- settings

if [ "$HAS_COMPOSE_FILE" -eq 1 ]; then
    # compose.yaml already holds the folders and port; don't ask for values that would be ignored.
    if [ -n "$DATA_DIR$PORT$MODELS_DIR" ]; then
        warn "compose.yaml already exists, so --data-dir/--port/--models-dir are ignored.
         Delete compose.yaml to have it written again with new values."
    fi
    DATA_DIR=$(sed -n 's/^ *SHEPHERD_DATA_DIR: *//p' "$HERE/compose.yaml" | head -n 1)
    PORT=$(sed -n 's/^ *- *"\([0-9]*\):8090".*/\1/p' "$HERE/compose.yaml" | head -n 1)
    DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}
    PORT=${PORT:-$DEFAULT_PORT}
else
    if [ "$INTERACTIVE" -eq 1 ]; then
        say "A few settings"
    fi

    # models folder: suggest the first common place that already has .gguf files
    if [ "$MODELS_SET" -eq 0 ] && [ "$INTERACTIVE" -eq 1 ]; then
        suggestion=""
        user_home=$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6 || true)
        for d in /srv/models /opt/models /data/models "$user_home/models" "$user_home/Models"; do
            if [ -d "$d" ] && [ -n "$(find "$d" -maxdepth 4 -name '*.gguf' -print -quit 2>/dev/null)" ]; then
                suggestion=$d; break
            fi
        done
        echo "    GPU Shepherd can list your .gguf model files so you pick them instead of typing paths."
        echo "    It only reads this folder. Leave it empty to skip; you can add folders later."
        while :; do
            ask MODELS_DIR "Folder with your model files" "$suggestion"
            [ -n "$MODELS_DIR" ] || break
            p=$(dir_problem "$MODELS_DIR" models)
            if [ -n "$p" ]; then echo "    $p."; continue; fi
            [ -d "$MODELS_DIR" ] && break
            if confirm "$MODELS_DIR doesn't exist. Create it?" Y; then
                mkdir -p "$MODELS_DIR"
                [ -z "${SUDO_USER:-}" ] || chown "$SUDO_USER:" "$MODELS_DIR"
                break
            fi
        done
        echo
    fi

    # port: default 8090, or the next free one
    if [ -z "$PORT" ]; then
        suggestion=$(free_port_from "$DEFAULT_PORT")
        if [ "$suggestion" != "$DEFAULT_PORT" ]; then
            echo "    Port $DEFAULT_PORT is already in use on this machine."
        fi
        if [ "$INTERACTIVE" -eq 1 ]; then
            while :; do
                ask PORT "Port for the web UI" "$suggestion"
                p=$(port_problem "$PORT")
                if [ -n "$p" ]; then echo "    $p."; continue; fi
                if port_in_use "$PORT"; then echo "    Port $PORT is in use; pick another."; continue; fi
                break
            done
            echo
        else
            PORT=$suggestion
        fi
    elif port_in_use "$PORT"; then
        warn "port $PORT is already in use; GPU Shepherd may fail to start."
    fi

    # settings folder
    if [ -z "$DATA_DIR" ]; then
        if [ "$INTERACTIVE" -eq 1 ]; then
            echo "    GPU Shepherd keeps its settings, model lists and first password in one folder."
            while :; do
                ask DATA_DIR "Folder for GPU Shepherd's settings" "$DEFAULT_DATA_DIR"
                p=$(dir_problem "$DATA_DIR" settings)
                [ -n "$p" ] && { echo "    $p."; continue; }
                break
            done
            echo
        else
            DATA_DIR=$DEFAULT_DATA_DIR
        fi
    fi
fi

# --------------------------------------------------------------------------- preset

first_run=1
if compgen -G "$DATA_DIR/instances/*.yaml" >/dev/null; then first_run=0; fi
if [ "$first_run" -eq 0 ]; then
    [ -z "$PRESET" ] || [ "$PRESET" = none ] || warn "--preset ignored: $DATA_DIR already has instances."
    PRESET=none
elif [ -z "$PRESET" ]; then
    PRESET=none
    mapfile -t presets < <(preset_list)
    if [ "$INTERACTIVE" -eq 1 ] && [ ${#presets[@]} -gt 0 ]; then
        echo "    GPU Shepherd starts empty. It can start you off with a preset: a set of instances"
        echo "    (and their settings) that you can change or delete later. Nothing in it starts until"
        echo "    you press Start."
        default=1
        for i in "${!presets[@]}"; do
            d="$PRESETS_DIR/${presets[$i]}"
            [ "${presets[$i]}" != starter ] || default=$((i + 1))
            printf '      %d) %s - %s\n         needs %s\n' "$((i + 1))" "$(preset_field "$d" name)" \
                "$(preset_field "$d" description)" "$(preset_field "$d" needs)"
        done
        echo "      0) None: start empty"
        echo "    Or type the folder of a preset of your own."
        choice=""
        while :; do
            ask choice "Preset" "$default"
            case "$choice" in
                0) break ;;
                *[!0-9]*)
                    if [ -n "$(preset_dir "$choice")" ]; then PRESET=$(preset_dir "$choice"); break; fi
                    echo "    $choice is not a preset folder (it needs an instances/ folder)." ;;
                *)
                    if [ "$choice" -ge 1 ] && [ "$choice" -le ${#presets[@]} ]; then
                        PRESET="$PRESETS_DIR/${presets[$((choice - 1))]}"; break
                    fi
                    echo "    Please pick a number from the list." ;;
            esac
        done
        echo
    fi
fi
PRESET_NAME=""
if [ "$PRESET" != none ]; then
    PRESET_NAME=$(preset_field "$PRESET" name); PRESET_NAME=${PRESET_NAME:-$(basename "$PRESET")}
    [ -n "$MODELS_DIR" ] || MODELS_DIR="$DATA_DIR/models"       # the preset's models need a home
fi

DOWNLOADER="$PRESETS_DIR/download-models.sh"
DOWNLOAD_GB=$( [ "$PRESET" = none ] || preset_field "$PRESET" download_gb )
if [ "$PRESET" = none ] || [ ! -f "$PRESET/downloads.txt" ]; then
    [ "$DOWNLOAD" != yes ] || warn "--download-models ignored: no preset with downloads was chosen."
    DOWNLOAD=no
elif [ -z "$DOWNLOAD" ]; then
    if [ "$INTERACTIVE" -eq 1 ]; then
        free_dir=$MODELS_DIR
        while [ ! -d "$free_dir" ]; do free_dir=$(dirname "$free_dir"); done
        free_gb=$(df -BG --output=avail "$free_dir" 2>/dev/null | tail -n 1 | tr -dc 0-9)
        echo "    The preset's models (about ${DOWNLOAD_GB:-?} GB) can be downloaded for you:"
        bash "$DOWNLOADER" "$PRESET" | tail -n +2
        echo "    ${free_gb:-?} GB is free in $free_dir. The download runs last, resumes if it"
        echo "    stops, and GPU Shepherd is usable meanwhile."
        default=Y
        if [ -n "$free_gb" ] && [ -n "$DOWNLOAD_GB" ] && [ "$free_gb" -lt "$DOWNLOAD_GB" ]; then
            warn "that is not enough room for all of them; $DOWNLOADER can fetch only some."
            default=N
        fi
        if confirm "Download the models?" "$default"; then DOWNLOAD=yes; else DOWNLOAD=no; fi
        echo
    else
        DOWNLOAD=no
    fi
fi

# --------------------------------------------------------------------------- docker group

DOCKER_USER=""
if [ "$DOCKER_GROUP" -eq 1 ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ] \
        && ! id -nG "$SUDO_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
    DOCKER_USER=$SUDO_USER
fi

# --------------------------------------------------------------------------- the plan

plan=()
if [ "$SKIP_DEPS" -eq 0 ]; then
    [ "$HAS_DOCKER" -eq 1 ] || plan+=("Install Docker Engine and Docker Compose (Ubuntu packages)")
    [ "$HAS_DOCKER" -eq 0 ] || [ "$HAS_COMPOSE" -eq 1 ] || plan+=("Install the Docker Compose plugin")
    [ "$HAS_CTK" -eq 1 ] || plan+=("Install the NVIDIA Container Toolkit (from NVIDIA's apt repository)")
    [ "$HAS_RUNTIME" -eq 1 ] || plan+=("Connect the NVIDIA runtime to Docker and restart Docker")
    [ "$HAS_CURL" -eq 1 ] || plan+=("Install curl")
    plan+=("Check that a test container ($GPU_TEST_IMAGE) can see the GPUs")
fi
if [ -n "$DOCKER_USER" ]; then
    plan+=("Add $DOCKER_USER to the docker group, so docker works without sudo (from your next login)")
fi
if [ "$HAS_COMPOSE_FILE" -eq 1 ]; then
    plan+=("Keep your compose.yaml (settings: $DATA_DIR, port $PORT)")
else
    plan+=("Write compose.yaml: settings in $DATA_DIR, port $PORT, models: ${MODELS_DIR:-none}")
fi
if [ "$PRESET" != none ]; then
    plan+=("Create the instances of the \"$PRESET_NAME\" preset, with new API keys and passwords (nothing starts yet)")
fi
if [ "$DOWNLOAD" = yes ]; then
    plan+=("At the end, download the preset's models into $MODELS_DIR (about ${DOWNLOAD_GB:-?} GB)")
fi
if [ "$NO_START" -eq 0 ]; then
    plan+=("Build and start GPU Shepherd (it restarts by itself after a reboot)")
fi

say "The plan:"
for step in "${plan[@]}"; do item "$NO" "$step"; done
echo
if [ "$INTERACTIVE" -eq 1 ]; then
    confirm "Go ahead?" Y || { echo "Nothing was changed."; exit 2; }
    echo
fi

# --------------------------------------------------------------------------- install

if [ "$SKIP_DEPS" -eq 0 ]; then
    if [ "$HAS_DOCKER" -eq 0 ]; then
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

    if ! have curl; then
        apt_install curl
    fi
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

if [ -n "$DOCKER_USER" ]; then
    getent group docker >/dev/null || groupadd docker
    usermod -aG docker "$DOCKER_USER"
    say "Added $DOCKER_USER to the docker group (takes effect at your next login)"
fi

# --------------------------------------------------------------------------- GPU Shepherd

mkdir -p "$DATA_DIR"
cd "$HERE"
[ -f compose.example.yaml ] || die "compose.example.yaml not found next to install.sh - run this from the repository"

if [ "$HAS_COMPOSE_FILE" -eq 1 ]; then
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

random_hex() { head -c "$1" /dev/urandom | od -An -tx1 | tr -d ' \n'; }
if [ "$PRESET" != none ]; then
    say "Creating the \"$PRESET_NAME\" preset"
    LLM_KEY="sk-$(random_hex 20)"
    HERMES_KEY="$(random_hex 24)"
    HERMES_PASSWORD="$(random_hex 8)"
    owner_uid=$(id -u "${SUDO_USER:-root}" 2>/dev/null || echo 0)
    owner_gid=$(id -g "${SUDO_USER:-root}" 2>/dev/null || echo 0)
    # Fill in this machine's folders, fresh secrets, and autostart off until you have tried it.
    fill() {
        sed -e "s#/srv/models#$MODELS_DIR#g" \
            -e "s#/opt/gpu-shepherd#$DATA_DIR#g" \
            -e "s#sk-demo-0*#$LLM_KEY#g" \
            -e "s#REPLACE_WITH_LLM_API_KEY#$LLM_KEY#g" \
            -e "s#change-me-hermes-api-key#$HERMES_KEY#g" \
            -e "s#change-me-please#$HERMES_PASSWORD#g" \
            -e "s#HERMES_UID: \"1000\"#HERMES_UID: \"$owner_uid\"#" \
            -e "s#HERMES_GID: \"1000\"#HERMES_GID: \"$owner_gid\"#" \
            -e "s#^autostart: true#autostart: false#" \
            "$1" > "$2"
    }
    mkdir -p "$DATA_DIR/instances" "$DATA_DIR/catalogs" "$DATA_DIR/appdata" "$MODELS_DIR"
    for f in "$PRESET"/instances/*.yaml "$PRESET"/catalogs/*.yaml; do
        [ -f "$f" ] || continue
        dest="$DATA_DIR/${f#"$PRESET"/}"
        if [ ! -e "$dest" ]; then fill "$f" "$dest"; chmod 600 "$dest"; fi
    done
    # appdata/ holds starting files for the instances' folders (configs, tools, notes). Copied as
    # the user, text files filled in; existing files are kept.
    if [ -d "$PRESET/appdata" ]; then
        while IFS= read -r -d '' f; do
            rel=${f#"$PRESET"/}; dest="$DATA_DIR/$rel"
            [ ! -e "$dest" ] || continue
            mkdir -p "$(dirname "$dest")"
            if ! grep -Iq . "$f"; then cp -p "$f" "$dest"
            else fill "$f" "$dest"; case "$f" in *.yaml) chmod 600 "$dest" ;; *) chmod --reference="$f" "$dest" ;; esac
            fi
        done < <(find "$PRESET/appdata" -type f -print0)
        for d in "$PRESET"/appdata/*/; do chown -R "$owner_uid:$owner_gid" "$DATA_DIR/appdata/$(basename "$d")"; done
    fi
    # Create every host folder the instances mount, so Docker doesn't make them root-only.
    grep -ho 'host: /[^,}]*' "$DATA_DIR"/instances/*.yaml | sed 's/^host: //' | sort -u | while read -r d; do
        case "$d" in "$DATA_DIR"/*|"$MODELS_DIR"|"$MODELS_DIR"/*) ;; *) continue ;; esac
        [ -e "$d" ] || { mkdir -p "$d"; chown "$owner_uid:$owner_gid" "$d"; }
    done
    if [ -n "${SUDO_USER:-}" ]; then chown "$owner_uid:$owner_gid" "$MODELS_DIR"; fi
    if [ -f "$PRESET/NOTES.txt" ]; then   # may hold a password
        fill "$PRESET/NOTES.txt" "$DATA_DIR/PRESET_NOTES.txt"; chmod 600 "$DATA_DIR/PRESET_NOTES.txt"
    fi
fi

# Run as the user who ran sudo, so the files are theirs. A failure is reported, not fatal.
download_models() {
    [ "$DOWNLOAD" = yes ] || return 0
    echo
    say "Downloading the preset's models into $MODELS_DIR (Ctrl+C stops it; run it again to resume)"
    local as=()
    if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then as=(sudo -u "$SUDO_USER"); fi
    if ! "${as[@]}" env HF_TOKEN="${HF_TOKEN:-}" bash "$DOWNLOADER" "$PRESET" "$MODELS_DIR"; then
        warn "some models were not downloaded. Finish with:  $DOWNLOADER $PRESET $MODELS_DIR"
    fi
}

if [ "$NO_START" -eq 1 ]; then
    say "Ready. Start GPU Shepherd with:  sudo docker compose up -d --build"
    download_models
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
    echo "    Lost it? sudo docker exec gpu-shepherd gpu-shepherd --reset-password"
fi
echo "    Keep it on a trusted network: it controls Docker, which is root-equivalent."
if [ -n "$DOCKER_USER" ]; then
    echo "    You ($DOCKER_USER) can run docker without sudo after logging out and back in. Being in"
    echo "    the docker group is root-equivalent too."
fi
if [ "$PRESET" != none ]; then
    echo
    say "\"$PRESET_NAME\" preset created (nothing is running yet)"
    if [ "$DOWNLOAD" != yes ] && [ -f "$PRESET/downloads.txt" ]; then
        echo "    Download its models (about ${DOWNLOAD_GB:-?} GB; run it without the folder to list them):"
        echo "      $DOWNLOADER $PRESET $MODELS_DIR"
    fi
    echo "    Press Start on each card in the dashboard (the first start downloads its image), and"
    echo "    turn on Autostart for the ones you want running after a reboot."
    if [ -f "$DATA_DIR/PRESET_NOTES.txt" ]; then
        sed -e "s#HOST#${host_ip:-<this-machine>}#g" -e 's/^/    /' "$DATA_DIR/PRESET_NOTES.txt"
    fi
fi

download_models
