#!/usr/bin/env bash
# GPU Shepherd installer for Linux: Ubuntu 24.04+, Debian 12/13, Fedora, RHEL/Rocky/Alma 9+ and Arch
# (and distros based on them). Elsewhere, install the four requirements yourself and use --skip-deps.
#
#   sudo ./install.sh              # guided: looks at the machine, asks a few questions, then installs
#   sudo ./install.sh --yes        # no questions: defaults plus whatever flags you give
#   sudo ./install.sh --help       # every flag (for scripted or advanced installs)
#   sudo ./install.sh --update     # update GPU Shepherd later (the same as ./update.sh)
#
# What it does, skipping anything already in place:
#   1. checks the NVIDIA driver (offers to install the recommended one)
#   2. installs Docker Engine and the Compose plugin (Ubuntu and Arch packages, or Docker's repository)
#   3. installs the NVIDIA Container Toolkit and connects it to Docker (and SELinux, where Docker uses it)
#   4. checks that a container can see the GPUs
#   5. adds you to the docker group, so docker works without sudo
#   6. on the first run, offers a preset: a ready-made set of instances to start from
#   7. writes compose.yaml (if there isn't one yet) and starts GPU Shepherd
#   8. with a preset, offers to download its models
#   9. offers to let the dashboard check for updates and update GPU Shepherd for you
#  10. if you ask for it (off by default), installs Tailscale to reach the dashboard and apps remotely
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
DASH_UPDATES=""         # yes / no: install the service that lets the dashboard update GPU Shepherd
TAILSCALE=""            # yes / no: install Tailscale on this machine and log it in to your tailnet
TS_HTTPS=0              # 1: also serve the dashboard over HTTPS on its Tailscale name
GPU_TEST_IMAGE=ubuntu:24.04
# Below this much installed RAM, models must fit entirely in GPU memory. Keeping part of a model in
# system RAM (MoE expert offload) needs room for those layers plus the OS, Docker and the other
# containers: a 35B-A3B model at Q4 is ~22 GB on its own, so 32 GB is the smallest size where offload
# is practical. 64 GB fits a large MoE coding model such as Qwen3 Coder Next (~49 GB at Q4).
MIN_RAM_GB=32
RECOMMENDED_RAM_GB=64
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")
if [ "${1:-}" = --update ]; then exec "$HERE/update.sh" "${@:2}"; fi

usage() {
    cat <<EOF
Usage: sudo ./install.sh [options]

Run it without options for a guided install. Every option below answers one of its questions
in advance, for scripted or advanced installs.

  --models-dir DIR    a folder of .gguf model files to pick from in the UI (mounted read-only);
                      pass "" for none
  --port N            web UI port (default: $DEFAULT_PORT)
  --data-dir DIR      where GPU Shepherd keeps its settings (default: $DEFAULT_DATA_DIR)
  --install-driver    install the distro's NVIDIA driver if none is working (needs a reboot;
                      not on RHEL-family systems or inside a container)
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
  --dashboard-updates     let the dashboard check for updates daily and update GPU Shepherd when
                          you press its button (a small systemd service; see update.sh);
                          with --yes this is only set up when given
  --no-dashboard-updates  don't offer that; update with ./update.sh instead
  --tailscale         install Tailscale and join this machine to your tailnet, to reach the
                      dashboard and apps from your other devices without opening any ports.
                      Logs in with a link it prints, or with an auth key in the TS_AUTHKEY
                      environment variable (sudo TS_AUTHKEY=tskey-... ./install.sh --tailscale);
                      the key is not saved. Off unless asked for.
  --tailscale-https   with --tailscale, also serve the dashboard at https://<name>.<tailnet>.ts.net
                      (needs HTTPS certificates turned on in Tailscale's admin console)
  --no-tailscale      don't offer Tailscale
  --update [OPTIONS]  update GPU Shepherd instead of installing: runs ./update.sh (see --help there)
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
        --dashboard-updates) DASH_UPDATES=yes; shift ;;
        --no-dashboard-updates) DASH_UPDATES=no; shift ;;
        --tailscale)      TAILSCALE=yes; shift ;;
        --tailscale-https) TAILSCALE=yes; TS_HTTPS=1; shift ;;
        --no-tailscale)   TAILSCALE=no; shift ;;
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

# The distro family decides the package manager and where each package comes from: ubuntu, debian,
# fedora, rhel (RHEL, Rocky, Alma, CentOS Stream, Oracle) or arch. Derivatives are matched through
# ID_LIKE. An unknown distro gets no package installs: everything must be in place (or --skip-deps).
ID=""; ID_LIKE=""; VERSION_ID=""; VERSION_CODENAME=""; DEBIAN_CODENAME=""; PRETTY_NAME=""
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
fi
family_of() {
    case "$1" in
        ubuntu|debian|fedora|arch) echo "$1" ;;
        rhel|centos|rocky|almalinux|ol) echo rhel ;;
    esac
}
FAMILY=$(family_of "$ID")
if [ -z "$FAMILY" ]; then
    for like in $ID_LIKE; do FAMILY=$(family_of "$like"); [ -z "$FAMILY" ] || break; done
fi
major=${VERSION_ID%%.*}
case "$FAMILY" in
    ubuntu) min_major=24 ;;
    debian) min_major=12 ;;
    rhel)   min_major=9 ;;
    *)      min_major="" ;;
esac
if [ -z "$FAMILY" ]; then
    warn "${PRETTY_NAME:-this distro} isn't one this script knows (Ubuntu, Debian, Fedora, RHEL-family, Arch).
         It can't install packages here: install Docker, the Compose plugin, the NVIDIA driver and
         the NVIDIA Container Toolkit yourself, then run: sudo ./install.sh --skip-deps"
else
    say "${PRETTY_NAME:-$ID} detected"
    # Only the distro itself is checked: a derivative's version numbers are its own.
    if [ -n "$min_major" ] && [ "$(family_of "$ID")" = "$FAMILY" ] \
            && [ -n "$major" ] && [ "$major" -eq "$major" ] 2>/dev/null && [ "$major" -lt "$min_major" ]; then
        warn "$PRETTY_NAME is older than this script was tested on; continuing anyway."
    fi
fi
# WSL2 (experimental): the GPU and its driver come from Windows, Docker usually from Docker Desktop.
WSL=0
if [ -e /proc/sys/fs/binfmt_misc/WSLInterop ] || grep -qi microsoft /proc/version 2>/dev/null; then
    WSL=1
    # WSL1 has no GPU support at all. Its kernel name ends in "-Microsoft" (WSL2's has "microsoft-standard").
    case "$(uname -r)" in
        *-Microsoft) die "this is WSL1, which can't use the GPU. Convert the distro to WSL2 in Windows with
       wsl --set-version <distro> 2   (wsl -l -v lists them), then run this script again." ;;
    esac
    warn "WSL2 detected. Support for it is experimental: the NVIDIA driver must be installed in Windows
         (not in Linux), and Docker Desktop with WSL integration, or Docker Engine in this distro, runs
         the containers. See docs/linux.md#wsl2."
    # The Windows driver puts nvidia-smi here, which isn't always on PATH.
    [ ! -x /usr/lib/wsl/lib/nvidia-smi ] || PATH="$PATH:/usr/lib/wsl/lib"
fi
# Inside a container (LXC on Proxmox, say) the driver belongs to the host, so it is never installed.
IN_CONTAINER=""
if have systemd-detect-virt; then
    IN_CONTAINER=$(systemd-detect-virt --container 2>/dev/null || true)
    case "$IN_CONTAINER" in none|wsl) IN_CONTAINER="" ;; esac
elif [ "$WSL" -eq 0 ] && { [ -f /.dockerenv ] || [ -f /run/.containerenv ]; }; then
    IN_CONTAINER=container
fi
[ -z "$IN_CONTAINER" ] || say "Running inside a container ($IN_CONTAINER)"
# Without systemd (WSL2 before it is turned on, some containers) there is no Docker service to
# start and no dashboard update service.
HAS_SYSTEMD=0
if [ -d /run/systemd/system ]; then HAS_SYSTEMD=1; fi
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

# Docker Desktop (WSL2) brings its own GPU support: no toolkit or runtime setup here.
docker_desktop() { [ "$WSL" -eq 1 ] && docker info --format '{{.OperatingSystem}}' 2>/dev/null | grep -q 'Docker Desktop'; }
HAS_DRIVER=0; HAS_DOCKER=0; HAS_COMPOSE=0; HAS_CTK=0; HAS_RUNTIME=0; HAS_CURL=0; HAS_COMPOSE_FILE=0
DOCKER_IS_PODMAN=0      # podman-docker's stand-in docker command: it can't run GPU Shepherd
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then HAS_DRIVER=1; fi
if have docker; then
    if docker --version 2>/dev/null | grep -qi podman; then DOCKER_IS_PODMAN=1; else HAS_DOCKER=1; fi
fi
if [ "$HAS_DOCKER" -eq 1 ] && docker compose version >/dev/null 2>&1; then HAS_COMPOSE=1; fi
if have nvidia-ctk; then HAS_CTK=1; fi
if [ "$HAS_DOCKER" -eq 1 ] && docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then HAS_RUNTIME=1; fi
if [ "$HAS_DOCKER" -eq 1 ] && docker_desktop; then HAS_CTK=1; HAS_RUNTIME=1; fi
if have curl; then HAS_CURL=1; fi
if [ -f "$HERE/compose.yaml" ]; then HAS_COMPOSE_FILE=1; fi
# AMD GPUs the amdgpu driver runs (GPU Shepherd gives them to apps through Vulkan; experimental),
# and whether there is any NVIDIA GPU at all. A machine with AMD GPUs only needs nothing from NVIDIA.
AMD_GPUS=0; HAS_NVIDIA_HW=0; AMD_ONLY=0
for d in /sys/class/drm/card[0-9]*; do
    if [ "$(cat "$d/device/vendor" 2>/dev/null)" = 0x1002 ] && [ -r "$d/device/mem_info_vram_total" ]; then
        AMD_GPUS=$((AMD_GPUS + 1))
    fi
done
for d in /sys/bus/pci/devices/*; do
    if [ "$(cat "$d/vendor" 2>/dev/null)" = 0x10de ]; then
        case "$(cat "$d/class" 2>/dev/null)" in 0x03*) HAS_NVIDIA_HW=1 ;; esac
    fi
done
if [ "$AMD_GPUS" -gt 0 ] && [ "$HAS_NVIDIA_HW" -eq 0 ] && [ "$HAS_DRIVER" -eq 0 ] && [ "$WSL" -eq 0 ]; then
    AMD_ONLY=1
fi

say "This machine has:"
if [ "$HAS_DRIVER" -eq 1 ]; then
    item "$OK" "NVIDIA driver, with:"
    nvidia-smi -L | sed 's/ (UUID.*//; s/^/        /'
elif [ "$AMD_ONLY" -eq 0 ]; then
    item "$NO" "no working NVIDIA driver"
fi
[ "$AMD_GPUS" -eq 0 ] || item "$OK" "$AMD_GPUS AMD GPU(s) with the amdgpu driver (used through Vulkan; experimental)"
status() { if [ "$1" -eq 1 ]; then item "$OK" "$2"; else item "$NO" "no $2"; fi; }
status "$HAS_DOCKER" "Docker"
[ "$DOCKER_IS_PODMAN" -eq 0 ] || item "$NO" "a docker command from Podman (podman-docker), not Docker"
status "$HAS_COMPOSE" "Docker Compose"
if [ "$HAS_DOCKER" -eq 1 ] && docker_desktop; then item "$OK" "GPU support from Docker Desktop"
elif [ "$AMD_ONLY" -eq 0 ]; then status "$HAS_CTK" "NVIDIA Container Toolkit"; fi
[ "$HAS_COMPOSE_FILE" -eq 1 ] && item "$OK" "a compose.yaml from an earlier run (it will be kept)"
echo

# --------------------------------------------------------------------------- NVIDIA driver

# Each family's package manager. apt's package lists are refreshed once, and again after a
# repository is added. Arch is never partly upgraded behind your back (no -Sy).
apt_fresh=0
pkg_install() {
    case "$FAMILY" in
        ubuntu|debian)
            if [ "$apt_fresh" -eq 0 ]; then DEBIAN_FRONTEND=noninteractive apt-get update -q; apt_fresh=1; fi
            DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$@" ;;
        fedora|rhel)
            dnf install -y "$@" ;;
        arch)
            pacman -S --needed --noconfirm "$@" || die "pacman could not install $*. If a download failed,
       the package lists are out of date: run  sudo pacman -Syu  (reboot if it updated the kernel),
       then run this again."
            ;;
        *)  die "this script can't install $* on ${PRETTY_NAME:-this distro}. Install it yourself and run
       this again with --skip-deps." ;;
    esac
}
pkg_remove() {
    case "$FAMILY" in
        ubuntu|debian) DEBIAN_FRONTEND=noninteractive apt-get remove -y -q "$@" ;;
        fedora|rhel)   dnf remove -y "$@" ;;
        arch)          pacman -R --noconfirm "$@" ;;
    esac
}

# Where the NVIDIA driver comes from on each family. RHEL-family systems have several equally
# official ways (NVIDIA's CUDA repository, precompiled or DKMS, open or proprietary), so the choice
# is left to you there; in a container the driver is the host's.
driver_cmd=""
case "$FAMILY" in
    ubuntu) driver_cmd="sudo ubuntu-drivers install" ;;
    debian) driver_cmd="sudo apt install nvidia-driver firmware-misc-nonfree linux-headers-amd64   (Debian's non-free)" ;;
    fedora) driver_cmd="sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda   (from RPM Fusion)" ;;
    arch)   driver_cmd="sudo pacman -S nvidia-open-dkms linux-headers" ;;
esac
[ -z "$IN_CONTAINER" ] && [ "$WSL" -eq 0 ] || driver_cmd=""
if [ "$WSL" -eq 1 ]; then
    no_driver_help="On WSL2 the driver comes from Windows: install NVIDIA's current Windows driver
       (never a Linux driver inside WSL), run  wsl --update  in Windows, then run this script again."
elif [ -n "$IN_CONTAINER" ]; then
    no_driver_help="This is a container ($IN_CONTAINER): install the NVIDIA driver on the host, pass the
       GPU devices into the container and install the same driver version's user-space files in it
       (docs/proxmox.md shows how for Proxmox LXC), then run this script again."
elif [ "$FAMILY" = rhel ]; then
    no_driver_help="Install it from NVIDIA's CUDA repository, following
       https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/
       then reboot and run this script again."
elif [ -z "$driver_cmd" ]; then
    no_driver_help="Install your distro's NVIDIA driver, reboot and run this script again."
else
    no_driver_help="Install one with
       sudo ./install.sh --install-driver
   or  $driver_cmd
then reboot and run this script again."
fi

install_driver() {
    case "$FAMILY" in
        ubuntu)
            pkg_install ubuntu-drivers-common
            ubuntu-drivers install ;;
        debian)
            DEBIAN_FRONTEND=noninteractive apt-get update -q; apt_fresh=1
            apt-cache policy nvidia-driver 2>/dev/null | grep -q 'Candidate: [0-9]' \
                || die "Debian's nvidia-driver package isn't available. Enable the contrib, non-free and
       non-free-firmware components (after \"main\" in /etc/apt/sources.list, or on the Components:
       line of /etc/apt/sources.list.d/debian.sources), then run this script again."
            pkg_install "linux-headers-$(dpkg --print-architecture)" nvidia-driver firmware-misc-nonfree ;;
        fedora)
            if ! rpm -q rpmfusion-nonfree-release >/dev/null 2>&1; then
                say "Enabling RPM Fusion (free and nonfree), where Fedora's NVIDIA driver comes from"
                local v; v=$(rpm -E %fedora)
                dnf install -y "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$v.noarch.rpm" \
                    "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$v.noarch.rpm"
            fi
            pkg_install akmod-nvidia xorg-x11-drv-nvidia-cuda
            say "Building the driver's kernel module (a few minutes)"
            akmods --force || warn "the kernel module didn't build yet; it is built again at the next boot" ;;
        arch)
            # nvidia-open covers Turing (GTX 16 / RTX 20) and newer; older cards need a legacy
            # driver from the AUR. DKMS builds it for every installed kernel, given its headers.
            local headers=() k
            for k in linux linux-lts linux-zen linux-hardened; do
                if pacman -Q "$k" >/dev/null 2>&1; then headers+=("$k-headers"); fi
            done
            pkg_install nvidia-open-dkms "${headers[@]}" ;;
    esac
}

if [ "$SKIP_DEPS" -eq 0 ] && [ "$HAS_DRIVER" -eq 0 ] && [ "$AMD_ONLY" -eq 0 ]; then
    if [ -z "$driver_cmd" ]; then
        die "no working NVIDIA driver (nvidia-smi failed). $no_driver_help"
    fi
    if [ "$INSTALL_DRIVER" -eq 0 ] && [ "$INTERACTIVE" -eq 1 ]; then
        echo "GPU Shepherd needs the NVIDIA driver. This script can install ${PRETTY_NAME:-your distro}'s driver"
        echo "for you; the machine then has to reboot before the rest of the install can continue."
        [ "$FAMILY" != arch ] || echo "(nvidia-open: for GPUs older than the GTX 16 / RTX 20 series, install a legacy driver from the AUR instead.)"
        if confirm "Install the NVIDIA driver now?" Y; then
            INSTALL_DRIVER=1
        else
            die "no working NVIDIA driver. $no_driver_help"
        fi
    fi
    [ "$INSTALL_DRIVER" -eq 1 ] || die "no working NVIDIA driver (nvidia-smi failed). $no_driver_help"
    say "Installing the NVIDIA driver"
    install_driver
    if [ -d /sys/firmware/efi ] && have mokutil && mokutil --sb-state 2>/dev/null | grep -qi enabled; then
        warn "Secure Boot is on: if the driver doesn't load after the reboot, enrol its signing key
         when the blue MOK screen asks (Ubuntu and Fedora set one up), or turn Secure Boot off."
    fi
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

# --------------------------------------------------------------------------- updates

if [ -z "$DASH_UPDATES" ]; then
    if systemctl is-enabled --quiet gpu-shepherd-update.path 2>/dev/null; then
        DASH_UPDATES=no                       # already on; nothing to do
    elif [ "$INTERACTIVE" -eq 1 ] && [ "$HAS_SYSTEMD" -eq 1 ] && [ -e "$HERE/.git" ]; then
        echo "    GPU Shepherd can check for updates once a day and show them on the dashboard, with a"
        echo "    button that updates it for you (a small systemd service runs ./update.sh). Without it,"
        echo "    update any time with:  sudo ./update.sh"
        if confirm "Let the dashboard check for updates and update GPU Shepherd?" Y; then
            DASH_UPDATES=yes; else DASH_UPDATES=no; fi
        echo
    else
        DASH_UPDATES=no
    fi
fi
if [ "$DASH_UPDATES" = yes ] && [ "$HAS_SYSTEMD" -eq 0 ]; then
    if [ "$WSL" -eq 1 ]; then
        warn "--dashboard-updates ignored: systemd isn't running in this WSL distro. Turn it on with
         [boot] systemd=true  in /etc/wsl.conf and  wsl --shutdown  in Windows, or update with ./update.sh."
    else
        warn "--dashboard-updates ignored: systemd isn't running here; update with ./update.sh."
    fi
    DASH_UPDATES=no
fi
if [ "$DASH_UPDATES" = yes ] && [ ! -e "$HERE/.git" ]; then
    warn "--dashboard-updates ignored: $HERE is not a git clone, so it can't be updated in place."
    DASH_UPDATES=no
fi

# --------------------------------------------------------------------------- remote access

# Tailscale goes on the host, not in a container: the dashboard and every app already publish
# their ports there, so one login reaches them all, and it keeps working while GPU Shepherd is
# restarted or updated. Never asked when it is already logged in.
HAS_TAILSCALE=0
if have tailscale; then HAS_TAILSCALE=1; fi
TS_UP=0
if [ "$HAS_TAILSCALE" -eq 1 ] && tailscale status >/dev/null 2>&1; then TS_UP=1; fi
if [ -z "$TAILSCALE" ]; then
    if [ "$INTERACTIVE" -eq 1 ] && [ "$TS_UP" -eq 0 ] && [ "$WSL" -eq 0 ]; then
        echo "    Tailscale (free for personal use) lets your phone and laptop reach the dashboard and"
        echo "    apps from anywhere, without opening ports on your router. Your other devices need it too."
        if confirm "Install Tailscale for remote access?" N; then TAILSCALE=yes; else TAILSCALE=no; fi
        echo
    else
        TAILSCALE=no
    fi
fi
if [ "$TAILSCALE" = yes ]; then
    if [ "$WSL" -eq 1 ]; then
        warn "--tailscale ignored in WSL2: install Tailscale for Windows instead. Ports published by
         Docker Desktop are then reachable on the Windows machine's Tailscale address."
        TAILSCALE=no
    elif [ "$HAS_TAILSCALE" -eq 0 ] && { [ "$SKIP_DEPS" -eq 1 ] || [ -z "$FAMILY" ]; }; then
        warn "--tailscale ignored: Tailscale isn't installed and this script won't install it here.
         Install it (https://tailscale.com/download), then run this script again."
        TAILSCALE=no
    elif [ "$HAS_SYSTEMD" -eq 0 ]; then
        warn "--tailscale ignored: systemd isn't running here, so tailscaled can't run as a service."
        TAILSCALE=no
    elif [ -n "$IN_CONTAINER" ] && [ ! -c /dev/net/tun ]; then
        warn "--tailscale ignored: this container has no /dev/net/tun. Pass it through (see
         https://tailscale.com/kb/1130/lxc-unprivileged) or install Tailscale on the host instead."
        TAILSCALE=no
    fi
fi

# --------------------------------------------------------------------------- docker group

DOCKER_USER=""
if [ "$DOCKER_GROUP" -eq 1 ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ] \
        && ! id -nG "$SUDO_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
    DOCKER_USER=$SUDO_USER
fi

# --------------------------------------------------------------------------- the plan

case "$FAMILY" in
    ubuntu) DOCKER_FROM="Ubuntu packages";  CTK_FROM="NVIDIA's apt repository" ;;
    debian) DOCKER_FROM="Docker's apt repository"; CTK_FROM="NVIDIA's apt repository" ;;
    fedora|rhel) DOCKER_FROM="Docker's rpm repository"; CTK_FROM="NVIDIA's rpm repository" ;;
    arch)   DOCKER_FROM="Arch packages";    CTK_FROM="Arch packages" ;;
    *)      DOCKER_FROM=""; CTK_FROM="" ;;
esac
# Docker's repositories name RHEL itself "rhel" and its rebuilds "centos"; Debian derivatives use
# the Debian release they are built on.
DOCKER_REPO=centos
[ "$ID" != rhel ] || DOCKER_REPO=rhel
DEBIAN_RELEASE=${DEBIAN_CODENAME:-$VERSION_CODENAME}
SELINUX_ON=0
if have selinuxenabled && selinuxenabled; then SELINUX_ON=1; fi

plan=()
if [ "$SKIP_DEPS" -eq 0 ]; then
    if [ "$WSL" -eq 1 ] && [ "$HAS_DOCKER" -eq 0 ] && [ "$HAS_SYSTEMD" -eq 0 ]; then
        die "no Docker. In WSL2 either install Docker Desktop and turn on its WSL integration for this
       distro, or turn systemd on ([boot] systemd=true in /etc/wsl.conf, then  wsl --shutdown  in
       Windows) so this script can install Docker Engine here. Then run it again."
    fi
    if [ -z "$FAMILY" ]; then
        missing=()
        [ "$HAS_DOCKER" -eq 1 ] || missing+=("Docker")
        [ "$HAS_DOCKER" -eq 0 ] || [ "$HAS_COMPOSE" -eq 1 ] || missing+=("the Docker Compose plugin")
        [ "$HAS_CTK" -eq 1 ] || [ "$AMD_ONLY" -eq 1 ] || missing+=("the NVIDIA Container Toolkit")
        [ "$HAS_CURL" -eq 1 ] || missing+=("curl")
        if [ ${#missing[@]} -gt 0 ]; then
            die "this script can't install packages on ${PRETTY_NAME:-this distro}. Install these yourself:
       $(IFS=,; echo "${missing[*]}" | sed 's/,/, /g')
       then run: sudo ./install.sh --skip-deps  (see docs/linux.md)"
        fi
    fi
    [ "$DOCKER_IS_PODMAN" -eq 0 ] || plan+=("Remove podman-docker (Podman's stand-in docker command), to make way for Docker")
    [ "$HAS_DOCKER" -eq 1 ] || plan+=("Install Docker Engine and Docker Compose ($DOCKER_FROM)")
    [ "$HAS_DOCKER" -eq 0 ] || [ "$HAS_COMPOSE" -eq 1 ] || plan+=("Install the Docker Compose plugin")
    if [ "$AMD_ONLY" -eq 0 ]; then
        [ "$HAS_CTK" -eq 1 ] || plan+=("Install the NVIDIA Container Toolkit ($CTK_FROM)")
        [ "$HAS_RUNTIME" -eq 1 ] || plan+=("Connect the NVIDIA runtime to Docker and restart Docker")
    fi
    [ "$HAS_CURL" -eq 1 ] || plan+=("Install curl")
    if [ "$SELINUX_ON" -eq 1 ]; then
        plan+=("If Docker uses SELinux: let containers use the GPUs (container_use_devices) and let GPU Shepherd reach Docker")
    fi
    [ "$AMD_ONLY" -eq 1 ] || plan+=("Check that a test container ($GPU_TEST_IMAGE) can see the GPUs")
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
    if [ "$WSL" -eq 1 ]; then
        plan+=("Build and start GPU Shepherd (it starts again when Docker does, once WSL is running)")
    else
        plan+=("Build and start GPU Shepherd (it restarts by itself after a reboot)")
    fi
fi
if [ "$DASH_UPDATES" = yes ]; then
    plan+=("Install the gpu-shepherd-update service: daily update checks, and updates from the dashboard")
fi
if [ "$TAILSCALE" = yes ]; then
    [ "$HAS_TAILSCALE" -eq 1 ] || plan+=("Install Tailscale (from Tailscale's package repository)")
    if [ "$TS_UP" -eq 0 ]; then
        if [ -n "${TS_AUTHKEY:-}" ]; then plan+=("Join your tailnet with the auth key in TS_AUTHKEY")
        else plan+=("Join your tailnet: you open a login link it prints"); fi
    fi
    [ "$TS_HTTPS" -eq 0 ] || plan+=("Serve the dashboard over HTTPS on this machine's Tailscale name")
fi

say "The plan:"
for step in "${plan[@]}"; do item "$NO" "$step"; done
echo
if [ "$INTERACTIVE" -eq 1 ]; then
    confirm "Go ahead?" Y || { echo "Nothing was changed."; exit 2; }
    echo
fi

# --------------------------------------------------------------------------- install

# Whether Docker labels containers for SELinux. GPU Shepherd then needs label=disable to reach the
# Docker socket (and it adds :z to the folders it mounts into app containers by itself).
docker_selinux() { docker info --format '{{json .SecurityOptions}}' 2>/dev/null | grep -q 'name=selinux'; }

install_docker() {
    case "$FAMILY" in
        ubuntu)
            pkg_install docker.io docker-compose-v2 \
                || die "could not install Docker from Ubuntu's packages (is the 'universe' repository enabled?
       sudo add-apt-repository universe). Or install Docker Engine yourself and run this again." ;;
        debian)
            pkg_install ca-certificates curl
            install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
            chmod a+r /etc/apt/keyrings/docker.asc
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc]" \
                "https://download.docker.com/linux/debian $DEBIAN_RELEASE stable" > /etc/apt/sources.list.d/docker.list
            apt_fresh=0
            pkg_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
                || die "could not install Docker from Docker's repository for Debian \"$DEBIAN_RELEASE\".
       Install Docker Engine yourself and run this again." ;;
        fedora|rhel)
            local repo=$DOCKER_REPO
            [ "$FAMILY" != fedora ] || repo=fedora
            curl -fsSL "https://download.docker.com/linux/$repo/docker-ce.repo" -o /etc/yum.repos.d/docker-ce.repo
            pkg_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
                || die "could not install Docker from Docker's repository ($repo). Install Docker Engine
       yourself and run this again." ;;
        arch)
            pkg_install docker docker-compose docker-buildx ;;
    esac
}
install_compose() {
    case "$FAMILY" in
        ubuntu) if dpkg -s docker-ce >/dev/null 2>&1; then pkg_install docker-compose-plugin
                else pkg_install docker-compose-v2; fi ;;
        debian) if dpkg -s docker-ce >/dev/null 2>&1; then pkg_install docker-compose-plugin
                else pkg_install docker-compose; fi ;;
        fedora|rhel) if rpm -q docker-ce >/dev/null 2>&1; then pkg_install docker-compose-plugin
                     else pkg_install docker-compose; fi ;;     # Fedora's own moby-engine
        arch)   pkg_install docker-compose ;;
    esac
    docker compose version >/dev/null 2>&1 || die "'docker compose' still doesn't work. Install the Docker
       Compose plugin (v2) yourself and run this again."
}
install_ctk() {
    case "$FAMILY" in
        ubuntu|debian)
            pkg_install ca-certificates curl gpg
            local keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o "$keyring"
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
                > /etc/apt/sources.list.d/nvidia-container-toolkit.list
            apt_fresh=0
            pkg_install nvidia-container-toolkit ;;
        fedora|rhel)
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
                -o /etc/yum.repos.d/nvidia-container-toolkit.repo
            pkg_install nvidia-container-toolkit ;;
        arch)
            pkg_install nvidia-container-toolkit ;;
    esac
}

if [ "$SKIP_DEPS" -eq 0 ]; then
    if [ "$DOCKER_IS_PODMAN" -eq 1 ]; then
        say "Removing podman-docker"
        pkg_remove podman-docker
    fi
    if ! have curl; then
        pkg_install curl
    fi
    if [ "$HAS_DOCKER" -eq 0 ]; then
        say "Installing Docker Engine and Compose ($DOCKER_FROM)"
        install_docker
    fi
    if ! docker compose version >/dev/null 2>&1; then
        say "Installing the Docker Compose plugin"
        install_compose
    fi
    systemctl enable --now docker >/dev/null 2>&1 || true
    if ! docker info >/dev/null 2>&1; then
        if [ "$WSL" -eq 1 ] && [ "$HAS_SYSTEMD" -eq 0 ]; then
            die "Docker isn't running. In WSL2 either start Docker Desktop with WSL integration on for
       this distro, or turn systemd on ([boot] systemd=true in /etc/wsl.conf, then  wsl --shutdown
       in Windows) so Docker Engine can run here. Then run this script again."
        fi
        die "Docker is installed but not running (try: sudo systemctl status docker)"
    fi
    say "Docker OK: $(docker --version)"

    if [ "$AMD_ONLY" -eq 1 ]; then
        say "AMD GPUs only: nothing from NVIDIA is needed (apps get the GPUs' /dev/dri render nodes)"
    elif docker_desktop; then
        say "Docker Desktop gives containers the GPUs itself; skipping the NVIDIA Container Toolkit"
    elif ! have nvidia-ctk; then
        say "Installing the NVIDIA Container Toolkit ($CTK_FROM)"
        install_ctk
    fi
    if [ "$AMD_ONLY" -eq 0 ] && ! docker_desktop && ! docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
        say "Connecting the NVIDIA runtime to Docker"
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
    fi
    # With SELinux on in Docker (Fedora's own moby-engine does that; Docker CE doesn't by default),
    # containers may only open the GPU device files once this boolean is on.
    if docker_selinux && have getsebool && getsebool container_use_devices >/dev/null 2>&1 \
            && ! getsebool container_use_devices | grep -q -- '--> on'; then
        say "Letting containers use the GPUs under SELinux (container_use_devices)"
        setsebool -P container_use_devices on
    fi
    # Without /dev/char links for the NVIDIA device files, any `systemctl daemon-reload` takes the
    # GPUs away from running containers ("Failed to initialize NVML: Unknown Error"), a known NVIDIA
    # Container Toolkit issue. Make the links now and after every boot (a udev rule). WSL2 has no
    # NVIDIA device files to link (the GPUs come through /dev/dxg), so nothing to do there.
    if [ "$WSL" -eq 0 ] && have nvidia-ctk && [ -d /run/systemd/system ]; then
        nvidia-ctk system create-dev-char-symlinks --create-all >/dev/null 2>&1 || true
        if [ ! -f /etc/udev/rules.d/71-nvidia-dev-char.rules ] && [ ! -f /lib/udev/rules.d/71-nvidia-dev-char.rules ]; then
            mkdir -p /etc/udev/rules.d
            printf '%s\n' "# Written by GPU Shepherd: keeps containers' GPUs through a systemd reload" \
                "ACTION==\"add\", DEVPATH==\"/bus/pci/drivers/nvidia\", RUN+=\"$(command -v nvidia-ctk) system create-dev-char-symlinks --create-all\"" \
                >/etc/udev/rules.d/71-nvidia-dev-char.rules || true
        fi
    fi

    if [ "$AMD_ONLY" -eq 1 ]; then
        :   # no NVIDIA GPU to test; AMD ones are plain device files
    elif say "Checking that containers can see the GPUs" && ! docker run --rm --gpus all "$GPU_TEST_IMAGE" nvidia-smi -L; then
        hint=""
        if [ -n "$IN_CONTAINER" ]; then
            hint="
       In an unprivileged container (LXC), set  no-cgroups = true  in
       /etc/nvidia-container-runtime/config.toml and restart Docker (see docs/proxmox.md)."
        fi
        die "a test container could not see the GPUs. Check the driver (nvidia-smi) and
       'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'.$hint"
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

DOCKER_SELINUX=0
if docker_selinux; then DOCKER_SELINUX=1; fi
if [ "$HAS_COMPOSE_FILE" -eq 1 ]; then
    say "Keeping your existing compose.yaml (delete it to have it written again)"
    if [ "$DOCKER_SELINUX" -eq 1 ] && ! grep -q '^ *security_opt:.*label=disable' compose.yaml; then
        warn "Docker uses SELinux here, and compose.yaml has no  security_opt: [label=disable]  for
         gpu-shepherd, so it can't reach Docker. Add it (see compose.example.yaml)."
    fi
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
    if [ "$AMD_ONLY" -eq 1 ]; then
        # no NVIDIA GPU to read stats from: Docker would refuse to start GPU Shepherd with the request
        sed -i '/^    deploy:$/,/capabilities: \[gpu, utility\]/d' compose.yaml
        ! grep -q 'driver: nvidia' compose.yaml \
            || warn "could not remove the NVIDIA GPU request from compose.yaml; delete its deploy: section by hand"
    fi
    if [ "$DOCKER_SELINUX" -eq 1 ]; then
        sed -i 's|# security_opt: \[label=disable\]|security_opt: [label=disable]|' compose.yaml
        grep -q '^ *security_opt: \[label=disable\]' compose.yaml \
            || warn "could not add security_opt: [label=disable] to compose.yaml; add it by hand"
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

# Installs and logs in Tailscale. Any failure is a warning: GPU Shepherd works without it.
TS_IP=""; TS_URL=""
setup_tailscale() {
    [ "$TAILSCALE" = yes ] || return 0
    echo
    if ! have tailscale; then
        say "Installing Tailscale"
        if [ "$FAMILY" = arch ]; then
            pkg_install tailscale || { warn "could not install Tailscale"; return 0; }
        else
            # Tailscale's own script adds its package repository for this distro (apt, dnf, ...).
            local script; script=$(mktemp)
            if ! curl -fsSL https://tailscale.com/install.sh -o "$script" || ! sh "$script"; then
                rm -f "$script"; warn "could not install Tailscale (see https://tailscale.com/download)"; return 0
            fi
            rm -f "$script"
        fi
    fi
    systemctl enable --now tailscaled >/dev/null 2>&1 || true
    if ! tailscale status >/dev/null 2>&1; then
        say "Joining your tailnet"
        local ok=0
        if [ -n "${TS_AUTHKEY:-}" ]; then
            # Handed over in a private temporary file, so the key never shows in the process list.
            local keyfile; keyfile=$(mktemp); chmod 600 "$keyfile"
            printf '%s' "$TS_AUTHKEY" > "$keyfile"
            if tailscale up --auth-key="file:$keyfile"; then ok=1; fi
            rm -f "$keyfile"
            unset TS_AUTHKEY
        elif [ "$INTERACTIVE" -eq 1 ]; then
            echo "    Open the link below in a browser and log in to Tailscale to add this machine."
            if tailscale up; then ok=1; fi
        fi
        if [ "$ok" -eq 0 ]; then
            warn "this machine hasn't joined a tailnet yet. Finish with:  sudo tailscale up"
            return 0
        fi
    fi
    TS_IP=$(tailscale ip -4 2>/dev/null | head -n 1)
    [ "$TS_HTTPS" -eq 1 ] || return 0
    # The certificate name is this machine's name in the tailnet; there is none until HTTPS
    # certificates are turned on, and serve would then wait for that instead of failing.
    local name
    name=$(tailscale status --json 2>/dev/null | tr -d ' \n' | sed -n 's/.*"CertDomains":\["\([^"]*\)".*/\1/p')
    if [ -n "$name" ] && tailscale serve --bg --https=443 "http://127.0.0.1:$PORT" >/dev/null; then
        TS_URL="https://$name"
    else
        warn "the dashboard isn't served over HTTPS yet. Turn on HTTPS certificates in Tailscale's admin
         console (DNS page), then run:  sudo tailscale serve --bg --https=443 http://127.0.0.1:$PORT"
    fi
}

if [ "$DASH_UPDATES" = yes ]; then
    "$HERE/update.sh" --enable-dashboard-updates || warn "could not set up dashboard updates; update with: sudo ./update.sh"
fi

if [ "$NO_START" -eq 1 ]; then
    setup_tailscale
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

setup_tailscale
host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
say "GPU Shepherd is running: http://${host_ip:-<this-machine>}:$PORT"
if [ -n "$TS_URL" ]; then
    echo "    From your devices on Tailscale: $TS_URL"
fi
if [ -n "$TS_IP" ]; then
    echo "    Over Tailscale: http://$TS_IP:$PORT, and each app on its usual port at $TS_IP"
fi
if [ -f "$DATA_DIR/INITIAL_PASSWORD.txt" ]; then
    echo "    First login (also saved in $DATA_DIR/INITIAL_PASSWORD.txt):"
    sed -n 's/^\(user\|password\): /    \1: /p' "$DATA_DIR/INITIAL_PASSWORD.txt"
    echo "    Change the password under Settings; that file is deleted when you do."
    echo "    Lost it? sudo docker exec -it gpu-shepherd gpu-shepherd --reset-password"
fi
echo "    Keep it on a trusted network: it controls Docker, which is root-equivalent."
echo "    Update it later with:  sudo $HERE/update.sh   (your settings and instances are kept)"
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
