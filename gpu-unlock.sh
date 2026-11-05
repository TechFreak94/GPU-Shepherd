#!/usr/bin/env bash
# GPU Shepherd's helper for gpu-unlock (https://github.com/Zagitta/gpu-unlock), an EXPERIMENTAL,
# optional extra. gpu-unlock is a Linux kernel module that lifts the half-rate cap GeForce cards
# have on FP16/BF16/TF32 tensor math. It runs at boot, before the NVIDIA driver, so every change
# here takes effect at the next reboot. It flashes nothing: turning it off and rebooting puts the
# cards back exactly as they were.
#
#   sudo ./gpu-unlock.sh --check            # what this machine has; changes nothing
#   sudo ./gpu-unlock.sh --install          # build and install the pinned gpu-unlock (no card unlocked yet)
#   sudo ./gpu-unlock.sh --set 0000:01:00.0,0000:02:00.0   # unlock these cards from the next boot
#   sudo ./gpu-unlock.sh --set none         # unlock nothing from the next boot (the rollback)
#   sudo ./gpu-unlock.sh --uninstall        # --set none, then remove the packages
#   sudo ./gpu-unlock.sh --install-service  # let the dashboard (Settings > Experimental) do the above
#
# Safety nets, because a bad unlock can stop a machine from booting:
#   - only the gpu-unlock commit pinned below is built, from a folder only root can write;
#   - the unlock goes on the normal boot entries only (GRUB_CMDLINE_LINUX_DEFAULT), so GRUB's
#     "Advanced options > recovery mode" always boots without it. Ubuntu shows the GRUB menu by
#     itself after a boot that did not finish;
#   - a boot that skipped the unlock (recovery mode, or the option deleted with 'e' in GRUB) while
#     it was on turns it off;
#   - after a change, the first boot with the unlock has 10 minutes to show every chosen card
#     unlocked and nvidia-smi seeing every GPU. If that boot dies first, or the check fails, the
#     unlock is turned off by itself for the boots after it.
# Exit status: 0 done, 1 an error.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")

# The gpu-unlock version GPU Shepherd installs. Moving the pin is a reviewed change to this file.
UNLOCK_REPO=https://github.com/Zagitta/gpu-unlock.git
UNLOCK_COMMIT=7eb51bccae0a0b7e599e084d971170e9a1e82d0c
UNLOCK_VERSION=2.0.0

T=${GPU_UNLOCK_TEST_ROOT:-}                  # tests only: a pretend machine in this folder
UNIT=gpu-shepherd-unlock
UNIT_DIR=$T/etc/systemd/system
HOST_DIR=$T/var/lib/gpu-shepherd-unlock     # root only: the source that is built and the safety state
SRC="$HOST_DIR/src"
STATE_FILE="$HOST_DIR/state"                # line 1 state, line 2 cards, line 3 boot id, line 4 note
GRUB_DEFAULT_FILE=$T/etc/default/grub
GRUB_DROPIN=$T/etc/default/grub.d/99-gpu-shepherd-unlock.cfg
PCI_DIR=$T/sys/bus/pci/devices
LOCK_FILE=$T/run/gpu-shepherd-unlock.lock
CONFIRM_AFTER=10min
PCI_RE='^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$'

ACTION=""
TARGETS_ARG=""
QUIET=0
FROM_DASHBOARD=0

usage() {
    cat <<EOF
Usage: sudo ./gpu-unlock.sh ACTION

EXPERIMENTAL. Installs and configures gpu-unlock $UNLOCK_VERSION (${UNLOCK_COMMIT:0:12}) for the GeForce
cards that support it. Every change takes effect at the next reboot.

  --check              show the cards, what is installed and what the next boot does; changes nothing
  --install            build and install gpu-unlock (it unlocks no card until you --set some)
  --set CARDS          the cards to unlock from the next boot: PCI addresses as  lspci -D  prints
                       them, comma separated; "all" (every supported card) or "none" (the rollback)
  --uninstall          unlock nothing from the next boot and remove gpu-unlock's packages
  --install-service    let the dashboard ask for these (a small systemd service, like update.sh's)
  --remove-service     remove that service again (the boot safety checks stay while a card is set)
  -q, --quiet          print less
  -h, --help           show this help
EOF
}

if [ -t 1 ]; then
    say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
    warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
    err()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; }
else
    say()  { printf '==> %s\n' "$*"; }
    warn() { printf 'warning: %s\n' "$*" >&2; }
    err()  { printf 'error: %s\n' "$*" >&2; }
fi
info() { [ "$QUIET" -eq 1 ] || say "$@"; }
have() { command -v "$1" >/dev/null 2>&1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --check|--install|--uninstall|--install-service|--remove-service|--boot|--confirm)
                          ACTION=${1#--}; shift ;;
        --set)            ACTION=set; TARGETS_ARG="${2:?--set needs PCI addresses, all or none}"; shift 2 ;;
        --from-dashboard) FROM_DASHBOARD=1; shift ;;
        -q|--quiet)       QUIET=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                usage >&2; err "unknown option: $1"; exit 1 ;;
    esac
done
[ -n "$ACTION" ] || [ "$FROM_DASHBOARD" -eq 1 ] || { usage; exit 0; }

if [ "$(id -u)" -ne 0 ] && [ -z "$T" ]; then
    have sudo || { err "run this as root"; exit 1; }
    exec sudo -- "$HERE/$(basename "$0")" "${ARGS[@]}"
fi

# The data folder comes from compose.yaml, the same way install.sh and update.sh read it. The
# dashboard and this script talk through files there (the same path inside the container):
#   status.json  written here after every action and at boot (see write_status)
#   request      written by the dashboard: line 1 check/install/set/uninstall, line 2 the cards
#   log.txt      the output of the last action the dashboard asked for
DATA_DIR=""
if [ -f "$HERE/compose.yaml" ]; then
    DATA_DIR=$(sed -n 's/^ *SHEPHERD_DATA_DIR: *//p' "$HERE/compose.yaml" | head -n 1)
fi
DATA_DIR=${DATA_DIR:-/opt/gpu-shepherd}
SHARED="$DATA_DIR/gpu-unlock"
STATUS="$SHARED/status.json"
REQUEST="$SHARED/request"
LOG="$SHARED/log.txt"

json_str() {
    local s=${1//\\/\\\\}
    s=${s//\"/\\\"}; s=${s//$'\t'/ }; s=${s//$'\r'/}; s=${s//$'\n'/\\n}
    printf '"%s"' "$s"
}

# --------------------------------------------------------------------------- what this machine has

kernel_log() {   # this boot's gpu_unlock lines
    { journalctl -k -b 0 -q --no-pager 2>/dev/null || dmesg 2>/dev/null || true; } | grep 'gpu_unlock' || true
}

# The supported device ids by family, from the pinned source's recipes.c (empty before the source is fetched).
supported_ids() {
    [ -f "$SRC/kernel/gpu_unlock/recipes.c" ] || return 0
    grep -oE '\b(GA10[0-9]|AD10[0-9]|TU10[0-9])\(0x[0-9a-fA-F]{4}\)' "$SRC/kernel/gpu_unlock/recipes.c" |
        sed 's/(0x/ /; s/)//' | tr 'A-F' 'a-f' | awk '{print $2, toupper($1)}'
}
family_of() {   # family_of 2484 -> GA104 (or nothing)
    supported_ids | awk -v id="$1" '$1 == id {print $2; exit}'
}
tier_of() {     # RTX 30 (GA10x) is gpu-unlock's stable tier; RTX 40 and RTX 20 are experimental there too
    case "$1" in GA*) echo stable ;; AD*|TU*) echo experimental ;; *) echo "" ;; esac
}

nvidia_cards() {   # PCI address and device id of every NVIDIA display card, one per line
    local d
    for d in "$PCI_DIR"/*; do
        [ "$(cat "$d/vendor" 2>/dev/null)" = 0x10de ] || continue
        case "$(cat "$d/class" 2>/dev/null)" in 0x0300*|0x0302*) ;; *) continue ;; esac
        printf '%s %s\n' "$(basename "$d")" "$(sed 's/^0x//' "$d/device")"
    done
}
supported_cards() {
    local pci id
    while read -r pci id; do
        if [ -n "$(family_of "$id")" ]; then echo "$pci"; fi
    done < <(nvidia_cards)
}

cmdline_targets() {   # the cards a kernel command line asks for (the running one by default)
    tr ' ' '\n' <"${1:-$T/proc/cmdline}" | sed -n 's/^gpu_unlock\.targets=//p' | tail -n 1
}

bootloader() {
    if have update-grub && [ -f "$GRUB_DEFAULT_FILE" ]; then echo grub
    elif have grubby; then echo grubby
    else echo ""; fi
}

configured_targets() {   # what the next boot unlocks
    case "$(bootloader)" in
        grub)   sed -n 's/.*gpu_unlock\.targets=\([^ "]*\).*/\1/p' "$GRUB_DROPIN" 2>/dev/null | tail -n 1 ;;
        grubby) grubby --info=DEFAULT 2>/dev/null | tr ' "' '\n\n' | sed -n 's/^gpu_unlock\.targets=//p' | tail -n 1 ;;
    esac
}

installed() { modinfo gpu_unlock >/dev/null 2>&1; }
installed_version() { modinfo -F version gpu_unlock 2>/dev/null || true; }
secure_boot() { have mokutil && mokutil --sb-state 2>/dev/null | grep -qi 'SecureBoot enabled'; }

# Why the dashboard can't install it here (nothing when it can).
blocker() {
    if grep -qi microsoft "$T/proc/version" 2>/dev/null; then
        echo "This is WSL: gpu-unlock needs a Linux machine that boots its own kernel."
    elif [ -z "$(bootloader)" ]; then
        echo "Only GRUB (Debian, Ubuntu) and grubby (Fedora, RHEL) boot setups can be changed from here. Follow gpu-unlock's README to set it up by hand."
    elif secure_boot; then
        echo "Secure Boot is on: gpu-unlock's module would have to be signed with a key you enroll at the machine's screen. Follow gpu-unlock's README to do it by hand, or turn Secure Boot off."
    elif grep -q 'gpu_unlock\.targets' "$GRUB_DEFAULT_FILE" 2>/dev/null; then
        echo "/etc/default/grub already sets gpu_unlock.targets by hand. Remove it there (and run update-grub) to let GPU Shepherd manage it."
    fi
}

# --------------------------------------------------------------------------- safety state

read_state() {   # sets ST_STATE ST_CARDS ST_BOOT ST_NOTE
    ST_STATE=off; ST_CARDS=""; ST_BOOT=""; ST_NOTE=""
    [ -f "$STATE_FILE" ] || return 0
    { read -r ST_STATE; read -r ST_CARDS; read -r ST_BOOT; read -r ST_NOTE; } <"$STATE_FILE" || true
    ST_STATE=${ST_STATE:-off}
}
save_state() {   # save_state STATE CARDS BOOT NOTE
    mkdir -p "$HOST_DIR"; chmod 700 "$HOST_DIR"
    printf '%s\n%s\n%s\n%s\n' "$1" "$2" "$3" "$4" >"$STATE_FILE.tmp"
    mv -f "$STATE_FILE.tmp" "$STATE_FILE"
}
boot_id() { cat "$T/proc/sys/kernel/random/boot_id" 2>/dev/null || echo unknown; }

# --------------------------------------------------------------------------- status for the dashboard

RUN=idle                # idle / working / done / failed: the action the dashboard asked for
RUN_MSG=""
write_status() {
    [ -d "$DATA_DIR" ] || return 0
    mkdir -p "$SHARED"
    read_state
    local cards="" first=1 pci id fam name unlocked note logs
    logs=$(kernel_log)
    while read -r pci id; do
        [ -n "$pci" ] || continue
        fam=$(family_of "$id")
        name=$({ lspci -s "$pci" 2>/dev/null || true; } | sed 's/^[^:]*:[^:]*: //' | head -n 1)
        note=$(printf '%s\n' "$logs" | grep -F "$pci" | tail -n 1 | sed 's/.*gpu_unlock: //' || true)
        unlocked=false
        printf '%s\n' "$logs" | grep -F "$pci" | grep -qE 'gpu_unlock: (unlocked|still unlocked|unlocked again)' && unlocked=true
        [ "$first" -eq 1 ] || cards+=","
        first=0
        cards+="{\"pci\":$(json_str "$pci"),\"device\":$(json_str "$id"),\"name\":$(json_str "$name"),"
        cards+="\"family\":$(json_str "$fam"),\"tier\":$(json_str "$(tier_of "$fam")"),\"unlocked\":$unlocked,\"note\":$(json_str "$note")}"
    done < <(nvidia_cards)
    local service=false
    have systemctl && systemctl is-enabled --quiet "$UNIT.path" 2>/dev/null && service=true
    {
        printf '{"checked_at":%s,"run":%s,"message":%s,' "$(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)")" \
            "$(json_str "$RUN")" "$(json_str "$RUN_MSG")"
        printf '"installed":%s,"installed_version":%s,"version":%s,"commit":%s,' \
            "$(installed && echo true || echo false)" "$(json_str "$(installed_version)")" \
            "$(json_str "$UNLOCK_VERSION")" "$(json_str "$UNLOCK_COMMIT")"
        printf '"source":%s,"bootloader":%s,"blocker":%s,"service":%s,' "$( [ -d "$SRC/.git" ] && echo true || echo false)" \
            "$(json_str "$(bootloader)")" "$(json_str "$(blocker)")" "$service"
        printf '"active":%s,"configured":%s,"state":%s,"note":%s,"repo":%s,"cards":[%s]}\n' \
            "$(json_str "$(cmdline_targets)")" "$(json_str "$(configured_targets)")" "$(json_str "$ST_STATE")" \
            "$(json_str "$ST_NOTE")" "$(json_str "$HERE")" "$cards"
    } >"$STATUS.tmp"
    mv -f "$STATUS.tmp" "$STATUS"
}

fail() {
    err "$1"
    RUN=failed; RUN_MSG=$1
    write_status || true
    exit 1
}

# --------------------------------------------------------------------------- the boot line

write_cmdline() {   # write_cmdline CARDS (empty: remove it)
    case "$(bootloader)" in
        grub)
            if [ -n "$1" ]; then
                mkdir -p "$(dirname "$GRUB_DROPIN")"
                cat >"$GRUB_DROPIN" <<EOF
# Written by $HERE/gpu-unlock.sh (GPU Shepherd, Settings > Experimental). Normal boot entries
# only: GRUB's recovery mode boots without gpu-unlock. Remove with: sudo $HERE/gpu-unlock.sh --set none
GRUB_CMDLINE_LINUX_DEFAULT="\${GRUB_CMDLINE_LINUX_DEFAULT:-} gpu_unlock.targets=$1"
EOF
            else
                rm -f "$GRUB_DROPIN"
            fi
            info "Updating the boot menu"
            update-grub >/dev/null 2>&1 || fail "update-grub failed: run it yourself to see why"
            ;;
        grubby)
            grubby --update-kernel=ALL --remove-args=gpu_unlock.targets >/dev/null 2>&1 || true
            if [ -n "$1" ]; then
                grubby --update-kernel=ALL --args="gpu_unlock.targets=$1" >/dev/null || fail "grubby could not change the boot line"
            fi
            ;;
        *) [ -z "$1" ] || fail "this machine's boot setup can't be changed from here: $(blocker)" ;;
    esac
}

ensure_safety_units() {   # the boot and 10-minute checks; harmless while nothing is unlocked
    have systemctl || return 0
    cat >"$UNIT_DIR/$UNIT-boot.service" <<EOF
# Written by $HERE/gpu-unlock.sh. Turns gpu-unlock off after a boot that skipped it or did not stay up.
[Unit]
Description=GPU Shepherd: gpu-unlock boot safety check
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/bash $HERE/gpu-unlock.sh --boot --quiet

[Install]
WantedBy=multi-user.target rescue.target
EOF
    cat >"$UNIT_DIR/$UNIT-confirm.service" <<EOF
# Written by $HERE/gpu-unlock.sh.
[Unit]
Description=GPU Shepherd: check that gpu-unlock worked on this boot

[Service]
Type=oneshot
ExecStart=/bin/bash $HERE/gpu-unlock.sh --confirm --quiet
EOF
    cat >"$UNIT_DIR/$UNIT-confirm.timer" <<EOF
# Written by $HERE/gpu-unlock.sh.
[Unit]
Description=GPU Shepherd: check gpu-unlock $CONFIRM_AFTER after boot

[Timer]
OnBootSec=$CONFIRM_AFTER

[Install]
WantedBy=timers.target
EOF
    systemctl daemon-reload
    systemctl enable "$UNIT-boot.service" "$UNIT-confirm.timer" >/dev/null 2>&1 || warn "could not enable the boot safety checks"
}

# rollback NOTE: unlock nothing from the next boot (this boot stays as it is)
rollback() {
    warn "$1"
    write_cmdline ""
    save_state off "" "" "$1"
}

# --------------------------------------------------------------------------- actions

fetch_source() {
    have git || fail "git is not installed"
    mkdir -p "$HOST_DIR"; chmod 700 "$HOST_DIR"
    if [ ! -d "$SRC/.git" ]; then
        rm -rf "$SRC"
        info "Downloading gpu-unlock $UNLOCK_VERSION"
        git clone --quiet "$UNLOCK_REPO" "$SRC" || { rm -rf "$SRC"; fail "could not download gpu-unlock from $UNLOCK_REPO (offline?)"; }
    fi
    if ! git -C "$SRC" cat-file -e "$UNLOCK_COMMIT^{commit}" 2>/dev/null; then
        git -C "$SRC" fetch --quiet origin || fail "could not download gpu-unlock's pinned version"
    fi
    git -C "$SRC" checkout --quiet --force --detach "$UNLOCK_COMMIT" 2>/dev/null ||
        fail "gpu-unlock's pinned version ${UNLOCK_COMMIT:0:12} is not in its repository any more"
    git -C "$SRC" clean -fdxq
    [ "$(git -C "$SRC" rev-parse HEAD)" = "$UNLOCK_COMMIT" ] || fail "gpu-unlock's source is not at the pinned version"
}

do_check() {
    if have git && [ ! -d "$SRC/.git" ]; then
        fetch_source || true          # only to know which cards are supported; nothing is built
    fi
    RUN_MSG="Checked."
}

do_install() {
    local b
    b=$(blocker); [ -z "$b" ] || fail "$b"
    fetch_source
    [ -n "$(supported_cards)" ] || fail "none of this machine's NVIDIA cards is supported by gpu-unlock $UNLOCK_VERSION (RTX 20, 30 and 40 are; RTX 50 isn't)"
    have make || { have apt-get && apt-get install -y -q make >/dev/null; } || { have dnf && dnf install -y -q make >/dev/null; } ||
        fail "make is not installed"
    RUN=working; RUN_MSG="Building and installing gpu-unlock (this downloads NVIDIA's driver package once, for its firmware)"; write_status
    say "$RUN_MSG"
    make -C "$SRC" install || fail "gpu-unlock did not build or install (see the log). Nothing was unlocked."
    installed || fail "gpu-unlock was built, but its kernel module is not there for kernel $(uname -r)"
    ensure_safety_units
    RUN_MSG="gpu-unlock $UNLOCK_VERSION is installed. No card is unlocked yet: choose the cards, then reboot."
    say "$RUN_MSG"
}

# do_set CARDS: CARDS is "all", "none" or comma separated PCI addresses
do_set() {
    local want=$1 c cards="" supported
    supported=$(supported_cards)
    case "$want" in
        none|"") cards="" ;;
        all)     cards=$(printf '%s\n' "$supported" | paste -sd, -) ;;
        *)
            for c in ${want//,/ }; do
                c=$(printf '%s' "$c" | tr 'A-F' 'a-f')
                [[ $c =~ $PCI_RE ]] || fail "'$c' is not a PCI address like 0000:01:00.0"
                printf '%s\n' "$supported" | grep -qxF "$c" || fail "$c is not a card gpu-unlock supports on this machine"
                case ",$cards," in *",$c,"*) ;; *) cards=${cards:+$cards,}$c ;; esac
            done ;;
    esac
    if [ -z "$cards" ]; then
        write_cmdline ""
        save_state off "" "" "Turned off. The cards run at their normal speed after the next reboot."
        RUN_MSG="No card is unlocked from the next boot. Reboot to put the cards back to normal."
    else
        local b; b=$(blocker); [ -z "$b" ] || fail "$b"
        installed || fail "install gpu-unlock first"
        ensure_safety_units
        write_cmdline "$cards"
        save_state armed "$cards" "" "Set for the next boot: $cards"
        RUN_MSG="Reboot to unlock $cards. If that boot hangs, power the machine off and on, and pick Advanced options > recovery mode in the boot menu: that boot turns the unlock off."
    fi
    say "$RUN_MSG"
}

do_uninstall() {
    do_set none
    say "Removing gpu-unlock's packages"
    if have dpkg-query && dpkg-query -W gpu-unlock-dkms >/dev/null 2>&1; then
        apt-get remove -y -q gpu-unlock-dkms gpu-unlock-firmware || fail "could not remove gpu-unlock's packages"
    elif have rpm && rpm -q gpu-unlock-dkms >/dev/null 2>&1; then
        dnf remove -y -q 'gpu-unlock*' || fail "could not remove gpu-unlock's packages"
    elif have pacman && pacman -Q gpu-unlock-dkms >/dev/null 2>&1; then
        pacman -R --noconfirm gpu-unlock-dkms gpu-unlock-firmware || fail "could not remove gpu-unlock's packages"
    fi
    rm -rf "$SRC"
    if have systemctl; then
        systemctl disable "$UNIT-boot.service" "$UNIT-confirm.timer" >/dev/null 2>&1 || true
        rm -f "$UNIT_DIR/$UNIT-boot.service" "$UNIT_DIR/$UNIT-confirm.service" "$UNIT_DIR/$UNIT-confirm.timer"
        systemctl daemon-reload
    fi
    RUN_MSG="gpu-unlock is removed. Reboot to put the cards back to normal."
    say "$RUN_MSG"
}

# Early in every boot: catch a boot that skipped the unlock, and an unlocked boot that never
# reached the 10-minute check.
do_boot() {
    read_state
    local active; active=$(cmdline_targets)
    case "$ST_STATE" in
        armed)
            if [ -n "$active" ]; then
                save_state trying "$ST_CARDS" "$(boot_id)" "First boot with the unlock: checking it for $CONFIRM_AFTER."
            else
                rollback "This boot started without the unlock (recovery mode or an edited boot line), so it was turned off."
            fi ;;
        trying)
            [ "$ST_BOOT" = "$(boot_id)" ] ||
                rollback "The last boot with the unlock stopped before its $CONFIRM_AFTER check, so the unlock was turned off. Reboot to run at normal speed." ;;
        on)
            [ -n "$active" ] || rollback "This boot started without the unlock (recovery mode or an edited boot line), so it was turned off." ;;
    esac
}

# 10 minutes after boot: did every chosen card unlock, and does the driver still see every GPU?
do_confirm() {
    read_state
    [ "$ST_STATE" = trying ] && [ "$ST_BOOT" = "$(boot_id)" ] || return 0
    local logs c missing="" want got
    logs=$(kernel_log)
    for c in ${ST_CARDS//,/ }; do
        printf '%s\n' "$logs" | grep -F "$c" | grep -qE 'gpu_unlock: (unlocked|still unlocked|unlocked again)' || missing="$missing $c"
    done
    if have nvidia-smi; then
        want=$(nvidia_cards | wc -l)
        got=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
        if [ "${got:-0}" -lt "$want" ]; then
            rollback "After the unlock the NVIDIA driver saw only ${got:-0} of $want GPUs, so the unlock was turned off. Reboot to run at normal speed."
            return 0
        fi
    fi
    if [ -n "$missing" ]; then
        save_state on "$ST_CARDS" "$(boot_id)" "Working, but not unlocked:$missing (see journalctl -k -b | grep gpu_unlock)."
    else
        save_state on "$ST_CARDS" "$(boot_id)" "Unlocked and checked on $(date -u +%Y-%m-%d)."
    fi
}

install_service() {
    have systemctl && [ -d "$T/run/systemd/system" ] || { err "systemd is needed for the dashboard to manage gpu-unlock"; exit 1; }
    [ -d "$DATA_DIR" ] || { err "$DATA_DIR does not exist yet; run install.sh first"; exit 1; }
    mkdir -p "$SHARED"
    cat >"$UNIT_DIR/$UNIT.path" <<EOF
# Written by $HERE/gpu-unlock.sh --install-service. Remove with --remove-service.
[Unit]
Description=GPU Shepherd: watch for gpu-unlock requests from the dashboard

[Path]
PathExists=$REQUEST

[Install]
WantedBy=multi-user.target
EOF
    cat >"$UNIT_DIR/$UNIT.service" <<EOF
# Written by $HERE/gpu-unlock.sh --install-service.
[Unit]
Description=GPU Shepherd: change gpu-unlock, as asked from the dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/bash $HERE/gpu-unlock.sh --from-dashboard
EOF
    systemctl daemon-reload
    systemctl enable --now "$UNIT.path" >/dev/null
    read_state
    [ "$ST_STATE" = off ] || ensure_safety_units
    write_status
    info "The dashboard can now manage gpu-unlock (Settings > Experimental). Remove with: sudo $HERE/gpu-unlock.sh --remove-service"
}

remove_service() {
    if have systemctl; then
        systemctl disable --now "$UNIT.path" >/dev/null 2>&1 || true
        rm -f "$UNIT_DIR/$UNIT.path" "$UNIT_DIR/$UNIT.service"
        systemctl daemon-reload
    fi
    rm -f "$REQUEST"
    write_status
    info "The dashboard can no longer change gpu-unlock."
}

# --------------------------------------------------------------------------- main

if [ "$FROM_DASHBOARD" -eq 1 ]; then
    ACTION=$(head -n 1 "$REQUEST" 2>/dev/null | head -c 16 | tr -dc '[:lower:]' || true)
    TARGETS_ARG=$(sed -n 2p "$REQUEST" 2>/dev/null | head -c 400 | tr -dc '0-9a-fA-F:.,lnoe' || true)
    rm -f "$REQUEST"
    mkdir -p "$SHARED"
    exec >"$LOG" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) requested from the dashboard: ${ACTION:-nothing} ${TARGETS_ARG}"
    case "$ACTION" in check|install|set|uninstall) ;; *) echo "unknown request; nothing done"; exit 0 ;; esac
fi

case "$ACTION" in
    install-service) install_service; exit 0 ;;
    remove-service)  remove_service; exit 0 ;;
    boot)            do_boot; write_status; exit 0 ;;
    confirm)         do_confirm; write_status; exit 0 ;;
esac

# one change at a time (the dashboard and you could overlap)
if have flock && { exec 9>"$LOCK_FILE"; } 2>/dev/null && ! flock -n 9; then
    fail "gpu-unlock is already being changed"
fi

RUN=working; RUN_MSG="Working on it"; write_status
case "$ACTION" in
    check)     do_check ;;
    install)   do_install ;;
    set)       do_set "$TARGETS_ARG" ;;
    uninstall) do_uninstall ;;
esac
RUN=done
write_status
if [ "$ACTION" = check ] && [ "$QUIET" -eq 0 ] && [ "$FROM_DASHBOARD" -eq 0 ]; then
    cat "$STATUS"
fi
