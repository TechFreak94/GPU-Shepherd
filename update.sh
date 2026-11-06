#!/usr/bin/env bash
# GPU Shepherd updater: brings this clone up to date and rebuilds GPU Shepherd. Your settings,
# instances, model lists and passwords live in the data folder, not here, so they are kept, and
# so is your compose.yaml. Running instances keep running while GPU Shepherd restarts.
#
#   sudo ./update.sh              # shows what is new, asks, then updates and rebuilds
#   sudo ./update.sh --check      # only looks: exit 0 up to date, 10 an update is available
#   sudo ./update.sh --rollback   # goes back to the version before the last update
#   sudo ./update.sh --restore-settings   # puts back the settings backed up before it
#   sudo ./update.sh --set-token USER     # a GitHub token for a private repository (asks for it)
#   sudo ./update.sh --help       # every option
#
# Updates are fast-forward only: if you changed files in this folder it stops and tells you
# (or sets them aside and puts them back with --stash), and it never discards commits of yours.
# Before updating it keeps the running image and backs up your settings; if the new version fails
# to build or start, it goes back to the old one by itself and says so.
# With --enable-dashboard-updates, the dashboard can check for updates and run this script for
# you through a small systemd service, since GPU Shepherd can't rebuild itself from inside its
# container. Exit status: 0 done or up to date, 10 (--check) update available, 2 you declined,
# 1 an error.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")
CHECK=0
ASSUME_YES=0
STASH=0
REBUILD=0
SWITCH_TO=""            # --branch: the branch to follow from now on
ROLLBACK=0
RESTORE=0
RESTORE_FILE=""
QUIET=0
FROM_DASHBOARD=0
FINISH=""               # internal: the commit we updated from, after re-running the new script
SERVICE=""              # enable / disable
TOKEN_ACTION=""         # set / clear: the GitHub login used to download updates
TOKEN_USER=""
UNIT=gpu-shepherd-update
UNIT_DIR=/etc/systemd/system
CONTAINER=gpu-shepherd
MAX_LISTED=30           # commits shown in "what's new"

usage() {
    cat <<EOF
Usage: sudo ./update.sh [options]

Without options: fetch the latest version, show what changed, ask, then update this clone and
rebuild GPU Shepherd.

  --check                      only check for an update (exit 10 if there is one); changes nothing
  -y, --yes                    don't ask before updating
  --stash                      set your own changes to files in this folder aside during the update
                               and put them back afterwards (otherwise the update stops)
  --branch NAME                follow another branch of the repository (for trying changes before they are
                               released): switches this clone to it and rebuilds; --branch main goes back
  --rebuild                    rebuild and restart GPU Shepherd even if nothing new was downloaded
  --rollback                   go back to the version that ran before the last update (uses the
                               image kept from before it, so no rebuild or network is needed)
  --restore-settings [FILE]    put back the settings, instances and model lists backed up just
                               before the last update (or from FILE); the current ones are kept
                               as *.before-restore
  --set-token USER             save a GitHub user name and access token for downloading updates
                               from a private repository (asks for the token, or reads it from
                               standard input); it is kept in the clone's .git folder, readable
                               only by the clone's owner and root, and never printed
  --clear-token                forget that token again
  --enable-dashboard-updates   install a systemd service so the dashboard can check for updates
                               daily and update when you press its button
  --disable-dashboard-updates  remove that service again
  -q, --quiet                  print less
  -h, --help                   show this help
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

# `systemctl daemon-reload` takes the GPUs away from every running container (nvidia-smi then fails
# with "Failed to initialize NVML: Unknown Error") unless /dev/char has links for the NVIDIA device
# files: a known NVIDIA Container Toolkit issue. So unit files are only rewritten when they change,
# and before a reload the links are made, with a udev rule that makes them again after a reboot.
# (gpu-unlock.sh has the same helpers.)
UDEV_RULE=/etc/udev/rules.d/71-nvidia-dev-char.rules
UNITS_CHANGED=0
put_unit() {   # put_unit FILE < text: write a unit file only if it is new or different
    local new; new=$(cat)
    if [ ! -f "$1" ] || [ "$(cat "$1")" != "$new" ]; then
        printf '%s\n' "$new" >"$1"
        UNITS_CHANGED=1
    fi
}
keep_gpus_through_reload() {
    have nvidia-ctk || return 0
    grep -qi microsoft /proc/version 2>/dev/null && return 0   # WSL2: no NVIDIA device files to link
    nvidia-ctk system create-dev-char-symlinks --create-all >/dev/null 2>&1 || true
    [ -f "$UDEV_RULE" ] || [ -f /lib/udev/rules.d/71-nvidia-dev-char.rules ] || {
        mkdir -p "$(dirname "$UDEV_RULE")" &&
        printf '%s\n' "# Written by GPU Shepherd: keeps containers' GPUs through a systemd reload" \
            "ACTION==\"add\", DEVPATH==\"/bus/pci/drivers/nvidia\", RUN+=\"$(command -v nvidia-ctk) system create-dev-char-symlinks --create-all\"" \
            >"$UDEV_RULE"
    } 2>/dev/null || true
}
reload_units() {   # only after a unit file changed or went away
    keep_gpus_through_reload
    systemctl daemon-reload
    UNITS_CHANGED=0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --check)        CHECK=1; shift ;;
        -y|--yes)       ASSUME_YES=1; shift ;;
        --stash)        STASH=1; shift ;;
        --rebuild)      REBUILD=1; shift ;;
        --branch)       SWITCH_TO="${2:?--branch needs a branch name}"; REBUILD=1; shift 2 ;;
        --rollback)     ROLLBACK=1; shift ;;
        --restore-settings) RESTORE=1; shift
                        if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then RESTORE_FILE=$1; shift; fi ;;
        --set-token)    TOKEN_ACTION=set; TOKEN_USER="${2:?--set-token needs your GitHub user name}"; shift 2 ;;
        --clear-token)  TOKEN_ACTION=clear; shift ;;
        --enable-dashboard-updates)  SERVICE=enable; shift ;;
        --disable-dashboard-updates) SERVICE=disable; shift ;;
        --from-dashboard) FROM_DASHBOARD=1; ASSUME_YES=1; shift ;;
        --finish)       FINISH="${2:?}"; shift 2 ;;
        -q|--quiet)     QUIET=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              usage >&2; err "unknown option: $1"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- where things are

# The data folder and port come from compose.yaml, the same way install.sh reads them.
DATA_DIR=""; PORT=""
if [ -f "$HERE/compose.yaml" ]; then
    DATA_DIR=$(sed -n 's/^ *SHEPHERD_DATA_DIR: *//p' "$HERE/compose.yaml" | head -n 1)
    PORT=$(sed -n 's/^ *- *"\([0-9]*\):8090".*/\1/p' "$HERE/compose.yaml" | head -n 1)
fi
DATA_DIR=${DATA_DIR:-/opt/gpu-shepherd}
PORT=${PORT:-8090}
STATE_DIR="$DATA_DIR/update"          # shared with the dashboard (same path inside the container)
STATUS="$STATE_DIR/status.json"
REQUEST="$STATE_DIR/request"
PREVIOUS="$STATE_DIR/previous-commit"
BACKUPS="$STATE_DIR/backups"
SAVED_IMAGE=gpu-shepherd:previous   # the image that ran before the last update
SAVED_IMAGE_FILE="$STATE_DIR/previous-image"
PREV_BRANCH_FILE="$STATE_DIR/previous-branch"   # "branch commit" the clone was on before --branch switched it

# Docker is root-equivalent, so this needs root unless you can already use docker.
if [ "$(id -u)" -ne 0 ] && { [ -n "$SERVICE" ] || ! docker info >/dev/null 2>&1; }; then
    have sudo || { err "run this as root"; exit 1; }
    exec sudo -- "$HERE/$(basename "$0")" "${ARGS[@]}"
fi

have git || { err "git is not installed (install it with your package manager: apt, dnf or pacman)"; exit 1; }
[ -e "$HERE/.git" ] || { err "$HERE is not a git clone, so it can't be updated in place. Clone the
       repository again, copy your compose.yaml into the new folder and run install.sh there."; exit 1; }

# git runs as the owner of the clone, so the files stay theirs and their saved credentials (for a
# private fork) are used. As root, git would also refuse a folder owned by someone else.
OWNER=$(stat -c %U "$HERE/.git")
GIT_AS=()
if [ "$(id -u)" -eq 0 ] && [ "$OWNER" != root ]; then
    owner_home=$(getent passwd "$OWNER" | cut -d: -f6)
    GIT_AS=(runuser -u "$OWNER" -- env HOME="$owner_home")
fi
# A GitHub token saved with --set-token (or from the dashboard) is used for downloads only, through
# git's own "store" helper reading this file; your git settings are left alone.
CREDENTIALS="$HERE/.git/gpu-shepherd-credentials"
g() {
    if [ -s "$CREDENTIALS" ]; then
        "${GIT_AS[@]}" git -c credential.helper= -c "credential.helper=store --file=$CREDENTIALS" -C "$HERE" "$@"
    else
        "${GIT_AS[@]}" git -C "$HERE" "$@"
    fi
}
if [ "$ASSUME_YES" -eq 1 ] || ! [ -t 0 ]; then export GIT_TERMINAL_PROMPT=0; fi   # never hang on a password prompt

ask_yes() {   # ask_yes "question" - returns 0 for yes; --yes answers yes
    [ "$ASSUME_YES" -eq 1 ] && return 0
    { : </dev/tty; } 2>/dev/null || return 0
    local answer
    while :; do
        read -r -p "$(printf '? %s [Y/n] ' "$1")" answer </dev/tty || answer=n
        case "${answer:-y}" in [Yy]|[Yy][Ee][Ss]) return 0 ;; [Nn]|[Nn][Oo]) return 1 ;; esac
        echo "    Please answer y or n."
    done
}

# --------------------------------------------------------------------------- status for the dashboard

json_str() {   # a JSON string literal
    local s=${1//\\/\\\\}
    s=${s//\"/\\\"}; s=${s//$'\t'/ }; s=${s//$'\r'/}; s=${s//$'\n'/\\n}
    printf '"%s"' "$s"
}
service_enabled() { have systemctl && systemctl is-enabled --quiet "$UNIT.path" 2>/dev/null; }
# The dashboard's (experimental) gpu-unlock setting goes through gpu-unlock.sh's own small service;
# it comes with dashboard updates, and an update brings in its latest version.
unlock_service() {
    [ -f "$HERE/gpu-unlock.sh" ] && service_enabled || return 0
    bash "$HERE/gpu-unlock.sh" --install-service --quiet || warn "could not set up the gpu-unlock service (Settings > Experimental)"
}

version_at() {   # version_at REF: the version number in gpushepherd/__init__.py at that commit
    { g show "$1:gpushepherd/__init__.py" 2>/dev/null || true; } | sed -n 's/^__version__ = "\(.*\)"$/\1/p' | head -n 1
}
STEP=""                 # what is being done now, for the dashboard's progress view

# write_status STATE MESSAGE [STEP]: what the last check or update found, for the dashboard and
# for you. STATE is one of: idle, checking, updating, done, failed. STEP is one of: check, backup,
# download, rebuild, start, recover (the dashboard lists them in this order; see updates.py), and
# stays the same until the next one is given.
write_status() {
    [ -d "$DATA_DIR" ] || return 0
    mkdir -p "$STATE_DIR"
    [ -z "${3:-}" ] || STEP=$3
    local tmp="$STATUS.tmp" commits="" sha date subject first=1 changes="" f
    if [ -n "${UPSTREAM:-}" ] && [ "$1" = idle ]; then
        # the newest changelog, so the dashboard can show what an update brings before installing it
        g show "$UPSTREAM:CHANGELOG.md" >"$STATE_DIR/changelog.md.tmp" 2>/dev/null &&
            mv -f "$STATE_DIR/changelog.md.tmp" "$STATE_DIR/changelog.md" || rm -f "$STATE_DIR/changelog.md.tmp"
    fi
    if [ -n "${UPSTREAM:-}" ] && [ "${BEHIND:-0}" -gt 0 ]; then
        while IFS=$'\t' read -r sha date subject; do
            [ "$first" -eq 1 ] || commits+=","
            first=0
            commits+="{\"sha\":$(json_str "$sha"),\"date\":$(json_str "$date"),\"subject\":$(json_str "$subject")}"
        done < <(g log --format='%h%x09%cs%x09%s' -n "$MAX_LISTED" "HEAD..$UPSTREAM" 2>/dev/null)
    fi
    first=1
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ "$first" -eq 1 ] || changes+=","
        first=0; changes+=$(json_str "$f")
    done <<<"${LOCAL_CHANGES:-}"
    {
        printf '{"checked_at":%s,"state":%s,"step":%s,"message":%s,' "$(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)")" \
            "$(json_str "$1")" "$(json_str "$STEP")" "$(json_str "$2")"
        printf '"version":%s,"latest_version":%s,' "$(json_str "$(version_at HEAD)")" \
            "$(json_str "$( [ -z "${UPSTREAM:-}" ] || version_at "$UPSTREAM")")"
        printf '"repo":%s,"branch":%s,"upstream":%s,' "$(json_str "$HERE")" "$(json_str "${BRANCH:-}")" \
            "$(json_str "${UPSTREAM:-}")"
        printf '"current":%s,"latest":%s,"previous":%s,"behind":%s,"ahead":%s,' \
            "$(json_str "$(g rev-parse HEAD 2>/dev/null || true)")" "$(json_str "${LATEST:-}")" \
            "$(json_str "$(cat "$PREVIOUS" 2>/dev/null || true)")" "${BEHIND:-0}" "${AHEAD:-0}"
        printf '"credentials":%s,' "$(json_str "$(token_user)")"
        printf '"branches":[%s],"branch_info":[%s],"github":%s,' "$(branch_list)" "$(branch_info)" "$(json_str "$(github_repo)")"
        printf '"local_changes":[%s],"commits":[%s],"dashboard_updates":%s}\n' "$changes" "$commits" \
            "$(service_enabled && echo true || echo false)"
    } >"$tmp"
    mv -f "$tmp" "$STATUS"
}

branch_list() {   # the remote's branches as a JSON list, newest first (from the last fetch)
    local remote=${UPSTREAM%%/*} out="" b first=1
    [ -n "$remote" ] || return 0
    while IFS= read -r b; do
        b=${b#"$remote"/}
        case "$b" in HEAD|"$remote"|"") continue ;; esac
        [ "$first" -eq 1 ] || out+=","
        first=0; out+=$(json_str "$b")
    done < <(g for-each-ref --sort=-committerdate --count=40 --format='%(refname:short)' "refs/remotes/$remote/" 2>/dev/null)
    printf '%s' "$out"
}

branch_info() {   # the same branches as JSON objects (name, date, subject), newest first, for the dashboard's picker
    local remote=${UPSTREAM%%/*} out="" b d s first=1
    [ -n "$remote" ] || return 0
    while IFS=$'\t' read -r b d s; do
        b=${b#"$remote"/}
        case "$b" in HEAD|"$remote"|"") continue ;; esac
        [ "$first" -eq 1 ] || out+=","
        first=0; out+="{\"name\":$(json_str "$b"),\"date\":$(json_str "$d"),\"subject\":$(json_str "$s"),\"version\":$(json_str "$(version_at "refs/remotes/$remote/$b")")}"
    done < <(g for-each-ref --sort=-committerdate --count=40 --format='%(refname:short)%09%(committerdate:iso-strict)%09%(contents:subject)' "refs/remotes/$remote/" 2>/dev/null)
    printf '%s' "$out"
}

github_repo() {   # owner/name when the remote is on github.com (for looking up pull requests), else nothing
    local url
    url=$(g config --get "remote.${UPSTREAM%%/*}.url" 2>/dev/null || true)
    case "$url" in
        https://github.com/*|https://*@github.com/*|git@github.com:*|ssh://git@github.com/*)
            url=${url#*github.com[:/]}; url=${url%.git}; url=${url%/}
            case "$url" in */*/*|*[!A-Za-z0-9._/-]*|"") ;; *) printf '%s' "$url" ;; esac ;;
    esac
}

fail() {   # fail "message": report it here and on the dashboard, then stop
    err "$1"
    write_status failed "$1" || true
    exit 1
}

# --------------------------------------------------------------------------- GitHub token

token_user() {   # the user name saved with the token, or nothing
    [ -s "$CREDENTIALS" ] && sed -n 's|^https://\([^:]*\):.*|\1|p' "$CREDENTIALS" | head -n 1 || true
}

# save_token USER TOKEN: GitHub user names are letters, digits and hyphens; tokens (ghp_..., github_pat_...)
# letters, digits and underscores. Checking that keeps the file's URL format safe without escaping.
save_token() {
    local user=$1 token=$2 url host
    [[ "$user" =~ ^[A-Za-z0-9][A-Za-z0-9-]{0,38}$ ]] || { err "that doesn't look like a GitHub user name"; return 1; }
    [[ "$token" =~ ^[A-Za-z0-9_]{20,255}$ ]] || { err "that doesn't look like a GitHub access token (ghp_... or github_pat_...)"; return 1; }
    url=$(g config --get "remote.$(g config --get "branch.$(g symbolic-ref --short -q HEAD).remote" || echo origin).url" 2>/dev/null || true)
    case "$url" in
        https://*) host=${url#https://}; host=${host%%/*}; host=${host##*@} ;;
        "")        host=github.com ;;
        *)         warn "this clone downloads from $url, which doesn't use HTTPS, so a token isn't used for it.
         Switch it with:  git -C $HERE remote set-url origin https://github.com/<owner>/<repo>.git"
                   host=github.com ;;
    esac
    ( umask 077; printf 'https://%s:%s@%s\n' "$user" "$token" "$host" >"$CREDENTIALS.tmp" )
    chown "$OWNER" "$CREDENTIALS.tmp" 2>/dev/null || true
    chmod 600 "$CREDENTIALS.tmp"
    mv -f "$CREDENTIALS.tmp" "$CREDENTIALS"
    say "Saved the GitHub token for $user (used for $host; only $OWNER and root can read it)"
}

clear_token() {
    rm -f "$CREDENTIALS" "$CREDENTIALS.tmp"
    say "Forgot the saved GitHub token"
}

if [ "$TOKEN_ACTION" = set ]; then
    if [ -t 0 ]; then
        echo "Create a token at https://github.com/settings/personal-access-tokens/new: a fine-grained token,"
        echo "only this repository, permission Contents: Read-only. (GitHub doesn't take passwords for git.)"
        read -r -s -p "GitHub token for $TOKEN_USER (not shown): " token </dev/tty; echo
    else
        read -r token || true
    fi
    save_token "$TOKEN_USER" "$token" || exit 1
    unset token
    CHECK=1                            # check right away, which shows whether the token works
elif [ "$TOKEN_ACTION" = clear ]; then
    clear_token
    CHECK=1
fi

# --------------------------------------------------------------------------- dashboard service

if [ "$SERVICE" = enable ]; then
    { have systemctl && [ -d /run/systemd/system ]; } || { err "systemd is needed for dashboard updates (on WSL2:
       set [boot] systemd=true in /etc/wsl.conf, then wsl --shutdown in Windows)"; exit 1; }
    [ -d "$DATA_DIR" ] || { err "$DATA_DIR does not exist yet; run install.sh first"; exit 1; }
    mkdir -p "$STATE_DIR"
    put_unit "$UNIT_DIR/$UNIT.path" <<EOF
# Written by $HERE/update.sh --enable-dashboard-updates. Remove with --disable-dashboard-updates.
[Unit]
Description=GPU Shepherd: watch for update requests from the dashboard

[Path]
PathExists=$REQUEST

[Install]
WantedBy=multi-user.target
EOF
    put_unit "$UNIT_DIR/$UNIT.service" <<EOF
# Written by $HERE/update.sh --enable-dashboard-updates.
[Unit]
Description=GPU Shepherd: check for an update or update, as asked from the dashboard
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
# Through bash, so SELinux (Fedora, RHEL) lets systemd run it from a clone in a home folder.
ExecStart=/bin/bash $HERE/update.sh --from-dashboard
EOF
    put_unit "$UNIT_DIR/$UNIT-check.service" <<EOF
# Written by $HERE/update.sh --enable-dashboard-updates.
[Unit]
Description=GPU Shepherd: check for an update (nothing is changed)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/bash $HERE/update.sh --check --quiet
SuccessExitStatus=10
EOF
    put_unit "$UNIT_DIR/$UNIT-check.timer" <<EOF
# Written by $HERE/update.sh --enable-dashboard-updates.
[Unit]
Description=GPU Shepherd: check for an update once a day

[Timer]
OnBootSec=10min
OnCalendar=daily
RandomizedDelaySec=1h
Persistent=true

[Install]
WantedBy=timers.target
EOF
    [ "$UNITS_CHANGED" -eq 0 ] || reload_units
    systemctl enable --no-reload --now "$UNIT.path" "$UNIT-check.timer" >/dev/null
    unlock_service
    say "Dashboard updates are on: GPU Shepherd checks for updates daily, and Settings > Updates"
    echo "    (or the notice on the dashboard) can check now or update. Turn it off with:"
    echo "      sudo $HERE/update.sh --disable-dashboard-updates"
    systemctl start --no-block "$UNIT-check.service" || true
    exit 0
elif [ "$SERVICE" = disable ]; then
    if have systemctl; then
        systemctl disable --no-reload --now "$UNIT.path" "$UNIT-check.timer" >/dev/null 2>&1 || true
        rm -f "$UNIT_DIR/$UNIT.path" "$UNIT_DIR/$UNIT.service" "$UNIT_DIR/$UNIT-check.service" \
              "$UNIT_DIR/$UNIT-check.timer"
        reload_units
    fi
    rm -f "$REQUEST"
    [ -f "$STATUS" ] && sed -i 's/"dashboard_updates":true/"dashboard_updates":false/' "$STATUS"
    say "Dashboard updates are off. Update from here with: sudo $HERE/update.sh"
    exit 0
fi

# A request from the dashboard: "check" or "update". Taken (deleted) first, so systemd doesn't
# start us again for the same request; the output goes to a log the dashboard shows.
if [ "$FROM_DASHBOARD" -eq 1 ]; then
    action=$(head -n 1 "$REQUEST" 2>/dev/null | head -c 16 | tr -dc "[:lower:]" || true)
    request_arg=$(sed -n 2p "$REQUEST" 2>/dev/null | head -c 100 | tr -dc 'A-Za-z0-9._/-' || true)
    rm -f "$REQUEST"
    exec >"$STATE_DIR/log.txt" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) requested from the dashboard: ${action:-nothing}"
    case "$action" in
        check)  CHECK=1 ;;
        update) REBUILD=1 ;;
        branch) [ -n "$request_arg" ] || fail "no branch name was given"; SWITCH_TO=$request_arg; REBUILD=1 ;;
        credentials)
            # the dashboard leaves "user" and "token" lines (or "clear") in a 0600 file; take them
            # out of the data folder at once, keep them in the clone, then check that they work
            creds="$STATE_DIR/credentials"
            if [ "$(head -n 1 "$creds" 2>/dev/null)" = clear ]; then
                clear_token
            else
                { read -r c_user; read -r c_token; } <"$creds" 2>/dev/null || true
                rm -f "$creds"
                save_token "${c_user:-}" "${c_token:-}" || { unset c_token; fail "The GitHub token was not saved: it doesn't look like a user name and token."; }
                unset c_token
            fi
            rm -f "$creds"
            CHECK=1 ;;
        *)      echo "unknown request; nothing done"; exit 0 ;;
    esac
fi

# One update at a time (the daily check, the dashboard and you could overlap).
if [ -z "$FINISH" ]; then
    if have flock && { exec 9>"/tmp/gpu-shepherd-update-$(id -u).lock"; } 2>/dev/null && ! flock -n 9; then
        err "another update is already running"; exit 1
    fi
fi

# --------------------------------------------------------------------------- rebuild

answers() {   # the web UI is up and its dashboard draws (older versions ignore ?full=1)
    curl -fsS "http://127.0.0.1:$PORT/healthz?full=1" >/dev/null 2>&1
}

restarts() { docker inspect -f '{{.RestartCount}}' "$CONTAINER" 2>/dev/null || echo 0; }

healthy() {   # wait up to 90 s for the web UI, then make sure it stays up; true if so (or curl is missing)
    local up=0 before
    have curl || return 0
    info "Waiting for the web UI"
    for _ in $(seq 1 90); do
        answers && { up=1; break; }
        sleep 1
    done
    [ "$up" -eq 1 ] || return 1
    # a version that answers once and then crashes would otherwise count as working
    before=$(restarts)
    sleep 15
    [ "$(restarts)" = "$before" ] && answers
}

# Before anything changes, keep what is needed to get back: the image GPU Shepherd is running now
# (tagged gpu-shepherd:previous, so going back needs no rebuild and no network) and a copy of the
# settings, instances and model lists (small YAML files; model files and app data are not copied).
save_recovery() {
    [ "${RECOVERY_SAVED:-0}" -eq 0 ] || return 0
    RECOVERY_SAVED=1
    local img name stamp
    write_status updating "Keeping the running version and backing up the settings" backup
    mkdir -p "$BACKUPS"
    img=$(docker inspect -f '{{.Image}}' "$CONTAINER" 2>/dev/null || true)
    name=$(docker inspect -f '{{.Config.Image}}' "$CONTAINER" 2>/dev/null || true)
    if [ -n "$img" ] && [ -n "$name" ] && docker tag "$img" "$SAVED_IMAGE" 2>/dev/null; then
        echo "$name $(g rev-parse HEAD)" >"$SAVED_IMAGE_FILE"
        info "Kept the running version as $SAVED_IMAGE, to go back to if needed"
    else
        rm -f "$SAVED_IMAGE_FILE"
    fi
    stamp="$(date -u +%Y%m%d-%H%M%S)-$(g rev-parse --short HEAD)"
    if (cd "$DATA_DIR" && tar -czf "$BACKUPS/settings-$stamp.tar.gz" --ignore-failed-read \
            settings.yaml instances catalogs 2>/dev/null); then
        chmod 600 "$BACKUPS/settings-$stamp.tar.gz"
        echo "$BACKUPS/settings-$stamp.tar.gz" >"$STATE_DIR/last-backup"
        info "Backed up your settings to $BACKUPS/settings-$stamp.tar.gz"
        # keep the newest 10
        find "$BACKUPS" -name 'settings-*.tar.gz' -printf '%T@ %p\n' | sort -rn | tail -n +11 | cut -d' ' -f2- \
            | xargs -r rm -f
    else
        warn "could not back up the settings in $DATA_DIR (continuing)"
    fi
}

# Every rebuild leaves the image it replaces behind, untagged. Only the running image and
# gpu-shepherd:previous (for --rollback) are worth keeping, so the older builds are removed.
tidy_images() {
    local freed
    freed=$(docker image prune -f --filter "label=org.opencontainers.image.title=GPU Shepherd" 2>/dev/null \
        | sed -n 's/^Total reclaimed space: *//p')
    if [ -n "$freed" ] && [ "$freed" != "0B" ]; then info "Removed older GPU Shepherd builds ($freed)"; fi
    return 0
}

restore_settings() {   # restore_settings FILE: put the backed-up YAML files back (current ones -> .before-restore)
    local f
    for f in $(tar -tzf "$1" | grep -v '/$'); do
        [ -e "$DATA_DIR/$f" ] && cp -p "$DATA_DIR/$f" "$DATA_DIR/$f.before-restore"
    done
    tar -xzf "$1" -C "$DATA_DIR"
}

start_saved_image() {   # put the image saved by save_recovery back in place and start it
    local name
    read -r name _ <"$SAVED_IMAGE_FILE" 2>/dev/null || return 1
    [ -n "$name" ] && docker image inspect "$SAVED_IMAGE" >/dev/null 2>&1 || return 1
    docker tag "$SAVED_IMAGE" "$name" &&
        (cd "$HERE" && docker compose up -d --no-build --force-recreate "$CONTAINER")
}

# recover OLD_COMMIT REASON: the update failed, so put back the version that worked: its code, its
# image and the settings from just before. Called once; it never starts another update.
recover() {
    local old=$1 reason=$2 code_back=1 how="" backup failed_step=$STEP
    err "$reason"
    say "Going back to the version that was running before"
    write_status updating "The update failed; going back to the version that worked" recover
    if [ -s "$PREV_BRANCH_FILE" ]; then      # a branch switch: back to the branch (and commit) it left
        local pb pc
        read -r pb pc <"$PREV_BRANCH_FILE" || true
        if [ -n "$pb" ] && g checkout --quiet "$pb"; then old=${pc:-$old}; rm -f "$PREV_BRANCH_FILE"
            BRANCH=$pb; UPSTREAM=$(g rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true); fi
    fi
    if [ -n "$old" ] && [ "$old" != "$(g rev-parse HEAD)" ]; then
        g reset --keep "$old" || code_back=0
    fi
    if start_saved_image; then
        how="the saved image"
    elif (cd "$HERE" && docker compose up -d --build "$CONTAINER"); then
        how="a rebuild of the old code"
    fi
    backup=$(cat "$STATE_DIR/last-backup" 2>/dev/null || true)
    if [ -n "$how" ] && [ -n "$old" ] && [ -n "$backup" ] && [ -f "$backup" ]; then
        restore_settings "$backup" || warn "could not restore the settings from $backup"
    fi
    BEHIND=$(g rev-list --count "HEAD..${UPSTREAM:-HEAD}" 2>/dev/null || echo 0)
    STEP=$failed_step                  # the dashboard marks the step that failed
    if [ -n "$how" ] && [ "$code_back" -eq 1 ] && healthy; then
        write_status failed "The update failed ($reason). GPU Shepherd went back to the version that was running before ($(g log -1 --format=%h HEAD)) and is working. Nothing was lost; see the log for what went wrong."
        say "Recovered: GPU Shepherd is running $(g log -1 --format='%h from %cs' HEAD) again, from $how."
        echo "    The update is still available; try again later, or report the error above."
        exit 1
    fi
    write_status failed "The update failed ($reason), and going back automatically did not work either. On the server run: sudo $HERE/update.sh --rollback (see Recovery in the README)."
    err "going back automatically did not work either. Try, in this order:
         sudo $HERE/update.sh --rollback             # previous version, from the saved image
         sudo $HERE/update.sh --restore-settings     # settings from just before the update
         sudo docker compose -f $HERE/compose.yaml logs $CONTAINER
       Your data folder ($DATA_DIR) is never deleted by updates."
    exit 1
}

rebuild() {   # rebuild OLD_COMMIT: rebuild and restart GPU Shepherd, then wait for its web UI
    local old=$1 new
    new=$(g rev-parse HEAD)
    if [ ! -f "$HERE/compose.yaml" ]; then
        write_status "done" "Updated the files. There is no compose.yaml yet, so nothing was rebuilt."
        warn "there is no compose.yaml here, so GPU Shepherd was not rebuilt. Run: sudo ./install.sh"
        return 0
    fi
    write_status updating "Rebuilding GPU Shepherd" rebuild
    if [ -n "$old" ] && [ "$old" != "$new" ] && ! g diff --quiet "$old" "$new" -- compose.example.yaml 2>/dev/null; then
        warn "compose.example.yaml changed. Your compose.yaml is kept as it is; compare them with:
         git -C $HERE diff ${old:0:12} ${new:0:12} -- compose.example.yaml"
    fi
    # Before the rebuild, so that if it ever has to reload systemd, the new container isn't the one
    # that loses its GPUs.
    unlock_service
    say "Rebuilding and restarting GPU Shepherd (your instances keep running)"
    # A failed build leaves the old container running; a new one that doesn't answer is replaced
    # by the old one again. Either way the clone goes back to the old code too.
    (cd "$HERE" && docker compose up -d --build "$CONTAINER") || recover "$old" "the rebuild failed"
    write_status updating "Starting the new version and waiting for the dashboard" start
    healthy || recover "$old" "GPU Shepherd did not start on port $PORT"
    tidy_images
    write_status "done" "GPU Shepherd $(version_at HEAD) is running ($(g log -1 --format='%h, %cs' HEAD))."
    say "GPU Shepherd $(version_at HEAD) is running $(g log -1 --format='%h from %cs: %s' HEAD)"
    if [ -n "$old" ] && [ "$old" != "$new" ]; then
        echo "    Something wrong? Go back to the previous version with: sudo $HERE/update.sh --rollback"
    fi
}

# After pulling, the new update.sh finishes the job, so a fix to the rebuild steps applies at once.
if [ -n "$FINISH" ]; then
    BRANCH=$(g symbolic-ref --short -q HEAD || true)
    UPSTREAM=$(g rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
    BEHIND=0; AHEAD=0; LATEST=$(g rev-parse HEAD)
    rebuild "$FINISH"
    exit 0
fi

# --------------------------------------------------------------------------- rollback

if [ "$ROLLBACK" -eq 1 ]; then
    prev=$(cat "$PREVIOUS" 2>/dev/null || true)
    [ -n "$prev" ] || { err "there is no earlier version to go back to (no update has run here yet)"; exit 1; }
    g cat-file -e "$prev^{commit}" 2>/dev/null || { err "the previous version ($prev) is no longer in this clone"; exit 1; }
    cur=$(g rev-parse HEAD)
    if [ -s "$PREV_BRANCH_FILE" ]; then      # the last change was --branch: go back to the branch it left
        read -r pb _ <"$PREV_BRANCH_FILE" || true
        say "Going back from branch $(g symbolic-ref --short -q HEAD) to branch $pb"
        ask_yes "Go back and rebuild?" || { echo "Nothing was changed."; exit 2; }
        g checkout --quiet "$pb" || { err "could not switch back to $pb: you changed files the other branch also changes"; exit 1; }
        rm -f "$PREV_BRANCH_FILE"
        prev=$(g rev-parse HEAD)
    else
        [ "$cur" != "$prev" ] || { say "Already on the previous version ${prev:0:12}"; exit 0; }
        say "Going back from $(g log -1 --format='%h (%cs)' "$cur") to $(g log -1 --format='%h (%cs)' "$prev")"
        ask_yes "Go back and rebuild?" || { echo "Nothing was changed."; exit 2; }
        # --keep refuses (and changes nothing) if it would overwrite a file you edited.
        g reset --keep "$prev" || { err "could not go back: you changed files the older version also changes"; exit 1; }
        echo "$cur" >"$PREVIOUS"           # rolling back again returns to where we were
    fi
    BRANCH=$(g symbolic-ref --short -q HEAD || true)
    UPSTREAM=$(g rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
    BEHIND=$(g rev-list --count "HEAD..$UPSTREAM" 2>/dev/null || echo 0); AHEAD=0
    LATEST=$(g rev-parse "$UPSTREAM" 2>/dev/null || true)
    saved_commit=$(cut -d' ' -f2 "$SAVED_IMAGE_FILE" 2>/dev/null || true)
    if [ "$saved_commit" = "$prev" ] && start_saved_image && healthy; then
        # the image kept from before the update: no rebuild, works offline and with a broken build
        write_status "done" "Went back to $(g log -1 --format='%h (%cs)' HEAD)."
        say "GPU Shepherd is running $(g log -1 --format='%h from %cs: %s' HEAD) again (the saved image)"
    else
        rebuild ""
    fi
    echo "    Settings from before the update, if you need them: sudo $HERE/update.sh --restore-settings"
    echo "    The next  sudo ./update.sh  updates to the latest version again."
    exit 0
fi

# --------------------------------------------------------------------------- restore settings

if [ "$RESTORE" -eq 1 ]; then
    backup=$(cat "$STATE_DIR/last-backup" 2>/dev/null || true)
    [ -n "$RESTORE_FILE" ] && backup=$RESTORE_FILE
    [ -n "$backup" ] && [ -f "$backup" ] || { err "no settings backup found in $BACKUPS (one is made before every update)"; exit 1; }
    say "Settings backups (newest first):"
    find "$BACKUPS" -name 'settings-*.tar.gz' -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2- | sed 's/^/    /'
    say "Restoring $backup: settings.yaml, instances/ and catalogs/ as they were then"
    echo "    Your current files are kept next to them as *.before-restore. Instances you added since"
    echo "    stay; restart the instances whose settings changed."
    ask_yes "Restore?" || { echo "Nothing was changed."; exit 2; }
    restore_settings "$backup"
    say "Restored. The dashboard shows the restored settings at once."
    exit 0
fi

# --------------------------------------------------------------------------- check

git_err=$(g rev-parse --git-dir 2>&1 >/dev/null) || fail "git can't read $HERE as $OWNER: $git_err"
BRANCH=$(g symbolic-ref --short -q HEAD || true)
[ -n "$BRANCH" ] || fail "this clone is not on a branch (detached HEAD), so there is nothing to follow.
       Switch to one first, e.g.:  git -C $HERE switch main"
UPSTREAM=$(g rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
[ -n "$UPSTREAM" ] || fail "branch $BRANCH does not follow a remote branch. Set one with:
       git -C $HERE branch --set-upstream-to=origin/$BRANCH"
REMOTE=${UPSTREAM%%/*}

# what the dashboard shows while this runs: an update it asked for, or a check
RUN_STATE=checking
[ "$FROM_DASHBOARD" -eq 1 ] && [ "$CHECK" -eq 0 ] && RUN_STATE=updating
BEHIND=$(g rev-list --count "HEAD..$UPSTREAM" 2>/dev/null || echo 0)     # as of the last check
write_status "$RUN_STATE" "Checking for updates" check
info "Checking for updates ($UPSTREAM)"
fetch_err=$(g fetch --quiet "$REMOTE" 2>&1) || {
    if printf '%s' "$fetch_err" | grep -qiE 'authentication|could not read username|terminal prompts disabled|403|401|not found'; then
        if [ -s "$CREDENTIALS" ]; then
            fail "GitHub refused the saved token for $(token_user) ($fetch_err). It may have expired or not cover this repository: save a new one in Settings > Private repository, or with  sudo $HERE/update.sh --set-token <user>"
        fi
        fail "could not download updates from '$REMOTE': the repository needs a login ($fetch_err). For a private repository, save a GitHub access token in Settings > Private repository, or with  sudo $HERE/update.sh --set-token <your GitHub user>"
    fi
    fail "could not download updates from '$REMOTE': ${fetch_err:-no details} (offline?)"
}
LATEST=$(g rev-parse "$UPSTREAM")
BEHIND=$(g rev-list --count "HEAD..$UPSTREAM")
AHEAD=$(g rev-list --count "$UPSTREAM..HEAD")
LOCAL_CHANGES=$(g status --porcelain --untracked-files=no | cut -c4-)

# Follow another branch (--branch, or the dashboard's branch picker): switch the clone to it. The
# rest of the run is the normal update, which rebuilds the image from the new code.
if [ -n "$SWITCH_TO" ] && [ "$SWITCH_TO" != "$BRANCH" ]; then
    case "$SWITCH_TO" in -*|*..*|*//*|*.lock|*@\{*) fail "'$SWITCH_TO' is not a usable branch name" ;; esac
    g show-ref --verify --quiet "refs/remotes/$REMOTE/$SWITCH_TO" || fail "there is no branch '$SWITCH_TO' on '$REMOTE'"
    [ -z "$LOCAL_CHANGES" ] || fail "you changed files in $HERE, so the branch was not switched. Run  sudo ./update.sh --stash  first, or undo them."
    say "Switching from $BRANCH to $SWITCH_TO"
    save_recovery
    write_status updating "Switching to branch $SWITCH_TO" download
    mkdir -p "$STATE_DIR" 2>/dev/null || true
    echo "$BRANCH $(g rev-parse HEAD)" >"$PREV_BRANCH_FILE"
    g rev-parse HEAD >"$PREVIOUS"
    if g show-ref --verify --quiet "refs/heads/$SWITCH_TO"; then
        g checkout --quiet "$SWITCH_TO" && g branch --quiet --set-upstream-to="$REMOTE/$SWITCH_TO" "$SWITCH_TO"
    else
        g checkout --quiet -b "$SWITCH_TO" --track "$REMOTE/$SWITCH_TO"
    fi || { g checkout --quiet "$BRANCH" 2>/dev/null; rm -f "$PREV_BRANCH_FILE"; fail "could not switch to branch $SWITCH_TO. Nothing was changed."; }
    BRANCH=$SWITCH_TO
    UPSTREAM="$REMOTE/$SWITCH_TO"
    LATEST=$(g rev-parse "$UPSTREAM")
    BEHIND=$(g rev-list --count "HEAD..$UPSTREAM")
    AHEAD=$(g rev-list --count "$UPSTREAM..HEAD")
elif [ "$ROLLBACK" -eq 0 ] && [ "$CHECK" -eq 0 ]; then
    rm -f "$PREV_BRANCH_FILE"        # an ordinary update: going back means this version, not a branch
fi

if [ "$BEHIND" -eq 0 ]; then
    if [ "$CHECK" -eq 1 ] || [ "$REBUILD" -eq 0 ]; then
        write_status idle "Up to date."
        info "Up to date: $(version_at HEAD) ($(g log -1 --format='%h from %cs: %s' HEAD))"
        [ "$REBUILD" -eq 0 ] && [ "$CHECK" -eq 0 ] && echo "    (sudo ./update.sh --rebuild rebuilds it anyway)"
        exit 0
    fi
    save_recovery
    rebuild ""
    exit 0
fi

new_version=$(version_at "$UPSTREAM")
if [ -n "$new_version" ] && [ "$new_version" != "$(version_at HEAD)" ]; then
    say "Version $new_version is available (you have $(version_at HEAD)). $BEHIND new change(s) on $UPSTREAM:"
else
    say "$BEHIND new change(s) on $UPSTREAM:"
fi
g log --format='    %h %cs  %s' -n "$MAX_LISTED" "HEAD..$UPSTREAM"
[ "$BEHIND" -le "$MAX_LISTED" ] || echo "    ... and $((BEHIND - MAX_LISTED)) more"
echo "    What changed, version by version:  git -C $HERE show $UPSTREAM:CHANGELOG.md | less"

# the dashboard shows "update available" unless it asked for this update, which carries on below
[ "$RUN_STATE" = updating ] || write_status idle "An update is available."
if [ "$CHECK" -eq 1 ]; then
    echo "    Update with: sudo $HERE/update.sh"
    exit 10
fi

# --------------------------------------------------------------------------- update

if [ "$AHEAD" -gt 0 ]; then
    fail "this clone has $AHEAD commit(s) of its own that $UPSTREAM doesn't have, so it can't simply
       move forward. Combine them yourself (e.g. git -C $HERE pull --rebase), then run this again."
fi
stashed=0
if [ -n "$LOCAL_CHANGES" ]; then
    if [ "$STASH" -eq 0 ]; then
        printf '%s\n' "$LOCAL_CHANGES" | sed 's/^/      /' >&2
        fail "you changed these files in $HERE, and the update could overwrite them. Either
       run  sudo ./update.sh --stash  (they are set aside and put back after the update), or undo
       them with  git -C $HERE checkout -- <file>. Your compose.yaml and data folder are not affected."
    fi
fi

echo
ask_yes "Update now and rebuild GPU Shepherd?" || { echo "Nothing was changed."; exit 2; }

if [ -n "$LOCAL_CHANGES" ]; then
    say "Setting your changes aside"
    g stash push --quiet -m "update.sh $(date -u +%Y-%m-%dT%H:%M:%SZ)" || fail "could not set your changes aside"
    stashed=1
fi

OLD=$(g rev-parse HEAD)
save_recovery
write_status updating "Downloading the new version" download
if ! g merge --ff-only --quiet "$UPSTREAM"; then
    [ "$stashed" -eq 0 ] || g stash pop --quiet || true
    fail "could not move to $UPSTREAM (see above). Nothing was changed."
fi
mkdir -p "$STATE_DIR" 2>/dev/null && echo "$OLD" >"$PREVIOUS" || true
say "Updated from $(g log -1 --format=%h "$OLD") to $(g log -1 --format='%h (%cs)' HEAD)"

if [ "$stashed" -eq 1 ]; then
    if g stash pop --quiet; then
        say "Put your changes back"
    else
        warn "your changes clash with the update. They are kept in git's stash: see them with
         git -C $HERE stash show -p    and drop them with    git -C $HERE stash drop"
        g reset --hard --quiet HEAD       # the clash only; your changes are still in the stash
    fi
fi

# Hand over to the update.sh we just downloaded for the rest. It does the rebuild and any going
# back, so if it can't even be read, go back to the old code now (nothing was rebuilt yet).
if ! bash -n "$HERE/update.sh" 2>/dev/null || ! grep -q -- '--finish)' "$HERE/update.sh"; then
    g reset --keep "$OLD" || true
    fail "the new update.sh is broken, so the update was undone and nothing was rebuilt. GPU Shepherd keeps running $(g log -1 --format=%h HEAD); try again later."
fi
next=(--finish "$OLD")
[ "$QUIET" -eq 0 ] || next+=(--quiet)
exec "$HERE/update.sh" "${next[@]}"
