#!/usr/bin/env bash
# GPU Shepherd updater: brings this clone up to date and rebuilds GPU Shepherd. Your settings,
# instances, model lists and passwords live in the data folder, not here, so they are kept, and
# so is your compose.yaml. Running instances keep running while GPU Shepherd restarts.
#
#   sudo ./update.sh              # shows what is new, asks, then updates and rebuilds
#   sudo ./update.sh --check      # only looks: exit 0 up to date, 10 an update is available
#   sudo ./update.sh --rollback   # goes back to the version before the last update
#   sudo ./update.sh --restore-settings   # puts back the settings backed up before it
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
ROLLBACK=0
RESTORE=0
RESTORE_FILE=""
QUIET=0
FROM_DASHBOARD=0
FINISH=""               # internal: the commit we updated from, after re-running the new script
SERVICE=""              # enable / disable
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
  --rebuild                    rebuild and restart GPU Shepherd even if nothing new was downloaded
  --rollback                   go back to the version that ran before the last update (uses the
                               image kept from before it, so no rebuild or network is needed)
  --restore-settings [FILE]    put back the settings, instances and model lists backed up just
                               before the last update (or from FILE); the current ones are kept
                               as *.before-restore
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

while [ $# -gt 0 ]; do
    case "$1" in
        --check)        CHECK=1; shift ;;
        -y|--yes)       ASSUME_YES=1; shift ;;
        --stash)        STASH=1; shift ;;
        --rebuild)      REBUILD=1; shift ;;
        --rollback)     ROLLBACK=1; shift ;;
        --restore-settings) RESTORE=1; shift
                        if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then RESTORE_FILE=$1; shift; fi ;;
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

# Docker is root-equivalent, so this needs root unless you can already use docker.
if [ "$(id -u)" -ne 0 ] && { [ -n "$SERVICE" ] || ! docker info >/dev/null 2>&1; }; then
    have sudo || { err "run this as root"; exit 1; }
    exec sudo -- "$HERE/$(basename "$0")" "${ARGS[@]}"
fi

have git || { err "git is not installed (sudo apt install git)"; exit 1; }
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
g() { "${GIT_AS[@]}" git -C "$HERE" "$@"; }
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

# write_status STATE MESSAGE: what the last check or update found, for the dashboard and for you.
# STATE is one of: idle, updating, done, failed.
write_status() {
    [ -d "$DATA_DIR" ] || return 0
    mkdir -p "$STATE_DIR"
    local tmp="$STATUS.tmp" commits="" sha date subject first=1 changes="" f
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
        printf '{"checked_at":%s,"state":%s,"message":%s,' "$(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)")" \
            "$(json_str "$1")" "$(json_str "$2")"
        printf '"repo":%s,"branch":%s,"upstream":%s,' "$(json_str "$HERE")" "$(json_str "${BRANCH:-}")" \
            "$(json_str "${UPSTREAM:-}")"
        printf '"current":%s,"latest":%s,"previous":%s,"behind":%s,"ahead":%s,' \
            "$(json_str "$(g rev-parse HEAD 2>/dev/null || true)")" "$(json_str "${LATEST:-}")" \
            "$(json_str "$(cat "$PREVIOUS" 2>/dev/null || true)")" "${BEHIND:-0}" "${AHEAD:-0}"
        printf '"local_changes":[%s],"commits":[%s],"dashboard_updates":%s}\n' "$changes" "$commits" \
            "$(service_enabled && echo true || echo false)"
    } >"$tmp"
    mv -f "$tmp" "$STATUS"
}

fail() {   # fail "message": report it here and on the dashboard, then stop
    err "$1"
    write_status failed "$1" || true
    exit 1
}

# --------------------------------------------------------------------------- dashboard service

if [ "$SERVICE" = enable ]; then
    have systemctl || { err "systemd is needed for dashboard updates"; exit 1; }
    [ -d "$DATA_DIR" ] || { err "$DATA_DIR does not exist yet; run install.sh first"; exit 1; }
    mkdir -p "$STATE_DIR"
    cat >"$UNIT_DIR/$UNIT.path" <<EOF
# Written by $HERE/update.sh --enable-dashboard-updates. Remove with --disable-dashboard-updates.
[Unit]
Description=GPU Shepherd: watch for update requests from the dashboard

[Path]
PathExists=$REQUEST

[Install]
WantedBy=multi-user.target
EOF
    cat >"$UNIT_DIR/$UNIT.service" <<EOF
# Written by $HERE/update.sh --enable-dashboard-updates.
[Unit]
Description=GPU Shepherd: check for an update or update, as asked from the dashboard
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=$HERE/update.sh --from-dashboard
EOF
    cat >"$UNIT_DIR/$UNIT-check.service" <<EOF
# Written by $HERE/update.sh --enable-dashboard-updates.
[Unit]
Description=GPU Shepherd: check for an update (nothing is changed)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=$HERE/update.sh --check --quiet
SuccessExitStatus=10
EOF
    cat >"$UNIT_DIR/$UNIT-check.timer" <<EOF
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
    systemctl daemon-reload
    systemctl enable --now "$UNIT.path" "$UNIT-check.timer" >/dev/null
    say "Dashboard updates are on: GPU Shepherd checks for updates daily, and Settings > Updates"
    echo "    (or the notice on the dashboard) can check now or update. Turn it off with:"
    echo "      sudo $HERE/update.sh --disable-dashboard-updates"
    systemctl start --no-block "$UNIT-check.service" || true
    exit 0
elif [ "$SERVICE" = disable ]; then
    if have systemctl; then
        systemctl disable --now "$UNIT.path" "$UNIT-check.timer" >/dev/null 2>&1 || true
        rm -f "$UNIT_DIR/$UNIT.path" "$UNIT_DIR/$UNIT.service" "$UNIT_DIR/$UNIT-check.service" \
              "$UNIT_DIR/$UNIT-check.timer"
        systemctl daemon-reload
    fi
    rm -f "$REQUEST"
    [ -f "$STATUS" ] && sed -i 's/"dashboard_updates":true/"dashboard_updates":false/' "$STATUS"
    say "Dashboard updates are off. Update from here with: sudo $HERE/update.sh"
    exit 0
fi

# A request from the dashboard: "check" or "update". Taken (deleted) first, so systemd doesn't
# start us again for the same request; the output goes to a log the dashboard shows.
if [ "$FROM_DASHBOARD" -eq 1 ]; then
    action=$(head -c 16 "$REQUEST" 2>/dev/null | tr -dc "[:lower:]" || true)
    rm -f "$REQUEST"
    exec >"$STATE_DIR/log.txt" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) requested from the dashboard: ${action:-nothing}"
    case "$action" in
        check)  CHECK=1 ;;
        update) REBUILD=1 ;;
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

healthy() {   # wait up to 90 s for the web UI; true when it answers (or curl is missing)
    have curl || return 0
    info "Waiting for the web UI"
    for _ in $(seq 1 90); do
        curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

# Before anything changes, keep what is needed to get back: the image GPU Shepherd is running now
# (tagged gpu-shepherd:previous, so going back needs no rebuild and no network) and a copy of the
# settings, instances and model lists (small YAML files; model files and app data are not copied).
save_recovery() {
    local img name stamp
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
    local old=$1 reason=$2 code_back=1 how="" backup
    err "$reason"
    say "Going back to the version that was running before"
    write_status updating "The update failed; going back to the version that worked"
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
    write_status updating "Rebuilding GPU Shepherd"
    if [ -n "$old" ] && [ "$old" != "$new" ] && ! g diff --quiet "$old" "$new" -- compose.example.yaml 2>/dev/null; then
        warn "compose.example.yaml changed. Your compose.yaml is kept as it is; compare them with:
         git -C $HERE diff ${old:0:12} ${new:0:12} -- compose.example.yaml"
    fi
    say "Rebuilding and restarting GPU Shepherd (your instances keep running)"
    # A failed build leaves the old container running; a new one that doesn't answer is replaced
    # by the old one again. Either way the clone goes back to the old code too.
    (cd "$HERE" && docker compose up -d --build "$CONTAINER") || recover "$old" "the rebuild failed"
    healthy || recover "$old" "GPU Shepherd did not start on port $PORT"
    write_status "done" "GPU Shepherd is running $(g log -1 --format='%h (%cs)' HEAD)."
    say "GPU Shepherd is running $(g log -1 --format='%h from %cs: %s' HEAD)"
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
    [ "$cur" != "$prev" ] || { say "Already on the previous version ${prev:0:12}"; exit 0; }
    say "Going back from $(g log -1 --format='%h (%cs)' "$cur") to $(g log -1 --format='%h (%cs)' "$prev")"
    ask_yes "Go back and rebuild?" || { echo "Nothing was changed."; exit 2; }
    # --keep refuses (and changes nothing) if it would overwrite a file you edited.
    g reset --keep "$prev" || { err "could not go back: you changed files the older version also changes"; exit 1; }
    echo "$cur" >"$PREVIOUS"           # rolling back again returns to where we were
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

info "Checking for updates ($UPSTREAM)"
fetch_err=$(g fetch --quiet "$REMOTE" 2>&1) || fail "could not download updates from '$REMOTE': ${fetch_err:-no details}
       (offline, or a private repository that needs the clone owner's saved credentials?)"
LATEST=$(g rev-parse "$UPSTREAM")
BEHIND=$(g rev-list --count "HEAD..$UPSTREAM")
AHEAD=$(g rev-list --count "$UPSTREAM..HEAD")
LOCAL_CHANGES=$(g status --porcelain --untracked-files=no | cut -c4-)

if [ "$BEHIND" -eq 0 ]; then
    if [ "$CHECK" -eq 1 ] || [ "$REBUILD" -eq 0 ]; then
        write_status idle "Up to date."
        info "Up to date: $(g log -1 --format='%h from %cs: %s' HEAD)"
        [ "$REBUILD" -eq 0 ] && [ "$CHECK" -eq 0 ] && echo "    (sudo ./update.sh --rebuild rebuilds it anyway)"
        exit 0
    fi
    save_recovery
    rebuild ""
    exit 0
fi

say "$BEHIND new change(s) on $UPSTREAM:"
g log --format='    %h %cs  %s' -n "$MAX_LISTED" "HEAD..$UPSTREAM"
[ "$BEHIND" -le "$MAX_LISTED" ] || echo "    ... and $((BEHIND - MAX_LISTED)) more"

if [ "$CHECK" -eq 1 ]; then
    write_status idle "An update is available."
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
write_status updating "Downloading the new version"
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

# Hand over to the update.sh we just downloaded for the rest.
next=(--finish "$OLD")
[ "$QUIET" -eq 0 ] || next+=(--quiet)
exec "$HERE/update.sh" "${next[@]}"
