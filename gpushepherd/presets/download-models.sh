#!/usr/bin/env bash
# Downloads a preset's model files (listed in its downloads.txt) into your models folder.
#
#   ./download-models.sh starter /srv/models               # the preset's default groups
#   ./download-models.sh starter /srv/models chat         # only some groups
#   ./download-models.sh ~/my-preset /srv/models           # a preset folder of your own
#   ./download-models.sh starter                           # list the preset's groups
#   HF_TOKEN=hf_xxx ./download-models.sh ...               # only if a download says 401/403
#
# Downloads resume where they stopped and files you already have are skipped, so run it again
# after a failure. Run it as yourself, not root, so the files are yours.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_TOKEN="${HF_TOKEN:-}"
FAILED=()

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
    ''|-h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; [ -n "${1:-}" ]; exit $? ;;
esac
preset=$1
case "$preset" in */*) ;; *) preset="$HERE/$preset" ;; esac
list="$preset/downloads.txt"
[ -f "$list" ] || die "no downloads.txt in $preset"

# group <name> <default|optional> <size> <description...>
names=(); defaults=()
declare -A size desc
while read -r kind name when sz rest; do
    [ "$kind" = group ] || continue
    names+=("$name"); size[$name]=$sz; desc[$name]=$rest
    [ "$when" = optional ] || defaults+=("$name")
done < "$list"

if [ $# -lt 2 ]; then
    echo "Groups in $(basename "$preset") (* = downloaded when you name none):"
    for g in "${names[@]}"; do
        mark=" "; case " ${defaults[*]} " in *" $g "*) mark="*" ;; esac
        printf '  %s %-12s %-6s %s\n' "$mark" "$g" "${size[$g]}" "${desc[$g]}"
    done
    exit 0
fi

MODELS="${2%/}"; shift 2
[ -d "$MODELS" ] || die "$MODELS does not exist"
[ -w "$MODELS" ] || die "you can't write to $MODELS (run it as the folder's owner)"
if [ $# -gt 0 ]; then groups=("$@"); else groups=("${defaults[@]}"); fi
for g in "${groups[@]}"; do
    [ -n "${size[$g]+x}" ] || die "unknown group '$g' (groups: ${names[*]})"
done
command -v curl >/dev/null || die "curl is not installed"

# get <dest> <url> [fallback-url ...]: resumable (.part file); returns 1 if every URL failed.
# -g so that [ ] in a file name is not read as a curl pattern.
get() {
    local dest="$1" url auth; shift
    mkdir -p "$(dirname "$dest")"
    if [ -s "$dest" ]; then say "Already have ${dest#"$MODELS"/}"; return 0; fi
    for url in "$@"; do
        auth=()
        if [ -n "$HF_TOKEN" ] && [[ "$url" == https://huggingface.co/* ]]; then
            auth=(-H "Authorization: Bearer $HF_TOKEN")
        fi
        say "Downloading ${dest#"$MODELS"/}"
        if curl -g -fL --retry 5 --retry-delay 5 --retry-connrefused -C - --progress-bar \
                "${auth[@]}" -o "$dest.part" "$url"; then
            mv -f "$dest.part" "$dest"
            return 0
        fi
        warn "failed: $url"
    done
    return 1
}

while read -r kind group dest urls; do
    [ "$kind" = file ] || continue
    case " ${groups[*]} " in *" $group "*) ;; *) continue ;; esac
    case "$dest" in /*|*..*) warn "skipping $dest: it must be a path inside the models folder"; continue ;; esac
    read -ra url_list <<< "$urls"
    get "$MODELS/$dest" "${url_list[@]}" || FAILED+=("$dest")
done < "$list"

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    warn "these could not be downloaded (run this again to retry; a 401/403 needs HF_TOKEN):"
    printf '    %s\n' "${FAILED[@]}" >&2
    exit 1
fi
say "All done: ${groups[*]}"
