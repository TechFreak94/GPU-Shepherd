#!/usr/bin/env bash
# Turn a folder of coloring pages into a print-ready KDP PDF (8.5x11 in, 300 DPI by default).
#
#   kdp-book.sh <folder of pages> <book.pdf> [options]     # options: kdp_compile.py --help
#
# Runs kdp_compile.py in a throwaway python container (installs Pillow and img2pdf there each
# time), so nothing is installed on this machine. Copy the pages you want into one folder first;
# they are ordered by file name. Page PNGs for review are written next to the PDF.
set -euo pipefail
if [ $# -lt 2 ]; then
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
pages=$(realpath "$1"); book=$(realpath -m "$2"); shift 2
[ -d "$pages" ] || { echo "error: $pages is not a folder" >&2; exit 1; }
mkdir -p "$(dirname "$book")"
tools=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$tools:/tools:ro" -v "$pages:/pages:ro" -v "$(dirname "$book"):/out" \
    python:3.13-slim sh -c '
        pip install -q --disable-pip-version-check --target /tmp/py pillow img2pdf &&
        PYTHONPATH=/tmp/py python /tools/kdp_compile.py /pages -o "/out/$0" "$@"' "$(basename "$book")" "$@"
