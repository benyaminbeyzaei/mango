#!/usr/bin/env bash
# Named submission: builds paper/report-named.pdf (authors, named references, no line numbers).
# The PDF is committed. Needs tectonic or latexmk.
set -euo pipefail
PAPER=$(cd "$(dirname "$0")/paper" && pwd)

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp -r "$PAPER"/. "$tmp"
cd "$tmp"
if command -v tectonic >/dev/null; then tectonic --chatter minimal report-named.tex
elif command -v latexmk >/dev/null; then latexmk -pdf -quiet report-named.tex
else echo "Install tectonic or latexmk first." >&2; exit 1; fi
cp report-named.pdf "$PAPER/report-named.pdf"
echo "Wrote $PAPER/report-named.pdf"
