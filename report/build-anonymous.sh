#!/usr/bin/env bash
# Anonymous submission: report/submission/mango-anonymous.zip (git-ignored), holding
#   report-anonymous.pdf  the review-mode report: no names, UUNs or repository link
#   code/                 the repository at HEAD without its git history, names and accounts replaced
#   README.txt            what was hidden and why
# It packs the last commit, so commit first. Needs tectonic or latexmk, pdftotext and zip.
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
PAPER=$ROOT/report/paper
OUT=$ROOT/report/submission
IDS='Beyzaei|Halvachi|benyaminbeyzaei|s2865916|s2984817|Benyamin|Paniz'

[ -z "$(git -C "$ROOT" status --porcelain)" ] || echo "Warning: uncommitted changes are not in the bundle." >&2

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
stage=$tmp/mango-anonymous
mkdir -p "$stage" "$tmp/tex"

# 1. The anonymous report: report.tex on its own stays anonymous.
cp -r "$PAPER"/. "$tmp/tex"
(cd "$tmp/tex"
 if command -v tectonic >/dev/null; then tectonic --chatter minimal report.tex
 elif command -v latexmk >/dev/null; then latexmk -pdf -quiet report.tex
 else echo "Install tectonic or latexmk first." >&2; exit 1; fi)
cp "$tmp/tex/report.pdf" "$stage/report-anonymous.pdf"

# 2. The code at HEAD, without history and without the named report.
git -C "$ROOT" archive --prefix=code/ HEAD -- . ':(exclude)report/paper/report-named.*' | tar -x -C "$stage"
grep -rlIE "$IDS" "$stage/code" | while read -r f; do
  sed -i -E 's/Beyzaei, Benyamin and Halvachi, Paniz/Anonymous/g; s/Benyamin Beyzaei|Paniz Halvachi/Anonymous/g;
             s/benyaminbeyzaei/anonymous/g; s/s2865916|s2984817/sXXXXXXX/g; s/Benyamin|Paniz/the author/g' "$f"
done

cat > "$stage/README.txt" <<'TXT'
Mango: anonymous submission for INFR11269 Machine Learning Systems.

report-anonymous.pdf is the report in review mode. code/ is the repository without its git
history. Author names, UUNs, the GitHub account and the repository link are hidden or replaced
with "Anonymous" in both, for anonymous review.
TXT

# 3. Nothing identifying may remain, in the files or in the PDF's text.
if grep -rIE "$IDS" "$stage" || pdftotext "$stage/report-anonymous.pdf" - | grep -E "$IDS"; then
  echo "Identifying text left in the bundle (above); nothing written." >&2; exit 1
fi

mkdir -p "$OUT"; rm -f "$OUT/mango-anonymous.zip"
(cd "$tmp" && zip -qr "$OUT/mango-anonymous.zip" mango-anonymous)
echo "Wrote $OUT/mango-anonymous.zip ($(du -h "$OUT/mango-anonymous.zip" | cut -f1))"
