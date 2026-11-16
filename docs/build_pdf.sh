#!/usr/bin/env bash
# docs/gdb.md -> docs/gdb.pdf (needs pandoc and Chrome/Chromium)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CHROME="${CHROME:-$(command -v google-chrome || command -v chromium || echo "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")}"
pandoc gdb.md -s --metadata pagetitle="gdb technical report" --css pdf.css --embed-resources --no-highlight -o /tmp/gdb_report.html
"$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/gdb.pdf" "file:///tmp/gdb_report.html"
