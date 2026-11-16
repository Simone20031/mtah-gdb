#!/usr/bin/env bash
set -euo pipefail

# run_all.sh - Team gdb: Reproduction script for More than a Hack 2027 Phase 1
# Regenerates outputs/gdb.csv deterministically from the raw data.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=== [1/5] Checking external sample-data ==="
mkdir -p data/external
if [ ! -d "data/external/sample-data" ]; then
    echo "Cloning metrica-sports/sample-data..."
    git clone https://github.com/metrica-sports/sample-data.git data/external/sample-data
    git -C data/external/sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996
else
    echo "Sample data repository already present."
fi

echo "=== [2/5] Building reference labels for external games (1 and 2) ==="
python src/ref_events.py 1
python src/ref_events.py 2

echo "=== [3/5] Preprocessing external games ==="
python src/prep_game.py 1
python src/prep_game.py 2

echo "=== [4/5] Fitting HMM and generating final submission (gdb.csv) ==="
mkdir -p outputs
python src/make_submission.py --out outputs/gdb.csv

echo "=== [5/5] Validation ==="
VALIDATOR="${MTAH_VALIDATOR:-}"
if [ -z "$VALIDATOR" ]; then
    for c in ../more-than-a-hack-2027/evaluation/validate_submission.py \
             ../more-than-a-hack-2027-main/evaluation/validate_submission.py; do
        if [ -f "$c" ]; then VALIDATOR="$c"; break; fi
    done
fi
if [ -n "$VALIDATOR" ] && [ -f "$VALIDATOR" ]; then
    echo "Running official submission validator..."
    python "$VALIDATOR" outputs/gdb.csv
elif [ -f "evaluation/validate_submission.py" ]; then
    python evaluation/validate_submission.py outputs/gdb.csv
else
    echo "Official validator not found. Submission generated at outputs/gdb.csv."
fi

echo "=== Done! Output successfully generated and validated at outputs/gdb.csv ==="
