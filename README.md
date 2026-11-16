# Team gdb — More than a Hack 2027, Phase 1 (Ball Possession)

Implementation for Phase 1 of the **More than a Hack 2027** competition (Barça Innovation Hub & Metrica Sports).  
This pipeline infers team ball possession (`H`, `A`, `D`, `X`) for each of the 199,319 video frames of FC Barcelona vs Athletic Club using a Hidden Markov Model (HMM) fitted on external match data (`metrica-sports/sample-data`) with Viterbi decoding.

Match data assets are licensed for this competition and are **not included** in this repository (see `DATA.md` of the challenge).

---

## Requirements & Environment

- **OS**: Linux / WSL2 (Ubuntu 22.04+ recommended)
- **Python**: 3.9+ (tested with Python 3.13 and 3.14)
- **Dependencies**: Listed in `requirements.txt`

### Setup

```bash
# 1. Clone repository
git clone https://github.com/Simone20031/mtah-gdb.git
cd mtah-gdb

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

### Data Placement

Unpack the official match data bundle into `data/` (or point the `MTAH_DATA_DIR` environment variable to its location):

```text
data/
└── tactical/
    ├── BAR-ATH_tactical_atd.txt
    ├── BAR-ATH_tactical_atd.xml
    ├── BAR-ATH_tactical_homography.csv
    └── BAR-ATH_tactical_smart_tagging.xml
```

---

## Reproducing the Submission File (`outputs/gdb.csv`)

To regenerate the official submission file `outputs/gdb.csv` from scratch in a single command:

```bash
bash run_all.sh
```

This automated script:
1. Clones the external reference dataset (`metrica-sports/sample-data` at pinned commit `e706dd506b360d69d9d123d5b8026e7294b13996`).
2. Generates empirical reference ground truths for Games 1 and 2 from synchronized event data (`src/ref_events.py`).
3. Preprocesses and verifies tracking and team arrays (`src/prep_game.py`).
4. Fits the HMM transition dynamics and emission distributions and performs Viterbi decoding on the match tracking (`src/make_submission.py --out outputs/gdb.csv`).
5. Validates the submission with the official `validate_submission.py` (199,319 rows, frame order, labels).
   Set `MTAH_VALIDATOR` to its path if the challenge repository is not next to this one.

The pipeline is deterministic. Expected MD5 of `outputs/gdb.csv`: `6d64a1731eb5c6651ff62629bea2c7f6`.

Options of `src/make_submission.py`: `--no-extras` (no ball-out / set-piece dead-ball signals),
`--no-kickoff` (no start-of-half rule).

---

## Repository Structure

```text
mtah-gdb/
├── docs/
│   ├── gdb.md               # Technical report (Deliverable 3)
│   ├── gdb.pdf              # PDF of the technical report (the file uploaded to the form)
│   ├── build_pdf.sh         # gdb.md -> gdb.pdf (pandoc + Chrome)
│   └── NOTES.md             # Experimental logs and design journal
├── outputs/                 # Generated submissions (gitignored)
│   └── gdb.csv              # Official submission file (Deliverable 1)
├── src/
│   ├── parse_atd.py         # Fast ATD streaming parser with numpy caching
│   ├── features.py          # Collective player velocity and ball displacement features
│   ├── ref_events.py        # External reference generator from Metrica event logs
│   ├── prep_game.py         # External sample games preprocessor
│   ├── load_metrica_csv.py  # Metrica sample-data CSV loader
│   ├── model_v2.py          # HMM training, synthetic ball masking, and Viterbi decoding
│   └── make_submission.py   # Full match submission generator with extras
├── README.md                # Reproduction and setup guide (Deliverable 2)
├── requirements.txt         # Pinned Python package dependencies
└── run_all.sh               # Single-command end-to-end reproduction script
```

---

## Deliverables Summary

1. **Submission File**: `outputs/gdb.csv` (199,319 frames, verified with `validate_submission.py`).
2. **Repository**: Self-contained with single-command reproduction via `run_all.sh`.
3. **Technical Document**: `docs/gdb.pdf` (source: `docs/gdb.md`).
