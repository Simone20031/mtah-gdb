"""Esplorazione (non tuning): come si relaziona D del GT pubblico con lo stato della palla nel tracking."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from parse_atd import load

GT = Path.home() / "more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public.csv"
LO, HI = 11185, 18684

ball, players, team = load()
gt = np.loadtxt(GT, delimiter=",", skiprows=1, dtype=str)[:, 1]

f = np.arange(LO, HI + 1)
lab = gt[f]
bx, by = ball[f, 0], ball[f, 1]
missing = np.isnan(bx)
out = ~missing & ((bx < 0) | (bx > 1) | (by < 0) | (by > 1))
inside = ~missing & ~out
state = np.where(missing, "assente", np.where(out, "fuori", "in campo"))

print("righe = stato palla nel tracking, colonne = etichetta GT")
print(f"{'':10}{'H':>8}{'A':>8}{'D':>8}")
for s in ("in campo", "fuori", "assente"):
    m = state == s
    print(f"{s:10}" + "".join(f"{(m & (lab == k)).sum():>8}" for k in "HAD"))

# durata dei tratti consecutivi di D nel GT
d = (lab == "D").astype(int)
edges = np.flatnonzero(np.diff(np.concatenate(([0], d, [0]))))
runs = edges[1::2] - edges[0::2]
print("\ntratti di D nel GT (frame):", runs.tolist())

# velocita' della palla nei frame D con palla visibile: e' ferma?
sp = np.hypot(np.diff(bx, prepend=np.nan) * 105.3, np.diff(by, prepend=np.nan) * 68.0) * 25
for k in "HAD":
    m = (lab == k) & inside & ~np.isnan(sp)
    if m.any():
        print(f"velocita' palla mediana (m/s) con GT={k}: {np.median(sp[m]):.2f}  (n={m.sum()})")
