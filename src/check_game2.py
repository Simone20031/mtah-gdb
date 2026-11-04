"""Partita 2: (1) controllo indipendente del riferimento con il tracking, (2) GT per lo scorer, (3) tre baseline."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from load_metrica_csv import load, EXT, ROOT

L, W = 105.0, 68.0
frame, ball, players, team = load(2)
n = len(frame)
ref = np.load(EXT / "ref_game2.npz")

# GT nel formato dello scorer: riga i <-> Frame i+1, X dove il riferimento non c'e'
idx = ref["frame"] - 1
ok_idx = (idx >= 0) & (idx < n)
gt = np.full(n, "X", dtype="<U1")
gt[idx[ok_idx]] = np.array(list("HAD"))[ref["label"][ok_idx]]
with open(EXT / "gt_game2.csv", "w") as f:
    f.write("frame,label\n")
    f.writelines(f"{i},{l}\n" for i, l in enumerate(gt))
print("GT scritto:", {k: int((gt == k).sum()) for k in "HADX"})

# squadra del giocatore piu' vicino alla palla
dx = players[:, :, 0] * L - (ball[:, 0] * L)[:, None]
dy = players[:, :, 1] * W - (ball[:, 1] * W)[:, None]
d = np.hypot(dx, dy)
d = np.where(np.isnan(d), np.inf, d)
j = d.argmin(axis=1)
dist = d[np.arange(n), j]
near = np.where(np.isfinite(dist), team[j], -1)

has = np.isfinite(dist)
for name, m in (("tutti", np.ones(n, bool)), ("entro 5 m dalla palla", dist <= 5), ("oltre 5 m", dist > 5)):
    mm = m & has & np.isin(gt, ["H", "A"])
    ref_t = (gt[mm] == "A").astype(int)
    print(f"[controllo indipendente] frame H/A con palla vista, {name}: n={mm.sum()}, "
          f"squadra del piu' vicino = riferimento: {np.mean(near[mm] == ref_t):.1%}")
print(f"palla visibile nei frame D del riferimento: {np.mean(has[gt == 'D']):.1%}")


def smooth(codes, w):
    if w <= 0:
        return np.where(codes == 0, "H", np.where(codes == 1, "A", "D"))
    h = np.concatenate(([0], np.cumsum(codes == 0)))
    a = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - w), np.minimum(len(codes), i + w + 1)
    home, away = h[hi] - h[lo], a[hi] - a[lo]
    lab = np.full(len(codes), "D", dtype="<U1")
    s = (home + away) > 0
    lab[s] = np.where(home[s] >= away[s], "H", "A")
    return lab


out = ROOT / "outputs" / "g2_results"
out.mkdir(parents=True, exist_ok=True)
for name, w in (("baseline_raw", 0), ("baseline_smooth50", 50), ("baseline_smooth125", 125)):
    lab = smooth(near, w)
    with open(out / f"{name}.csv", "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(lab))
print("baseline scritti in", out)
