"""Esporta il riferimento della partita N nel formato dello scorer e fa il controllo con il tracking.
Prerequisito: python src/ref_events.py N.  Uso: python src/prep_game.py N"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from load_metrica_csv import load, EXT

G = int(sys.argv[1])
frame, ball, players, team = load(G)
n = len(frame)
ref = np.load(EXT / f"ref_game{G}.npz")
idx = ref["frame"] - 1
ok = (idx >= 0) & (idx < n)
gt = np.full(n, "X", dtype="<U1")
gt[idx[ok]] = np.array(list("HAD"))[ref["label"][ok]]
with open(EXT / f"gt_game{G}.csv", "w") as f:
    f.write("frame,label\n")
    f.writelines(f"{i},{l}\n" for i, l in enumerate(gt))
print(f"partita {G}: righe {n}, slot {players.shape[1]}, palla visibile {np.mean(~np.isnan(ball[:, 0])):.1%}")
print("GT scritto:", {k: int((gt == k).sum()) for k in "HADX"})

d = np.hypot(players[:, :, 0] * 105 - (ball[:, 0] * 105)[:, None], players[:, :, 1] * 68 - (ball[:, 1] * 68)[:, None])
d = np.where(np.isnan(d), np.inf, d)
j = d.argmin(axis=1)
dist = d[np.arange(n), j]
near = np.where(np.isfinite(dist), team[j], -1)
m = np.isfinite(dist) & np.isin(gt, ["H", "A"])
print(f"controllo: squadra del piu' vicino = riferimento nel {np.mean(near[m] == (gt[m] == 'A').astype(int)):.1%} dei frame H/A (n={m.sum()})")
print(f"palla visibile nei frame D: {np.mean(np.isfinite(dist[gt == 'D'])):.1%}")
