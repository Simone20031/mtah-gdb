"""Riproduce baseline_smooth50 con gli array in cache e lo confronta con il file ufficiale."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from parse_atd import load, graded_mask, N_FRAMES

PITCH = (105.3, 68.0)  # metri, da games.py degli organizzatori
OFFICIAL = Path.home() / "more-than-a-hack-2027/evaluation/results/baseline_smooth50.csv"
NOBODY = -1


def nearest_team(ball, players, team):
    out = np.full(N_FRAMES, NOBODY, dtype=np.int8)
    seen = ~np.isnan(ball[:, 0])
    bx = ball[:, 0] * PITCH[0]
    by = ball[:, 1] * PITCH[1]
    dx = players[:, :, 0] * PITCH[0] - bx[:, None]
    dy = players[:, :, 1] * PITCH[1] - by[:, None]
    d2 = dx * dx + dy * dy
    d2 = np.where(np.isnan(d2), np.inf, d2)
    idx = d2.argmin(axis=1)
    ok = seen & np.isfinite(d2.min(axis=1))
    out[ok] = team[idx[ok]]
    return out


def smooth(codes, window):
    h = np.concatenate(([0], np.cumsum(codes == 0)))
    a = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo = np.maximum(0, i - window)
    hi = np.minimum(len(codes), i + window + 1)
    home, away = h[hi] - h[lo], a[hi] - a[lo]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


if __name__ == "__main__":
    ball, players, team = load()
    mine = smooth(nearest_team(ball, players, team), 50)
    ref = np.loadtxt(OFFICIAL, delimiter=",", skiprows=1, dtype=str)[:, 1]
    g = graded_mask()
    print(f"uguali su tutti i frame:   {(mine == ref).mean():.5f}")
    print(f"uguali sui frame valutati: {(mine[g] == ref[g]).mean():.5f}")
    print("mio    :", {k: int((mine == k).sum()) for k in "HAD"})
    print("ufficiale:", {k: int((ref == k).sum()) for k in "HAD"})
