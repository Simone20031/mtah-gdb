"""Diagnostica partita 2: visibilita' della palla vs stato, buchi nei frame vivi, D dai soli giocatori."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from load_metrica_csv import load, EXT

L, W = 105.0, 68.0
frame, ball, players, team = load(2)
n = len(frame)
gt = np.loadtxt(EXT / "gt_game2.csv", delimiter=",", skiprows=1, dtype=str)[:, 1]
live, dead = np.isin(gt, ["H", "A"]), gt == "D"
vis = ~np.isnan(ball[:, 0])
print(f"palla visibile: frame vivi {vis[live].mean():.1%} | frame D {vis[dead].mean():.1%}")

miss = live & ~vis
edges = np.flatnonzero(np.diff(np.concatenate(([0], miss.astype(int), [0]))))
runs = (edges[1::2] - edges[0::2]) / 25
if len(runs):
    print(f"buchi di palla dentro il gioco vivo: {len(runs)}  durata s (mediana, p90, p99): "
          f"{np.percentile(runs, 50):.1f}, {np.percentile(runs, 90):.1f}, {np.percentile(runs, 99):.1f}")

lag, win = 25, 75
d = players[lag:] - players[:-lag]
step = np.hypot(d[..., 0] * L, d[..., 1] * W)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    med = np.nanmedian(step, axis=1)
spd = np.full(n, np.nan)
spd[lag // 2: lag // 2 + len(med)] = med
spd = pd.Series(spd).rolling(win, center=True, min_periods=win // 3).mean().to_numpy()

m = np.isin(gt, ["H", "A", "D"]) & ~np.isnan(spd)
print(f"\nvelocita' mediana giocatori (m/s): vivi {np.median(spd[m & live]):.2f} | D {np.median(spd[m & dead]):.2f}")
print(f"AUC (velocita' bassa => D), solo giocatori: {roc_auc_score(dead[m], -spd[m]):.3f}   (0.5 = caso)")
for p in (1, 2):
    pm = m & (np.arange(n) < n // 2 if p == 1 else np.arange(n) >= n // 2)
    print(f"  meta' {p}: {roc_auc_score(dead[pm], -spd[pm]):.3f}")
