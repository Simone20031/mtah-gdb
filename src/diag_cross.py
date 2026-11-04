"""Trasferibilita' della feature 'velocita' mediana dei giocatori' tra partite (Metrica 1 e 2) e confronto
con la distribuzione dell'ATD della gara (percentili senza etichette)."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from load_metrica_csv import load, EXT

L, W = 105.0, 68.0
ATD_PCT = [0.35, 0.96, 1.36, 2.00, 3.48]      # ATD gara, frame valutati, percentili 5/25/50/75/95 (da features.py)


def speed(players, lag=25, win=75):
    n = len(players)
    d = players[lag:] - players[:-lag]
    step = np.hypot(d[..., 0] * L, d[..., 1] * W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(step, axis=1)
    out = np.full(n, np.nan)
    out[lag // 2: lag // 2 + len(med)] = med
    return pd.Series(out).rolling(win, center=True, min_periods=win // 3).mean().to_numpy()


data = {}
for g in (1, 2):
    frame, ball, players, team = load(g)
    gt = np.loadtxt(EXT / f"gt_game{g}.csv", delimiter=",", skiprows=1, dtype=str)[:, 1]
    spd = speed(players)
    m = np.isin(gt, ["H", "A", "D"]) & ~np.isnan(spd)
    data[g] = (spd[m], gt[m] == "D")
    print(f"partita {g}: percentili 5/25/50/75/95 della velocita': {np.round(np.percentile(spd[m], [5, 25, 50, 75, 95]), 2)}"
          f"  | AUC {roc_auc_score(data[g][1], -data[g][0]):.3f}"
          f"  | mediana vivi {np.median(spd[m][~data[g][1]]):.2f}, D {np.median(spd[m][data[g][1]]):.2f}")
print("ATD gara (valutati):                       ", ATD_PCT, "\n")


def best_threshold(s, dead):
    ts = np.quantile(s, np.linspace(0.02, 0.98, 200))
    bal = [(np.mean(s[dead] < t) + np.mean(s[~dead] >= t)) / 2 for t in ts]
    return ts[int(np.argmax(bal))], max(bal)


for a, b in ((2, 1), (1, 2)):
    t, bal_a = best_threshold(*data[a])
    s, dead = data[b]
    tpr, tnr = np.mean(s[dead] < t), np.mean(s[~dead] >= t)
    print(f"soglia scelta sulla partita {a}: {t:.2f} m/s (balanced acc {bal_a:.3f})  ->  partita {b}: "
          f"D riconosciuti {tpr:.1%}, vivi riconosciuti {tnr:.1%}, balanced acc {(tpr + tnr) / 2:.3f}")
