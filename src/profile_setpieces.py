"""Profilo medio attorno a f0 dei set piece (esclusi i kickoff): quando e' 'fermo'?
Solo Smart Tagging + tracking, nessun GT."""
import sys
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from kloppy import sportscode
sys.path.insert(0, str(Path(__file__).parent))
from parse_atd import graded_mask, N_FRAMES, TAC
from features import compute

spread, speed = compute()
g = graded_mask()
ok = g & ~np.isnan(spread) & ~np.isnan(speed)
X = np.column_stack([np.log1p(spread[ok]), np.log(speed[ok] + 0.05)])
gm = GaussianMixture(3, random_state=0, n_init=3).fit(X)
c = int(np.argmin(gm.means_[:, 0]))
still = np.full(N_FRAMES, np.nan)
still[ok] = (gm.predict(X) == c).astype(float)

ev = sportscode.load(str(TAC / "BAR-ATH_tactical_smart_tagging.xml")).to_df()
sp = ev[ev["code"].str.upper() == "SET PIECES"].copy()
sp = sp[~sp["Type"].astype(str).str.contains("Kick Off", case=False)]
f0 = (sp["timestamp"].dt.total_seconds() * 25).round().astype(int).to_numpy()
half = sp["Half"].astype(str).to_numpy()


def mean_or_nan(a):
    return np.nan if np.all(np.isnan(a)) else np.nanmean(a)


def profile(mask, label):
    print(f"\n{label}  (n={mask.sum()})   baseline fermo = {np.nanmean(still[g]):.1%}")
    print(f"{'offset (s)':>11}{'% fermo':>10}{'giocatori m/s':>16}")
    for off in range(-30, 60, 5):
        vs, vv = [], []
        for x in f0[mask] + off * 25:
            if x < 0 or x + 125 > N_FRAMES:
                continue
            vs.append(mean_or_nan(still[x:x + 125]))
            vv.append(mean_or_nan(speed[x:x + 125]))
        print(f"{off:>11}{np.nanmean(vs):>10.1%}{np.nanmean(vv):>16.2f}")


profile(np.ones(len(f0), dtype=bool), "tutti i set piece")
profile(half == "1st Half", "1T")
profile(half == "2nd Half", "2T")
