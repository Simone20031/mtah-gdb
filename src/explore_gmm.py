"""Esplorazione non supervisionata: le feature formano gruppi sull'intera partita?
Nessuna etichetta nel fit. Il GT pubblico e' usato solo alla fine come controllo."""
import sys
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
sys.path.insert(0, str(Path(__file__).parent))
from parse_atd import graded_mask
from features import compute

spread, speed = compute()
g = graded_mask()
ok = g & ~np.isnan(spread) & ~np.isnan(speed)
X = np.column_stack([np.log1p(spread[ok]), np.log(speed[ok] + 0.05)])
print("frame usati:", ok.sum(), "di", g.sum())

for k in (2, 3, 4):
    gm = GaussianMixture(k, random_state=0, n_init=3).fit(X)
    print(f"\nK={k}  BIC={gm.bic(X):.0f}")
    order = np.argsort(gm.means_[:, 1])
    for c in order:
        m_ball = np.expm1(gm.means_[c, 0])
        m_spd = np.exp(gm.means_[c, 1]) - 0.05
        print(f"  comp {c}: peso {gm.weights_[c]:.2f}  palla {m_ball:5.1f} m  giocatori {m_spd:4.2f} m/s")

# controllo con il GT pubblico (non tuning)
GT = Path.home() / "more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public.csv"
lab = np.loadtxt(GT, delimiter=",", skiprows=1, dtype=str)[:, 1]
gm = GaussianMixture(3, random_state=0, n_init=3).fit(X)
comp = np.full(len(spread), -1)
comp[ok] = gm.predict(X)
f = np.arange(11185, 18685)
print("\nK=3: componente x etichetta GT, finestra pubblica (controllo)")
print(f"{'':8}{'H':>7}{'A':>7}{'D':>7}")
for c in range(3):
    print(f"comp {c}: " + "".join(f"{((comp[f] == c) & (lab[f] == k)).sum():>7}" for k in "HAD"))
