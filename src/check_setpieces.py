"""Controllo: la componente 'palla ferma' coincide con i set piece dello Smart Tagging?
Nessuna etichetta GT usata. Smart Tagging: secondi dall'inizio del video, frame = round(s*25)."""
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
c = int(np.argmin(gm.means_[:, 0]))          # componente con palla piu' ferma
still = np.zeros(N_FRAMES, dtype=bool)
still[ok] = gm.predict(X) == c
print(f"componente 'ferma': {still[ok].mean():.1%} dei frame con feature disponibili\n")

ev = sportscode.load(str(TAC / "BAR-ATH_tactical_smart_tagging.xml")).to_df()
sp = ev[ev["code"].str.upper() == "SET PIECES"].copy()
sp["f0"] = (sp["timestamp"].dt.total_seconds() * 25).round().astype(int)
sp["f1"] = (sp["end_timestamp"].dt.total_seconds() * 25).round().astype(int)

print("sanity check kickoff (attesi f0 vicino a 11185 e 110808):")
ko = sp[sp["Type"].astype(str).str.contains("Kick Off", case=False)]
print(ko[["Type", "f0", "f1"]].to_string(index=False), "\n")

inside = np.zeros(N_FRAMES, dtype=bool)
print(f"{'tipo':26}{'n':>4}{'durata media (s)':>18}{'% frame fermi':>15}")
for t, grp in sp.groupby(sp["Type"].astype(str)):
    fr, tot = 0, 0
    for _, r in grp.iterrows():
        m = np.zeros(N_FRAMES, dtype=bool)
        m[max(r.f0, 0):min(r.f1, N_FRAMES - 1) + 1] = True
        inside |= m
        m &= ok
        fr += still[m].sum(); tot += m.sum()
    dur = ((grp.f1 - grp.f0) / 25).mean()
    print(f"{t:26}{len(grp):>4}{dur:>18.1f}{(fr / tot if tot else float('nan')):>15.1%}")

print(f"\nbaseline: frame fermi sul totale = {still[ok].mean():.1%}")
print(f"frame fermi dentro un set piece:   {(still & inside).sum() / still.sum():.1%} di tutti i frame fermi")
