"""Modello v1: evidenza per frame (H/A/D) + decodifica Viterbi con costi di transizione.
Nessuna etichetta usata. Parametri scelti a giudizio, non tarati sul GT pubblico.
Uso: python src/model_v1.py --out outputs/gdb_v1.csv
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from kloppy import sportscode

sys.path.insert(0, str(Path(__file__).parent))
from parse_atd import load, graded_mask, N_FRAMES, TAC, H1, H2
from features import compute

L, W, FPS = 105.3, 68.0, 25

# --- parametri (giudizio, non tarati) ---
CTRL_R = 5.0                      # m: entro questa distanza il controllo ha peso pieno
W_OUT, W_STILL, W_SP = 2.0, 0.8, 1.0
D_BASE = -0.5                     # D parte svantaggiato: serve evidenza
COST_HA, COST_D = 30.0, 20.0      # costo di cambio stato (in unita' di evidenza per frame)
SP_PRE, SP_POST = 7, 4            # s attorno a f0 dei set piece


def nearest_team(ball, players, team):
    bx, by = ball[:, 0] * L, ball[:, 1] * W
    d = np.hypot(players[:, :, 0] * L - bx[:, None], players[:, :, 1] * W - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    idx = d.argmin(axis=1)
    dist = d[np.arange(len(d)), idx]
    return team[idx], dist, np.isfinite(dist)


def still_component(spread, speed):
    g = graded_mask()
    ok = g & ~np.isnan(spread) & ~np.isnan(speed)
    X = np.column_stack([np.log1p(spread[ok]), np.log(speed[ok] + 0.05)])
    gm = GaussianMixture(3, random_state=0, n_init=3).fit(X)
    c = int(np.argmin(gm.means_[:, 0]))
    still = np.zeros(N_FRAMES)
    still[ok] = (gm.predict(X) == c).astype(float)
    return still


def setpiece_prior():
    ev = sportscode.load(str(TAC / "BAR-ATH_tactical_smart_tagging.xml")).to_df()
    sp = ev[ev["code"].str.upper() == "SET PIECES"]
    sp = sp[~sp["Type"].astype(str).str.contains("Kick Off (Start)", case=False, regex=False)]
    f0 = (sp["timestamp"].dt.total_seconds() * FPS).round().astype(int).to_numpy()
    prior = np.zeros(N_FRAMES)
    for f in f0:
        prior[max(0, f - SP_PRE * FPS): min(N_FRAMES, f + SP_POST * FPS + 1)] = 1.0
    return prior


def emissions():
    ball, players, team = load()
    spread, speed = compute()
    t, dist, ok = nearest_team(ball, players, team)
    seen = ~np.isnan(ball[:, 0])
    out = seen & ((ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1))
    strength = np.where(dist <= CTRL_R, 1.0, 0.5)
    live = ok & seen & ~out
    e = np.zeros((N_FRAMES, 3))
    e[:, 0] = np.where(live & (t == 0), strength, 0.0)
    e[:, 1] = np.where(live & (t == 1), strength, 0.0)
    e[:, 2] = D_BASE + W_OUT * out + W_STILL * still_component(spread, speed) + W_SP * setpiece_prior()
    return e


def viterbi(e):
    cost = np.array([[0, COST_HA, COST_D], [COST_HA, 0, COST_D], [COST_D, COST_D, 0]], dtype=float)
    n = len(e)
    delta = e[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for i in range(1, n):
        cand = delta[:, None] - cost
        back[i] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + e[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = delta.argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def main(out_path):
    e = emissions()
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    for a, b in (H1, H2):                       # si decodifica ogni tempo; fuori dal valutato: X
        labels[a:b + 1] = np.array(list("HAD"))[viterbi(e[a:b + 1])]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    g = graded_mask()
    print("scritto", out_path)
    print({k: int((labels == k).sum()) for k in "HADX"},
          f"| D sui frame valutati: {(labels[g] == 'D').mean():.1%}")
    runs = np.flatnonzero(np.diff(labels[g] != "D", prepend=0)) 
    print("blocchi di cambio D/non-D nei frame valutati:", len(runs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/gdb_v1.csv")
    main(p.parse_args().out)
