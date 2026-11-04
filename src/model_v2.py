"""Modello v2: HMM H/A/D con parametri stimati dai riferimenti (eventi) di ALTRE partite.
Evidenza D: LLR della velocita' mediana dei giocatori. Evidenza H/A: squadra del piu' vicino alla palla
(penalizza l'altra squadra; non favorisce mai il vivo su D, cosi' la presenza della palla non rivela lo stato).
Ball masking: tratti con lunghezze campionate dall'ATD della gara (senza etichette).
Uso: python src/model_v2.py TRAIN TEST     (es. 1 2)"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_metrica_csv import load as load_metrica, EXT, ROOT

LW = (105.0, 68.0)
CLIP, NEAR_R = 3.0, 5.0
CODE = {"H": 0, "A": 1, "D": 2}


def speed_feature(players, L, W, lag=25, win=75):
    n = len(players)
    d = players[lag:] - players[:-lag]
    step = np.hypot(d[..., 0] * L, d[..., 1] * W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(step, axis=1)
    out = np.full(n, np.nan)
    out[lag // 2: lag // 2 + len(med)] = med
    return pd.Series(out).rolling(win, center=True, min_periods=win // 3).mean().to_numpy()


def nearest(ball, players, team, L, W):
    dx = players[:, :, 0] * L - (ball[:, 0] * L)[:, None]
    dy = players[:, :, 1] * W - (ball[:, 1] * W)[:, None]
    d = np.hypot(dx, dy)
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(axis=1)
    dist = d[np.arange(len(d)), j]
    return np.where(np.isfinite(dist), team[j], -1).astype(int), dist


def gauss_logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)


def load_gt(g):
    gt = np.loadtxt(EXT / f"gt_game{g}.csv", delimiter=",", skiprows=1, dtype=str)[:, 1]
    code = np.full(len(gt), -1, dtype=int)
    for k, v in CODE.items():
        code[gt == k] = v
    return code


def fit(games):
    xd, xl = [], []
    hits = {"near": [0, 0], "far": [0, 0]}
    counts = np.zeros((3, 3))
    for g in games:
        frame, ball, players, team = load_metrica(g)
        code = load_gt(g)
        spd = speed_feature(players, *LW)
        m = (code >= 0) & ~np.isnan(spd)
        x = np.log(spd[m] + 0.05)
        xd.append(x[code[m] == 2]); xl.append(x[code[m] < 2])
        near, dist = nearest(ball, players, team, *LW)
        mm = (code >= 0) & (code < 2) & (near >= 0)
        ok = near[mm] == code[mm]
        for name, sel in (("near", dist[mm] <= NEAR_R), ("far", dist[mm] > NEAR_R)):
            hits[name][0] += int(ok[sel].sum()); hits[name][1] += int(sel.sum())
        a, b = code[:-1], code[1:]
        v = (a >= 0) & (b >= 0)
        np.add.at(counts, (a[v], b[v]), 1)
    xd, xl = np.concatenate(xd), np.concatenate(xl)
    p = {k: hits[k][0] / hits[k][1] for k in hits}
    return {"D": (xd.mean(), xd.std()), "live": (xl.mean(), xl.std()),
            "pen": {k: float(np.log(p[k] / (1 - p[k]))) for k in p}, "p": p,
            "logT": np.log((counts + 1) / (counts + 1).sum(1, keepdims=True))}


def emissions(ball, players, team, par, L=LW[0], W=LW[1]):
    """Evidenza per frame (non tempered). Colonne: H, A, D."""
    spd = speed_feature(players, L, W)
    x = np.log(spd + 0.05)
    llr = gauss_logpdf(x, *par["D"]) - gauss_logpdf(x, *par["live"])
    llr = np.where(np.isnan(llr), 0.0, np.clip(llr, -CLIP, CLIP))
    near, dist = nearest(ball, players, team, L, W)
    pen = np.where(dist <= NEAR_R, par["pen"]["near"], par["pen"]["far"])
    e = np.zeros((len(ball), 3))
    e[:, 0] = np.where(near == 1, -pen, 0.0)
    e[:, 1] = np.where(near == 0, -pen, 0.0)
    e[:, 2] = llr
    return e


def viterbi(e, logT):
    n = len(e)
    delta = e[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for i in range(1, n):
        cand = delta[:, None] + logT
        back[i] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + e[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = delta.argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def segments(g):
    ref = np.load(EXT / f"ref_game{g}.npz")
    return [(int(ref["frame"][ref["period"] == p].min()) - 1, int(ref["frame"][ref["period"] == p].max()) - 1)
            for p in np.unique(ref["period"])]


def decode(e, segs, logT):
    lab = np.full(len(e), "X", dtype="<U1")
    for a, b in segs:
        lab[a:b + 1] = np.array(list("HAD"))[viterbi(e[a:b + 1], logT)]
    return lab


def atd_run_lengths():
    """Lunghezze dei tratti con/senza palla nell'ATD della gara (frame valutati, nessuna etichetta)."""
    from parse_atd import load, H1, H2
    ball, _, _ = load()
    vis_runs, miss_runs = [], []
    for a, b in (H1, H2):
        v = ~np.isnan(ball[a:b + 1, 0])
        for s in np.split(v, np.flatnonzero(np.diff(v.astype(int))) + 1):
            (vis_runs if s[0] else miss_runs).append(len(s))
    return np.array(vis_runs), np.array(miss_runs)


def synth_mask(n, vis_runs, miss_runs, rng):
    m, i, vis = np.zeros(n, dtype=bool), 0, True
    while i < n:
        k = int(rng.choice(vis_runs if vis else miss_runs))
        if not vis:
            m[i:i + k] = True
        i += k
        vis = not vis
    return m


def smooth(codes, w):
    h = np.concatenate(([0], np.cumsum(codes == 0)))
    a = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - w), np.minimum(len(codes), i + w + 1)
    home, away = h[hi] - h[lo], a[hi] - a[lo]
    lab = np.full(len(codes), "D", dtype="<U1")
    s = (home + away) > 0
    lab[s] = np.where(home[s] >= away[s], "H", "A")
    return lab


def write(path, lab):
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(lab))


if __name__ == "__main__":
    train, test = int(sys.argv[1]), int(sys.argv[2])
    par = fit([train])
    print(f"addestrato sulla partita {train}")
    print(f"  log-velocita' D: mu {par['D'][0]:.2f} sd {par['D'][1]:.2f} | vivo: mu {par['live'][0]:.2f} sd {par['live'][1]:.2f}")
    print(f"  P(piu' vicino = squadra giusta): entro 5 m {par['p']['near']:.2f}, oltre {par['p']['far']:.2f}")
    print("  logT (righe da H,A,D; colonne a H,A,D):\n", np.round(par["logT"], 2))

    frame, ball, players, team = load_metrica(test)
    vis_r, miss_r = atd_run_lengths()
    mask = synth_mask(len(ball), vis_r, miss_r, np.random.default_rng(0))
    ball_d = ball.copy()
    ball_d[mask] = np.nan
    code = load_gt(test)
    live = (code == 0) | (code == 1)
    print(f"\npartita {test}: palla visibile nei frame vivi, originale {np.mean(~np.isnan(ball[live, 0])):.1%} -> degradata {np.mean(~np.isnan(ball_d[live, 0])):.1%}")

    out = ROOT / "outputs" / f"v2_train{train}_test{test}"
    out.mkdir(parents=True, exist_ok=True)
    segs = segments(test)
    e1 = emissions(ball_d, players, team, par)
    for k in (75, 25, 10, 5):
        lab = decode(e1 / k, segs, par["logT"])
        write(out / f"v2_alpha_1su{k}.csv", lab)
        print(f"  alpha=1/{k}: D {np.mean(lab == 'D'):.1%}")
    near, _ = nearest(ball_d, players, team, *LW)
    write(out / "baseline_smooth50.csv", smooth(near, 50))
    print("scritto in", out)
