"""Applica il modello v2 (parametri stimati dalle partite Metrica 1+2) all'ATD della gara e scrive il CSV.
Uso: python src/make_submission.py --out outputs/gdb.csv [--no-extras] [--no-kickoff]"""
import argparse
import sys
from pathlib import Path
import numpy as np
from kloppy import sportscode

sys.path.insert(0, str(Path(__file__).parent))
import model_v2 as m
from parse_atd import load as load_atd, N_FRAMES, TAC, H1, H2, graded_mask

L, W, FPS = 105.3, 68.0, 25
ALPHA = 1 / 5      # unico parametro scelto guardando le partite Metrica 1 e 2
OUT_W = 6.0        # palla fuori dal campo => morta (peso raw, come l'LLR di velocita')
SP_W = 1.0         # prior set piece, peso morbido
SP_PRE, SP_POST = 7, 4
KICKOFF_START = "Kick Off (Start)"


def load_tagging():
    return sportscode.load(str(TAC / "BAR-ATH_tactical_smart_tagging.xml")).to_df()


def set_piece_frames(ev, kickoff):
    """Frame di inizio dei tag SET PIECES: solo i calci d'inizio di tempo (kickoff=True) o tutti gli altri."""
    sp = ev[ev["code"].str.upper() == "SET PIECES"]
    is_ko = sp["Type"].astype(str).str.contains(KICKOFF_START, case=False, regex=False)
    sp = sp[is_ko] if kickoff else sp[~is_ko]
    return np.sort((sp["timestamp"].dt.total_seconds() * FPS).round().astype(int).to_numpy())


def extras(ball, ev):
    seen = ~np.isnan(ball[:, 0])
    out = seen & ((ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1))
    prior = np.zeros(N_FRAMES)
    for f in set_piece_frames(ev, kickoff=False):
        prior[max(0, f - SP_PRE * FPS): min(N_FRAMES, f + SP_POST * FPS + 1)] = 1.0
    return out, prior


def decode_half(e, s, t, logT, kicks=()):
    """Viterbi su [s, t]. Se un calcio d'inizio di tempo (tag Smart Tagging) cade dentro il tratto valutato,
    prima del tag la palla non e' ancora in gioco: quei frame sono D e la decodifica riparte dal tag in stato D
    (il tag inizia durante la sosta, prima del calcio d'inizio vero)."""
    lab = np.empty(t - s + 1, dtype="<U1")
    k = next((int(f) for f in kicks if s < f <= t), None)
    a = s if k is None else k
    if k is not None:
        lab[: k - s] = "D"
    seg = e[a:t + 1].copy()
    if k is not None:
        seg[0, :2] = -1e9                        # al tag la palla e' ancora ferma
    lab[a - s:] = np.array(list("HAD"))[m.viterbi(seg, logT)]
    return lab, (0 if k is None else k - s)


def possessions(lab):
    out, cur, n = [], None, 0
    for l in lab:
        if l == cur:
            n += 1
        else:
            if cur in ("H", "A"):
                out.append(n)
            cur, n = l, 1
    if cur in ("H", "A"):
        out.append(n)
    return np.array(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/gdb.csv")
    ap.add_argument("--no-extras", action="store_true", help="senza palla fuori campo e prior set piece")
    ap.add_argument("--no-kickoff", action="store_true", help="senza la regola del calcio d'inizio di tempo")
    a = ap.parse_args()

    par = m.fit([1, 2])
    ball, players, team = load_atd()
    ev = load_tagging()
    e = m.emissions(ball, players, team, par, L, W)
    if not a.no_extras:
        out, prior = extras(ball, ev)
        e[:, 2] += OUT_W * out + SP_W * prior
        print(f"palla fuori campo: {int(out.sum())} frame | frame nei set piece: {int((prior > 0).sum())}")
    e = e * ALPHA

    kicks = () if a.no_kickoff else set_piece_frames(ev, kickoff=True)
    lab = np.full(N_FRAMES, "X", dtype="<U1")
    for s, t in (H1, H2):
        lab[s:t + 1], n_pre = decode_half(e, s, t, par["logT"], kicks)
        if n_pre:
            print(f"tempo {s}-{t}: {n_pre} frame valutati prima del calcio d'inizio -> D")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(lab))

    g = graded_mask()
    pos = np.concatenate([possessions(lab[s:t + 1]) for s, t in (H1, H2)])
    print("scritto", a.out, {k: int((lab == k).sum()) for k in "HADX"})
    print(f"D sui frame valutati: {np.mean(lab[g] == 'D'):.1%} | possessioni: {len(pos)}, durata media {pos.mean() / FPS:.1f} s")
