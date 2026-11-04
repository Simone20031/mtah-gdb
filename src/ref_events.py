"""Riferimento di possesso H/A/D da eventi Metrica (sample-data), altra partita rispetto a quella della gara.
Regole (da documentare): vedi docs/NOTES.md. Uso: python src/ref_events.py 2"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "sample-data" / "data"
FPS = 25
H, A, D = 0, 1, 2
OWNING = {"PASS", "SHOT", "SET PIECE", "RECOVERY"}


def is_trigger(r):
    t = r["Type"]
    s = str(r["Subtype"]) if pd.notna(r["Subtype"]) else ""
    if t in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if t == "SHOT":
        return "GOAL" in s or s.endswith("OUT")
    if t == "CHALLENGE":
        return "FAULT" in s
    if t == "BALL LOST":
        return any(k in s for k in ("OFFSIDE", "HAND BALL", "END HALF"))
    return False


def period_changes(records):
    changes, dead = [], True
    for r in records:
        sf, ef = int(r["Start Frame"]), int(r["End Frame"])
        ty = r["Type"]
        team = {"Home": H, "Away": A}.get(r["Team"])
        if ty in OWNING and team is not None:
            dead = False
            changes.append((sf, team))
        elif ty == "BALL LOST" and team is not None:
            if is_trigger(r):
                if not dead:
                    changes.append((max(sf, ef), D))
                dead = True
            elif not dead:
                changes.append((sf, 1 - team))
        elif is_trigger(r) and not dead:
            dead = True
            changes.append((max(sf, ef), D))
    return changes


def fill(changes, f0, f1):
    lab = np.full(f1 - f0 + 1, D, dtype=np.int8)
    changes = sorted(changes, key=lambda c: c[0])
    for i, (f, s) in enumerate(changes):
        nxt = changes[i + 1][0] if i + 1 < len(changes) else f1 + 1
        a, b = max(f, f0) - f0, min(nxt, f1 + 1) - f0
        if b > a:
            lab[a:b] = s
    return lab


def runs(lab, pred):
    out, i, n = [], 0, len(lab)
    while i < n:
        if not pred(lab[i]):
            i += 1
            continue
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        out.append(j - i)
        i = j
    return np.array(out)


def build(game):
    ev = pd.read_csv(EXT / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawEventsData.csv")
    print("valori Team:", ev["Team"].unique().tolist(), "| periodi:", sorted(ev["Period"].unique().tolist()))
    frames, labels, periods = [], [], []
    for p, g in ev.groupby("Period"):
        g = g.sort_values("Start Frame", kind="stable")
        recs = g.to_dict("records")
        f0, f1 = int(g["Start Frame"].min()), int(g["End Frame"].max())
        lab = fill(period_changes(recs), f0, f1)
        frames.append(np.arange(f0, f1 + 1)); labels.append(lab); periods.append(np.full(len(lab), p))
        pos = runs(lab, lambda x: x != D)
        dead = runs(lab, lambda x: x == D)
        print(f"\nperiodo {p}: frame {f0}..{f1} ({len(lab) / FPS / 60:.1f} min)  D {np.mean(lab == D):.1%}")
        print(f"  possessioni {len(pos)}  durata media {pos.mean() / FPS:.1f} s  mediana {np.median(pos) / FPS:.1f} s")
        print(f"  tratti D {len(dead)}  durata media {dead.mean() / FPS:.1f} s  mediana {np.median(dead) / FPS:.1f} s")
    # esempi per controllare il ritardo tra interruzione e ripartenza
    print("\nprimi 6 BALL OUT: (start, end) -> primo evento che ricomincia")
    ev = ev.sort_values("Start Frame", kind="stable").reset_index(drop=True)
    for i in ev.index[ev["Type"] == "BALL OUT"][:6]:
        nxt = ev.loc[i + 1:][ev.loc[i + 1:, "Type"].isin(OWNING)].head(1)
        n = (f"{nxt.iloc[0]['Type']}/{nxt.iloc[0]['Subtype']} @ {int(nxt.iloc[0]['Start Frame'])}" if len(nxt) else "-")
        print(f"  {int(ev.loc[i, 'Start Frame'])}, {int(ev.loc[i, 'End Frame'])} -> {n}")
    out = EXT.parent.parent / f"ref_game{game}.npz"
    np.savez_compressed(out, frame=np.concatenate(frames), label=np.concatenate(labels), period=np.concatenate(periods))
    print("\nsalvato", out)


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
