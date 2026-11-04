"""Legge il raw ATD (tactical) in array numpy e li salva in cache (data/cache, gitignored).

Formato riga: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>"  (palla per ultima, 'NaN' se mancante).
frame_id parte da 1, la submission da 0.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data"))
TAC = DATA / "tactical"
CACHE = DATA / "cache" / "atd.npz"
N_FRAMES = 199_319
H1 = (11185, 90651)
H2 = (110808, 190197)


def team_of_slots(meta_path):
    """0 = casa (Barcellona), 1 = ospiti. La prima squadra nel file e' la casa."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    assert len(team_ids) == 2, "attese 2 squadre"
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def parse(raw_path, n_frames=N_FRAMES):
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = None
    with open(raw_path) as f:
        for line in f:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            groups = parts[1].split(";")
            if players is None:
                players = np.full((n_frames, len(groups), 2), np.nan, dtype=np.float32)
            for s, g in enumerate(groups):
                if g[0] != "N":
                    v = g.split(",")
                    players[i, s, 0] = float(v[0])
                    players[i, s, 1] = float(v[1])
            b = parts[2]
            if b[0] != "N":
                v = b.split(",")
                ball[i, 0] = float(v[0])
                ball[i, 1] = float(v[1])
    return ball, players


def load(force=False):
    if CACHE.exists() and not force:
        z = np.load(CACHE)
        return z["ball"], z["players"], z["team"]
    ball, players = parse(TAC / "BAR-ATH_tactical_atd.txt")
    team = team_of_slots(TAC / "BAR-ATH_tactical_atd.xml")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, ball=ball, players=players, team=team)
    return ball, players, team


def graded_mask(n=N_FRAMES):
    f = np.arange(n)
    return ((f >= H1[0]) & (f <= H1[1])) | ((f >= H2[0]) & (f <= H2[1]))


if __name__ == "__main__":
    ball, players, team = load()
    g = graded_mask()
    seen = ~np.isnan(ball[:, 0])
    print("ball", ball.shape, "players", players.shape, "team slots", team.shape)
    print(f"palla visibile (valutati): {seen[g].mean():.3f}  (atteso 0.593)")
    print(f"slot casa/ospiti: {(team == 0).sum()}/{(team == 1).sum()}")
    bx, by = ball[g & seen, 0], ball[g & seen, 1]
    out = (bx < 0) | (bx > 1) | (by < 0) | (by > 1)
    print(f"palla con coordinate fuori 0-1 (frame valutati): {out.sum()} ({out.mean():.3%} delle viste)")
