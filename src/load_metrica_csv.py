"""Loader dei CSV Metrica sample-data (partite 1-2) negli stessi array di parse_atd: ball, players, team.
Frame del CSV parte da 1: riga i <-> Frame i+1 (stessa convenzione degli eventi)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
SRC = EXT / "sample-data" / "data"


def _read(path):
    names = pd.read_csv(path, header=None, skiprows=2, nrows=1).iloc[0].tolist()
    k = sum(str(n).startswith("Player") for n in names)
    df = pd.read_csv(path, header=None, skiprows=3, na_values=["NaN"])
    frame = df.iloc[:, 1].to_numpy().astype(int)
    pl = df.iloc[:, 3:3 + 2 * k].to_numpy(dtype=np.float32).reshape(len(df), k, 2)
    ball = df.iloc[:, 3 + 2 * k:5 + 2 * k].to_numpy(dtype=np.float32)
    return frame, pl, ball


def load(game=2):
    cache = EXT / f"tracking_game{game}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["frame"], z["ball"], z["players"], z["team"]
    d = SRC / f"Sample_Game_{game}"
    fh, ph, bh = _read(d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
    fa, pa, ba = _read(d / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
    assert len(fh) == len(fa) and np.all(fh == fa), "frame Home/Away non allineati"
    if not np.all(fh == np.arange(len(fh)) + 1):
        print("ATTENZIONE: i frame non sono contigui da 1; riga i != Frame i+1")
    players = np.concatenate([ph, pa], axis=1)
    team = np.array([0] * ph.shape[1] + [1] * pa.shape[1], dtype=np.int8)
    np.savez_compressed(cache, frame=fh, ball=bh, players=players, team=team)
    return fh, bh, players, team


if __name__ == "__main__":
    frame, ball, players, team = load(2)
    print("righe:", len(frame), "frame", frame[0], "..", frame[-1])
    print("slot giocatori:", players.shape[1], "(casa", int((team == 0).sum()), "/ ospiti", int((team == 1).sum()), ")")
    print(f"palla visibile: {np.mean(~np.isnan(ball[:, 0])):.1%}")
