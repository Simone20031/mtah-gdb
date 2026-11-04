"""Feature per riconoscere palla morta: movimento della palla e velocita' mediana dei giocatori,
su finestre temporali. Nessuna etichetta usata per calcolarle."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from parse_atd import load, graded_mask, N_FRAMES

L, W = 105.3, 68.0
GT = Path.home() / "more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public.csv"


def ball_spread(ball, win=75, minp=25):
    """Quanto si e' mossa la palla (m) nella finestra di +/-1.5 s. NaN se la vediamo troppo poco."""
    rx = pd.Series(ball[:, 0] * L).rolling(win, center=True, min_periods=minp)
    ry = pd.Series(ball[:, 1] * W).rolling(win, center=True, min_periods=minp)
    return np.hypot(rx.max() - rx.min(), ry.max() - ry.min()).to_numpy()


def player_speed(players, lag=25, win=75):
    """Velocita' mediana dei giocatori (m/s): spostamento in 1 s, mediana sugli slot, poi media mobile."""
    d = players[lag:] - players[:-lag]
    step = np.hypot(d[..., 0] * L, d[..., 1] * W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(step, axis=1)
    out = np.full(N_FRAMES, np.nan)
    out[lag // 2: lag // 2 + len(med)] = med
    return pd.Series(out).rolling(win, center=True, min_periods=win // 3).mean().to_numpy()


def compute():
    ball, players, team = load()
    return ball_spread(ball), player_speed(players)


if __name__ == "__main__":
    spread, speed = compute()
    g = graded_mask()
    q = [5, 25, 50, 75, 95]
    print("percentili sui frame valutati (tutta la partita, senza etichette)")
    print("  spostamento palla in 3 s (m):", np.round(np.nanpercentile(spread[g], q), 1))
    print("  velocita' mediana giocatori (m/s):", np.round(np.nanpercentile(speed[g], q), 2))
    print(f"  spostamento palla NaN: {np.isnan(spread[g]).mean():.1%}")

    # solo per capire se le feature separano: mediane per etichetta nella finestra pubblica
    lab = np.loadtxt(GT, delimiter=",", skiprows=1, dtype=str)[:, 1]
    f = np.arange(11185, 18685)
    print("\nmediane per etichetta nei 5 minuti pubblici (controllo, non tuning)")
    for k in "HAD":
        m = lab[f] == k
        print(f"  {k}: palla {np.nanmedian(spread[f][m]):6.1f} m   giocatori {np.nanmedian(speed[f][m]):5.2f} m/s")
