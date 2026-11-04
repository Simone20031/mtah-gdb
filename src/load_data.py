"""Carica il tracking ATD (tactical) e stampa controlli di coerenza."""
from pathlib import Path
import os
import numpy as np
from kloppy import metrica

DATA = Path(os.environ.get("MTAH_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
TAC = DATA / "tactical"

H1 = (11185, 90651)
H2 = (110808, 190197)


def load_tracking():
    ds = metrica.load_tracking_epts(
        meta_data=str(TAC / "BAR-ATH_tactical_atd.xml"),
        raw_data=str(TAC / "BAR-ATH_tactical_atd.txt"),
    )
    df = ds.to_df()
    df["video_frame"] = df["frame_id"] - 1  # frame_id parte da 1, la submission da 0
    return df.set_index("video_frame")


def graded_mask(index):
    f = np.asarray(index)
    return ((f >= H1[0]) & (f <= H1[1])) | ((f >= H2[0]) & (f <= H2[1]))


if __name__ == "__main__":
    df = load_tracking()
    print("righe:", len(df), "(attese 199319)")
    print("video_frame:", df.index.min(), "..", df.index.max())
    ball = df["ball_x"].notna()
    g = graded_mask(df.index)
    print(f"palla visibile, tutti i frame:  {ball.mean():.3f} (atteso 0.569)")
    print(f"palla visibile, frame valutati: {ball[g].mean():.3f} (atteso 0.593)")
    xcols = [c for c in df.columns if c.endswith("_x") and c != "ball_x"]
    print("slot di tracking:", len(xcols), "(attesi 46)")
    print(f"oggetti tracciati in media (frame valutati): {df.loc[g, xcols].notna().sum(axis=1).mean():.1f} (atteso 20.5)")
