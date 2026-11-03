"""Scrive un CSV di prova con il formato di consegna. Non usa dati del match."""
import csv
import sys

N_FRAMES = 199_319
out = sys.argv[1] if len(sys.argv) > 1 else "outputs/dummy.csv"
labels = "HAD"

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["frame", "label"])
    for i in range(N_FRAMES):
        w.writerow([i, labels[(i // 500) % 3]])

print(f"scritto {out} con {N_FRAMES} righe")
