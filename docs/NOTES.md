
## 2. Caricamento dati (verificato)
- Tracking ATD letto con kloppy, `video_frame = frame_id - 1`.
- Controlli OK: 199319 righe, palla visibile 56.9% (tutti) / 59.3% (valutati), 46 slot, 20.5 oggetti medi.
- Frame valutati: 1T 11185-90651, 2T 110808-190197.

## 3. Validazione pipeline di lettura
- Baseline smooth50 riprodotto al 100% dai miei array numpy (src/baseline_repro.py): stesse etichette frame per frame del file ufficiale.
- Allineamento indici, assegnazione squadre e scala del campo confermati.
- Osservazione: nel 4.2% dei frame valutati con palla visibile le coordinate sono fuori da 0-1 (palla fuori campo). Da verificare se è un segnale utile per D.
- Correzione: l'"anomalia" del kickoff 2T era un mio confronto sbagliato. Kickoff reale 2T = 113742 (tag 113660-113785, OK); 110808 e' solo l'inizio dell'intervallo valutato. Tempi Smart Tagging validati in entrambi i tempi.
- Aperto: nel 2T ~2934 frame valutati precedono il kickoff reale; non so cosa contenga il GT li'.
