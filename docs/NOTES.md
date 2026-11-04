
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

## 6. Profilo dei set piece (Smart Tagging + tracking, nessun GT)
- Correzione alla nota 5: la finestra di 35 s diluiva il segnale. A blocchi di 5 s attorno a f0, i frame "fermi" sono ~38% a [-5,+5) s contro 14.8% di baseline, con i giocatori al minimo di velocita' (~1.0 m/s); dopo +5 s il gioco riparte (2.2 m/s). Vale in entrambi i tempi.
- Il tag inizia a meta' dello stop (kickoff reale ~3.4 s dopo f0). Stima grezza del dead ball: da f0-7 s a f0+4 s.
- I set piece coprono ~8% dei frame valutati; il resto dei D (falli, ecc.) va rilevato con le feature di movimento.

## 7. Modello v1 (src/model_v1.py)
- Evidenza per frame (squadra del giocatore piu' vicino alla palla; D da palla fuori campo, componente "ferma" GMM, prior set piece [-7,+4] s) + Viterbi con costi di cambio (HA 30, D 20).
- Parametri scelti a giudizio, non tarati sul GT pubblico. Prima versione: serve per avere un CSV valido e un punto di confronto.

## 8. Risultato v1 sul GT pubblico (solo controllo, non tuning)
- macroF1 0.719, spellF1 0.439 (rumore ~±0.35: indistinguibile da smooth125), 14 possessioni vs 17, durata media 19.8 s vs 14.2 s, transF1 0.29.
- D: precisione 0.98, recall 0.39. Debolezze: cambi di possesso persi, D di tipo non-rimessa.
- Limite: 17 possessioni non permettono di confrontare versioni. Serve un'altra partita con riferimenti.
