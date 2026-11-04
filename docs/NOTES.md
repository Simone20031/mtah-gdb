
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

## 9. Banco di prova esterno: metrica-sports/sample-data
- 3 partite Metrica anonimizzate, tracking+eventi sincronizzati. Nessun file di licenza visibile; il README chiede uso responsabile e citazione della fonte. Da citare nel documento tecnico; dati in data/external (gitignored).
- Limiti: tracking corretto (non automatico) e riferimento ricostruito da eventi con regole mie, diverse dal GT della gara.

## 10. Riferimento di possesso da eventi (sample-data, partita 2; src/ref_events.py)
- Regole: il possesso passa alla squadra di un evento PASS/SHOT/SET PIECE/RECOVERY; BALL LOST lo assegna all'altra squadra; D da un evento di interruzione (BALL OUT, FAULT RECEIVED, CHALLENGE con FAULT, tiro in gol o fuori, fuorigioco, fallo di mano, fine tempo) al primo evento che riparte con la palla. Regole mie, non quelle del GT della gara.
- Risultato: 1T 166 possessioni (media 12.8 s), D 21.8%; 2T 153 possessioni, D 32.9%. Coerente col GT pubblico (media 14.2 s, D 19.6%), ma D del 2T probabilmente sovrastimato dove manca l'evento di ripartenza.
- Da fare: controllo indipendente con il tracking (squadra del giocatore piu' vicino alla palla nei frame H/A).

## 11. Partita 2 come banco di prova (src/load_metrica_csv.py, src/check_game2.py)
- Riferimento coerente col tracking: squadra del giocatore piu' vicino = riferimento nell'82.6% dei frame H/A (84.9% entro 5 m).
- Scorer ufficiale su GT esterno: 319 possessioni, rumore spellF1 ~±0.07. Baseline: smooth50 0.426 > smooth125 0.289 > raw 0.098 (stesso ordine della gara).
- Correzione: in questo tracking la palla e' assente nel 97% dei frame D (visibile 59% in totale), quindi l'assenza della palla rivela D. Non e' un banco di prova per il rilevamento di D; vale per la parte H/A e la struttura temporale. Per D servono feature dei soli giocatori.

## 12. Diagnostica partita 2 (src/diag_game2.py)
- Palla visibile: 80.9% nei frame vivi, 2.9% nei D. 98 buchi nel gioco vivo: mediana 0.2 s, ma p90 31 s e p99 68 s; i buchi lunghi sono quasi certamente D non riconosciuti dal riferimento (manca l'evento di ripartenza). Il riferimento ha quindi errori in entrambi i versi.
- Velocita' mediana giocatori (lag 1 s, media su 3 s): vivi 1.64 m/s, D 0.93. AUC 0.83 (0.832 nel 1T, 0.818 nel 2T). Rapporto D/vivo ~0.57, simile al GT pubblico della gara (~0.56), che ha pero' solo 17 possessioni.
- AUC = accordo con il mio riferimento da eventi, non con la verita'.
- Prossimo: stesso procedimento sulla partita 1; stima su una partita, test sull'altra.

## 13. Partita 1 (src/prep_game.py)
- Controllo con il tracking: 81.2% (partita 2: 82.6%). Palla visibile nei D: 9.4% (partita 2: 2.9%) => il D del riferimento e' piu' rumoroso nella partita 1 (manca piu' spesso l'evento di ripartenza). Quota D 38%/28% (1T/2T), possessioni 9.4 s e 13.3 s di media.
- Uso del riferimento: affidabile per la struttura H/A; per D solo come coerenza dei segnali, non come precisione.
