# Team gdb — Technical report

**More than a Hack 2027 · Phase 1: ball possession** · FC Barcelona vs Athletic Club\
Submission: `gdb.csv` · Code: <https://github.com/Simone20031/mtah-gdb>

## 1. Approach in one paragraph

We treat possession as a three-state sequence (`H`, `A`, `D`) and decode it with a hidden Markov
model (HMM) and the Viterbi algorithm, separately for each half. Per-frame evidence comes from the
tactical automatic tracking (ATD): the collective speed of the players says whether the ball is
alive or dead, and the team of the player nearest to the ball says which side has it. Two extra
signals push towards `D`: the ball outside the pitch, and a window around each Smart Tagging set
piece. **Every parameter of the HMM is estimated on two other matches** (Metrica's public
`sample-data`), never on labels of the competition match. Ungraded frames are `X`; graded frames
are always `H`, `A` or `D`.

**Why this rather than something else.** The metric counts whole possessions, so the main risk is
a system that flickers. Majority-vote smoothing (the provided baseline) reduces flicker but blurs
every boundary symmetrically and has no notion of a dead ball beyond "the ball is not tracked". An
HMM makes persistence explicit (a change of state has a cost, learned from real matches), lets
evidence accumulate over time instead of voting in a fixed window, and simply holds the current
state when the ball is not visible — which is 41% of graded frames.

## 2. Assets used and dropped

| asset | used? | how |
|---|---|---|
| tactical ATD (players + ball) | **yes** | player speeds; nearest player to the ball and its team; ball outside [0,1]² |
| tactical Smart Tagging | **yes, only `SET PIECES`** | start time of each set piece → dead-ball prior; `Kick Off (Start)` → start of play in each half |
| Metrica `sample-data`, games 1–2 (other matches) | **yes** | to fit all HMM parameters and to choose the one tempering constant (Sec. 4) |
| other Smart Tagging codes (phases, team tags) | no | phases are not possessions; not enough time to validate them as evidence |
| homography | no | the ATD is already in pitch coordinates; we never go back to pixels |
| tactical and broadcast video | no | the ATD already summarises them; the broadcast is not synchronised to the tactical timeline |
| broadcast ATD | no | own timeline with cuts and replays, no mapping to tactical frames |

## 3. The model

**Features (ATD, 25 fps, pitch 105.3 × 68.0 m).**

- *Collective speed* `v_t`: for every track slot, displacement over 1 s; median over slots; centred
  rolling mean over 3 s. Used as `x_t = log(v_t + 0.05)`.
- *Nearest player*: Euclidean distance in metres from the ball to every tracked player; the team of
  the closest one and the distance `d_t`. Undefined when the ball is not tracked.
- *Ball out*: ball tracked and outside [0,1] on either axis.
- *Set-piece window*: frames in [f₀ − 7 s, f₀ + 4 s] around the start f₀ of each `SET PIECES` tag
  other than `Kick Off (Start)` (57 tags). The window comes from profiling player speed around the
  tags on unlabelled tracking: players are at their slowest in that interval and play resumes
  after about +4 s.

**Emissions** (log-scores per frame, columns `H`, `A`, `D`).

- `D` gets the log-likelihood ratio of `x_t` under two Gaussians, dead vs alive, clipped to ±3.
- When the ball is tracked, the team that is *not* nearest gets a penalty
  `log(p / (1 − p))`, with `p` = P(nearest team = team in possession), estimated separately for
  `d_t ≤ 5 m` and `> 5 m`. The penalty never rewards `H`/`A` over `D`, so the mere presence of a
  tracked ball does not reveal whether play is alive.
- Extras on `D`: `+6` while the ball is out, `+1` inside a set-piece window. These two weights
  were set by judgement and not tuned.
- All emissions are multiplied by **α = 1/5** (Sec. 4) before decoding: consecutive frames are far
  from independent, and untempered evidence over-segments.

**Transitions.** The 3×3 transition matrix is the frame-to-frame transition count of the
reference labels of the two external games (add-one smoothed).

**Fitted values** (games 1 + 2, output of `model_v2.fit([1, 2])`):

| | value |
|---|---|
| log-speed, dead | mean −0.107, sd 0.484 |
| log-speed, alive | mean 0.559, sd 0.481 |
| P(nearest = possessor), ball within 5 m | 0.844 → penalty 1.69 |
| P(nearest = possessor), beyond 5 m | 0.582 → penalty 0.33 |
| P(H→H), P(H→A), P(H→D) | 0.99691, 0.00235, 0.00074 |
| P(A→H), P(A→A), P(A→D) | 0.00272, 0.99628, 0.00100 |
| P(D→H), P(D→A), P(D→D) | 0.00090, 0.00103, 0.99807 |

**Decoding.** Viterbi runs independently on each graded half. If a `Kick Off (Start)` tag falls
inside a graded half, every frame before it is `D` and decoding starts from the tag in state `D`
(the tag starts during the stoppage that precedes the kick-off). This affects the second half only,
where the graded interval starts before the kick-off.

## 4. Training data and the one tuned constant

**External data.** Metrica Sports `sample-data` (public GitHub repository, pinned commit
`e706dd5`), games 1 and 2: anonymised professional matches with synchronised tracking and event
data. The event data was produced by Metrica; we did not label anything ourselves. **No data,
label or statistic of the competition match was used to fit or choose anything.**

**From events to frame labels** (`src/ref_events.py`, our rules): possession passes to the team of
any `PASS`, `SHOT`, `SET PIECE` or `RECOVERY`; a `BALL LOST` gives it to the other team; the ball
is dead from a stoppage event (`BALL OUT`, `FAULT RECEIVED`, a `CHALLENGE` with a foul, a shot into
goal or out, offside, hand ball, end of half) until the next event that restarts play.
Independent check against the tracking: the team of the player nearest to the ball matches this
reference on 82.6% of alive frames of game 2 (84.9% within 5 m) and 81.2% of game 1.

**Choosing α.** α is the only constant chosen by looking at scores, always on the external games
(fit on one, score on the other with the official scorer). With the ball as tracked in those
games:

| α | spellF1, fit 1 → test 2 | possessions (truth 319) | spellF1, fit 2 → test 1 |
|---|---|---|---|
| 1 | 0.157 | 1088 | 0.179 |
| 1/2 | 0.284 | 570 | 0.277 |
| 1/3 | 0.385 | 388 | 0.387 |
| 1/4 | 0.444 | 304 | 0.459 |
| **1/5** | **0.448** | 272 | **0.483** |
| 1/7 | 0.444 | 214 | 0.489 |
| 1/10 | 0.422 | 164 | 0.428 |
| baseline smooth50 | 0.426 | 464 | 0.439 |

The curve is flat between 1/4 and 1/7 (differences well inside the ±0.07 noise of these
matches); we kept 1/5. Because the external tracking is corrected and the ball is visible far more
often than in the competition ATD, we repeated the comparison after removing the ball in stretches
whose lengths are sampled from the run lengths of ball visibility in the competition ATD (no labels
involved), bringing ball visibility on alive frames down to 54–57%. There α = 1/5 scored 0.368 vs
0.325 for smooth50 (test on game 2) and 0.362 vs 0.314 (test on game 1): the same direction, but
within noise.

## 5. Possession rules as implemented

- **Change of possession.** It requires sustained evidence. With the fitted values, the opposing
  team must be nearest to the ball, within 5 m, for about 18 consecutive tracked frames (~0.7 s)
  before switching pays off; beyond 5 m it takes about 90 frames. A touch, a deflection or a lost
  aerial duel does not last that long and does not change possession.
- **Loose and contested balls.** Possession stays with the last team, through the transition
  inertia, unless the evidence above builds up.
- **Ball not tracked.** No `H`/`A` evidence at all: the current state is kept. We never write `X`
  on graded frames, since on the ranking metric `X` means "repeat the previous label" anyway.
- **Dead ball.** Collective slowdown of the players, a ball outside the pitch, or a set-piece
  window. Entering `D` needs about 12 frames of maximal slowdown evidence, or a handful of frames
  with the ball out of the pitch.
- **Restarts.** The team that restarts is the one the nearest-player evidence points to after the
  `D` stretch; set-piece team tags are not used.
- **Start of each half.** `D` until the `Kick Off (Start)` tag, then decoding from `D`.

## 6. Checks on the public 5 minutes

The public ground truth was used only to verify the pipeline: frame alignment, file format, label
meaning (our re-implementation of `baseline_smooth50` reproduces the official file frame by frame).
No parameter, weight or design choice was selected on it. With 17 possessions its noise is roughly
±0.35 spellF1, so the numbers below rank nothing.

| submission | macroF1 | spellF1 |
|---|---|---|
| baseline smooth50 (official) | 0.621 | 0.326 |
| baseline smooth125 (official) | 0.581 | 0.424 |
| our v1 (hand-set parameters, earlier version) | 0.719 | 0.439 |
| **final `gdb.csv`** | **0.825** | **0.400** |

On the full match the final file contains **311** possessions of mean **11.5** s and **43.9**% of
graded frames labelled `D`.

## 7. Limitations and what we would do next

- **Blurred boundaries.** The speed feature is a 3-second centred average, so the start and end of
  dead-ball stretches can be off by one to two seconds, and spellF1 drops sharply with boundary
  error (1 s late ≈ 0.65 on a perfect file). Next: refine `D` boundaries with ball displacement
  when the ball is tracked.
- **Occlusions hide turnovers.** With the ball missing on 41% of graded frames, a change of
  possession that happens out of sight is detected late or missed. Next: team-shape cues (width,
  depth, direction of movement) as ball-free evidence, and the Smart Tagging team tags.
- **Restarting team.** Not taken from the set-piece team tags, which would be free evidence for
  corners, goal kicks and free kicks.
- **Reference mismatch.** Our external labels follow our own event rules on corrected tracking;
  the competition truth is derived from corrected tracking with its own definition of "owning". The
  transition rates and the emission parameters inherit that mismatch.
- **Small external sample.** Two matches; α was chosen on them, so their scores are slightly
  optimistic.

## 8. Reproduction

```bash
git clone https://github.com/Simone20031/mtah-gdb.git && cd mtah-gdb
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned versions; tested with Python 3.13 and 3.14
# competition data in data/tactical/ (or export MTAH_DATA_DIR=...)
bash run_all.sh                          # -> outputs/gdb.csv, then the official validator
```

`run_all.sh` clones `sample-data` at the pinned commit, builds the external reference labels, fits
the model and writes `outputs/gdb.csv`. The pipeline has no random component; the expected MD5 of
the output is **6d64a1731eb5c6651ff62629bea2c7f6**. `src/make_submission.py --no-extras` and `--no-kickoff` switch off the
extra dead-ball signals and the kick-off rule.
