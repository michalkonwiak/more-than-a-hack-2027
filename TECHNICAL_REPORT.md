# jamnik team

More Than A Hack 2027

Initial Challenge

Michał Konwiak, Kamil Brzoza, Maciej Zapała

## Method

We estimate possession from player and ball tracking. Two regularised logistic models predict whether the ball is live and, when live, which team has possession. A three-state Viterbi decoder combines these probabilities into a sequence of home possession, away possession and dead ball. We decode each half separately and assign X outside the official half ranges.

The features describe each team's distance to the ball, player movement, ball speed, pitch position and tracking coverage. We add one-second and three-second averages to capture context around each frame. Team features change sign when the teams are exchanged, and the ownership model enforces the same symmetry. Player identities, score and match clock are not predictive features.

Temporal decoding penalises switches between states. It can therefore retain possession through a pass or a tracking gap when the surrounding observations support the same team. We do not detect tackles, loose balls or restarts as separate events, impose a minimum possession duration, or correct individual segments by hand.

## Competition inputs

The submitted model uses tactical ATD tracking and metadata. We convert normalised coordinates to centred metres using the pitch dimensions. ATD player speeds already use metres per second. Ball speed comes from consecutive positions within each half.

Video, broadcast tracking, homography, Smart Tagging and annotation proposals are not inputs to this model. Earlier experiments used other workflows; their provenance is described at the end of this report.

## Training data

We use IDSSE, a dataset of seven Bundesliga and 2. Bundesliga matches from the 2022/23 season. The data belongs to DFL, was collected by Sportec Solutions and is distributed under CC-BY 4.0. The source and pinned download fingerprints are recorded in protocols/.

Labels come from the provider's BallStatus and BallPossession fields. An inactive ball maps to D. Active home and away possession map to H and A. Unknown values map to X. We did not annotate these matches. Their label definitions may differ from those of the competition.

Training uses three matches at 5 Hz. Parameter selection uses two other matches, and the final test uses the remaining two. Inference and evaluation run at 25 Hz. The competition match is not used to fit or select this model.

---

## Experiment

The match split, parameter grid and selection metric were fixed before reading labels or scoring models.

| Split | Match IDs | Frames |
|---|---|---:|
| Training | J03WMX, J03WOH, J03WOY | 425,717 |
| Development | J03WN1, J03WPY | 287,772 |
| Holdout | J03WQQ, J03WR9 | 289,155 |

We train and evaluate with both complete tracking and simulated ball loss. The latter hides two-second blocks of ball positions, each with probability 0.4, using a fixed seed. The labels remain unchanged. Both our model and the reference receive the same observations.

We compare three ridge penalties (0.01, 0.1, 1.0) and three state-transition costs (0, 0.5, 2.0). Selection uses mean spellF1, weighted equally across development matches and conditions. The selected ridge penalty is 0.01 and the transition cost is 0.5. We freeze the model before evaluating the holdout once.

## Results

The reference assigns possession to the nearest player and applies the organisers' ±50-frame smoothing within each half. We use the unchanged official scoring code for both methods.

| Holdout match | Tracking | Our spellF1 | Reference |
|---|---|---:|---:|
| J03WQQ | Complete | 0.5160 | 0.3608 |
| J03WQQ | Ball loss | 0.4470 | 0.2711 |
| J03WR9 | Complete | 0.4875 | 0.3152 |
| J03WR9 | Ball loss | 0.3777 | 0.2331 |
| Mean | Both conditions | 0.4571 | 0.2950 |

These are results on other matches. We do not know the private competition score.

## Limitations

Both holdout matches include Fortuna Düsseldorf, so this test says little about transfer to unseen clubs or leagues. IDSSE tracking is cleaner than the competition tracking. Simulated ball loss does not cover duplicate tracks, unstable identities, missing players or camera errors.

The model also has no explicit notion of controlled possession. A nearby opponent, loose ball or restart can produce similar geometric features. Further evaluation needs additional labelled matches with these tracking errors.

---

## Reproduction

Repository: https://github.com/michalkonwiak/more-than-a-hack-2027

Use Linux x86_64, Python 3.12 and uv. Download the competition dataset separately, then run these commands from the repository root:

```bash
make install
make candidate DATA_DIR="/absolute/path/to/dataset"
```

The repository includes the weights in models/external_idsse_v1/. No external match download or retraining is needed for prediction. The command runs the tests, computes the full prediction twice, compares the CSV bytes and invokes the official format validator.

Output: artifacts/candidates/external_idsse_v1/submission.csv. The submitted copy is named jamnik team.csv and contains 199,319 rows.

Expected SHA-256:

```text
721c7d7e2d7e7aa4559ecfe8edbf78f386fd56fed0416c0bfd45fdc267a511e3
```

All 48 tests included in the published source passed in an isolated copy. Full inference from that copy reproduced the same CSV. A separate retraining run also reproduced identical weights and predictions. These checks reused the installed Python environment; a fresh dependency installation was not tested. Dependencies and hashes are recorded in requirements.lock.

EXTERNAL_EXPERIMENT.md describes how to download the external data, retrain the model and reproduce the evaluation.

## Prior work

Earlier work trained on the public five-minute competition labels and used diagnostics from that window to develop heuristics under an older local rules copy. We retired those workflows after checking the updated rules. A manual annotation interface and local vision-language annotation drafts were also explored. Those drafts were not used to train, select or repair the submission.

The submitted model uses newly fitted external-data parameters. It reuses general optimisation, decoding and data-loading functions, but not the earlier weights or heuristic configurations. We disclose the earlier work in PROVENANCE_AUDIT.md for organiser review. This report does not assert a ruling on team eligibility.

## Data source

Bassek, Rein, Weber and Memmert (2025). An integrated dataset of spatiotemporal and event data in elite soccer. Scientific Data 12, 195.

Paper: https://doi.org/10.1038/s41597-025-04505-y

Dataset: https://doi.org/10.6084/m9.figshare.28196177
