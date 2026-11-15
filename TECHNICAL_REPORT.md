# Ball possession from an externally trained temporal model

Candidate: `external_idsse_v1`. This report describes the experimental CSV.
The competition's private performance is unknown. Earlier experiments and their
rule-provenance issues are disclosed in `PROVENANCE_AUDIT.md`.

## Approach

Two regularised logistic classifiers estimate live/dead probability and conditional
home/away probability. Their joint state probabilities feed an exact offline
three-state Viterbi decoder applied separately to each half.

Inputs describe team distances to the ball, collective player movement, ball motion,
absolute pitch position and observation coverage. One- and three-second context means
supplement instantaneous values. Home/away features change sign when teams are swapped;
the classifier enforces that symmetry. Names, clock, score, player identities and
reference labels are not features. Generic numerical optimisation, decoding and I/O
utilities are reused; historical fitted weights and heuristic configurations are not.

## Inputs and label provenance

Competition inference consumes tactical ATD metadata and raw tracking only.
Coordinates are converted from 0–1 to centred metres using supplied pitch dimensions;
player speeds already use m/s. Axis reflection leaves the features unchanged.
Ball speed is derived from coordinate differences within each half. Broadcast data,
video, homography, Smart Tagging, manual labels and VLM proposals are not consumed.
Official half ranges define prediction context; all other frames receive X.

Training data: IDSSE, seven German Bundesliga/2. Bundesliga matches from 2022/23.
Owner DFL; collected by Sportec Solutions; CC-BY 4.0. Attribution:
Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and
event data in elite soccer*, Scientific Data 12, 195,
https://doi.org/10.1038/s41597-025-04505-y.
Original data: https://doi.org/10.6084/m9.figshare.28196177.
Pinned mirror revision and source fingerprints are in `protocols/`.

Reference labels are the provider's BallStatus/BallPossession fields. We did not
manually annotate these games and do not claim identical annotation procedures to
the competition. Inactive ball maps to D; active home/away possession maps to H/A;
unknown attributes map to X. Speeds in this external source use km/h and are converted.

| Role | Match IDs | Timeline frames |
|---|---|---:|
| Training | J03WMX, J03WOH, J03WOY | 425,717 |
| Selection | J03WN1, J03WPY | 287,772 |
| Holdout | J03WQQ, J03WR9 | 289,155 |

All source timeline frames have known reference labels. Training samples the three
training matches at 5 Hz in clean and artificially masked conditions. The competition
game is not a training or parameter-selection example.

The split, three ridge candidates, three transition costs and selection objective
were specified before reading labels or scoring models. Selection averages spellF1
equally over two development matches and two observation conditions. Selected ridge:
0.01; shared transition cost: 0.5. Holdout was evaluated once after model/code/protocol
freeze. Subsequent identical retraining checked reproducibility, without additional
tuning or another holdout evaluation.

## Possession rules

Live/dead and ownership follow externally learned probabilities and geometric evidence,
not manually specified frame boundaries. Temporal decoding can retain a team through
passes or observation gaps when surrounding evidence supports it. There is no semantic
detector for tackles, loose balls or restarts, so ambiguous geometry can cause errors.
There is no hard minimum possession duration or manual correction of output segments.

## External evaluation

The fixed reference is nearest-player inference with ±50-frame smoothing, adapted to
the external coordinate format and applied separately within each half. The unchanged
official metric scores both methods. Artificial dropout hides independently selected
two-second blocks with probability 0.4, using a fixed seed unrelated to labels.
Both methods receive the same observations. This coarse stress setting reflects the
documented missing-ball problem, not all target tracking errors.

| Held-out match | Observations | Candidate spellF1 | Reference spellF1 |
|---|---|---:|---:|
| J03WQQ | Clean | 0.5160 | 0.3608 |
| J03WQQ | Ball dropout | 0.4470 | 0.2711 |
| J03WR9 | Clean | 0.4875 | 0.3152 |
| J03WR9 | Ball dropout | 0.3777 | 0.2331 |
| **Mean across matches and conditions** | | **0.4571** | **0.2950** |

Development mean spellF1 was 0.4180 versus 0.2630. These are external benchmark
results, not public/private competition scores. Competition ground truth was not
used to select or tune this candidate.

## Reproduce

Reference environment: Linux x86_64, CPython 3.12; hash-pinned dependencies.

```bash
uv venv --python 3.12
uv pip sync requirements.lock --require-hashes
make candidate DATA_DIR="/absolute/path/to/competition/dataset"
```

Included weights and aggregate reports are in `models/external_idsse_v1/`.
The command runs tests, repeats full inference, compares CSV bytes and invokes the
official format validator. Output:
`artifacts/candidates/external_idsse_v1/submission.csv`.

CSV SHA-256:
`721c7d7e2d7e7aa4559ecfe8edbf78f386fd56fed0416c0bfd45fdc267a511e3`.

`EXTERNAL_EXPERIMENT.md` documents full external retraining. Retraining produced
identical model JSON, selection report and freeze. A separate inference process
using the retrained model reproduced identical competition CSV bytes.

The same CSV was also reproduced from an isolated source copy with no feature caches
or public ground-truth directory, using the included model and explicit ATD input path.
That check reused the existing locked Python environment; a fresh dependency installation
is not claimed. `make candidate` passed end to end, including 151 tests and official
CSV validation.

## Limitations and prior work

The two holdout games share Fortuna Düsseldorf; they do not establish broad
generalisation. External tracking is cleaner and its possession definition may differ.
Ball dropout does not simulate duplicate tracks, unstable IDs, missing players, camera
errors or correlated failures. Logistic models also have limited ability to represent
nonlinear interactions and distinguish passes, challenges and controlled possession.

Earlier repository work included public-label training and heuristics informed by
public-window diagnostics under an older local rules copy. Those workflows and
artifacts were preserved and retired. This candidate uses newly fitted external-data
parameters and separate selection; earlier history remains disclosed for organiser
review. No decision about the team's eligibility is asserted here.

The publication subset retains 48 tests for the model, parser, external-data pipeline
and CSV export. All 48 passed in an isolated copy of the staged source. Full inference
from that copy reproduced the submission SHA-256 above. The earlier count of 151
also included tests for local research tools omitted from this publication.
