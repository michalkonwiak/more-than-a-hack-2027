# External possession experiment v1

This is a new, separate experiment. It does not read the competition match, its
public labels, manual annotations, VLM proposals, or historical model artifacts.
The default competition reproduction remains the unchanged official reference.

## Data and attribution

Source: Bassek, Rein, Weber and Memmert, *An integrated dataset of spatiotemporal
and event data in elite soccer*, Scientific Data 12, 195 (2025),
[paper](https://doi.org/10.1038/s41597-025-04505-y),
[original data](https://doi.org/10.6084/m9.figshare.28196177).
Data owner: Deutsche Fußball Liga; collected by Sportec Solutions. License: CC-BY 4.0.
Downloaded from the [PySport mirror](https://huggingface.co/datasets/pysport/idsse-data)
at revision `a715a38dfbaf5f58e431727c2b78d174101a703c`.

The seven German 2022/23 matches contain positions and provider reference labels.
`BallStatus=0` maps to D. With `BallStatus=1`, possession 1 maps to H, 2 to A;
missing or unknown values map to X. These are provider labels, not a claim of the
same annotation procedure as the competition's corrected-tracking truth.
Coordinates are centred metres; player speeds are converted from km/h to m/s.
Event files are not used. Ball height, results, clock and player identities are
not model features. Player IDs are used only to assemble observations correctly.

Downloaded files, caches, model weights and reports stay under ignored `artifacts/`.
The public protocol contains URLs and fingerprints, not downloaded observations.
All 14 metadata/tracking files are verified against pinned SHA-256 or Git blob IDs.

## Protocol fixed before inspecting labels or scoring

| Purpose | Matches |
|---|---|
| Training | J03WMX, J03WOH, J03WOY |
| Development / parameter selection | J03WN1, J03WPY |
| Final holdout | J03WQQ, J03WR9 |

The executable specification is `protocols/idsse_v1.json`; source fingerprints
are in `protocols/idsse_sources_v1.json`. XML for held-out matches can be downloaded
and checksummed, but the normal workflow parses its labels only after model freeze.
Matches never cross splits. Several matches share Fortuna Düsseldorf, including
both holdout games; this tests new matches, not unseen clubs or leagues.

Each match is evaluated with clean tracking and a fixed artificial ball-dropout
condition: independently masked two-second blocks, with masking probability 0.4.
The mask depends on a fixed seed and match ID, never on labels or prediction errors.
The same masked input is supplied to both candidate and reference. This approximates
missing observations; it does not reproduce all competition tracking failures.

Training uses both conditions, sampled at 5 Hz. Inference and scoring use 25 Hz.
Whole halves are separate contexts; scoring inserts an unknown separator so that
possessions cannot merge through halftime. Naturally unknown reference intervals
are excluded and coverage is reported.

## Model and selection

The model fits live/dead and conditional home/away logistic classifiers on external
training matches only. Input standardisation is fitted on training samples only.
Home/away fitting is antisymmetric under swapping teams. The feature function
accepts positions, player speeds, team assignment, pitch size and half boundaries;
it has no label argument. Tests change all label fields and require identical features.

Live/dead features describe distances to the ball, absolute pitch position, ball
motion, collective player speed and observation coverage. Team features describe
relative distances and team observation counts. Instantaneous values are supplemented
by one- and three-second means within each half. The classifier is decoded with
an exact three-state Viterbi objective.

Nine combinations are specified before evaluation: ridge in `{0.01, 0.1, 1.0}`,
and a shared state-transition cost in `{0, 0.5, 2.0}`. Selection maximises mean
spellF1 across development matches and conditions with equal weights; ties follow
protocol order. No competition public-window score enters this procedure.
The pure logistic optimiser and exact decoder are reused mathematical utilities;
historical fitted weights, heuristic configurations and feature recipes are not reused.

The comparison is fixed nearest-player inference with the organisers' ±50-frame
smoothing. It is applied independently to each external half. It is not selected
from several baselines using the competition's public labels.

After selection, `freeze.json` fixes the model and protocol hashes. Holdout evaluation
checks code, development report and training-cache hashes, evaluates only the frozen
candidate and reference, and refuses a second evaluation or retraining into that run.
These checks protect against accidental workflow changes; they cannot prove that a
person never inspected a file outside the workflow.

## Reproduce

Use the project's locked Python environment. No additional Python package or GPU is required.
The source download is approximately 2.6 GB; XML is parsed in two streaming passes per match.

```bash
.venv/bin/python -m possession.external_data download
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m possession.external_data prepare
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m possession.external_model train
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m possession.external_model test
```

Outputs: `artifacts/external_runs/idsse_v1/development.json`, `selected.json`,
`freeze.json`, and `holdout.json`. Cached features validate source, protocol, code
and archive fingerprints. A changed cache is rejected, not silently reused.

## Interpretation

This is an external pilot, not a private competition score or a qualification ruling.
The source tracking is cleaner, the provider's possession definition can differ,
and the holdout contains only two matches with a shared home club. In particular,
duplicate tracks, unstable IDs and simultaneous player/ball omissions in the target
match are not represented by the artificial ball-dropout condition.

No automatic promotion to a competition submission follows from this experiment.
The previous provenance audit and the history of earlier work remain disclosed.

## Completed run

Selected ridge: **0.01**; transition cost: **0.5**. Development mean spellF1:
**0.4180** for the candidate versus **0.2630** for the fixed reference.

| Holdout | Condition | Candidate spellF1 | Reference spellF1 |
|---|---|---:|---:|
| J03WQQ | clean | 0.5160 | 0.3608 |
| J03WQQ | ball dropout | 0.4470 | 0.2711 |
| J03WR9 | clean | 0.4875 | 0.3152 |
| J03WR9 | ball dropout | 0.3777 | 0.2331 |
| **Mean** | | **0.4571** | **0.2950** |

Full retraining in a separate run directory reproduced byte-identical `selected.json`,
`development.json` and `freeze.json`. It was a determinism check using the unchanged
protocol, not another tuning iteration. Holdout was not re-evaluated during that check.

An ATD adapter converts coordinates to centred metres and preserves speeds in m/s,
as specified in the official DATA.md. The frozen model produced an experimental
competition CSV; no competition labels were read or scored. Two full inference runs,
plus a separate inference process using the independently retrained model, reproduced
identical CSV bytes. These checks establish reproducibility, not target accuracy.

The frozen weights and aggregate reports are included in `models/external_idsse_v1/`.
Run `make candidate` for inference, repeated-byte verification and official format
validation. Output: `artifacts/candidates/external_idsse_v1/submission.csv`.
See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the candidate's technical document.
An isolated source copy with no public GT or feature caches reproduced the same CSV
using the bundled model. The existing locked venv was reused for that check.
The complete `make candidate` target passed, including 151 tests and format validation.

The publication subset retains 48 tests for the model, parser, external-data pipeline
and CSV export. All 48 passed in an isolated copy of the staged source. Full inference
from that copy reproduced the submission SHA-256 above. The earlier count of 151
also included tests for local research tools omitted from this publication.
