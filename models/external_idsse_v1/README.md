# External IDSSE model v1

This directory contains model parameters and aggregate reports, not match observations
or frame-level labels. Files are identical to the frozen experiment artifacts:

- `selected.json`: fitted weights, training/selection match IDs and provenance hashes.
- `freeze.json`: selected-model and protocol fingerprints.
- `development.json`: nine predetermined comparisons on external development games.
- `holdout.json`: one evaluation of the frozen winner on two held-out games.

Training data: IDSSE, collected by Sportec Solutions for DFL, CC-BY 4.0.
Attribution: Bassek et al., *An integrated dataset of spatiotemporal and event data
in elite soccer*, Scientific Data 12, 195 (2025),
https://doi.org/10.1038/s41597-025-04505-y.
Source: https://doi.org/10.6084/m9.figshare.28196177.
Mirror revision: `a715a38dfbaf5f58e431727c2b78d174101a703c` in `pysport/idsse-data`.
Retain the dataset attribution when sharing this model and its reports.

Run `make candidate` to apply these weights to the competition observations.
See [the experiment](../../EXTERNAL_EXPERIMENT.md) for complete retraining.
No external dataset download is needed when using the included frozen parameters.

`competition_data_used: false` refers to training/selection examples: no competition
observations or labels were used as those examples. The experimental design does
acknowledge the competition's documented missing-ball problem.

This is an experimental candidate, not a private-score or eligibility claim.
Earlier project history remains in the provenance audit.
