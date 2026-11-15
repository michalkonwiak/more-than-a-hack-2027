# jamnik team: ball possession inference

More Than A Hack 2027, Initial Challenge.

Team: Michał Konwiak, Kamil Brzoza, Maciej Zapała.

This solution predicts home possession, away possession and dead ball for each tactical
video frame. Two logistic classifiers trained on other matches feed a temporal Viterbi
decoder. Competition inference uses the supplied tactical tracking and metadata.

## Reproduce the submission

Use Linux x86_64, Python 3.12 and `uv`. Download and extract the competition data separately.

```bash
make install
make candidate DATA_DIR="/absolute/path/to/competition/dataset"
```

The command runs the tests, repeats full inference, checks identical CSV bytes and
runs the official format validator. Weights are included; retraining is not required.
Output: `artifacts/candidates/external_idsse_v1/submission.csv`.
Name a copy `jamnik team.csv` for the submission form.

Expected SHA-256:

```text
721c7d7e2d7e7aa4559ecfe8edbf78f386fd56fed0416c0bfd45fdc267a511e3
```

Dependencies are pinned with hashes in `requirements.lock`. The source data, videos,
feature caches and generated submissions are not included. `make reproduce` reproduces
the unchanged official smooth50 reference rather than the team candidate.

## Method and evidence

- [Technical report](TECHNICAL_REPORT.md): inputs, model, possession rules and limitations.
- [External experiment](EXTERNAL_EXPERIMENT.md): data attribution, splits, training and evaluation.
- [Provenance audit](PROVENANCE_AUDIT.md): earlier experiments and their exclusion from this candidate.
- `protocols/`: fixed experiment settings and source fingerprints.
- `models/external_idsse_v1/`: frozen weights and aggregate evaluation reports.

Mean external holdout spellF1 is 0.4571 for the candidate and 0.2950 for the fixed
reference across two matches in clean and artificial ball-dropout conditions.
This is not a competition score. The private competition result is unknown.

The original starter, evaluation code and competition documents are retained from
[the organisers' repository](https://github.com/metrica-sports/more-than-a-hack-2027).
Consult its current rules; the inherited local documentation may describe older rules.
