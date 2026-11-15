# Model provenance

## Earlier experiments

Earlier work used public five-minute competition labels to train a logistic model
and to diagnose and select heuristic changes. Later heuristic variants inherited
some of those settings. This work followed an older local copy of the rules.
On 24 October 2026, the current upstream rules were checked and the affected workflows
were retired. Public labels are for pipeline checks, not training or tuning.

A manual annotation interface was also explored. No completed independent human
validation result was established. A local vision-language model produced draft
annotations; those drafts were not used to train, select or repair a submission.
The earlier code, configurations and outputs remain in the local research archive.
They are not the submitted model or the published reproduction workflow.

## Submitted candidate

The external_idsse_v1 candidate uses newly fitted parameters from seven other matches
in the IDSSE dataset. The training, development and holdout splits are disjoint by match.
The protocol was specified before label inspection and scoring. Model selection used
external development matches, followed by one evaluation on the frozen external holdout.

The candidate reuses general numerical optimisation, Viterbi decoding and data-loading
utilities. It does not use the earlier fitted weights, heuristic configurations,
competition public labels, manual annotations or vision-language model proposals.
Some shared modules retain historical functions, but external inference does not call
the historical heuristic inference or label-training workflow. Retired training entry
points reject calls before reading competition labels.

Competition inference reads tactical tracking and metadata. It does not read video,
broadcast tracking, Smart Tagging, homography or competition ground truth.

The external experiment does not erase the prior work. This disclosure is provided
for organiser review and does not assert a decision about team eligibility.
See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) and
[EXTERNAL_EXPERIMENT.md](EXTERNAL_EXPERIMENT.md) for the candidate and its limitations.
