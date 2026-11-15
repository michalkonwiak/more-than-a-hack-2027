"""Prevent accidental use of retired competition workflows, not a compliance certificate."""

RULES_URL = "https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md"
AUDIT_MESSAGE = (
    "This workflow is retired after the rules audit. Training/tuning on the public match "
    "labels and human validation labels on this match are not allowed. Existing heuristic "
    "candidates also require provenance review. See PROVENANCE_AUDIT.md. "
    "Use 'python -m possession.reference --verify' for the unchanged official reference."
)


def refuse_retired_workflow():
    raise ValueError(AUDIT_MESSAGE)
