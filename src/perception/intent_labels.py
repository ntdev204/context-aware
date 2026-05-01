"""Canonical intent label ontology for runtime, dataset, and training.

`FOLLOWING`/`FOLLOW` is intentionally absent. Person-following is not part of
the robot behavior anymore, and ambiguous same-direction motion must be treated
as `UNCERTAIN` until reviewed.
"""

from __future__ import annotations

STATIONARY = 0
APPROACHING = 1
DEPARTING = 2
CROSSING = 3
ERRATIC = 4
UNCERTAIN = 5

TRAINABLE_INTENT_NAMES = [
    "STATIONARY",
    "APPROACHING",
    "DEPARTING",
    "CROSSING",
    "ERRATIC",
]

INTENT_NAMES = TRAINABLE_INTENT_NAMES + ["UNCERTAIN"]

NUM_TRAINABLE_INTENT_CLASSES = len(TRAINABLE_INTENT_NAMES)
NUM_INTENT_CLASSES = len(INTENT_NAMES)

LABEL_TO_ID = {name: idx for idx, name in enumerate(INTENT_NAMES)}
TRAINABLE_LABEL_TO_ID = {name: idx for idx, name in enumerate(TRAINABLE_INTENT_NAMES)}

REVIEW_REQUIRED_LABELS = {"UNCERTAIN", "ERRATIC"}
REVIEW_ACCEPTED_STATUSES = {"auto_accepted", "human_verified", "imported", "accepted"}

LEGACY_LABEL_ALIASES = {
    "FOLLOW": "UNCERTAIN",
    "FOLLOWING": "UNCERTAIN",
}


def canonical_label(label: str | None) -> str:
    """Return the canonical uppercase label, mapping legacy labels to UNCERTAIN."""
    if not label:
        return "UNCERTAIN"
    label_up = str(label).strip().upper()
    return LEGACY_LABEL_ALIASES.get(label_up, label_up)


def is_trainable_label(label: str | None) -> bool:
    return canonical_label(label) in TRAINABLE_LABEL_TO_ID


def needs_human_review(label: str | None) -> bool:
    return canonical_label(label) in REVIEW_REQUIRED_LABELS
