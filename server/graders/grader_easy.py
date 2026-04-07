"""
Grader for task: easy
One intern (Carlos Méndez), no exceptions.
Each of the 5 required checklist items is worth 0.20.
Score is always deterministic — no datetime.now() used.
"""
from __future__ import annotations

from models import PersonState

STEP_WEIGHT = 0.20
REQUIRED_STEPS = ["welcome_email_sent", "docs_shared", "intro_scheduled", "account_created", "access_granted"]


def grade(state: list[PersonState]) -> float:
    """
    Return episode score in [0.0, 1.0].
    Each completed required checklist item adds STEP_WEIGHT.
    """
    if not state:
        return 0.0

    intern = state[0]  # Easy task: single intern
    score = sum(
        STEP_WEIGHT
        for step in REQUIRED_STEPS
        if intern.checklist.get(step, False)
    )
    return round(min(score, 1.0), 2)
