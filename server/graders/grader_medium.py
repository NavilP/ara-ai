"""
Grader for difficulty: medium
Applies to any batch where at least one intern has days_without_response >= 1
but no hard exceptions (international, access_delayed, must-escalate).

Scoring: completed_steps / total_expected_steps  → always [0.0, 1.0]

Expected steps per intern:
  · Standard: 5 steps (welcome, docs, intro, account, access)
  · +1 followup if intern started with days_without_response >= 1

The total is computed dynamically from the batch — no hardcoded names or IDs.
No datetime.now() used anywhere.
"""
from __future__ import annotations

from models import PersonState

STANDARD_STEPS = [
    "welcome_email_sent",
    "docs_shared",
    "intro_scheduled",
    "account_created",
    "access_granted",
]


def _expected_steps(intern: PersonState) -> int:
    """How many steps are expected for this intern to be fully onboarded."""
    steps = len(STANDARD_STEPS)
    if intern.days_without_response >= 1:
        steps += 1  # followup required
    return steps


def _completed_steps(intern: PersonState) -> int:
    """How many steps has this intern completed so far."""
    count = sum(1 for s in STANDARD_STEPS if intern.checklist.get(s, False))
    if intern.days_without_response >= 1 and intern.checklist.get("followup_sent", False):
        count += 1
    return count


def grade(state: list[PersonState]) -> float:
    if not state:
        return 0.0

    total_expected = sum(_expected_steps(i) for i in state)
    total_completed = sum(_completed_steps(i) for i in state)

    if total_expected == 0:
        return 0.0

    score = total_completed / total_expected
    return round(min(score, 1.0), 2)
