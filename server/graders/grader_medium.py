"""
Grader for task: medium
Three interns in different states. Score = completed_steps / total_expected_steps.

Total expected steps:
  Carlos (welcome already sent): 4 remaining standard steps
  Priya  (welcome sent, 2 days no response): 1 followup + 4 standard = 5
  Ana    (everything from scratch): 5 standard steps
  Total = 14

Wait — per the plan: Carlos=5, Priya=6 (incl. followup), Ana=5 → total=16.
Carlos already has welcome_email_sent=true in the fixture, so his checklist
starts with 1 done. The grader measures steps *completed during the episode*
relative to the expected total (16).
"""
from __future__ import annotations

from models import PersonState

# Expected total steps per intern as defined in the plan
CARLOS_EXPECTED = 5   # 4 remaining + welcome (already done counts in total)
PRIYA_EXPECTED  = 6   # 5 standard + 1 followup
ANA_EXPECTED    = 5

TOTAL_EXPECTED = CARLOS_EXPECTED + PRIYA_EXPECTED + ANA_EXPECTED  # 16

STANDARD_STEPS = ["welcome_email_sent", "docs_shared", "intro_scheduled", "account_created", "access_granted"]


def _count_completed(intern: PersonState, include_followup: bool = False) -> int:
    count = sum(1 for s in STANDARD_STEPS if intern.checklist.get(s, False))
    if include_followup and intern.checklist.get("followup_sent", False):
        count += 1
    return count


def grade(state: list[PersonState]) -> float:
    if not state:
        return 0.0

    completed = 0
    for intern in state:
        # Priya (intern_002) needs followup counted
        include_followup = (intern.id == "intern_002")
        completed += _count_completed(intern, include_followup=include_followup)

    score = completed / TOTAL_EXPECTED
    return round(min(score, 1.0), 2)
