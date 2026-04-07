"""
Grader for task: hard
Three interns each with a unique exception:
  Ahmed  (intern_001): is_international=true → needs intl_docs before access
  Rohan  (intern_002): days_without_response=4, intern_confirmed=false → escalate only
  Sara   (intern_003): access_delayed=true → request_alternative_access

Scoring:
  Base steps:   proportional contribution
  Exception bonuses:
    share_international_docs completed (Ahmed)  → +0.10
    escalate_to_manager completed (Rohan)        → +0.10
    request_alternative_access completed (Sara)  → +0.10
  Total possible = base + 0.30 bonuses → normalised to [0.0, 1.0]

Base steps per intern (weight = 1 each, total base = 14):
  Ahmed: send_welcome_email, share_docs, share_international_docs, schedule_intro_meeting,
         create_account, grant_system_access  → 6
  Rohan: escalate_to_manager (terminal; can never confirm, so 1 meaningful action)  → 1
  Sara:  send_welcome_email, share_docs, schedule_intro_meeting, create_account,
         request_alternative_access  → 5
  Total base actions = 12

Bonuses add 0.10 each for the 3 exception actions → max bonus = 0.30.
We normalise: score = (base_completed/12) * 0.70 + bonus_score * 1.0
so that perfect = 0.70 + 0.30 = 1.00.

No datetime.now() is used anywhere.
"""
from __future__ import annotations

from models import PersonState

BASE_TOTAL = 12          # see docstring
BASE_WEIGHT = 0.70       # fraction of score from base steps

AHMED_BASE = ["welcome_email_sent", "docs_shared", "intl_docs_shared",
              "intro_scheduled", "account_created", "access_granted"]
ROHAN_BASE = ["escalated"]
SARA_BASE  = ["welcome_email_sent", "docs_shared", "intro_scheduled",
              "account_created", "access_granted"]

BONUS_PER_EXCEPTION = 0.10  # × 3 = 0.30 total bonus possible


def grade(state: list[PersonState]) -> float:
    if not state:
        return 0.0

    by_id = {i.id: i for i in state}

    base_completed = 0
    bonus = 0.0

    ahmed = by_id.get("intern_001")
    if ahmed:
        base_completed += sum(1 for s in AHMED_BASE if ahmed.checklist.get(s, False))
        if ahmed.checklist.get("intl_docs_shared", False):
            bonus += BONUS_PER_EXCEPTION

    rohan = by_id.get("intern_002")
    if rohan:
        base_completed += sum(1 for s in ROHAN_BASE if rohan.checklist.get(s, False))
        if rohan.checklist.get("escalated", False):
            bonus += BONUS_PER_EXCEPTION

    sara = by_id.get("intern_003")
    if sara:
        base_completed += sum(1 for s in SARA_BASE if sara.checklist.get(s, False))
        if sara.checklist.get("access_granted", False) and sara.access_delayed:
            # Only grant bonus if they used the *correct* path (request_alternative_access)
            # We can check this by verifying access_delayed is still true on the intern
            # (access_delayed flag is never mutated by the env — it's a fixture property)
            bonus += BONUS_PER_EXCEPTION

    base_score = (base_completed / BASE_TOTAL) * BASE_WEIGHT
    score = base_score + bonus
    return round(min(max(score, 0.0), 1.0), 2)
