"""
Grader for difficulty: hard
Applies to any batch that contains at least one hard exception:
  · is_international=true         → needs intl_docs before access
  · access_delayed=true           → must use request_alternative_access
  · days_without_response >= 3
    AND intern_confirmed=false    → can never confirm; must escalate (terminal)

Scoring: two components summed → always [0.0, 1.0]

  base_score  = (base_steps_completed / total_base_expected) * BASE_WEIGHT   (0.70)
  bonus_score = exceptions_resolved * BONUS_PER_EXCEPTION                    (0.10 each)

The total_base_expected and bonuses are computed from the batch properties,
not from hardcoded intern IDs.
No datetime.now() used anywhere.
"""
from __future__ import annotations

from models import PersonState

BASE_WEIGHT        = 0.70
BONUS_PER_EXCEPTION = 0.10

# Steps expected per intern type
STANDARD_STEPS = ["welcome_email_sent", "docs_shared", "intro_scheduled",
                  "account_created", "access_granted"]


def _is_blocked(intern: PersonState) -> bool:
    """Intern can never confirm — only escalation makes sense."""
    return intern.days_without_response >= 3 and not intern.intern_confirmed


def _expected_base_steps(intern: PersonState) -> int:
    """
    Base steps expected for scoring purposes.
    Blocked interns: only escalation counts (1 step).
    International: standard + intl_docs (6 steps).
    Others: standard 5 steps.
    """
    if _is_blocked(intern):
        return 1
    if intern.is_international:
        return len(STANDARD_STEPS) + 1  # +intl_docs_shared
    return len(STANDARD_STEPS)


def _completed_base_steps(intern: PersonState) -> int:
    if _is_blocked(intern):
        return 1 if intern.checklist.get("escalated", False) else 0

    count = sum(1 for s in STANDARD_STEPS if intern.checklist.get(s, False))
    if intern.is_international and intern.checklist.get("intl_docs_shared", False):
        count += 1
    return count


def _exception_resolved(intern: PersonState) -> bool:
    """
    Did the agent correctly handle this intern's specific hard exception?
    · International  → intl_docs_shared completed
    · Blocked        → escalated
    · Access delayed → access_granted AND access_delayed=true
                       (proxy for having used request_alternative_access,
                        since grant_system_access is blocked with -0.05 penalty
                        when access_delayed=true)
    """
    if _is_blocked(intern):
        return intern.checklist.get("escalated", False)
    if intern.is_international:
        return intern.checklist.get("intl_docs_shared", False)
    if intern.access_delayed:
        return intern.checklist.get("access_granted", False)
    return False


def _has_hard_exception(intern: PersonState) -> bool:
    return _is_blocked(intern) or intern.is_international or intern.access_delayed


def grade(state: list[PersonState]) -> float:
    if not state:
        return 0.0

    total_base_expected = sum(_expected_base_steps(i) for i in state)
    total_base_completed = sum(_completed_base_steps(i) for i in state)

    # Count bonus only for interns that actually have a hard exception
    exceptions_resolved = sum(
        1 for i in state
        if _has_hard_exception(i) and _exception_resolved(i)
    )

    base_score  = (total_base_completed / total_base_expected) * BASE_WEIGHT if total_base_expected else 0.0
    bonus_score = exceptions_resolved * BONUS_PER_EXCEPTION

    score = base_score + bonus_score
    return round(min(max(score, 0.0), 1.0), 2)
