"""
Unified state-based grader.

Deterministic guarantee: same final state + same step_count = same score.
No references to action names or intern IDs anywhere in this file.

Formula:
    Without exceptions: score = core * 0.90 + efficiency * 0.10
    With exceptions:    score = core * 0.70 + exceptions * 0.20 + efficiency * 0.10
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import PersonState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# States every non-blocked intern must reach (order doesn't matter for scoring)
REQUIRED_STATES: list[str] = [
    "welcome_email_sent",
    "docs_shared",
    "intro_scheduled",
    "account_created",
    "access_granted",
]

# Precomputed optimal step counts per task (used for efficiency bonus)
OPTIMAL_STEPS: dict[str, int] = {
    "easy":   5,
    "medium": 14,
    "hard":   12,
}

# ---------------------------------------------------------------------------
# State helpers — all property-based, no IDs
# ---------------------------------------------------------------------------

def _is_blocked(intern: "PersonState") -> bool:
    """Intern cannot complete standard flow — needs escalation."""
    return intern.days_without_response >= 3 and not intern.intern_confirmed


def _has_exception(intern: "PersonState") -> bool:
    return _is_blocked(intern) or intern.is_international or intern.access_delayed


def _exception_resolved(intern: "PersonState") -> bool:
    if _is_blocked(intern):
        return bool(intern.checklist.get("escalated", False))
    if intern.is_international:
        return bool(intern.checklist.get("intl_docs_shared", False))
    if intern.access_delayed:
        return bool(intern.checklist.get("access_granted", False))
    return False


def _required_states(intern: "PersonState") -> list[str]:
    """Dynamic required state list based on intern properties."""
    req = list(REQUIRED_STATES)
    if intern.is_international:
        req.append("intl_docs_shared")
    if intern.days_without_response >= 1:
        req.append("followup_sent")
    return req


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade(task: str, interns: list["PersonState"], step_count: int) -> float:
    """
    Compute the episode score.

    Args:
        task:       "easy" | "medium" | "hard"
        interns:    current state of all interns
        step_count: total steps taken in this episode

    Returns:
        float in [0.05, 0.95], rounded to 2 decimal places
    """
    if not interns:
        return 0.0

    # --- Component 1: core progress ---
    total_required = 0
    total_achieved = 0

    for intern in interns:
        if _is_blocked(intern):
            # Blocked intern: only escalation counts as completion
            total_required += 1
            total_achieved += 1 if intern.checklist.get("escalated", False) else 0
        else:
            req            = _required_states(intern)
            total_required += len(req)
            total_achieved += sum(1 for k in req if intern.checklist.get(k, False))

    core = total_achieved / total_required if total_required > 0 else 0.0

    # --- Component 2: exception handling ---
    exc_total    = sum(1 for i in interns if _has_exception(i))
    exc_resolved = sum(1 for i in interns if _has_exception(i) and _exception_resolved(i))
    exc_ratio    = (exc_resolved / exc_total) if exc_total > 0 else 0.0

    # --- Component 3: efficiency bonus ---
    optimal    = OPTIMAL_STEPS.get(task, 20)
    efficiency = min(optimal / max(step_count, 1), 1.0)

    # --- Weighted total ---
    if exc_total > 0:
        score = core * 0.70 + exc_ratio * 0.20 + efficiency * 0.10
    else:
        score = core * 0.90 + efficiency * 0.10
        
    score = 0.05 + score * 0.90
    return round(min(max(score, 0.05), 0.95), 2)