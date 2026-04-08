"""
Heuristic++ agent — deterministic DAG-based decision.

Handles the happy path without any LLM call.
Returns None when the state has exceptions or ambiguity → triggers LLM.

This is the fast path: every step where there are no exceptions
is resolved here without spending a token.
"""
from __future__ import annotations

from typing import Optional

# Standard flow priority order — mirrors the DAG dependencies
FLOW_ORDER: list[str] = [
    "send_welcome_email",
    "share_docs",
    "share_international_docs",
    "schedule_intro_meeting",
    "create_account",
    "request_alternative_access",
    "grant_system_access",
]

# Exception-handling actions — always deferred to LLM
EXCEPTION_ACTIONS: set[str] = {
    "send_followup_email",
    "escalate_to_manager",
}


def _intern_has_exception(intern: dict) -> bool:
    """True when intern state requires reasoning beyond the simple DAG."""
    return (
        intern.get("is_international", False)
        or intern.get("access_delayed", False)
        or intern.get("days_without_response", 0) >= 1
    )


def get_action(observation: dict) -> Optional[tuple[str, str]]:
    """
    Pick the next deterministic action from the DAG.

    Returns (intern_id, action) when the path is unambiguous.
    Returns None when any intern has an exception — caller must use LLM.
    """
    available = observation.get("available_actions", {})
    interns   = observation.get("interns", [])

    # Any exception in the batch → defer entirely to the LLM
    for intern in interns:
        if _intern_has_exception(intern):
            return None

    # No exceptions: pick most-advanced intern, apply next DAG step
    def _progress(intern: dict) -> int:
        return sum(1 for v in intern.get("checklist", {}).values() if v)

    for intern in sorted(interns, key=_progress, reverse=True):
        actions = available.get(intern["id"], [])
        for action in FLOW_ORDER:
            if action in actions:
                return intern["id"], action

    return None