"""
Semantic validator — rule-based, deterministic.
Interface: validate(action, intern_dict) → ValidationResult

Same inputs always produce the same result.
Rules are defined in rules.py — edit that file to adapt to a new context.
"""
from __future__ import annotations

from dataclasses import dataclass

from .rules import (
    ALLOWED_PREFIXES,
    KNOWN_ACTIONS,
    PRECONDITION_RULES,
    PROHIBITED_VERBS,
)


@dataclass
class ValidationResult:
    valid:            bool
    zone:             int    # 1=prohibited  2=wrong domain  3=valid onboarding
    reason:           str
    reward_modifier:  float  # -0.05 zone1 | 0.0 zone2&3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_prohibited(action: str) -> str | None:
    al = action.lower()
    for verb in PROHIBITED_VERBS:
        if verb in al:
            return (
                f"Action '{action}' contains prohibited verb '{verb}' — "
                "this is outside the onboarding scope."
            )
    return None


def _valid_prefix(action: str) -> bool:
    if action in KNOWN_ACTIONS:
        return True
    return any(action.startswith(p) for p in ALLOWED_PREFIXES)


def _eval(field: str, intern: dict) -> bool:
    """Evaluate a single field name against intern dict."""
    # Direct intern attributes
    if field in ("intern_confirmed", "is_international", "access_delayed"):
        return bool(intern.get(field, False))
    # Checklist items
    return bool(intern.get("checklist", {}).get(field, False))


def _check_preconditions(action: str, intern: dict) -> str | None:
    """Return an error message if preconditions fail, else None."""
    rule = PRECONDITION_RULES.get(action)
    if not rule:
        return None

    for field in rule.get("requires_checklist", []):
        if not intern.get("checklist", {}).get(field, False):
            return rule["message"]

    for field in rule.get("requires_intern", []):
        if not _eval(field, intern):
            return rule["message"]

    for field in rule.get("forbidden_checklist", []):
        if intern.get("checklist", {}).get(field, False):
            return rule["message"]

    for field in rule.get("forbidden_intern", []):
        if _eval(field, intern):
            return rule["message"]

    days_req = rule.get("requires_days_gte")
    if days_req is not None:
        if intern.get("days_without_response", 0) < days_req:
            return rule["message"]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_onboarding(action: str, intern: dict) -> ValidationResult:
    """
    Validate an action against the current intern state.

    Zone 1 — prohibited (reward_modifier = -0.05):
        Action contains a verb that is outside the onboarding domain.

    Zone 2 — wrong category (reward_modifier = 0.0):
        Dynamic action that doesn't use an allowed prefix.

    Zone 3 — valid onboarding action (reward_modifier = 0.0):
        Passes all checks; environment may still reject on dependencies.
    """
    # Zone 1
    prohibited_msg = _check_prohibited(action)
    if prohibited_msg:
        return ValidationResult(
            valid=False, zone=1,
            reason=prohibited_msg,
            reward_modifier=-0.05,
        )

    # Zone 2
    if not _valid_prefix(action):
        return ValidationResult(
            valid=False, zone=2,
            reason=(
                f"Dynamic action '{action}' must use an allowed prefix "
                f"(send_, share_, comm_, doc_, access_, escalate_, etc.)."
            ),
            reward_modifier=0.0,
        )

    # Zone 3 — business rule check
    err = _check_preconditions(action, intern)
    if err:
        return ValidationResult(
            valid=False, zone=3,
            reason=err,
            reward_modifier=0.0,
        )

    return ValidationResult(valid=True, zone=3, reason="ok", reward_modifier=0.0)