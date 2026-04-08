from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from models import PersonState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
    "send_welcome_email",
    "share_docs",
    "share_international_docs",
    "schedule_intro_meeting",
    "create_account",
    "grant_system_access",
    "request_alternative_access",
    "send_followup_email",
    "escalate_to_manager",
]

DATA_DIR = Path(__file__).parent / "data"
FIXTURES_PATH = DATA_DIR / "fixtures.json"


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_task(task_name: str) -> list[PersonState]:
    """Load intern fixtures for a given task. Always returns a deep copy."""
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if task_name not in raw:
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {list(raw.keys())}")

    return [PersonState(**intern) for intern in raw[task_name]]


# ---------------------------------------------------------------------------
# Available-action filtering
# ---------------------------------------------------------------------------

def _check_dep(dep: str, intern: PersonState) -> bool:
    """Resolve a single dependency string against the intern state."""
    if dep == "intern_confirmed":
        return intern.intern_confirmed
    if dep == "is_international":
        return intern.is_international
    if dep == "access_delayed":
        return intern.access_delayed
    if dep == "docs_shared":
        return intern.checklist.get("docs_shared", False)
    if dep == "intl_docs_shared":
        return intern.checklist.get("intl_docs_shared", False)
    if dep == "account_created":
        return intern.checklist.get("account_created", False)
    if dep == "welcome_email_sent":
        return intern.checklist.get("welcome_email_sent", False)
    if dep == "days_without_response_gte_1":
        return intern.days_without_response >= 1
    if dep == "days_without_response_gte_3":
        return intern.days_without_response >= 3
    if dep == "not_intern_confirmed":
        return not intern.intern_confirmed
    return False


def get_available_actions(intern: PersonState) -> list[str]:
    """Return the list of actions whose dependencies are all satisfied."""
    # If this intern is already complete, no further actions are needed
    if _intern_complete(intern):
        return []
    available = []
    for action in VALID_ACTIONS:
        deps = intern.dependencies.get(action, [])
        if all(_check_dep(d, intern) for d in deps):
            # Skip already-completed actions
            if action == "send_welcome_email" and intern.checklist.get("welcome_email_sent"):
                continue
            if action == "share_docs" and intern.checklist.get("docs_shared"):
                continue
            if action == "share_international_docs" and intern.checklist.get("intl_docs_shared"):
                continue
            if action == "schedule_intro_meeting" and intern.checklist.get("intro_scheduled"):
                continue
            if action == "create_account" and intern.checklist.get("account_created"):
                continue
            if action == "grant_system_access" and intern.checklist.get("access_granted"):
                continue
            if action == "request_alternative_access" and intern.checklist.get("access_granted"):
                continue
            if action == "send_followup_email" and intern.checklist.get("followup_sent"):
                continue
            if action == "escalate_to_manager" and intern.checklist.get("escalated"):
                continue
            available.append(action)
    return available


# ---------------------------------------------------------------------------
# Email preview generator
# ---------------------------------------------------------------------------

EMAIL_TEMPLATES: dict[str, dict[str, str]] = {
    "send_welcome_email": {
        "subject": "Welcome to the team, {name}!",
        "body": (
            "Hi {name},\n\nWe're excited to have you joining us on {start_date}. "
            "Please reply to confirm your start date and we'll get everything ready for you.\n\n"
            "Best regards,\nHR Team"
        ),
    },
    "share_docs": {
        "subject": "Your onboarding documents — {name}",
        "body": (
            "Hi {name},\n\nPlease find your onboarding documents attached. "
            "Review them before your first day.\n\nBest regards,\nHR Team"
        ),
    },
    "share_international_docs": {
        "subject": "International onboarding package — {name}",
        "body": (
            "Hi {name},\n\nAs an international hire, please find your visa and "
            "relocation documents attached. Contact us with any questions.\n\n"
            "Best regards,\nHR Team"
        ),
    },
    "schedule_intro_meeting": {
        "subject": "Intro meeting invitation — {name}",
        "body": (
            "Hi {name},\n\nYou're invited to a team intro meeting on your first week. "
            "A calendar invite will follow shortly.\n\nBest regards,\nHR Team"
        ),
    },
    "send_followup_email": {
        "subject": "Following up — {name}",
        "body": (
            "Hi {name},\n\nWe haven't heard back from you yet. "
            "Please reply at your earliest convenience so we can proceed with your onboarding.\n\n"
            "Best regards,\nHR Team"
        ),
    },
    "escalate_to_manager": {
        "subject": "Escalation: no response from {name}",
        "body": (
            "Hi Manager,\n\n{name} has not responded for {days} days. "
            "Manual intervention may be required to proceed with their onboarding.\n\n"
            "Best regards,\nHR System"
        ),
    },
}


def build_email_preview(action: str, intern: PersonState) -> dict[str, Any] | None:
    tmpl = EMAIL_TEMPLATES.get(action)
    if not tmpl:
        return None
    return {
        "type": action,
        "intern_id": intern.id,
        "intern_name": intern.name,
        "subject": tmpl["subject"].format(name=intern.name, start_date=intern.start_date),
        "body": tmpl["body"].format(
            name=intern.name,
            start_date=intern.start_date,
            days=intern.days_without_response,
        ),
    }


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------

def apply_action(
    intern: PersonState,
    action: str,
) -> tuple[float, bool, str | None]:
    """
    Mutate intern state and return (step_reward, done, error_message).
    'done' here means this intern's checklist is complete.
    No datetime.now() is used anywhere.
    """
    if action not in VALID_ACTIONS:
        return -0.05, False, f"unknown action '{action}'"

    deps = intern.dependencies.get(action, [])
    missing = [d for d in deps if not _check_dep(d, intern)]

    # Special case: grant_system_access when access_delayed → hard error
    if action == "grant_system_access" and intern.access_delayed:
        return -0.05, False, "access_delayed — use request_alternative_access"

    # Penalize escalate_to_manager before 3 days
    if action == "escalate_to_manager" and intern.days_without_response < 3:
        return -0.05, False, "escalate_to_manager requires days_without_response >= 3"

    if missing:
        return -0.05, False, f"missing dependencies: {missing}"

    email_preview = None

    if action == "send_welcome_email":
        intern.checklist["welcome_email_sent"] = True
        intern.intern_confirmed = True          # auto-confirm per spec
        email_preview = build_email_preview(action, intern)
        reward = 0.20

    elif action == "share_docs":
        intern.checklist["docs_shared"] = True
        email_preview = build_email_preview(action, intern)
        reward = 0.20

    elif action == "share_international_docs":
        intern.checklist["intl_docs_shared"] = True
        email_preview = build_email_preview(action, intern)
        reward = 0.10

    elif action == "schedule_intro_meeting":
        intern.checklist["intro_scheduled"] = True
        email_preview = build_email_preview(action, intern)
        reward = 0.20

    elif action == "create_account":
        intern.checklist["account_created"] = True
        reward = 0.20

    elif action == "grant_system_access":
        intern.checklist["access_granted"] = True
        reward = 0.20

    elif action == "request_alternative_access":
        intern.checklist["access_granted"] = True
        reward = 0.20

    elif action == "send_followup_email":
        intern.checklist["followup_sent"] = True
        email_preview = build_email_preview(action, intern)
        reward = 0.05

    elif action == "escalate_to_manager":
        intern.checklist["escalated"] = True
        email_preview = build_email_preview(action, intern)
        reward = 0.05

    else:
        return -0.05, False, f"unhandled action '{action}'"

    # Check if intern's required steps are done
    done = _intern_complete(intern)
    return reward, done, None


def _intern_complete(intern: PersonState) -> bool:
    """An intern is complete when all required checklist items are ticked."""
    required = ["welcome_email_sent", "docs_shared", "intro_scheduled", "account_created", "access_granted"]
    if intern.is_international:
        required.append("intl_docs_shared")
    if intern.checklist.get("escalated") or (
        intern.days_without_response >= 3 and not intern.intern_confirmed
    ):
        # For interns who can never confirm (Rohan-like), escalation counts as their terminal state
        if intern.checklist.get("escalated"):
            return True
    return all(intern.checklist.get(k, False) for k in required)


# ---------------------------------------------------------------------------
# Difficulty classification — based purely on exception types, not intern count
# ---------------------------------------------------------------------------

def classify_difficulty(interns: list[PersonState]) -> str:
    """
    Derive difficulty from the types of exceptions present in the batch.
    Number of interns is irrelevant — 10 normal interns is still EASY.

    EASY   → no exceptions in any intern
    MEDIUM → at least one intern needs followup (days_without_response >= 1, < 3)
             but no hard exceptions
    HARD   → at least one intern has a hard exception:
               · is_international=true
               · access_delayed=true
               · days_without_response >= 3 AND intern_confirmed=false (must escalate)
    """
    has_hard = any(
        intern.is_international
        or intern.access_delayed
        or (intern.days_without_response >= 3 and not intern.intern_confirmed)
        for intern in interns
    )
    if has_hard:
        return "hard"

    has_medium = any(
        intern.days_without_response >= 1
        for intern in interns
    )
    if has_medium:
        return "medium"

    return "easy"


# ---------------------------------------------------------------------------
# Episode-level helper
# ---------------------------------------------------------------------------

def all_complete(interns: list[PersonState]) -> bool:
    return all(_intern_complete(i) for i in interns)
