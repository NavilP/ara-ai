from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from models import PersonState
from onboarding_validator.semantic_validator import validate_onboarding, ValidationResult

STANDARD_ACTIONS: list[str] = [
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

DATA_DIR      = Path(__file__).parent / "data"
FIXTURES_PATH = DATA_DIR / "fixtures.json"


def load_task(task_name: str) -> list[PersonState]:
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if task_name not in raw:
        raise ValueError(f"Unknown task '{task_name}'. Valid: {list(raw.keys())}")
    return [PersonState(**intern) for intern in copy.deepcopy(raw[task_name])]


def _check_dep(dep: str, intern: PersonState) -> bool:
    if dep == "intern_confirmed":            return intern.intern_confirmed
    if dep == "is_international":            return intern.is_international
    if dep == "access_delayed":              return intern.access_delayed
    if dep == "docs_shared":                 return intern.checklist.get("docs_shared", False)
    if dep == "intl_docs_shared":            return intern.checklist.get("intl_docs_shared", False)
    if dep == "account_created":             return intern.checklist.get("account_created", False)
    if dep == "welcome_email_sent":          return intern.checklist.get("welcome_email_sent", False)
    if dep == "days_without_response_gte_1": return intern.days_without_response >= 1
    if dep == "days_without_response_gte_3": return intern.days_without_response >= 3
    if dep == "not_intern_confirmed":        return not intern.intern_confirmed
    return False


def _intern_complete(intern: PersonState) -> bool:
    required = ["welcome_email_sent", "docs_shared", "intro_scheduled", "account_created", "access_granted"]
    if intern.is_international:
        required.append("intl_docs_shared")
    if intern.checklist.get("escalated"):
        return True
    return all(intern.checklist.get(k, False) for k in required)


def get_available_actions(intern: PersonState) -> list[str]:
    if _intern_complete(intern):
        return []
    skip_map = {
        "send_welcome_email":        "welcome_email_sent",
        "share_docs":                "docs_shared",
        "share_international_docs":  "intl_docs_shared",
        "schedule_intro_meeting":    "intro_scheduled",
        "create_account":            "account_created",
        "grant_system_access":       "access_granted",
        "request_alternative_access":"access_granted",
        "send_followup_email":       "followup_sent",
        "escalate_to_manager":       "escalated",
    }
    available = []
    for action in STANDARD_ACTIONS:
        deps = intern.dependencies.get(action, [])
        if not all(_check_dep(d, intern) for d in deps):
            continue
        if action in skip_map and intern.checklist.get(skip_map[action]):
            continue
        available.append(action)
    return available


EMAIL_TEMPLATES: dict[str, dict[str, str]] = {
    "send_welcome_email": {
        "subject": "Welcome to the team, {name}!",
        "body": "Hi {name},\n\nWe're excited to have you joining us on {start_date}. Please reply to confirm.\n\nBest regards,\nHR Team",
    },
    "share_docs": {
        "subject": "Your onboarding documents — {name}",
        "body": "Hi {name},\n\nPlease find your onboarding documents attached.\n\nBest regards,\nHR Team",
    },
    "share_international_docs": {
        "subject": "International onboarding package — {name}",
        "body": "Hi {name},\n\nPlease find your international onboarding documents attached.\n\nBest regards,\nHR Team",
    },
    "schedule_intro_meeting": {
        "subject": "Intro meeting invitation — {name}",
        "body": "Hi {name},\n\nYou're invited to a team intro meeting. A calendar invite will follow.\n\nBest regards,\nHR Team",
    },
    "send_followup_email": {
        "subject": "Following up — {name}",
        "body": "Hi {name},\n\nWe haven't heard back. Please reply so we can continue your onboarding.\n\nBest regards,\nHR Team",
    },
    "escalate_to_manager": {
        "subject": "Escalation: no response from {name}",
        "body": "Hi Manager,\n\n{name} has not responded for {days} days. Manual intervention may be required.\n\nBest regards,\nHR System",
    },
}


def build_email_preview(action: str, intern: PersonState) -> dict[str, Any] | None:
    tmpl = EMAIL_TEMPLATES.get(action)
    if not tmpl:
        return None
    return {
        "type":        action,
        "intern_id":   intern.id,
        "intern_name": intern.name,
        "subject": tmpl["subject"].format(name=intern.name, start_date=intern.start_date),
        "body":    tmpl["body"].format(name=intern.name, start_date=intern.start_date, days=intern.days_without_response),
    }


def validate_action(action: str, intern: PersonState) -> ValidationResult:
    return validate_onboarding(action, intern.model_dump())


def apply_action(intern: PersonState, action: str) -> tuple[float, bool, str | None]:
    if action not in STANDARD_ACTIONS:
        # Dynamic action passed semantic validation → sandbox (log, no state change)
        return 0.02, False, None

    deps    = intern.dependencies.get(action, [])
    missing = [d for d in deps if not _check_dep(d, intern)]

    if action == "grant_system_access" and intern.access_delayed:
        return -0.05, False, "access_delayed — use request_alternative_access"
    if action == "escalate_to_manager" and intern.days_without_response < 3:
        return -0.05, False, "escalate_to_manager requires days_without_response >= 3"
    if missing:
        return -0.05, False, f"missing dependencies: {missing}"

    reward = 0.0
    if action == "send_welcome_email":
        intern.checklist["welcome_email_sent"] = True
        intern.intern_confirmed = True
        reward = 0.20
    elif action == "share_docs":
        intern.checklist["docs_shared"] = True
        reward = 0.20
    elif action == "share_international_docs":
        intern.checklist["intl_docs_shared"] = True
        reward = 0.10
    elif action == "schedule_intro_meeting":
        intern.checklist["intro_scheduled"] = True
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
        reward = 0.05
    elif action == "escalate_to_manager":
        intern.checklist["escalated"] = True
        reward = 0.05
    else:
        return -0.05, False, f"unhandled action '{action}'"

    return reward, _intern_complete(intern), None


def classify_difficulty(interns: list[PersonState]) -> str:
    has_hard = any(
        intern.is_international
        or intern.access_delayed
        or (intern.days_without_response >= 3 and not intern.intern_confirmed)
        for intern in interns
    )
    if has_hard:
        return "hard"
    return "medium" if any(i.days_without_response >= 1 for i in interns) else "easy"


def all_complete(interns: list[PersonState]) -> bool:
    return all(_intern_complete(i) for i in interns)