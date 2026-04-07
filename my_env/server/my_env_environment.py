"""
OpenEnv onboarding environment implementation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        OnboardingAction,
        OnboardingObservation,
        OnboardingReward,
        OnboardingState,
        PersonState,
        TaskId,
    )
except ImportError:
    from models import (  # type: ignore
        OnboardingAction,
        OnboardingObservation,
        OnboardingReward,
        OnboardingState,
        PersonState,
        TaskId,
    )

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
TASKS: list[TaskId] = ["easy", "medium", "hard"]
FIXTURES_PATH = Path(__file__).resolve().parent.parent / "data" / "fixtures.json"

EMAIL_TEMPLATES: dict[str, dict[str, str]] = {
    "send_welcome_email": {
        "subject": "Welcome to the team, {name}!",
        "body": (
            "Hi {name},\n\nWe're excited to have you joining us on {start_date}. "
            "Please reply to confirm your start date so we can finalize onboarding.\n\n"
            "Best regards,\nHR Team"
        ),
    },
    "share_docs": {
        "subject": "Your onboarding documents - {name}",
        "body": (
            "Hi {name},\n\nPlease review your onboarding documents before day one.\n\n"
            "Best regards,\nHR Team"
        ),
    },
    "share_international_docs": {
        "subject": "International onboarding package - {name}",
        "body": (
            "Hi {name},\n\nPlease review the immigration and relocation packet attached "
            "to your onboarding case.\n\nBest regards,\nHR Team"
        ),
    },
    "schedule_intro_meeting": {
        "subject": "Intro meeting invitation - {name}",
        "body": (
            "Hi {name},\n\nWe've reserved time for your team introduction during your "
            "first week. A calendar invite will follow.\n\nBest regards,\nHR Team"
        ),
    },
    "send_followup_email": {
        "subject": "Following up on your onboarding - {name}",
        "body": (
            "Hi {name},\n\nWe haven't heard back from you yet. Please reply so we can "
            "continue your onboarding process.\n\nBest regards,\nHR Team"
        ),
    },
    "escalate_to_manager": {
        "subject": "Escalation: no response from {name}",
        "body": (
            "Hi Manager,\n\n{name} has not responded for {days} days. Manual support is "
            "required to continue this onboarding case.\n\nBest regards,\nHR System"
        ),
    },
}


def _load_task(task_name: TaskId) -> list[PersonState]:
    raw = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    return [PersonState(**person) for person in copy.deepcopy(raw[task_name])]


def _check_dep(dep: str, intern: PersonState) -> bool:
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


def _intern_complete(intern: PersonState) -> bool:
    required = [
        "welcome_email_sent",
        "docs_shared",
        "intro_scheduled",
        "account_created",
        "access_granted",
    ]
    if intern.is_international:
        required.append("intl_docs_shared")
    if intern.checklist.get("escalated"):
        return True
    return all(intern.checklist.get(item, False) for item in required)


def _all_complete(interns: list[PersonState]) -> bool:
    return all(_intern_complete(intern) for intern in interns)


def _classify_difficulty(interns: list[PersonState]) -> TaskId:
    has_hard = any(
        intern.is_international
        or intern.access_delayed
        or (intern.days_without_response >= 3 and not intern.intern_confirmed)
        for intern in interns
    )
    if has_hard:
        return "hard"

    has_medium = any(intern.days_without_response >= 1 for intern in interns)
    return "medium" if has_medium else "easy"


def _build_email_preview(action: str, intern: PersonState) -> dict[str, Any] | None:
    template = EMAIL_TEMPLATES.get(action)
    if not template:
        return None

    return {
        "type": action,
        "intern_id": intern.id,
        "intern_name": intern.name,
        "subject": template["subject"].format(name=intern.name, start_date=intern.start_date),
        "body": template["body"].format(
            name=intern.name,
            start_date=intern.start_date,
            days=intern.days_without_response,
        ),
    }


def _available_actions(intern: PersonState) -> list[str]:
    if _intern_complete(intern):
        return []

    available: list[str] = []
    for action in VALID_ACTIONS:
        deps = intern.dependencies.get(action, [])
        if not all(_check_dep(dep, intern) for dep in deps):
            continue
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
        if action in {"grant_system_access", "request_alternative_access"} and intern.checklist.get("access_granted"):
            continue
        if action == "send_followup_email" and intern.checklist.get("followup_sent"):
            continue
        if action == "escalate_to_manager" and intern.checklist.get("escalated"):
            continue
        available.append(action)
    return available


def _grade_easy(interns: list[PersonState]) -> float:
    if not interns:
        return 0.0
    required = ["welcome_email_sent", "docs_shared", "intro_scheduled", "account_created", "access_granted"]
    completed = sum(1 for key in required if interns[0].checklist.get(key, False))
    return round(min(completed / len(required), 1.0), 2)


def _grade_medium(interns: list[PersonState]) -> float:
    if not interns:
        return 0.0
    total_expected = 16
    standard = ["welcome_email_sent", "docs_shared", "intro_scheduled", "account_created", "access_granted"]
    completed = 0
    for intern in interns:
        completed += sum(1 for key in standard if intern.checklist.get(key, False))
        if intern.id == "intern_002" and intern.checklist.get("followup_sent", False):
            completed += 1
    return round(min(completed / total_expected, 1.0), 2)


def _grade_hard(interns: list[PersonState]) -> float:
    if not interns:
        return 0.0
    by_id = {intern.id: intern for intern in interns}
    base_total = 12
    base_weight = 0.70
    base_completed = 0
    bonus = 0.0

    ahmed = by_id.get("intern_001")
    if ahmed:
        for key in ["welcome_email_sent", "docs_shared", "intl_docs_shared", "intro_scheduled", "account_created", "access_granted"]:
            base_completed += int(ahmed.checklist.get(key, False))
        if ahmed.checklist.get("intl_docs_shared", False):
            bonus += 0.10

    rohan = by_id.get("intern_002")
    if rohan:
        base_completed += int(rohan.checklist.get("escalated", False))
        if rohan.checklist.get("escalated", False):
            bonus += 0.10

    sara = by_id.get("intern_003")
    if sara:
        for key in ["welcome_email_sent", "docs_shared", "intro_scheduled", "account_created", "access_granted"]:
            base_completed += int(sara.checklist.get(key, False))
        if sara.access_delayed and sara.checklist.get("access_granted", False):
            bonus += 0.10

    score = ((base_completed / base_total) * base_weight) + bonus
    return round(min(max(score, 0.0), 1.0), 2)


def _grade(task: TaskId, interns: list[PersonState]) -> float:
    if task == "easy":
        return _grade_easy(interns)
    if task == "medium":
        return _grade_medium(interns)
    return _grade_hard(interns)


def _apply_action(intern: PersonState, action: str) -> tuple[float, bool, str | None]:
    if action not in VALID_ACTIONS:
        return 0.0, False, f"unknown action '{action}'"

    deps = intern.dependencies.get(action, [])
    missing = [dep for dep in deps if not _check_dep(dep, intern)]
    if action == "grant_system_access" and intern.access_delayed:
        return 0.0, False, "access_delayed - use request_alternative_access"
    if action == "escalate_to_manager" and intern.days_without_response < 3:
        return 0.0, False, "escalate_to_manager requires days_without_response >= 3"
    if missing:
        return 0.0, False, f"missing dependencies: {missing}"

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

    return reward, _intern_complete(intern), None


def task_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for task in TASKS:
        interns = _load_task(task)
        summaries.append(
            {
                "id": task,
                "difficulty": _classify_difficulty(interns),
                "intern_count": len(interns),
                "intern_names": [intern.name for intern in interns],
                "max_steps": 20,
            }
        )
    return summaries


class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    DEFAULT_TASK: ClassVar[TaskId] = "easy"

    @classmethod
    def set_default_task(cls, task: TaskId) -> None:
        cls.DEFAULT_TASK = task

    def __init__(self):
        self._episode_id = str(uuid4())
        self._task: TaskId = self.DEFAULT_TASK
        self._difficulty: TaskId = self.DEFAULT_TASK
        self._interns: list[PersonState] = []
        self._communications_log: list[dict[str, Any]] = []
        self._step_count = 0
        self._cumulative_score = 0.0
        self._done = False
        self._last_reward = OnboardingReward(step_reward=0.0, score=0.0, done=False, info={})

    def _snapshot(self) -> OnboardingState:
        return OnboardingState(
            task=self._task,
            difficulty=self._difficulty,
            interns=self._interns,
            communications_log=self._communications_log,
            step_count=self._step_count,
            cumulative_score=round(self._cumulative_score, 2),
            done=self._done,
        )

    def _observation(self, message: str, reward: OnboardingReward | None = None) -> OnboardingObservation:
        reward_payload = reward or self._last_reward
        return OnboardingObservation(
            task=self._task,
            difficulty=self._difficulty,
            interns=self._interns,
            available_actions={intern.id: _available_actions(intern) for intern in self._interns},
            communications_log=self._communications_log,
            step_count=self._step_count,
            message=message,
            done=self._done,
            reward=round(reward_payload.score, 2),
            metadata={
                "episode_id": self._episode_id,
                "state": self._snapshot().model_dump(),
                "valid_tasks": TASKS,
                "reward": reward_payload.model_dump(),
            },
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: TaskId | None = None,
        **kwargs: Any,
    ) -> OnboardingObservation:
        del seed, kwargs
        selected_task = task if task in TASKS else self.DEFAULT_TASK
        self._episode_id = episode_id or str(uuid4())
        self._task = selected_task
        self._interns = _load_task(self._task)
        self._difficulty = _classify_difficulty(self._interns)
        self._communications_log = []
        self._step_count = 0
        self._cumulative_score = 0.0
        self._done = False
        self._last_reward = OnboardingReward(step_reward=0.0, score=0.0, done=False, info={"task": self._task})
        return self._observation(
            message=f"Episode reset for task '{self._task}'. Difficulty: {self._difficulty}.",
            reward=self._last_reward,
        )

    def step(self, action: OnboardingAction) -> OnboardingObservation:  # type: ignore[override]
        if not self._interns:
            self.reset()

        if self._done:
            reward = OnboardingReward(
                step_reward=0.0,
                score=round(self._cumulative_score, 2),
                done=True,
                info={"error": "Episode is done. Reset before taking more actions."},
            )
            self._last_reward = reward
            return self._observation(message="Episode already complete.", reward=reward)

        intern = next((person for person in self._interns if person.id == action.intern_id), None)
        if intern is None:
            reward = OnboardingReward(
                step_reward=0.0,
                score=round(self._cumulative_score, 2),
                done=False,
                info={"error": f"intern_id '{action.intern_id}' not found"},
            )
            self._last_reward = reward
            return self._observation(message=reward.info["error"], reward=reward)

        step_reward, intern_done, error = _apply_action(intern, action.action)
        self._step_count += 1

        email_preview = _build_email_preview(action.action, intern)
        if email_preview and error is None:
            self._communications_log.append(email_preview)

        self._cumulative_score = _grade(self._task, self._interns)
        self._done = _all_complete(self._interns) or self._step_count >= 20

        reward = OnboardingReward(
            step_reward=round(step_reward, 2),
            score=round(self._cumulative_score, 2),
            done=self._done,
            info={"error": error, "intern_done": intern_done},
        )
        self._last_reward = reward
        message = error or f"Applied '{action.action}' to {intern.name}."
        return self._observation(message=message, reward=reward)

    @property
    def state(self) -> OnboardingState:
        return self._snapshot()
