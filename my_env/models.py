"""
Typed models for the onboarding environment.
"""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

TaskId = Literal["easy", "medium", "hard"]
ActionName = Literal[
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


class PersonState(BaseModel):
    id: str
    name: str
    type: Literal["intern", "employee", "client"]
    start_date: str
    is_international: bool
    intern_confirmed: bool
    days_without_response: int
    access_delayed: bool
    checklist: dict[str, bool]
    dependencies: dict[str, list[str]]


class OnboardingReward(BaseModel):
    step_reward: float = Field(..., ge=0.0, le=1.0)
    score: float = Field(..., ge=0.0, le=1.0)
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class OnboardingState(BaseModel):
    task: TaskId
    difficulty: TaskId
    interns: list[PersonState]
    communications_log: list[dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
    cumulative_score: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = False


class OnboardingAction(Action):
    intern_id: str = Field(..., description="Identifier of the intern to act on")
    action: ActionName = Field(..., description="Workflow action to execute")


class OnboardingObservation(Observation):
    task: TaskId = "easy"
    difficulty: TaskId = "easy"
    interns: list[PersonState] = Field(default_factory=list)
    available_actions: dict[str, list[str]] = Field(default_factory=dict)
    communications_log: list[dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
    message: str = ""


MyAction = OnboardingAction
MyObservation = OnboardingObservation
