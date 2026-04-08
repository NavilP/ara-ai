from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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


class OnboardingObservation(BaseModel):
    interns: list[PersonState]
    available_actions: dict[str, list[str]]   # intern_id → valid actions right now
    communications_log: list[dict[str, Any]]
    step_count: int
    difficulty: str                           # computed from data: "easy" | "medium" | "hard"
    message: str


class InternAction(BaseModel):
    intern_id: str
    action: str


class OnboardingReward(BaseModel):
    step_reward: float
    score: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)