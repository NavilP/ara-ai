from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from environment import (
    all_complete,
    apply_action,
    build_email_preview,
    classify_difficulty,
    get_available_actions,
    load_task,
)
from graders import grader_easy, grader_medium, grader_hard
from models import InternAction, OnboardingObservation, OnboardingReward, PersonState

app = FastAPI(title="onboarding-env", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRADERS = {
    "easy":   grader_easy.grade,
    "medium": grader_medium.grade,
    "hard":   grader_hard.grade,
}

KNOWN_TASKS = ["easy", "medium", "hard"]


class EpisodeState:
    def __init__(self) -> None:
        self.task: str = ""
        self.difficulty: str = ""
        self.interns: list[PersonState] = []
        self.communications_log: list[dict[str, Any]] = []
        self.step_count: int = 0
        self.cumulative_score: float = 0.0
        self.done: bool = False


_episode = EpisodeState()


def _build_observation(message: str = "") -> OnboardingObservation:
    return OnboardingObservation(
        interns=_episode.interns,
        available_actions={
            intern.id: get_available_actions(intern)
            for intern in _episode.interns
        },
        communications_log=_episode.communications_log,
        step_count=_episode.step_count,
        difficulty=_episode.difficulty,
        message=message,
    )


@app.post("/reset")
async def reset(body: dict[str, Any] = {}) -> dict[str, Any]:
    global _episode
    task = (body or {}).get("task", "easy")
    if task not in KNOWN_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Valid: {KNOWN_TASKS}")

    interns = load_task(task)
    difficulty = classify_difficulty(interns)

    _episode = EpisodeState()
    _episode.task = task
    _episode.difficulty = difficulty
    _episode.interns = interns

    obs = _build_observation(message=f"Episode reset. Task: {task} | Difficulty: {difficulty}")
    return obs.model_dump()


@app.post("/step")
async def step(action: InternAction) -> dict[str, Any]:
    if _episode.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")
    if not _episode.interns:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    intern = next((i for i in _episode.interns if i.id == action.intern_id), None)
    if intern is None:
        raise HTTPException(status_code=404, detail=f"intern_id '{action.intern_id}' not found.")

    step_reward, intern_done, error = apply_action(intern, action.action)
    _episode.step_count += 1

    email = build_email_preview(action.action, intern)
    if email and error is None:
        _episode.communications_log.append(email)

    grader = GRADERS[_episode.difficulty]
    _episode.cumulative_score = grader(_episode.interns)

    done = all_complete(_episode.interns)
    _episode.done = done

    info: dict[str, Any] = {"error": error, "intern_done": intern_done}
    obs = _build_observation(
        message=error if error else f"Step {_episode.step_count} — '{action.action}' applied."
    )
    reward = OnboardingReward(
        step_reward=step_reward,
        score=_episode.cumulative_score,
        done=done,
        info=info,
    )
    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}


@app.get("/state")
async def state() -> dict[str, Any]:
    if not _episode.interns:
        return {"message": "No active episode. Call /reset first."}
    obs = _build_observation(message="Current state")
    return {
        "observation": obs.model_dump(),
        "score": _episode.cumulative_score,
        "done": _episode.done,
        "task": _episode.task,
        "difficulty": _episode.difficulty,
    }


@app.get("/tasks")
async def tasks() -> dict[str, Any]:
    result = []
    for task_id in KNOWN_TASKS:
        interns = load_task(task_id)
        difficulty = classify_difficulty(interns)
        result.append({
            "id": task_id,
            "difficulty": difficulty,
            "intern_count": len(interns),
            "intern_names": [i.name for i in interns],
            "exceptions": _describe_exceptions(interns),
            "max_steps": 20,
        })
    return {"tasks": result}


@app.post("/agent/next")
async def agent_next() -> dict[str, Any]:
    if not _episode.interns or _episode.done:
        return {"intern_id": None, "action": None, "reason": "Episode done or not started"}

    available = {
        intern.id: get_available_actions(intern)
        for intern in _episode.interns
    }

    for intern in _episode.interns:
        if "escalate_to_manager" in available.get(intern.id, []):
            return {
                "intern_id": intern.id,
                "action": "escalate_to_manager",
                "reason": f"{intern.name} hasn't responded in {intern.days_without_response} days",
            }

    for intern in _episode.interns:
        if "send_followup_email" in available.get(intern.id, []):
            return {
                "intern_id": intern.id,
                "action": "send_followup_email",
                "reason": f"{intern.name} hasn't responded in {intern.days_without_response} day(s)",
            }

    FLOW_ORDER = [
        "send_welcome_email", "share_docs", "share_international_docs",
        "schedule_intro_meeting", "create_account",
        "request_alternative_access", "grant_system_access",
    ]

    def progress(intern: PersonState) -> int:
        return sum(1 for v in intern.checklist.values() if v)

    for intern in sorted(_episode.interns, key=progress, reverse=True):
        for preferred in FLOW_ORDER:
            if preferred in available.get(intern.id, []):
                return {
                    "intern_id": intern.id,
                    "action": preferred,
                    "reason": f"Continuing onboarding for {intern.name}",
                }

    return {"intern_id": None, "action": None, "reason": "All interns are complete"}


def _describe_exceptions(interns: list[PersonState]) -> list[str]:
    found = []
    for i in interns:
        if i.is_international:
            found.append(f"{i.name}: needs international documents")
        if i.access_delayed:
            found.append(f"{i.name}: system access delayed")
        if i.days_without_response >= 3 and not i.intern_confirmed:
            found.append(f"{i.name}: {i.days_without_response} days without response — needs escalation")
        elif i.days_without_response >= 1:
            found.append(f"{i.name}: {i.days_without_response} day(s) without response — needs follow-up")
    return found


@app.get("/")
async def index() -> HTMLResponse:
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>onboarding-env</h1><p>UI not found.</p>")


_ui_dir = Path(__file__).parent / "ui"
if _ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_ui_dir), html=True), name="ui")
