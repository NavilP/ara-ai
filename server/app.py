from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
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
    validate_action,
)
from graders.grader import grade
from models import InternAction, OnboardingObservation, OnboardingReward, PersonState

app = FastAPI(title="onboarding-env", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

KNOWN_TASKS = ["easy", "medium", "hard"]


class EpisodeState:
    def __init__(self) -> None:
        self.task:               str                    = ""
        self.difficulty:         str                    = ""
        self.interns:            list[PersonState]      = []
        self.communications_log: list[dict[str, Any]]  = []
        self.step_count:         int                    = 0
        self.cumulative_score:   float                  = 0.0
        self.done:               bool                   = False


_episode = EpisodeState()


def _build_observation(message: str = "") -> OnboardingObservation:
    return OnboardingObservation(
        interns=_episode.interns,
        available_actions={i.id: get_available_actions(i) for i in _episode.interns},
        communications_log=_episode.communications_log,
        step_count=_episode.step_count,
        difficulty=_episode.difficulty,
        task=_episode.task,
        message=message,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset(body: dict = None) -> dict[str, Any]:
    body   = body or {}
    task   = body.get("task", "easy")
    if task not in KNOWN_TASKS:
        task = "easy"

    _episode.task               = task
    _episode.interns            = load_task(task)
    _episode.difficulty         = classify_difficulty(_episode.interns)
    _episode.communications_log = []
    _episode.step_count         = 0
    _episode.cumulative_score   = 0.0
    _episode.done               = False

    return _build_observation(f"Episode reset for task '{task}'.").model_dump()


@app.post("/step")
def step(action: InternAction) -> dict[str, Any]:
    if _episode.done or not _episode.interns:
        reward = OnboardingReward(step_reward=0.0, score=_episode.cumulative_score, done=True,
                                  info={"error": "Episode done. Call /reset first."})
        return {"observation": _build_observation("Episode done.").model_dump(),
                "reward": reward.model_dump(), "done": True, "info": reward.info}

    intern = next((i for i in _episode.interns if i.id == action.intern_id), None)
    if not intern:
        reward = OnboardingReward(step_reward=0.0, score=_episode.cumulative_score, done=False,
                                  info={"error": f"intern_id '{action.intern_id}' not found"})
        return {"observation": _build_observation(reward.info["error"]).model_dump(),
                "reward": reward.model_dump(), "done": False, "info": reward.info}

    # --- Semantic validation (zone 1 / 2 / 3) ---
    validation = validate_action(action.action, intern)
    if not validation.valid:
        step_reward = max(0.0, 0.0 + validation.reward_modifier)
        reward = OnboardingReward(
            step_reward=step_reward,
            score=_episode.cumulative_score,
            done=False,
            info={
                "error": validation.reason,
                "zone":  validation.zone,
                "dynamic": action.action not in {
                    "send_welcome_email", "share_docs", "share_international_docs",
                    "schedule_intro_meeting", "create_account", "grant_system_access",
                    "request_alternative_access", "send_followup_email", "escalate_to_manager",
                }
            },
        )
        return {"observation": _build_observation(validation.reason).model_dump(),
                "reward": reward.model_dump(), "done": False, "info": reward.info}

    # --- Apply action ---
    step_reward, _intern_done, error = apply_action(intern, action.action)

    # Log email if generated
    email = build_email_preview(action.action, intern)
    if email and error is None:
        _episode.communications_log.append(email)

    _episode.step_count      += 1
    _episode.cumulative_score = grade(_episode.task, _episode.interns, _episode.step_count)
    _episode.done             = all_complete(_episode.interns) or _episode.step_count >= 20

    is_dynamic = action.action not in {
        "send_welcome_email", "share_docs", "share_international_docs",
        "schedule_intro_meeting", "create_account", "grant_system_access",
        "request_alternative_access", "send_followup_email", "escalate_to_manager",
    }

    reward = OnboardingReward(
        step_reward=step_reward,
        score=_episode.cumulative_score,
        done=_episode.done,
        info={
            "error":   error,
            "dynamic": is_dynamic,
            "zone":    3,
        },
    )

    msg = error or f"Applied '{action.action}' to {intern.name}."
    return {
        "observation": _build_observation(msg).model_dump(),
        "reward":      reward.model_dump(),
        "done":        _episode.done,
        "info":        reward.info,
    }


@app.post("/agent/next")
def agent_next() -> dict[str, Any]:
    """Greedy baseline — always uses Heuristic++ for the demo UI."""
    if not _episode.interns or _episode.done:
        return {"intern_id": None, "action": None, "reason": "No active episode."}

    flow_order = [
        "send_welcome_email", "share_docs", "share_international_docs",
        "schedule_intro_meeting", "create_account",
        "request_alternative_access", "grant_system_access",
    ]

    available = {i.id: get_available_actions(i) for i in _episode.interns}

    # Priority 1: escalate blocked interns
    for intern in _episode.interns:
        if "escalate_to_manager" in available.get(intern.id, []):
            return {"intern_id": intern.id, "action": "escalate_to_manager",
                    "reason": f"{intern.name} needs escalation."}

    # Priority 2: followup
    for intern in _episode.interns:
        if "send_followup_email" in available.get(intern.id, []):
            return {"intern_id": intern.id, "action": "send_followup_email",
                    "reason": f"{intern.name} needs a follow-up."}

    # Priority 3: standard flow, most advanced intern first
    def progress(i: PersonState) -> int:
        return sum(1 for v in i.checklist.values() if v)

    for intern in sorted(_episode.interns, key=progress, reverse=True):
        for act in flow_order:
            if act in available.get(intern.id, []):
                return {"intern_id": intern.id, "action": act,
                        "reason": f"Continuing onboarding for {intern.name}."}

    return {"intern_id": None, "action": None, "reason": "No more actions available."}


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    tasks = []
    for task in KNOWN_TASKS:
        interns    = load_task(task)
        difficulty = classify_difficulty(interns)
        tasks.append({
            "id":           task,
            "difficulty":   difficulty,
            "intern_count": len(interns),
            "intern_names": [i.name for i in interns],
        })
    return {"tasks": tasks}


@app.get("/state")
def get_state() -> dict[str, Any]:
    return {
        "task":               _episode.task,
        "difficulty":         _episode.difficulty,
        "step_count":         _episode.step_count,
        "cumulative_score":   _episode.cumulative_score,
        "done":               _episode.done,
        "interns": [
            {
                "id":       i.id,
                "name":     i.name,
                "checklist":i.checklist,
                "complete": all(i.checklist.get(k, False) for k in
                                ["welcome_email_sent","docs_shared","intro_scheduled",
                                 "account_created","access_granted"]),
            }
            for i in _episode.interns
        ],
        "available_actions":  {i.id: get_available_actions(i) for i in _episode.interns},
    }


@app.get("/")
def ui_root() -> HTMLResponse:
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>onboarding-env</h1><p>UI not found.</p>")


_ui_dir = Path(__file__).parent / "ui"
if _ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_ui_dir), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
