# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the onboarding OpenEnv environment."""

from pathlib import Path
from typing import Any

from fastapi import Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import OnboardingAction, TaskId
    from .my_env_environment import MyEnvironment, TASKS, task_summaries
except ModuleNotFoundError:
    from models import OnboardingAction, TaskId  # type: ignore
    from server.my_env_environment import MyEnvironment, TASKS, task_summaries
try:
    from ..models import OnboardingObservation
except ModuleNotFoundError:
    from models import OnboardingObservation  # type: ignore


app = create_app(
    MyEnvironment,
    OnboardingAction,
    OnboardingObservation,
    env_name="my_env",
    max_concurrent_envs=1,
)

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"
_ui_env = MyEnvironment()


def _ui_reward_payload(observation: Any) -> dict[str, Any]:
    metadata = getattr(observation, "metadata", {}) or {}
    reward = metadata.get("reward", {})
    return reward if isinstance(reward, dict) else {}


def _ui_state_response(observation: Any) -> dict[str, Any]:
    reward = _ui_reward_payload(observation)
    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": bool(getattr(observation, "done", False)),
        "info": reward.get("info", {}),
    }


def _choose_next_action(observation: OnboardingObservation) -> dict[str, Any]:
    available = observation.available_actions or {}
    interns = observation.interns or []

    for intern in interns:
        actions = available.get(intern.id, [])
        if "escalate_to_manager" in actions:
            return {
                "intern_id": intern.id,
                "action": "escalate_to_manager",
                "reason": f"{intern.name} has not responded in {intern.days_without_response} days.",
            }

    for intern in interns:
        actions = available.get(intern.id, [])
        if "send_followup_email" in actions:
            return {
                "intern_id": intern.id,
                "action": "send_followup_email",
                "reason": f"{intern.name} needs a follow-up reminder.",
            }

    flow_order = [
        "send_welcome_email",
        "share_docs",
        "share_international_docs",
        "schedule_intro_meeting",
        "create_account",
        "request_alternative_access",
        "grant_system_access",
    ]

    def progress(intern: Any) -> int:
        return sum(1 for done in intern.checklist.values() if done)

    for intern in sorted(interns, key=progress, reverse=True):
        actions = available.get(intern.id, [])
        for action in flow_order:
            if action in actions:
                return {
                    "intern_id": intern.id,
                    "action": action,
                    "reason": f"Continuing onboarding for {intern.name}.",
                }

    return {"intern_id": None, "action": None, "reason": "No more actions available."}


@app.get("/tasks")
def list_tasks() -> dict[str, object]:
    return {"tasks": task_summaries(), "active_task": MyEnvironment.DEFAULT_TASK}


@app.post("/task/{task_id}")
def select_task(task_id: TaskId) -> dict[str, object]:
    if task_id not in TASKS:
        return {"ok": False, "error": f"Unknown task '{task_id}'", "valid_tasks": TASKS}
    MyEnvironment.set_default_task(task_id)
    return {"ok": True, "active_task": task_id, "message": "Call /reset to start the selected task."}


@app.get("/ui")
def ui_index() -> FileResponse:
    return FileResponse(_UI_DIR / "index.html")


@app.get("/ui-api/tasks")
def ui_tasks() -> dict[str, object]:
    return list_tasks()


@app.post("/ui-api/task/{task_id}")
def ui_select_task(task_id: TaskId) -> dict[str, object]:
    return select_task(task_id)


@app.post("/ui-api/reset")
def ui_reset(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    task = body.get("task", MyEnvironment.DEFAULT_TASK)
    if task not in TASKS:
        return {"detail": f"Unknown task '{task}'"}
    MyEnvironment.set_default_task(task)
    observation = _ui_env.reset()
    return observation.model_dump()


@app.post("/ui-api/step")
def ui_step(action: OnboardingAction) -> dict[str, Any]:
    observation = _ui_env.step(action)
    return _ui_state_response(observation)


@app.get("/ui-api/state")
def ui_state() -> dict[str, Any]:
    state = _ui_env.state
    return state.model_dump()


@app.post("/ui-api/agent/next")
def ui_agent_next() -> dict[str, Any]:
    state = _ui_env.state
    if not state.interns or state.done:
        return {"intern_id": None, "action": None, "reason": "Episode done or not started."}
    observation = _ui_env._observation(message="UI agent inspection")
    return _choose_next_action(observation)


if _UI_DIR.exists():
    app.mount("/ui/assets", StaticFiles(directory=str(_UI_DIR)), name="ui-assets")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m my_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn my_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
