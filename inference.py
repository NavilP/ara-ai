"""
Inference script for the OpenEnv onboarding environment.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Optional

import requests
from openai import OpenAI
from openenv.core.containers.runtime import LocalDockerProvider

from my_env import MyEnv, OnboardingAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME", "")
ENV_BASE_URL = os.getenv("ONBOARDING_URL") or os.getenv("ENV_BASE_URL", "http://localhost:8000")
ENV_BASE_URL = ENV_BASE_URL.rstrip("/")
BENCHMARK = os.getenv("BENCHMARK", "onboarding-env")
MAX_STEPS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.80

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are operating an HR onboarding coordination environment.
    Pick exactly one next action that maximizes progress and avoids invalid transitions.

    Rules:
    - Only choose actions that appear in available_actions for that intern.
    - Escalate blocked interns when escalate_to_manager is available.
    - Send follow-up emails when send_followup_email is available.
    - For access_delayed interns, prefer request_alternative_access over grant_system_access.
    - Return only compact JSON on one line:
      {"intern_id":"<id>","action":"<action>"}
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def choose_fallback_action(observation: dict[str, Any]) -> tuple[str, str]:
    available = observation.get("available_actions", {})
    interns = observation.get("interns", [])

    for intern in interns:
        actions = available.get(intern["id"], [])
        if "escalate_to_manager" in actions:
            return intern["id"], "escalate_to_manager"

    for intern in interns:
        actions = available.get(intern["id"], [])
        if "send_followup_email" in actions:
            return intern["id"], "send_followup_email"

    preferred = [
        "send_welcome_email",
        "share_docs",
        "share_international_docs",
        "schedule_intro_meeting",
        "create_account",
        "request_alternative_access",
        "grant_system_access",
    ]
    for intern in interns:
        actions = available.get(intern["id"], [])
        for action in preferred:
            if action in actions:
                return intern["id"], action

    return "", ""


def get_model_action(client: OpenAI, observation: dict[str, Any]) -> tuple[str, str]:
    state_summary = {
        "task": observation.get("task"),
        "difficulty": observation.get("difficulty"),
        "step_count": observation.get("step_count", 0),
        "interns": [
            {
                "id": intern["id"],
                "name": intern["name"],
                "is_international": intern["is_international"],
                "intern_confirmed": intern["intern_confirmed"],
                "days_without_response": intern["days_without_response"],
                "access_delayed": intern["access_delayed"],
                "checklist": intern["checklist"],
            }
            for intern in observation.get("interns", [])
        ],
        "available_actions": observation.get("available_actions", {}),
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(state_summary, ensure_ascii=True)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        return parsed["intern_id"], parsed["action"]
    except Exception:
        return choose_fallback_action(observation)


def build_env() -> tuple[Any, str]:
    if LOCAL_IMAGE_NAME:
        provider = LocalDockerProvider()
        base_url = provider.start_container(LOCAL_IMAGE_NAME)
        provider.wait_for_ready(base_url)
        return MyEnv(base_url=base_url, provider=provider).sync(), base_url
    return MyEnv(base_url=ENV_BASE_URL).sync(), ENV_BASE_URL


def fetch_tasks(base_url: str) -> list[str]:
    try:
        response = requests.get(f"{base_url}/tasks", timeout=30)
        response.raise_for_status()
        payload = response.json()
        return [task["id"] for task in payload.get("tasks", [])] or ["easy", "medium", "hard"]
    except Exception:
        return ["easy", "medium", "hard"]


def select_task(base_url: str, task: str) -> None:
    response = requests.post(f"{base_url}/task/{task}", timeout=30)
    response.raise_for_status()


def reward_parts(result: Any) -> tuple[float, float, Optional[str]]:
    reward = result.reward if isinstance(result.reward, dict) else {}
    step_reward = float(reward.get("step_reward", 0.0))
    score = float(reward.get("score", 0.0))
    info = reward.get("info", {})
    error = info.get("error") if isinstance(info, dict) else None
    return step_reward, score, error


def run_task(client: OpenAI, env: Any, base_url: str, task: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        select_task(base_url, task)
        result = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            observation = result.observation.model_dump()
            intern_id, action_name = get_model_action(client, observation)
            if not intern_id or not action_name:
                break

            action = OnboardingAction(intern_id=intern_id, action=action_name)
            result = env.step(action)
            step_reward, score, error = reward_parts(result)

            rewards.append(step_reward)
            steps_taken = step

            action_str = json.dumps(
                {"intern_id": intern_id, "action": action_name},
                separators=(",", ":"),
            )
            log_step(
                step=step,
                action=action_str,
                reward=step_reward,
                done=result.done,
                error=error,
            )

            if result.done:
                break

        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env, base_url = build_env()

    try:
        with env:
            for task in fetch_tasks(base_url):
                run_task(client, env, base_url, task)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
