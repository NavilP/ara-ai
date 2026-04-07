"""
inference.py — onboarding-env
==============================
Runs the LLM agent against all 3 tasks using OnboardingEnvClient.

MANDATORY environment variables:
    API_BASE_URL    LLM endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME      Model identifier
    HF_TOKEN        API key
    ONBOARDING_URL  Server URL    (default: http://localhost:7860)

STDOUT FORMAT (exact — OpenEnv spec):
    [START] task=<task> env=onboarding-env model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Optional

from openai import OpenAI

from client import OnboardingEnvClient

# ---------------------------------------------------------------------------
# Configuration — all from environment variables, nothing hardcoded
# ---------------------------------------------------------------------------

API_BASE_URL   = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME     = os.getenv("MODEL_NAME",   "<your-active-model>")
API_KEY        = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ONBOARDING_URL = os.getenv("ONBOARDING_URL", "http://localhost:7860")

TASKS             = ["easy", "medium", "hard"]
MAX_STEPS         = 20
TEMPERATURE       = 0.0
MAX_TOKENS        = 512
SUCCESS_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Log helpers — exact OpenEnv spec format, no deviation allowed
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=onboarding-env model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM decision
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an HR onboarding coordinator agent.
    You receive the current state of interns and available actions per intern.

    Pick ONE action to execute next to maximise onboarding progress.

    Priority rules (in order):
    1. Escalate interns with days_without_response >= 3 AND intern_confirmed=false.
    2. Send followup to interns with days_without_response >= 1.
    3. For access_delayed interns use request_alternative_access, never grant_system_access.
    4. Continue standard flow for remaining interns, most advanced first.

    Respond ONLY with a single JSON object on one line:
    {"intern_id": "<id>", "action": "<action>"}
    No explanation, no markdown, no extra text.
""").strip()


def decide_action(client: OpenAI, observation: dict) -> tuple[str, str]:
    """Ask the LLM to pick the next action. Returns (intern_id, action)."""
    state_summary = {
        "interns": [
            {
                "id": i["id"],
                "name": i["name"],
                "intern_confirmed": i["intern_confirmed"],
                "is_international": i["is_international"],
                "days_without_response": i["days_without_response"],
                "access_delayed": i["access_delayed"],
                "checklist": i["checklist"],
            }
            for i in observation.get("interns", [])
        ],
        "available_actions": observation.get("available_actions", {}),
        "step_count": observation.get("step_count", 0),
    }

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current state:\n{json.dumps(state_summary, indent=2)}\n\nPick the next action."},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)
        return parsed["intern_id"], parsed["action"]

    except Exception as exc:
        print(f"[DEBUG] LLM failed: {exc}", flush=True)
        # Fallback — pick first available action from first intern
        for iid, actions in observation.get("available_actions", {}).items():
            if actions:
                return iid, actions[0]
        return "", ""


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(llm: OpenAI, env: OnboardingEnvClient, task: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, model=MODEL_NAME)

    try:
        observation = env.reset(task=task)

        for step in range(1, MAX_STEPS + 1):
            intern_id, action = decide_action(llm, observation)
            if not intern_id or not action:
                print("[DEBUG] No action available — ending episode.", flush=True)
                break

            action_str = json.dumps({"intern_id": intern_id, "action": action})

            result   = env.step(intern_id=intern_id, action=action)
            reward   = env.get_step_reward(result)
            done     = env.is_done(result)
            error    = env.get_error(result)
            score    = env.get_score(result)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            observation = result["observation"]
            observation["_done"] = done

            if done:
                break

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task '{task}' crashed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = OnboardingEnvClient(base_url=ONBOARDING_URL)

    for task in TASKS:
        run_task(llm=llm, env=env, task=task)
        print("", flush=True)


if __name__ == "__main__":
    main()