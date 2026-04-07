"""
Inference Script — onboarding-env
==================================
Runs the agent against all 3 tasks (easy, medium, hard) using an LLM
to decide which action to take at each step.

MANDATORY environment variables:
    API_BASE_URL      LLM endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          API key / HuggingFace token
    ONBOARDING_URL    Base URL of the onboarding-env server (default: http://localhost:7860)

STDOUT FORMAT (exact, no deviation):
    [START] task=<task> env=onboarding-env model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
from typing import Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL      = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME        = os.getenv("MODEL_NAME", "<your-active-model>")
API_KEY           = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ONBOARDING_URL    = os.getenv("ONBOARDING_URL", "http://localhost:7860").rstrip("/")

TASKS             = ["easy", "medium", "hard"]
MAX_STEPS         = 20
TEMPERATURE       = 0.0
MAX_TOKENS        = 512
SUCCESS_THRESHOLD = 0.5   # minimum score to count as success

# ---------------------------------------------------------------------------
# Logging helpers  (exact spec format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=onboarding-env model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def env_reset(task: str) -> dict:
    resp = requests.post(f"{ONBOARDING_URL}/reset", json={"task": task}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(intern_id: str, action: str) -> dict:
    resp = requests.post(
        f"{ONBOARDING_URL}/step",
        json={"intern_id": intern_id, "action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM decision
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an HR onboarding coordinator agent.
    You will receive the current state of interns and a list of available actions per intern.

    Your job: pick ONE action to execute next to maximise onboarding progress.

    Rules:
    - Only pick actions from the available_actions list.
    - Prioritise: complete blocked interns first, handle exceptions (escalate, alternative access).
    - Send followup_email to interns with days_without_response >= 1 before trying to advance them.
    - Escalate interns with days_without_response >= 3 and intern_confirmed=false.
    - For access_delayed interns, use request_alternative_access (never grant_system_access).

    Respond ONLY with a JSON object on a single line:
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
    user_msg = f"Current state:\n{json.dumps(state_summary, indent=2)}\n\nPick the next action."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)
        intern_id = parsed["intern_id"]
        action = parsed["action"]
        return intern_id, action
    except Exception as exc:
        print(f"[DEBUG] LLM decision failed: {exc}", flush=True)
        # Fallback: pick first available action from first intern
        avail = observation.get("available_actions", {})
        for iid, actions in avail.items():
            if actions:
                return iid, actions[0]
        return "", ""


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, model=MODEL_NAME)

    try:
        observation = env_reset(task)

        for step in range(1, MAX_STEPS + 1):
            # Check if already done
            if observation.get("_done"):
                break

            intern_id, action = decide_action(client, observation)
            if not intern_id or not action:
                print(f"[DEBUG] No action available — ending episode.", flush=True)
                break

            action_str = json.dumps({"intern_id": intern_id, "action": action})

            try:
                result = env_step(intern_id, action)
            except requests.HTTPError as exc:
                log_step(step=step, action=action_str, reward=0.0, done=False, error=str(exc))
                steps_taken = step
                break

            step_reward: float = result["reward"]["step_reward"]
            done: bool = result["done"]
            error: Optional[str] = result["info"].get("error")
            score = result["reward"]["score"]

            rewards.append(step_reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=step_reward, done=done, error=error)

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
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        run_task(client, task)
        print("", flush=True)  # blank line between tasks


if __name__ == "__main__":
    main()
