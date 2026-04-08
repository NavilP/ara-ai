"""
inference.py — onboarding-env
==============================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    ONBOARDING_URL      The URL of the running onboarding-env server.

- All variables are required and must be set in a .env file or exported:
    cp .env.example .env   # then fill in your values

- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using the above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=easy env=onboarding-env model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action={"intern_id":"intern_001","action":"send_welcome_email"} reward=0.20 done=false error=null
    [STEP] step=2 action={"intern_id":"intern_001","action":"share_docs"} reward=0.20 done=false error=null
    [END] success=true steps=5 score=1.000 rewards=0.20,0.20,0.20,0.20,0.20
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from client import OnboardingEnvClient

# Load variables from .env file if it exists (ignored in production/HF Spaces)
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — all from environment variables, nothing hardcoded
# ---------------------------------------------------------------------------

API_BASE_URL   = os.getenv("API_BASE_URL")
MODEL_NAME     = os.getenv("MODEL_NAME")
API_KEY        = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ONBOARDING_URL = os.getenv("ONBOARDING_URL", "http://localhost:7860")
BENCHMARK      = os.getenv("ONBOARDING_BENCHMARK", "onboarding-env")

# Validate required variables are set
_missing = [k for k, v in {
    "API_BASE_URL": API_BASE_URL,
    "MODEL_NAME":   MODEL_NAME,
    "HF_TOKEN / API_KEY": API_KEY,
}.items() if not v]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(_missing)}\n"
        f"Set them in a .env file or export them before running."
    )

TASKS             = ["easy", "medium", "hard"]
MAX_STEPS         = 20
TEMPERATURE       = 0.0
MAX_TOKENS        = 512
SUCCESS_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Log helpers — exact OpenEnv spec format, no deviation allowed
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM decision
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an HR onboarding coordinator agent.
    You receive the current state of interns and a list of available actions per intern.

    Your job: pick ONE action to execute next to maximise onboarding progress.

    Priority rules (in order):
    1. Escalate interns with days_without_response >= 3 AND intern_confirmed=false.
    2. Send followup_email to interns with days_without_response >= 1.
    3. For access_delayed interns use request_alternative_access — never grant_system_access.
    4. Continue standard flow for remaining interns, most advanced intern first.

    Respond ONLY with a single JSON object on one line:
    {"intern_id": "<id>", "action": "<action>"}
    No explanation, no markdown, no extra text.
""").strip()


def get_model_action(
    client: OpenAI,
    observation: dict,
    history: List[str],
) -> tuple[str, str]:
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

    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
        Current state:
        {json.dumps(state_summary, indent=2)}

        Previous steps:
        {history_block}

        Pick the next action.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        parsed    = json.loads(raw)
        intern_id = parsed["intern_id"]
        action    = parsed["action"]
        return intern_id, action

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback — pick first available action from first intern
        for iid, actions in observation.get("available_actions", {}).items():
            if actions:
                return iid, actions[0]
        return "", ""


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

async def run_task(llm: OpenAI, task: str) -> None:
    env = OnboardingEnvClient(base_url=ONBOARDING_URL)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = await env.reset(task=task)

        for step in range(1, MAX_STEPS + 1):
            if observation.get("_done"):
                break

            intern_id, action = get_model_action(llm, observation, history)
            if not intern_id or not action:
                print("[DEBUG] No action available — ending episode.", flush=True)
                break

            action_str = json.dumps({"intern_id": intern_id, "action": action})

            result  = await env.step(intern_id=intern_id, action=action)
            reward  = env.get_step_reward(result)
            done    = env.is_done(result)
            error   = env.get_error(result)
            score   = env.get_score(result)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: intern={intern_id} action={action} reward={reward:+.2f} error={error or 'null'}"
            )

            observation          = result["observation"]
            observation["_done"] = done

            if done:
                break

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task '{task}' crashed: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        await run_task(llm=llm, task=task)
        print("", flush=True)  # blank line between tasks


if __name__ == "__main__":
    asyncio.run(main())