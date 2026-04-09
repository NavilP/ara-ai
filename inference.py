"""
inference.py — onboarding-env
==============================

MANDATORY env vars (set in .env or HuggingFace Space Secrets):
    API_BASE_URL    LLM endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME      Model identifier  (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN        Your Hugging Face API key

ARCHITECTURE
    Router: Heuristic++ for happy path, Hybrid RL+LLM for exceptions.
    Semantic validator on server rejects incoherent/out-of-domain actions.
    N_EPISODES per task. First N-1 episodes run silently to build memory
    (in-context RL). Only the final episode emits [START]/[STEP]/[END].

STDOUT FORMAT (OpenEnv spec — do not modify)
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,...>
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import List, Optional

import requests as _requests
from dotenv import load_dotenv
from openai import OpenAI

from agent.heuristic import get_action as heuristic_action
from agent.llm_agent  import get_action as llm_action
from agent.memory     import EpisodeRecord, StepRecord, TrajectoryMemory
from client           import OnboardingEnvClient

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL   = os.getenv("API_BASE_URL")
MODEL_NAME     = os.getenv("MODEL_NAME")
API_KEY        = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ONBOARDING_URL = os.getenv("ONBOARDING_URL", "http://localhost:7860").rstrip("/")
BENCHMARK      = os.getenv("BENCHMARK", "onboarding-env")

TASKS       = ["easy", "medium", "hard"]
N_EPISODES  = 3       # first N-1 are silent warmup; only last emits logs
MAX_STEPS   = 20
SUCCESS_THR = 0.5

_missing = [k for k, v in {
    "API_BASE_URL": API_BASE_URL,
    "MODEL_NAME":   MODEL_NAME,
    "HF_TOKEN":     API_KEY,

}.items() if not v]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(_missing)}\n"
        "Set them in a .env file or as HuggingFace Space Secrets."
    )

# ---------------------------------------------------------------------------
# Log helpers — exact OpenEnv spec, never modify
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# ---------------------------------------------------------------------------
# Reward extraction
# ---------------------------------------------------------------------------

def extract_reward(result: dict) -> tuple[float, float, Optional[str], int, bool]:
    """Returns (step_reward, score, error, zone, is_dynamic)."""
    reward = result.get("reward", {})
    if isinstance(reward, dict):
        step_r  = float(reward.get("step_reward", 0.0))
        score   = float(reward.get("score", 0.0))
        info    = reward.get("info", {})
        error   = info.get("error") if isinstance(info, dict) else None
        zone    = int(info.get("zone", 3)) if isinstance(info, dict) else 3
        dynamic = bool(info.get("dynamic", False)) if isinstance(info, dict) else False
        return step_r, score, error, zone, dynamic
    v = float(reward or 0.0)
    return v, v, None, 3, False

# ---------------------------------------------------------------------------
# Router: Heuristic++ first, LLM for ambiguous/exception states
# ---------------------------------------------------------------------------

def decide(
    llm:            OpenAI,
    observation:    dict,
    step_hist:      List[str],
    memory:         TrajectoryMemory,
    task:           str,
    failed_actions: Optional[set] = None,
) -> tuple[str, str, str, bool]:
    """Returns (intern_id, action, reasoning, used_llm)."""
    h = heuristic_action(observation)
    if h is not None:
        intern_id, action = h
        return intern_id, action, "heuristic", False

    intern_id, action, reasoning = llm_action(
        client=llm, model=MODEL_NAME,
        observation=observation,
        step_history=step_hist,
        memory_context=memory.get_context(task),
    )
    if intern_id and action:
        return intern_id, action, reasoning, True

    # Last resort: first available action not previously failed
    blocked = failed_actions or set()
    for iid, actions in observation.get("available_actions", {}).items():
        for act in actions:
            if f"{iid}:{act}" not in blocked:
                return iid, act, "fallback", False
    return "", "", "", False

# ---------------------------------------------------------------------------
# Single episode — emit_logs controls whether [STEP] lines are printed
# ---------------------------------------------------------------------------

async def run_episode(
    llm:        OpenAI,
    env:        OnboardingEnvClient,
    task:       str,
    episode_n:  int,
    memory:     TrajectoryMemory,
    emit_logs:  bool = False,       # True only for the final episode
) -> EpisodeRecord:
    record         = EpisodeRecord(task=task, episode_num=episode_n)
    failed_actions: set = set()

    try:
        _requests.post(f"{ONBOARDING_URL}/task/{task}", timeout=10)
    except Exception:
        pass
    try:
        observation = await env.reset(task=task)
    except Exception as exc:
        print(f"[DEBUG] env.reset failed for task '{task}': {exc}", flush=True)
        return record
    step_hist: List[str] = []
    score = 0.0

    for step in range(1, MAX_STEPS + 1):
        if observation.get("_done"):
            break

        intern_id, action, reasoning, used_llm = decide(
            llm, observation, step_hist, memory, task, failed_actions
        )
        if not intern_id or not action:
            break

        action_str = json.dumps(
            {"intern_id": intern_id, "action": action}, separators=(",", ":")
        )
        try:
            result = await env.step(intern_id=intern_id, action=action)
        except Exception as exc:
            print(f"[DEBUG] env.step failed at step {step}: {exc}", flush=True)
            break
        step_r, score, error, zone, is_dynamic = extract_reward(result)
        done       = env.is_done(result)

        if error:
            failed_actions.add(f"{intern_id}:{action}")

        # Only the final episode writes [STEP] to stdout
        if emit_logs:
            log_step(step=step, action=action_str, reward=step_r, done=done, error=error)

        record.steps.append(StepRecord(
            intern_id=intern_id, action=action,
            reward=step_r, error=error, zone=zone,
            dynamic=is_dynamic, reasoning=reasoning,
        ))
        step_hist.append(
            f"Step {step}: intern={intern_id} action={action} "
            f"reward={step_r:+.2f} zone={zone} "
            f"{'[dynamic]' if is_dynamic else '[standard]'} error={error or 'null'}"
        )
        try:
            observation = result["observation"]
        except (KeyError, TypeError) as exc:
            print(f"[DEBUG] Bad result format at step {step}: {exc}", flush=True)
            break
        observation["_done"] = done
        if done:
            break

    record.final_score = score
    record.success     = score >= SUCCESS_THR
    return record

# ---------------------------------------------------------------------------
# Task runner: N_EPISODES per task, only last episode is logged
# ---------------------------------------------------------------------------

async def run_task(llm: OpenAI, task: str, memory: TrajectoryMemory) -> None:
    env: Optional[OnboardingEnvClient] = None
    final_rewards: List[float] = []
    final_score   = 0.0
    final_steps   = 0
    success       = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = OnboardingEnvClient(base_url=ONBOARDING_URL)

        for episode_n in range(1, N_EPISODES + 1):
            is_last    = (episode_n == N_EPISODES)
            emit_logs  = is_last          # only final episode emits [STEP]
            record     = await run_episode(llm, env, task, episode_n, memory, emit_logs)
            memory.add(record)

            if not is_last:
                # Warmup summary — visible for debugging but not parsed by evaluator
                print(f"[DEBUG] warmup episode {episode_n}/{N_EPISODES - 1}"
                      f" score={record.final_score:.2f}", flush=True)
            else:
                final_rewards = [s.reward for s in record.steps]
                final_score   = record.final_score
                final_steps   = len(record.steps)
                success       = record.success

    except Exception as exc:
        print(f"[DEBUG] Task '{task}' crashed: {exc}", flush=True)
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=final_steps, score=final_score, rewards=final_rewards)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    llm    = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    memory = TrajectoryMemory(max_episodes=N_EPISODES)
    for task in TASKS:
        await run_task(llm=llm, task=task, memory=memory)
        print("", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
