"""
Hybrid RL + LLM agent.

Receives the current state and trajectory memory from previous episodes.
The system prompt intentionally does NOT hardcode rules — the LLM must
infer correct behavior from context and from what the reward signal
taught it in prior episodes.

May propose dynamic actions (prefixed: comm_, doc_, access_, escalate_)
when no standard action fits the situation.
"""
from __future__ import annotations

import json
import textwrap
from typing import Optional

from openai import OpenAI

SYSTEM_PROMPT = textwrap.dedent("""
    You are an HR onboarding coordinator agent operating in a structured environment.

    You will receive:
    - The current state of one or more interns (checklist, flags, days without response)
    - A list of available standard actions per intern
    - A memory of what happened in previous episodes (rewards, mistakes, what worked)

    Your task: choose ONE action to best move the onboarding forward.

    You may choose from available_actions (standard) OR propose a new action when
    the situation genuinely requires it. New actions must use one of these prefixes:
        comm_       — communication with intern or manager
        doc_        — document-related action
        access_     — access or account action
        escalate_   — escalation action

    IMPORTANT: proposed actions must make sense in an onboarding context.
    Do not propose actions that terminate, fire, suspend, or remove an intern.

    Respond with exactly one JSON object on one line:
    {"intern_id": "<id>", "action": "<action>", "reasoning": "<one sentence>"}
    No markdown, no explanation, no extra text.
""").strip()


def get_action(
    client:         OpenAI,
    model:          str,
    observation:    dict,
    step_history:   list[str],
    memory_context: str,
) -> tuple[str, str, str]:
    """
    Ask the LLM to decide the next action.

    Returns (intern_id, action, reasoning).
    Returns ("", "", "") on any failure — caller should fall back to heuristic.
    """
    obs_summary = {
        "task":       observation.get("task"),
        "step_count": observation.get("step_count", 0),
        "interns": [
            {
                "id":                    i["id"],
                "name":                  i["name"],
                "is_international":      i["is_international"],
                "intern_confirmed":      i["intern_confirmed"],
                "days_without_response": i["days_without_response"],
                "access_delayed":        i["access_delayed"],
                "checklist":             i["checklist"],
            }
            for i in observation.get("interns", [])
        ],
        "available_actions": observation.get("available_actions", {}),
    }

    history_block = "\n".join(step_history[-4:]) if step_history else "None"
    memory_block  = memory_context or "No previous episodes recorded yet."

    user_prompt = textwrap.dedent(f"""
        === Prior episode memory ===
        {memory_block}

        === Current state ===
        {json.dumps(obs_summary, indent=2)}

        === Recent steps this episode ===
        {history_block}

        Choose the next action.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        parsed    = json.loads(raw)
        intern_id = parsed["intern_id"]
        action    = parsed["action"]
        reasoning = parsed.get("reasoning", "")
        return intern_id, action, reasoning

    except Exception as exc:
        print(f"[DEBUG] LLM failed: {exc}", flush=True)
        return "", "", ""