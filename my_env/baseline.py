"""Deterministic baseline for the onboarding OpenEnv environment."""

from __future__ import annotations

from typing import Iterable

from .models import OnboardingAction, TaskId
from .server.my_env_environment import MyEnvironment, TASKS

FLOW_ORDER = [
    "escalate_to_manager",
    "send_followup_email",
    "send_welcome_email",
    "share_docs",
    "share_international_docs",
    "schedule_intro_meeting",
    "create_account",
    "request_alternative_access",
    "grant_system_access",
]


def _choose_action(available_actions: dict[str, list[str]]) -> OnboardingAction | None:
    for preferred in FLOW_ORDER:
        for intern_id, actions in available_actions.items():
            if preferred in actions:
                return OnboardingAction(intern_id=intern_id, action=preferred)
    return None


def run_task(task: TaskId) -> None:
    MyEnvironment.set_default_task(task)
    env = MyEnvironment()
    observation = env.reset()
    rewards: list[float] = []

    print(f"[START] task={task}")
    while not observation.done and observation.step_count < 20:
        action = _choose_action(observation.available_actions)
        if action is None:
            break
        observation = env.step(action)
        reward = observation.metadata.get("reward", {}) if isinstance(observation.metadata, dict) else {}
        rewards.append(float(reward.get("step_reward", 0.0)))
        print(
            f"[STEP] step={observation.step_count} intern_id={action.intern_id} "
            f"action={action.action} step_reward={reward.get('step_reward', 0.0):.2f} "
            f"score={reward.get('score', 0.0):.2f} done={observation.done}"
        )

    final_reward = observation.metadata.get("reward", {}) if isinstance(observation.metadata, dict) else {}
    print(
        f"[END] task={task} steps={observation.step_count} "
        f"score={final_reward.get('score', 0.0):.2f} rewards={','.join(f'{value:.2f}' for value in rewards)}"
    )


def main(tasks: Iterable[TaskId] = TASKS) -> None:
    for task in tasks:
        run_task(task)


if __name__ == "__main__":
    main()
