"""
client.py — OnboardingEnvClient
================================
HTTPEnvClient implementation for the onboarding-env OpenEnv environment.
Import this in inference.py instead of making raw HTTP calls.

Usage:
    from client import OnboardingEnvClient

    env = OnboardingEnvClient(base_url="http://localhost:7860")
    obs    = env.reset(task="easy")
    result = env.step(intern_id="intern_001", action="send_welcome_email")
    state  = env.state()
    next_a = env.agent_next()
"""

from __future__ import annotations

from typing import Any, Optional

import requests


class OnboardingEnvClient:
    """
    Thin HTTP wrapper around the onboarding-env FastAPI server.
    Mirrors the server endpoints as Python methods.
    """

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    # ------------------------------------------------------------------
    # Core OpenEnv methods
    # ------------------------------------------------------------------

    def reset(self, task: str = "easy") -> dict[str, Any]:
        """
        Start a new episode for the given task.
        Returns the initial OnboardingObservation as a dict.
        """
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task": task},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, intern_id: str, action: str) -> dict[str, Any]:
        """
        Execute one action for a given intern.
        Returns {"observation": ..., "reward": ..., "done": ..., "info": ...}
        """
        resp = requests.post(
            f"{self.base_url}/step",
            json={"intern_id": intern_id, "action": action},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict[str, Any]:
        """Return the current episode state without executing any action."""
        resp = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def tasks(self) -> list[dict[str, Any]]:
        """Return available tasks with metadata derived from the fixtures."""
        resp = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("tasks", [])

    def agent_next(self) -> dict[str, Any]:
        """
        Ask the server's greedy agent for the next recommended action.
        Returns {"intern_id": ..., "action": ..., "reason": ...}
        Useful for testing without an LLM.
        """
        resp = requests.post(f"{self.base_url}/agent/next", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience properties from last observation
    # ------------------------------------------------------------------

    def get_available_actions(self, observation: dict[str, Any]) -> dict[str, list[str]]:
        """Extract available_actions from an observation dict."""
        return observation.get("available_actions", {})

    def is_done(self, result: dict[str, Any]) -> bool:
        """Check if the episode is finished from a /step result."""
        return result.get("done", False)

    def get_score(self, result: dict[str, Any]) -> float:
        """Extract the current score from a /step result."""
        return result.get("reward", {}).get("score", 0.0)

    def get_step_reward(self, result: dict[str, Any]) -> float:
        """Extract the step reward from a /step result."""
        return result.get("reward", {}).get("step_reward", 0.0)

    def get_error(self, result: dict[str, Any]) -> Optional[str]:
        """Extract the error message from a /step result, or None."""
        return result.get("info", {}).get("error")