"""
client.py — OnboardingEnvClient
================================
HTTPEnvClient implementation for onboarding-env.
What YOU import in inference.py.

Usage:
    from client import OnboardingEnvClient

    env = OnboardingEnvClient(base_url="http://localhost:7860")
    obs    = await env.reset(task="easy")
    result = await env.step(intern_id="intern_001", action="send_welcome_email")
    await env.close()
"""
from __future__ import annotations

from typing import Any, Optional

import httpx


class OnboardingEnvClient:
    """
    Async HTTP wrapper around the onboarding-env FastAPI server.
    Mirrors the server endpoints as async Python methods.
    """

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    # ------------------------------------------------------------------
    # Core OpenEnv methods
    # ------------------------------------------------------------------

    async def reset(self, task: str = "easy") -> dict[str, Any]:
        """Start a new episode. Returns the initial OnboardingObservation."""
        client = await self._get_client()
        resp = await client.post(f"{self.base_url}/reset", json={"task": task})
        resp.raise_for_status()
        return resp.json()

    async def step(self, intern_id: str, action: str) -> dict[str, Any]:
        """
        Execute one action.
        Returns {"observation": ..., "reward": ..., "done": ..., "info": ...}
        """
        client = await self._get_client()
        resp = await client.post(
            f"{self.base_url}/step",
            json={"intern_id": intern_id, "action": action},
        )
        resp.raise_for_status()
        return resp.json()

    async def state(self) -> dict[str, Any]:
        """Return current episode state without executing any action."""
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        """Close the HTTP client. Always call this in your finally block."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    async def tasks(self) -> list[dict[str, Any]]:
        """Return available tasks with metadata from the server."""
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json().get("tasks", [])

    async def agent_next(self) -> dict[str, Any]:
        """
        Ask the server's greedy agent for the next recommended action.
        Returns {"intern_id": ..., "action": ..., "reason": ...}
        """
        client = await self._get_client()
        resp = await client.post(f"{self.base_url}/agent/next")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience extractors from step results
    # ------------------------------------------------------------------

    def is_done(self, result: dict[str, Any]) -> bool:
        return result.get("done", False)

    def get_score(self, result: dict[str, Any]) -> float:
        return result.get("reward", {}).get("score", 0.0)

    def get_step_reward(self, result: dict[str, Any]) -> float:
        return result.get("reward", {}).get("step_reward", 0.0)

    def get_error(self, result: dict[str, Any]) -> Optional[str]:
        return result.get("info", {}).get("error")
