"""
Trajectory memory — in-context RL between episodes.

Stores a summary of each episode per task.
On the next episode the LLM receives this context and can learn
from what worked, what failed, and what zone violations occurred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StepRecord:
    intern_id: str
    action:    str
    reward:    float
    error:     Optional[str]
    zone:      int            # 1=prohibited  2=wrong domain  3=valid
    dynamic:   bool           # True if LLM invented this action
    reasoning: str = ""


@dataclass
class EpisodeRecord:
    task:        str
    episode_num: int
    steps:       list[StepRecord] = field(default_factory=list)
    final_score: float = 0.0
    success:     bool  = False


class TrajectoryMemory:
    """
    Stores up to max_episodes per task.
    Generates a concise text context for the LLM prompt.
    """

    def __init__(self, max_episodes: int = 3) -> None:
        self.max_episodes = max_episodes
        self._records: dict[str, list[EpisodeRecord]] = {}

    def add(self, record: EpisodeRecord) -> None:
        task = record.task
        if task not in self._records:
            self._records[task] = []
        self._records[task].append(record)
        if len(self._records[task]) > self.max_episodes:
            self._records[task].pop(0)

    def get_context(self, task: str) -> str:
        """Generate a text summary suitable for injection into the LLM prompt."""
        records = self._records.get(task, [])
        if not records:
            return ""

        lines: list[str] = []
        for rec in records:
            lines.append(f"Episode {rec.episode_num} (score {rec.final_score:.2f}):")

            # Zone violations — the most important feedback signal
            violations = [
                f"  '{s.action}' blocked zone={s.zone}: {s.error}"
                for s in rec.steps if not (s.error is None and s.zone == 3) and s.error
            ]
            if violations:
                lines.append("  Blocked actions:")
                lines.extend(violations[:4])

            # Dynamic actions that worked (non-standard, no error, positive reward)
            dynamic_good = [
                f"  '{s.action}' (+{s.reward:.2f}) — {s.reasoning}"
                for s in rec.steps if s.dynamic and s.reward > 0 and s.error is None
            ]
            if dynamic_good:
                lines.append("  Novel actions that worked:")
                lines.extend(dynamic_good[:3])

            # Highest-reward steps (what the grader rewarded most)
            top_steps = sorted(
                [s for s in rec.steps if s.reward >= 0.10],
                key=lambda s: s.reward, reverse=True
            )[:3]
            if top_steps:
                lines.append("  Best steps: " +
                    ", ".join(f"{s.action}(+{s.reward:.2f})" for s in top_steps))

        return "\n".join(lines)

    def episode_count(self, task: str) -> int:
        return len(self._records.get(task, []))