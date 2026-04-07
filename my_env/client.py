# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the onboarding OpenEnv environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import OnboardingAction, OnboardingObservation, OnboardingState


class MyEnv(
    EnvClient[OnboardingAction, OnboardingObservation, OnboardingState]
):
    def _step_payload(self, action: OnboardingAction) -> Dict:
        return {
            "intern_id": action.intern_id,
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OnboardingObservation]:
        obs_data = payload.get("observation", {})
        observation = OnboardingObservation(
            task=obs_data.get("task", "easy"),
            difficulty=obs_data.get("difficulty", "easy"),
            interns=obs_data.get("interns", []),
            available_actions=obs_data.get("available_actions", {}),
            communications_log=obs_data.get("communications_log", []),
            step_count=obs_data.get("step_count", 0),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.metadata.get("reward", observation.reward)),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> OnboardingState:
        return OnboardingState(
            task=payload.get("task", "easy"),
            difficulty=payload.get("difficulty", "easy"),
            interns=payload.get("interns", []),
            communications_log=payload.get("communications_log", []),
            step_count=payload.get("step_count", 0),
            cumulative_score=payload.get("cumulative_score", 0.0),
            done=payload.get("done", False),
        )


OnboardingEnv = MyEnv
