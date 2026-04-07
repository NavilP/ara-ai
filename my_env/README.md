# My Env

`my_env/` is the OpenEnv source of truth for this hackathon submission. The earlier root-level onboarding prototype has been migrated into the scaffold created by `openenv init my_env` instead of replacing that structure.

## Scenario

This environment simulates a real HR onboarding coordinator workflow. An agent must complete onboarding for interns while handling follow-ups, international documentation, delayed access, and escalation paths.

## OpenEnv Interface

- `reset()` returns the initial typed observation for the active task
- `step(action)` applies one typed `OnboardingAction`
- `state()` returns the current typed `OnboardingState`
- reward values stay within `0.0` to `1.0`
- partial progress is surfaced on each step before the final score reaches `1.0`

## Typed Models

- `OnboardingAction` in `models.py`
- `OnboardingObservation` in `models.py`
- `OnboardingReward` in `models.py`
- `OnboardingState` in `models.py`

## Tasks

- `easy`: one standard onboarding case
- `medium`: multiple interns including a follow-up case
- `hard`: international docs, blocked response escalation, and alternative access routing

The server exposes `/tasks` to list tasks and `POST /task/{task_id}` to select the task used by the next `reset()`.

## Run Locally

```bash
cd my_env
pip install -e .
python -m my_env.server.app
```

Example:

```bash
curl -X POST http://localhost:8000/task/hard
curl -X POST http://localhost:8000/reset
curl http://localhost:8000/state
```

Custom UI:

```bash
http://localhost:8000/ui
```

The UI talks to `/ui-api/*`, which keeps a persistent local episode for the dashboard without changing the standard OpenEnv endpoints used for validation.

## Reproducible Baseline

```bash
cd my_env
python -m my_env.baseline
```

The baseline uses a deterministic priority policy and runs all three tasks in sequence.

## Layout

- `models.py`: typed OpenEnv-facing models
- `data/fixtures.json`: task fixtures and initial environment state
- `server/my_env_environment.py`: environment logic, task flow, and reward shaping
- `server/app.py`: scaffold-aligned OpenEnv server entrypoint with task-selection endpoints
- `baseline.py`: deterministic baseline runner
