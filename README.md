---
title: Ara AI
emoji: ✨
colorFrom: green
colorTo: blue
sdk: docker
tags: 
  - openenv
pinned: false
license: mit
app_port: 7860
---

# onboarding-env
> An OpenEnv reinforcement learning environment that simulates HR onboarding coordination. An AI agent manages interns through a structured workflow with real-world exceptions: international documentation, system access delays, and non-responsive candidates.

---

## Contents

- [Overview](#overview)
- [Environment Design](#environment-design)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Reward Function](#reward-function)
- [Tasks](#tasks)
- [Baseline Scores](#baseline-scores)
- [Setup](#setup)
- [Running with Docker](#running-with-docker)
- [Project Structure](#project-structure)

---

## Overview

**Domain:** Human Resources — intern onboarding coordination  
**Why it matters:** Onboarding coordination is a high-volume, exception-heavy process that current HR tools handle poorly. This environment lets agents learn to handle both the standard flow and edge cases (international hires, delayed IT access, unresponsive candidates) in a realistic, structured setting.

**Agent role:** The agent acts as an HR coordinator. It receives the current state of one or more interns and must choose the next action to advance their onboarding while respecting process dependencies and handling exceptions appropriately.

---

## Environment Design

### State management

Each episode contains one or more interns, each with:
- A checklist of required steps (`welcome_email_sent`, `docs_shared`, `intro_scheduled`, `account_created`, `access_granted`)
- Exception flags (`is_international`, `access_delayed`, `days_without_response`)
- A dependency graph that determines which actions are currently valid

`reset()` loads a fresh copy of the task fixtures. `step()` applies the action, runs the semantic validator, updates the checklist, and returns the new observation with reward. `state()` returns the current episode state without advancing it.

### Semantic validator

A three-zone validator intercepts every action before it reaches the environment:

| Zone | Condition | Effect |
|------|-----------|--------|
| 1 | Prohibited verb (`terminate`, `fire`, `delete`, ...) | Rejected, `reward = -0.05` |
| 2 | Invalid action prefix for dynamic actions | Rejected, `reward = 0.0` |
| 3 | Business rule violation (e.g. `share_docs` before confirmation) | Rejected with explanation |

This allows the agent to propose novel actions (prefixed `comm_`, `doc_`, `access_`, `escalate_`) while enforcing HR process integrity.

### Agent architecture

```
Observation
    │
    ▼
Heuristic++ ──── no exceptions? ──▶ DAG action (no LLM call)
    │
    │ exception detected
    ▼
Hybrid RL + LLM ─────────────────▶ reasoned action + memory context
    │
    ▼
Semantic Validator ───────────────▶ enforce business rules
    │
    ▼
Environment step()
```

The `inference.py` runs **3 episodes per task**. Episodes 1–2 are silent warmup that build trajectory memory. Episode 3 is the evaluated run — the LLM receives a summary of what worked and what failed in prior episodes (in-context RL).

---

## Action Space
9 standard actions, each targeting a specific `intern_id`:

| Action | Description | Prerequisite |
|--------|-------------|--------------|
| `send_welcome_email` | Initiates onboarding, sets `intern_confirmed` | None |
| `share_docs` | Shares onboarding documents | `intern_confirmed` |
| `share_international_docs` | Shares visa/relocation packet | `intern_confirmed`, `is_international` |
| `schedule_intro_meeting` | Books team introduction | `intern_confirmed` |
| `create_account` | Creates system account | `intern_confirmed`, `docs_shared` |
| `grant_system_access` | Grants full access | `account_created`, `docs_shared` |
| `request_alternative_access` | Alternative path when IT is delayed | `account_created`, `access_delayed` |
| `send_followup_email` | Follows up with unresponsive intern | `days_without_response >= 1` |
| `escalate_to_manager` | Escalates blocked intern | `days_without_response >= 3`, `not confirmed` |

---

## Observation Space

```json
{
  "task": "easy | medium | hard",
  "difficulty": "easy | medium | hard",
  "interns": [
    {
      "id": "intern_001",
      "name": "string",
      "is_international": false,
      "intern_confirmed": false,
      "days_without_response": 0,
      "access_delayed": false,
      "checklist": {
        "welcome_email_sent": false,
        "docs_shared": false,
        "intro_scheduled": false,
        "account_created": false,
        "access_granted": false,
        "followup_sent": false,
        "escalated": false
      }
    }
  ],
  "available_actions": {
    "intern_001": ["send_welcome_email"]
  },
  "communications_log": [],
  "step_count": 0,
  "message": "string"
}
```

---

## Reward Function

Rewards are **dense** — partial progress is rewarded at each step.

| Action | Step reward |
|--------|-------------|
| `send_welcome_email` | +0.20 |
| `share_docs` | +0.20 |
| `schedule_intro_meeting` | +0.20 |
| `create_account` | +0.20 |
| `grant_system_access` | +0.20 |
| `request_alternative_access` | +0.20 |
| `share_international_docs` | +0.10 |
| `send_followup_email` | +0.05 |
| `escalate_to_manager` | +0.05 |
| Missing dependency / wrong action | -0.05 |
| Prohibited verb (zone 1) | -0.05 |

**Episode score formula:**

```
# Without exceptions
score = core_progress × 0.90 + efficiency × 0.10

# With exceptions (international, delayed access, escalation needed)
score = core_progress × 0.70 + exceptions_resolved × 0.20 + efficiency × 0.10
```

Where `efficiency = min(optimal_steps / actual_steps, 1.0)`.

---

## Tasks

### easy
**1 intern, no exceptions.** Standard happy path — 5 steps, Heuristic++ resolves without LLM.

```
send_welcome_email → share_docs → schedule_intro_meeting → create_account → grant_system_access
```

### medium
**3 interns, mixed state.** One intern partially completed, one needs a follow-up (LLM decides when), one brand new. Tests parallel coordination and follow-up timing.

### hard
**3 interns, one each with a different hard exception:**

| Intern | Exception | Required handling |
|--------|-----------|-------------------|
| Ahmed Al-Rashid | `is_international` | Must include `share_international_docs` |
| Rohan Iyer | `days_without_response=4` | Must escalate — standard flow impossible |
| Sara Kim | `access_delayed` | Must use `request_alternative_access`, not `grant_system_access` |

---

## Baseline Scores

Measured with **GPT-4o** via Azure OpenAI. Agent: Heuristic++ (happy path) + LLM (exceptions).

```
[START] task=easy env=onboarding-env model=gpt-4o
[STEP] step=1 action={"intern_id":"intern_001","action":"send_welcome_email"} reward=0.20 done=false error=null
[STEP] step=2 action={"intern_id":"intern_001","action":"share_docs"} reward=0.20 done=false error=null
[STEP] step=3 action={"intern_id":"intern_001","action":"schedule_intro_meeting"} reward=0.20 done=false error=null
[STEP] step=4 action={"intern_id":"intern_001","action":"create_account"} reward=0.20 done=false error=null
[STEP] step=5 action={"intern_id":"intern_001","action":"grant_system_access"} reward=0.20 done=true error=null
[END] success=true steps=5 score=1.00 rewards=0.20,0.20,0.20,0.20,0.20

[START] task=medium env=onboarding-env model=gpt-4o
[STEP] step=1 action={"intern_id":"intern_002","action":"send_followup_email"} reward=0.05 done=false error=null
[STEP] step=13 action={"intern_id":"intern_003","action":"grant_system_access"} reward=0.20 done=true error=null
[END] success=true steps=13 score=1.00 rewards=0.20,0.20,0.20,0.20,0.20,0.20,0.05,0.20,0.20,0.20,0.20,0.20,0.20

[START] task=hard env=onboarding-env model=gpt-4o
[STEP] step=12 action={"intern_id":"intern_003","action":"grant_system_access"} reward=0.00 done=false error=Cannot grant system access when access is delayed. Use request_alternative_access instead.
[STEP] step=13 action={"intern_id":"intern_002","action":"escalate_to_manager"} reward=0.05 done=true error=null
[END] success=true steps=13 score=0.99 rewards=0.20,0.20,0.10,0.20,0.20,0.05,0.20,0.20,0.20,0.20,0.20,0.20,0.05
```

| Task | Steps | Score | Notes |
|------|-------|-------|-------|
| easy | 5 | **1.00** | Perfect — Heuristic++ only |
| medium | 13 | **1.00** | LLM correctly sequenced follow-up |
| hard | 13 | **0.99** | One recoverable error on Sara Kim (zone 3), self-corrected |

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for containerized run)

### 1. Install dependencies

```bash
# Server
cd server
pip install -r requirements.txt

# Agent (from root)
cd ..
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env:
#   API_BASE_URL=https://router.huggingface.co/v1
#   MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
#   HF_TOKEN=hf_your_token_here
#   ONBOARDING_URL=http://localhost:7860
```

### 3. Start the server

```bash
cd server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Verify it's running:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"easy"}'
```

### 4. Run the agent

```bash
# From root, in a second terminal
python inference.py
```

---

## Running with Docker

```bash
docker build -t onboarding-env ./server
docker run -p 7860:7860 onboarding-env
```

---

## Project Structure

```
ara-ai/
├── agent/
│   ├── heuristic.py          Heuristic++ DAG — no LLM, handles happy path
│   ├── llm_agent.py          Hybrid RL+LLM — reasons over exceptions
│   └── memory.py             Trajectory memory for in-context RL
├── server/
│   ├── app.py                FastAPI endpoints: /reset /step /state /tasks
│   ├── environment.py        State transitions + validator integration
│   ├── models.py             Pydantic models (PersonState, Observation, Reward)
│   ├── graders/
│   │   └── grader.py         Unified state-based grader, deterministic
│   ├── onboarding_validator/
│   │   ├── rules.py          Business rules (data only, no logic)
│   │   └── semantic_validator.py  validate(action, intern) → zone 1/2/3
│   ├── data/
│   │   └── fixtures.json     Task definitions (easy / medium / hard)
│   ├── Dockerfile
│   └── requirements.txt
├── client.py                 Async HTTP client for inference.py
├── inference.py              Agent entry point — router + 3 episodes + memory
├── openenv.yaml              Environment spec
├── pyproject.toml            Package metadata
├── requirements.txt          Root dependencies (agent + client)
├── validate-submission.sh    Pre-submission validator
└── .env.example              Environment variable template
```
