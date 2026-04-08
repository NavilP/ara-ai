# onboarding-env

> **HR onboarding coordinator simulation for AI agents**

An [OpenEnv](https://openenv.dev) environment where an AI agent acts as an HR coordinator managing interns through a structured onboarding pipeline — prioritising people, respecting dependencies, and handling real-world exceptions like non-responses and access delays.

---

## Problem Statement

Every company runs intern onboarding manually. This environment simulates that process: the agent sees intern states, decides which onboarding step to execute next, and receives reward proportional to real progress. The goal is to complete all interns' checklists correctly and efficiently.

---

## Observation Space

The observation is a JSON object returned by `/reset` and `/step`:

| Field | Type | Description |
|---|---|---|
| `interns` | `list[PersonState]` | Current state of each intern |
| `available_actions` | `dict[str, list[str]]` | Valid actions per intern right now |
| `communications_log` | `list[dict]` | Email previews generated so far |
| `step_count` | `int` | Steps taken in this episode |
| `message` | `str` | Human-readable status message |

Each `PersonState` includes: `id`, `name`, `is_international`, `intern_confirmed`, `days_without_response`, `access_delayed`, `checklist`.

---

## Action Space

9 actions, each identified by `{"intern_id": "<id>", "action": "<name>"}`:

| Action | Reward | Requires |
|---|---|---|
| `send_welcome_email` | +0.20 | — |
| `share_docs` | +0.20 | `intern_confirmed` |
| `share_international_docs` | +0.10 | `intern_confirmed` + `is_international` |
| `schedule_intro_meeting` | +0.20 | `intern_confirmed` |
| `create_account` | +0.20 | `intern_confirmed` + `docs_shared` |
| `grant_system_access` | +0.20 | `account_created` + `docs_shared` |
| `request_alternative_access` | +0.20 | `account_created` + `docs_shared` + `access_delayed` |
| `send_followup_email` | +0.05 | `welcome_email_sent` + `days >= 1` |
| `escalate_to_manager` | +0.05 | `welcome_email_sent` + `days >= 3` + `not confirmed` |

**Penalties:** -0.05 for missing dependencies, calling `grant_system_access` when `access_delayed`, or escalating before 3 days.

---

## Tasks

| Task | Interns | Description |
|---|---|---|
| `easy` | 1 | Carlos Méndez. Linear flow, no exceptions. |
| `medium` | 3 | Carlos (in progress), Priya (2 days no reply → needs followup), Ana (fresh start). |
| `hard` | 3 | Ahmed (international docs required), Rohan (4 days no reply, can never confirm → escalate), Sara (access_delayed → alternative access only). |

All graders are **fully deterministic** — no `datetime.now()`, no external state. Scores always land in `[0.0, 1.0]`.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start episode. Body: `{"task": "easy\|medium\|hard"}` |
| `POST` | `/step` | Execute action. Body: `{"intern_id": "...", "action": "..."}` |
| `GET` | `/state` | Current episode state |
| `GET` | `/tasks` | Available tasks metadata |
| `GET` | `/` | Visual dashboard |

---

## Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn main:app --host 0.0.0.0 --port 7860

# 3. Open dashboard
open http://localhost:7860

# 4. Test reset endpoint
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"easy"}'
```

---

## Running inference.py

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
export ONBOARDING_URL="http://localhost:7860"

python inference.py
```

Expected output format:
```
[START] task=easy env=onboarding-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"intern_id":"intern_001","action":"send_welcome_email"} reward=0.20 done=false error=null
...
[END] success=true steps=5 score=1.000 rewards=0.20,0.20,0.20,0.20,0.20
```

---

## Docker

```bash
docker build -t onboarding-env .
docker run -p 7860:7860 onboarding-env
```

---

## Deploy on Hugging Face Spaces

1. Create a new Space with **Docker** SDK.
2. Add the `openenv` tag in the Space config.
3. Push this repo to the Space.
4. Set environment variables in Space Settings: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.

---

## Pre-submission Validation

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://<your-space>.hf.space .
```

---

## Baseline Scores

Obtained with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Score | Steps |
|---|---|---|
| easy | 1.00 | 5 |
| medium | ~0.88 | 14 |
| hard | ~0.85 | 15 |

---

## File Structure

```
onboarding-env/
├── main.py                  FastAPI server
├── environment.py           Core logic, actions, email previews
├── models.py                Pydantic v2 models
├── inference.py             Baseline agent script
├── openenv.yaml             OpenEnv spec metadata
├── Dockerfile               Container, port 7860
├── requirements.txt
├── README.md
├── validate-submission.sh   Pre-submission validator
├── sample_interns.csv       Sample data
├── data/
│   └── fixtures.json        Task fixtures (deterministic)
├── graders/
│   ├── grader_easy.py
│   ├── grader_medium.py
│   └── grader_hard.py
└── ui/
    └── index.html           Visual dashboard
```
