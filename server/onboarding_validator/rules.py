"""
Validator rules loader.

Rules now live in rules.json so non-Python users can edit them safely.
This module keeps the same exported names used by semantic_validator.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_RULES_PATH = Path(__file__).with_name("rules.json")

with _RULES_PATH.open("r", encoding="utf-8") as f:
    _raw: dict[str, Any] = json.load(f)

PROHIBITED_VERBS: list[str] = list(_raw.get("prohibited_verbs", []))
ALLOWED_PREFIXES: list[str] = list(_raw.get("allowed_prefixes", []))
KNOWN_ACTIONS: frozenset[str] = frozenset(_raw.get("known_actions", []))
PRECONDITION_RULES: dict[str, dict] = dict(_raw.get("precondition_rules", {}))