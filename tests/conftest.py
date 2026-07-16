from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm_parser import load_cached_demo_criteria, reset_runtime_guards


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def criteria():
    return load_cached_demo_criteria()


@pytest.fixture
def edge_cases():
    return json.loads((ROOT / "data" / "golden4_edge_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def reset_llm_state():
    reset_runtime_guards()
    yield
    reset_runtime_guards()
