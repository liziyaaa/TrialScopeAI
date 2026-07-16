from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.llm_parser import (
    HourlyBudget,
    LLMParseError,
    LLMRateLimitError,
    parse_with_deepseek,
    split_for_llm,
)


def _payload():
    return {
        "criteria": [
            {
                "criterion_id": "i1",
                "kind": "inclusion",
                "source_text": "Age at least 40 years",
                "source_reference": "will-be-replaced",
                "field": "age",
                "operator": "gte",
                "value": 40,
                "unit": "year",
                "confidence": 0.97,
                "execution_status": "automated",
            }
        ]
    }


class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    def create(self, **kwargs):
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self, contents):
        self.chat = SimpleNamespace(completions=FakeCompletions(contents))


def test_live_parser_validates_json_and_caches():
    client = FakeClient([json.dumps(_payload())])
    first = parse_with_deepseek(
        "Inclusion Criteria:\nAge at least 40 years.",
        "test-key",
        "https://example.test/study",
        client=client,
    )
    second = parse_with_deepseek(
        "Inclusion Criteria:\nAge at least 40 years.",
        "test-key",
        "https://example.test/study",
        client=client,
    )
    assert first.from_cache is False
    assert second.from_cache is True
    assert client.chat.completions.calls == 1
    assert first.criteria[0].criterion_id == "C001"
    assert first.criteria[0].source_reference == "https://example.test/study"


def test_empty_response_retries_then_succeeds():
    client = FakeClient(["", "", json.dumps(_payload())])
    outcome = parse_with_deepseek(
        "Inclusion Criteria:\nAge at least 40 years.",
        "test-key",
        "test-source",
        client=client,
    )
    assert outcome.criteria
    assert client.chat.completions.calls == 3


def test_invalid_response_exhausts_retries():
    client = FakeClient(["not-json"])
    with pytest.raises(LLMParseError, match="解析失败"):
        parse_with_deepseek(
            "Inclusion Criteria:\nAge at least 40 years.",
            "test-key",
            "test-source",
            client=client,
        )
    assert client.chat.completions.calls == 3


def test_missing_key_is_rejected_without_call():
    with pytest.raises(LLMParseError, match="DEEPSEEK_API_KEY"):
        parse_with_deepseek("criteria text", "", "test")


def test_long_input_is_split_and_hard_capped():
    text = ("Inclusion criterion paragraph.\n\n" * 2500) + ("Exclusion criterion.\n\n" * 2500)
    chunks = split_for_llm(text)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 30_000 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) <= 60_000


def test_hourly_budget_enforces_limit():
    budget = HourlyBudget(limit=1)
    budget.consume(now=100)
    with pytest.raises(LLMRateLimitError):
        budget.consume(now=101)
    budget.consume(now=3701)
