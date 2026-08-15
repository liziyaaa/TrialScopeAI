from src.history import (
    build_workspace_snapshot,
    deserialize_history,
    serialize_history,
    upsert_workspace,
    workspace_id,
)
from src.llm_parser import load_cached_demo_criteria
from src.trial_sources import source_from_text


def _source(text: str = "Inclusion Criteria: age 40 years or older"):
    return source_from_text(text)


def test_workspace_id_distinguishes_different_pasted_protocols():
    assert workspace_id(_source("Inclusion Criteria: age 40 years or older")) != workspace_id(
        _source("Inclusion Criteria: age 50 years or older")
    )


def test_upsert_keeps_latest_snapshot_and_activity_timeline():
    source = _source()
    first = build_workspace_snapshot(
        source=source,
        criteria_text=source.criteria_text,
        criteria=[],
        results=None,
        scenario_parameters={},
        action="方案已导入",
        saved_at="2026-08-15T10:00:00+08:00",
    )
    second = build_workspace_snapshot(
        source=source,
        criteria_text=source.criteria_text,
        criteria=load_cached_demo_criteria()[:2],
        results=None,
        scenario_parameters={},
        action="标准审核已保存",
        last_page="协作审核",
        saved_at="2026-08-15T10:05:00+08:00",
    )
    history = upsert_workspace(upsert_workspace([], first), second)
    assert len(history) == 1
    assert history[0]["criterion_count"] == 2
    assert [event["action"] for event in history[0]["events"]] == ["方案已导入", "标准审核已保存"]


def test_history_serialization_round_trip_and_invalid_payload():
    source = _source()
    snapshot = build_workspace_snapshot(
        source=source,
        criteria_text=source.criteria_text,
        criteria=[],
        results=None,
        scenario_parameters={},
        action="方案已导入",
    )
    restored = deserialize_history(serialize_history([snapshot]))
    assert restored[0]["workspace_id"] == snapshot["workspace_id"]
    assert deserialize_history("not-json") == []
    assert deserialize_history({"schema_version": 999, "workspaces": [snapshot]}) == []
