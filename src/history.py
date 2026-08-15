"""Browser-safe research workspace history helpers.

The module keeps history payloads small and portable: protocol metadata,
reviewed criteria, aggregate result counts and scenario parameters are stored;
uploaded files and patient-level rows are deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Mapping

from .models import Criterion, MatchResult, TrialSource


HISTORY_SCHEMA_VERSION = 1
MAX_WORKSPACES = 12
MAX_EVENTS_PER_WORKSPACE = 30


def workspace_id(source: TrialSource) -> str:
    """Return a stable identifier for a protocol source."""

    identity = "\n".join(
        [
            source.identifier.strip().upper(),
            source.source_reference.strip(),
            source.title.strip(),
            source.criteria_text.strip(),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def result_summary(results: Iterable[MatchResult] | None) -> dict[str, int]:
    counts = Counter(item.overall_status for item in (results or []))
    return {
        "total": sum(counts.values()),
        "eligible": counts.get("eligible", 0),
        "ineligible": counts.get("ineligible", 0),
        "missing_data": counts.get("missing_data", 0),
        "needs_review": counts.get("needs_review", 0),
    }


def build_workspace_snapshot(
    *,
    source: TrialSource,
    criteria_text: str,
    criteria: Iterable[Criterion],
    results: Iterable[MatchResult] | None,
    scenario_parameters: Mapping[str, Any] | None,
    action: str,
    detail: str = "",
    last_page: str = "研究工作台",
    saved_at: str | None = None,
) -> dict[str, Any]:
    timestamp = saved_at or datetime.now().astimezone().isoformat(timespec="seconds")
    criteria_items = list(criteria)
    event = {"at": timestamp, "action": action, "detail": detail}
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "workspace_id": workspace_id(source),
        "identifier": source.identifier,
        "title": source.title,
        "source_type": source.source_type,
        "source": source.model_dump(mode="json"),
        "criteria_text": criteria_text,
        "criteria": [item.model_dump(mode="json") for item in criteria_items],
        "criterion_count": len(criteria_items),
        "result_summary": result_summary(results),
        "scenario_parameters": dict(scenario_parameters or {}),
        "last_page": last_page,
        "last_action": action,
        "updated_at": timestamp,
        "events": [event],
    }


def upsert_workspace(
    workspaces: Iterable[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    *,
    max_workspaces: int = MAX_WORKSPACES,
    max_events: int = MAX_EVENTS_PER_WORKSPACE,
) -> list[dict[str, Any]]:
    """Insert or update one workspace while retaining a compact activity log."""

    incoming = dict(snapshot)
    existing_items = [dict(item) for item in workspaces]
    previous = next(
        (item for item in existing_items if item.get("workspace_id") == incoming.get("workspace_id")),
        None,
    )
    if previous:
        events = list(previous.get("events") or []) + list(incoming.get("events") or [])
        incoming["events"] = events[-max_events:]

    remaining = [
        item for item in existing_items if item.get("workspace_id") != incoming.get("workspace_id")
    ]
    combined = [incoming, *remaining]
    combined.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return combined[:max_workspaces]


def serialize_history(workspaces: Iterable[Mapping[str, Any]]) -> str:
    payload = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "workspaces": list(workspaces),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def deserialize_history(payload: Any) -> list[dict[str, Any]]:
    """Read browser history defensively and ignore unsupported payloads."""

    if payload in (None, "", {}):
        return []
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict) or data.get("schema_version") != HISTORY_SCHEMA_VERSION:
        return []
    items = data.get("workspaces")
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict) and item.get("workspace_id")]
