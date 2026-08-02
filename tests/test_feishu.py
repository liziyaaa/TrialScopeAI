from __future__ import annotations

import json

import httpx

from src.feishu import (
    FeishuClient,
    FeishuSettings,
    apply_reviewed_records,
    criterion_to_feishu_fields,
)
from src.models import Criterion


def _criterion() -> Criterion:
    return Criterion.model_validate(
        {
            "criterion_id": "I06",
            "kind": "inclusion",
            "source_text": "Post-bronchodilator FEV1/FVC ratio < 0.70 during Screening.",
            "source_reference": "https://clinicaltrials.gov/study/NCT02347774",
            "field": "post_bd_fev1_fvc",
            "operator": "lt",
            "value": 0.7,
            "unit": "ratio",
            "logic_group": "lung_function",
            "confidence": 1.0,
            "execution_status": "automated",
        }
    )


def test_criterion_mapping_keeps_traceability():
    fields = criterion_to_feishu_fields("NCT02347774", _criterion())
    assert fields["同步键"] == "NCT02347774:I06"
    assert fields["类型"] == "入组"
    assert fields["运算符"] == "小于"
    assert fields["阈值"] == "0.7"
    assert fields["来源链接"].endswith("NCT02347774")


def test_reviewed_record_builds_diff_without_mutating_original():
    original = _criterion()
    reviewed, diffs = apply_reviewed_records(
        [original],
        [
            {
                "record_id": "rec_1",
                "fields": {
                    "标准编号": "I06",
                    "审核状态": "已确认",
                    "审核后指标": "post_bd_fev1_fvc",
                    "审核后运算符": "小于等于",
                    "审核后阈值": "0.72",
                    "审核后单位": "ratio",
                },
            }
        ],
    )
    assert original.operator == "lt"
    assert reviewed[0].operator == "lte"
    assert reviewed[0].value == 0.72
    assert {item["字段"] for item in diffs} == {"operator", "value"}


def test_sync_creates_and_updates_while_preserving_review_fields():
    calls: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path.endswith("tenant_access_token/internal/"):
            return httpx.Response(
                200,
                json={"code": 0, "tenant_access_token": "token", "expire": 7200},
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "has_more": False,
                        "items": [
                            {
                                "record_id": "rec_existing",
                                "fields": {
                                    "同步键": "NCT02347774:I06",
                                    "标准编号": "I06",
                                    "标准原文": "old text",
                                    "审核状态": "已确认",
                                    "修改意见": "keep me",
                                },
                            }
                        ],
                    },
                },
            )
        return httpx.Response(200, json={"code": 0, "data": {}})

    settings = FeishuSettings("app", "secret", "base", "table")
    with FeishuClient(settings, transport=httpx.MockTransport(handler)) as client:
        summary = client.sync_criteria("NCT02347774", [_criterion()])

    assert summary.updated == 1
    update_call = next(call for call in calls if call[1].endswith("batch_update"))
    update_fields = update_call[2]["records"][0]["fields"]
    assert "审核状态" not in update_fields
    assert "修改意见" not in update_fields


def test_upsert_record_creates_aggregate_in_target_table():
    calls: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path.endswith("tenant_access_token/internal/"):
            return httpx.Response(
                200,
                json={"code": 0, "tenant_access_token": "token", "expire": 7200},
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"code": 0, "data": {"has_more": False, "items": []}},
            )
        return httpx.Response(200, json={"code": 0, "data": {}})

    settings = FeishuSettings("app", "secret", "base", "criteria")
    with FeishuClient(settings, transport=httpx.MockTransport(handler)) as client:
        action = client.upsert_record(
            "snapshots",
            "快照键",
            {"快照键": "NCT:scenario:1", "模拟符合人数": 12},
        )

    assert action == "created"
    create_call = next(call for call in calls if call[1].endswith("batch_create"))
    assert "/tables/snapshots/" in create_call[1]
    assert create_call[2]["records"][0]["fields"]["模拟符合人数"] == 12


def test_upsert_record_updates_matching_business_key():
    calls: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path.endswith("tenant_access_token/internal/"):
            return httpx.Response(
                200,
                json={"code": 0, "tenant_access_token": "token", "expire": 7200},
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "has_more": False,
                        "items": [
                            {
                                "record_id": "rec_snapshot",
                                "fields": {"快照键": "NCT:scenario:1"},
                            }
                        ],
                    },
                },
            )
        return httpx.Response(200, json={"code": 0, "data": {}})

    settings = FeishuSettings("app", "secret", "base", "criteria")
    with FeishuClient(settings, transport=httpx.MockTransport(handler)) as client:
        action = client.upsert_record(
            "snapshots",
            "快照键",
            {"快照键": "NCT:scenario:1", "模拟符合人数": 14},
        )

    assert action == "updated"
    update_call = next(call for call in calls if call[1].endswith("batch_update"))
    assert update_call[2]["records"][0]["record_id"] == "rec_snapshot"
