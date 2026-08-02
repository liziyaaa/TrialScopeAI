"""Feishu Base synchronization for collaborative eligibility review.

Only structured trial criteria and review metadata are synchronized. Uploaded
protocol files and patient-level data are deliberately outside this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

from src.models import Criterion


FEISHU_BASE_URL = "https://open.feishu.cn"


class FeishuError(RuntimeError):
    """Base class for Feishu configuration and API failures."""


class FeishuConfigurationError(FeishuError):
    """Raised when required Feishu settings are missing."""


class FeishuAPIError(FeishuError):
    """Raised when Feishu returns an unsuccessful response."""


@dataclass(frozen=True)
class FeishuSettings:
    app_id: str
    app_secret: str
    base_token: str
    criteria_table_id: str
    snapshot_table_id: str = ""
    validation_table_id: str = ""
    base_url: str = FEISHU_BASE_URL
    workspace_url: str = ""

    @property
    def configured(self) -> bool:
        return all(
            value.strip()
            for value in (
                self.app_id,
                self.app_secret,
                self.base_token,
                self.criteria_table_id,
            )
        )

    def require_configured(self) -> None:
        if not self.configured:
            raise FeishuConfigurationError(
                "飞书协同尚未完成配置，请在 Streamlit Secrets 中填写应用凭证和多维表格标识。"
            )


@dataclass(frozen=True)
class FeishuSyncSummary:
    created: int
    updated: int
    unchanged: int
    total: int


KIND_TO_FEISHU = {"inclusion": "入组", "exclusion": "排除"}
OPERATOR_TO_FEISHU = {
    "eq": "等于",
    "neq": "不等于",
    "lt": "小于",
    "lte": "小于等于",
    "gt": "大于",
    "gte": "大于等于",
    "between": "区间",
    "in": "包含",
    "not_in": "不包含",
    "is_true": "等于",
    "is_false": "等于",
    "within_days": "区间",
    "exists": "存在",
    "human_review": "存在",
}
FEISHU_TO_OPERATOR = {
    value: key
    for key, value in OPERATOR_TO_FEISHU.items()
    if key not in {"is_true", "is_false", "within_days", "human_review"}
}
EXECUTION_TO_FEISHU = {
    "automated": "自动规则",
    "human_review": "人工复核",
}

SYSTEM_FIELD_NAMES = {
    "同步键",
    "试验编号",
    "标准编号",
    "类型",
    "标准原文",
    "结构化指标",
    "运算符",
    "阈值",
    "单位",
    "时间窗",
    "逻辑组",
    "执行方式",
    "置信度",
    "版本",
    "来源链接",
}


def _json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def criterion_to_feishu_fields(
    trial_id: str,
    criterion: Criterion,
    *,
    version: int = 1,
) -> dict[str, Any]:
    """Convert one criterion into system-owned Feishu fields."""

    time_window = (
        f"{criterion.time_window_days} days"
        if criterion.time_window_days is not None
        else ""
    )
    return {
        "同步键": f"{trial_id}:{criterion.criterion_id}",
        "试验编号": trial_id,
        "标准编号": criterion.criterion_id,
        "类型": KIND_TO_FEISHU[str(criterion.kind)],
        "标准原文": criterion.source_text,
        "结构化指标": criterion.field or "",
        "运算符": OPERATOR_TO_FEISHU[str(criterion.operator)],
        "阈值": _json_value(criterion.value),
        "单位": criterion.unit or "",
        "时间窗": time_window,
        "逻辑组": criterion.logic_group or "",
        "执行方式": EXECUTION_TO_FEISHU[str(criterion.execution_status)],
        "置信度": float(criterion.confidence),
        "版本": int(version),
        "来源链接": criterion.source_reference,
    }


def _record_fields(record: dict[str, Any]) -> dict[str, Any]:
    fields = record.get("fields", {})
    return fields if isinstance(fields, dict) else {}


def _select_value(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _coerce_review_value(raw: Any, original: Any) -> Any:
    if raw in (None, ""):
        return original
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_reviewed_records(
    criteria: Sequence[Criterion],
    records: Sequence[dict[str, Any]],
    *,
    trial_id: str | None = None,
) -> tuple[list[Criterion], list[dict[str, str]]]:
    """Prepare reviewed criteria and a human-readable diff without mutating input."""

    scoped_records = [
        record
        for record in records
        if trial_id is None
        or str(_record_fields(record).get("试验编号", "")) == trial_id
    ]
    records_by_id = {
        str(_record_fields(record).get("标准编号", "")).upper(): record
        for record in scoped_records
        if _record_fields(record).get("标准编号")
    }
    output: list[Criterion] = []
    diffs: list[dict[str, str]] = []

    for criterion in criteria:
        record = records_by_id.get(criterion.criterion_id)
        if not record:
            output.append(criterion.model_copy(deep=True))
            continue
        fields = _record_fields(record)
        review_status = _select_value(fields.get("审核状态"))
        updated = criterion.model_copy(deep=True)

        if review_status == "需专家复核":
            if updated.execution_status != "human_review":
                diffs.append(
                    {
                        "标准编号": criterion.criterion_id,
                        "字段": "执行方式",
                        "原值": "自动规则",
                        "审核值": "人工复核",
                    }
                )
            updated.execution_status = "human_review"
            output.append(updated)
            continue

        if review_status != "已确认":
            output.append(updated)
            continue

        proposed = {
            "field": fields.get("审核后指标") or updated.field,
            "operator": FEISHU_TO_OPERATOR.get(
                _select_value(fields.get("审核后运算符")), updated.operator
            ),
            "value": _coerce_review_value(fields.get("审核后阈值"), updated.value),
            "unit": fields.get("审核后单位") or updated.unit,
        }
        for attribute, value in proposed.items():
            old_value = getattr(updated, attribute)
            if value != old_value:
                diffs.append(
                    {
                        "标准编号": criterion.criterion_id,
                        "字段": attribute,
                        "原值": _json_value(old_value),
                        "审核值": _json_value(value),
                    }
                )
                setattr(updated, attribute, value)
        output.append(Criterion.model_validate(updated.model_dump()))

    return output, diffs


class FeishuClient:
    """Minimal Feishu Base client using application identity."""

    def __init__(
        self,
        settings: FeishuSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.base_url,
            timeout=timeout,
            transport=transport,
        )
        self._token = ""
        self._token_expires_at = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FeishuClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _access_token(self) -> str:
        self.settings.require_configured()
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        response = self._client.post(
            "/open-apis/auth/v3/tenant_access_token/internal/",
            json={
                "app_id": self.settings.app_id,
                "app_secret": self.settings.app_secret,
            },
        )
        payload = self._decode(response)
        token = str(payload.get("tenant_access_token", ""))
        if not token:
            raise FeishuAPIError("飞书未返回应用访问凭证。")
        expires = max(60, int(payload.get("expire", 7200)))
        self._token = token
        self._token_expires_at = time.monotonic() + expires - 60
        return token

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise FeishuAPIError("飞书返回了无法解析的响应。") from exc
        code = payload.get("code", 0)
        if response.is_error or code not in (0, None):
            message = payload.get("msg") or response.reason_phrase or "未知错误"
            raise FeishuAPIError(f"飞书接口调用失败：{message}")
        return payload

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self._access_token()}"
        response = self._client.request(method, path, headers=headers, **kwargs)
        return self._decode(response)

    def list_records(self, table_id: str | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        page_token = ""
        target_table_id = table_id or self.settings.criteria_table_id
        path = (
            f"/open-apis/bitable/v1/apps/{self.settings.base_token}/tables/"
            f"{target_table_id}/records"
        )
        while True:
            params: dict[str, Any] = {"page_size": 500}
            if page_token:
                params["page_token"] = page_token
            payload = self._request("GET", path, params=params)
            data = payload.get("data", {})
            records.extend(data.get("items", []))
            if not data.get("has_more"):
                break
            page_token = str(data.get("page_token", ""))
            if not page_token:
                break
        return records

    def _batch_write(
        self,
        action: str,
        records: Iterable[dict[str, Any]],
        *,
        table_id: str | None = None,
    ) -> None:
        items = list(records)
        if not items:
            return
        target_table_id = table_id or self.settings.criteria_table_id
        path = (
            f"/open-apis/bitable/v1/apps/{self.settings.base_token}/tables/"
            f"{target_table_id}/records/{action}"
        )
        for start in range(0, len(items), 200):
            self._request("POST", path, json={"records": items[start : start + 200]})

    def upsert_record(
        self,
        table_id: str,
        key_field: str,
        fields: dict[str, Any],
    ) -> str:
        """Create or update one aggregate record using a stable business key."""

        if not table_id.strip():
            raise FeishuConfigurationError("目标飞书数据表尚未配置。")
        key_value = fields.get(key_field)
        if key_value in (None, ""):
            raise FeishuConfigurationError(f"飞书记录缺少唯一键：{key_field}。")
        existing = self.list_records(table_id)
        current = next(
            (
                record
                for record in existing
                if _record_fields(record).get(key_field) == key_value
            ),
            None,
        )
        if current is None:
            self._batch_write("batch_create", [{"fields": fields}], table_id=table_id)
            return "created"
        self._batch_write(
            "batch_update",
            [{"record_id": current.get("record_id"), "fields": fields}],
            table_id=table_id,
        )
        return "updated"

    def sync_criteria(
        self,
        trial_id: str,
        criteria: Sequence[Criterion],
        *,
        version: int = 1,
    ) -> FeishuSyncSummary:
        """Create or update system-owned fields while preserving review fields."""

        existing = self.list_records()
        by_key = {
            str(_record_fields(record).get("同步键", "")): record
            for record in existing
            if _record_fields(record).get("同步键")
        }
        creates: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        unchanged = 0
        for criterion in criteria:
            fields = criterion_to_feishu_fields(trial_id, criterion, version=version)
            current = by_key.get(fields["同步键"])
            if current is None:
                creates.append({"fields": fields})
                continue
            current_fields = _record_fields(current)
            changed = {
                name: value
                for name, value in fields.items()
                if name in SYSTEM_FIELD_NAMES and current_fields.get(name) != value
            }
            if changed:
                updates.append(
                    {"record_id": current.get("record_id"), "fields": changed}
                )
            else:
                unchanged += 1
        self._batch_write("batch_create", creates)
        self._batch_write("batch_update", updates)
        return FeishuSyncSummary(
            created=len(creates),
            updated=len(updates),
            unchanged=unchanged,
            total=len(criteria),
        )
