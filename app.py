"""TrialScope clinical recruitment feasibility workspace."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_local_storage import LocalStorage

from src.analytics import (
    STATUS_LABELS as STATUS_LABELS_ZH,
    apply_scenario,
    blocker_counts,
    build_funnel,
    build_markdown_report,
    criterion_marginal_impact,
    missing_field_counts,
    representation_table,
    scenario_comparison,
    scenario_tradeoff,
)
from src.feishu import (
    FeishuClient,
    FeishuError,
    FeishuSettings,
    apply_reviewed_records,
    criterion_to_feishu_fields,
    feishu_url_value,
    records_for_trial,
)
from src.config import (
    DEEPSEEK_DEFAULT_MODEL,
    MAX_LIVE_CALLS_PER_SESSION,
)
from src.llm_parser import (
    LLMParseError,
    parse_with_deepseek,
    split_for_llm,
)
from src.history import (
    build_workspace_snapshot,
    deserialize_history,
    serialize_history,
    upsert_workspace,
)
from src.models import Criterion, TrialSource
from src.rules import match_dataframe, results_dataframe
from src.trial_sources import (
    PDFScannedError,
    SourceError,
    extract_searchable_pdf,
    fetch_nct_study,
    source_from_text,
)


st.set_page_config(
    page_title="TrialScope | Clinical Feasibility Workspace",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_CSS = (Path(__file__).parent / "assets" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


def empty_source() -> TrialSource:
    """Return a neutral source object for a new, unconfigured workspace."""

    return TrialSource(
        source_type="text",
        identifier="",
        title="",
        source_reference="",
        criteria_text="",
        metadata={"empty_workspace": True},
    )


def has_active_study() -> bool:
    source = st.session_state.get("source")
    return bool(
        isinstance(source, TrialSource)
        and source.identifier.strip()
        and not source.metadata.get("empty_workspace")
    )


@st.cache_data(show_spinner=False)
def cached_marginal_impact(
    patients: pd.DataFrame,
    criteria_payload: str,
) -> pd.DataFrame:
    criteria = [Criterion.model_validate(item) for item in json.loads(criteria_payload)]
    return criterion_marginal_impact(patients, criteria)


def read_setting(name: str, default: Any = None) -> Any:
    if name in os.environ:
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def bool_setting(name: str, default: bool = True) -> bool:
    raw = read_setting(name, default)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def init_state() -> None:
    source = empty_source()
    defaults = {
        "language": "zh",
        "navigation": "项目说明",
        "import_method": "NCT 编号",
        "selected_patient_id": None,
        "source": source,
        "criteria_text": "",
        "criteria": [],
        "patients": pd.DataFrame(),
        "cohort_file_name": "",
        "results": None,
        "live_calls": 0,
        "scenario_comparison": None,
        "scenario_results": None,
        "scenario_parameters": {},
        "scenario_snapshot_key": "",
        "feishu_pending_criteria": None,
        "feishu_review_diffs": [],
        "feishu_sync_note": "",
        "last_parse_note": "尚未生成结构化标准。",
        "scroll_to_top": False,
        "history_workspaces": [],
        "history_browser_payload": None,
        "history_dirty": False,
        "history_write_revision": 0,
        "history_storage_available": True,
        "history_v1_removed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("import_method") not in {"NCT 编号", "粘贴文本", "上传 PDF"}:
        st.session_state.import_method = "NCT 编号"


def set_source(source: TrialSource, criteria_text: str | None = None) -> None:
    text = criteria_text or source.criteria_text
    st.session_state.source = source
    st.session_state.criteria_text = text
    st.session_state.criteria_editor = text
    st.session_state.patients = pd.DataFrame()
    st.session_state.cohort_file_name = ""
    st.session_state.results = None
    st.session_state.scenario_comparison = None
    st.session_state.scenario_results = None
    st.session_state.scenario_parameters = {}
    st.session_state.scenario_snapshot_key = ""
    st.session_state.criteria = []
    st.session_state.last_parse_note = tr(
        "方案原文已就绪，请进入标准审核生成结构化约束。",
        "The source is ready. Continue to rule review to generate structured constraints.",
    )
    record_workspace_event(
        tr("方案已导入", "Protocol imported"),
        source.source_type,
    )


def current_language() -> str:
    return str(st.session_state.get("language", "zh"))


def tr(zh: str, en: str) -> str:
    return en if current_language() == "en" else zh


HISTORY_STORAGE_KEY = "trialscopeai.workspace-history.v2"
LEGACY_HISTORY_STORAGE_KEY = "trialscopeai.workspace-history.v1"


def hydrate_browser_history() -> LocalStorage | None:
    """Merge this browser's persisted workspaces into the active session."""

    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        storage = LocalStorage(key="trialscope_browser_storage")
        if not st.session_state.history_v1_removed:
            if storage.getItem(LEGACY_HISTORY_STORAGE_KEY) is not None:
                storage.eraseItem(
                    LEGACY_HISTORY_STORAGE_KEY,
                    key="trialscope_remove_legacy_history",
                )
            st.session_state.history_v1_removed = True
        payload = storage.getItem(HISTORY_STORAGE_KEY)
        if (
            payload is not None
            and payload != st.session_state.history_browser_payload
            and not st.session_state.history_dirty
        ):
            st.session_state.history_workspaces = deserialize_history(payload)
            st.session_state.history_browser_payload = payload
        st.session_state.history_storage_available = True
        return storage
    except Exception:
        st.session_state.history_storage_available = False
        return None


def record_workspace_event(action: str, detail: str = "") -> None:
    """Save the current protocol state without patient-level records."""

    if "source" not in st.session_state or not has_active_study():
        return
    snapshot = build_workspace_snapshot(
        source=st.session_state.source,
        criteria_text=str(st.session_state.get("criteria_text", "")),
        criteria=st.session_state.get("criteria", []),
        results=st.session_state.get("results"),
        scenario_parameters=st.session_state.get("scenario_parameters", {}),
        action=action,
        detail=detail,
        last_page=str(st.session_state.get("navigation", "研究工作台")),
    )
    st.session_state.history_workspaces = upsert_workspace(
        st.session_state.get("history_workspaces", []), snapshot
    )
    st.session_state.history_dirty = True


def flush_browser_history(storage: LocalStorage | None) -> None:
    if not st.session_state.get("history_dirty"):
        return
    payload = serialize_history(st.session_state.get("history_workspaces", []))
    if storage is not None:
        try:
            revision = int(st.session_state.get("history_write_revision", 0)) + 1
            storage.setItem(
                HISTORY_STORAGE_KEY,
                payload,
                key=f"trialscope_history_write_{revision}",
            )
            st.session_state.history_write_revision = revision
            st.session_state.history_browser_payload = payload
        except Exception:
            st.session_state.history_storage_available = False
    st.session_state.history_dirty = False


def delete_history_workspace(workspace_id: str) -> None:
    st.session_state.history_workspaces = [
        item
        for item in st.session_state.get("history_workspaces", [])
        if item.get("workspace_id") != workspace_id
    ]
    st.session_state.history_dirty = True


def restore_history_workspace(workspace_id: str) -> None:
    record = next(
        (
            item
            for item in st.session_state.get("history_workspaces", [])
            if item.get("workspace_id") == workspace_id
        ),
        None,
    )
    if not record:
        return
    source = TrialSource.model_validate(record["source"])
    criteria = [Criterion.model_validate(item) for item in record.get("criteria", [])]
    st.session_state.source = source
    st.session_state.criteria_text = record.get("criteria_text") or source.criteria_text
    st.session_state.criteria_editor = st.session_state.criteria_text
    st.session_state.criteria = criteria
    st.session_state.patients = pd.DataFrame()
    st.session_state.cohort_file_name = ""
    st.session_state.results = None
    st.session_state.scenario_parameters = dict(record.get("scenario_parameters") or {})
    st.session_state.scenario_comparison = None
    st.session_state.scenario_results = None
    st.session_state.scenario_snapshot_key = ""
    last_page = str(record.get("last_page") or "研究工作台")
    st.session_state.navigation = (
        last_page
        if last_page not in {"项目说明", "历史记录"}
        else "研究工作台"
    )
    st.session_state.scroll_to_top = True
    record_workspace_event(tr("恢复历史工作区", "Workspace restored"))


KIND_LABELS_ZH = {"inclusion": "入组", "exclusion": "排除"}
KIND_LABELS_EN = {"inclusion": "Inclusion", "exclusion": "Exclusion"}
EXECUTION_LABELS_ZH = {"automated": "自动判断", "human_review": "人工确认"}
EXECUTION_LABELS_EN = {"automated": "Rule-executable", "human_review": "Clinical review"}
OPERATOR_LABELS_ZH = {
    "eq": "等于",
    "neq": "不等于",
    "lt": "小于",
    "lte": "小于等于",
    "gt": "大于",
    "gte": "大于等于",
    "between": "区间",
    "in": "属于",
    "not_in": "不属于",
    "is_true": "是",
    "is_false": "否",
    "within_days": "时间窗内",
    "exists": "需要记录",
    "human_review": "人工判断",
}
OPERATOR_LABELS_EN = {
    "eq": "Equals",
    "neq": "Does not equal",
    "lt": "Less than",
    "lte": "At most",
    "gt": "Greater than",
    "gte": "At least",
    "between": "Within range",
    "in": "In set",
    "not_in": "Not in set",
    "is_true": "Yes",
    "is_false": "No",
    "within_days": "Within window",
    "exists": "Documented",
    "human_review": "Clinical judgement",
}
FIELD_LABELS_ZH = {
    "age": "年龄",
    "copd_diagnosis": "COPD 诊断",
    "smoking_pack_years": "吸烟包年",
    "post_bd_fev1_pct_predicted": "支气管舒张后 FEV1 预计值百分比",
    "post_bd_fev1_liters": "支气管舒张后 FEV1 容量",
    "post_bd_fev1_fvc": "支气管舒张后 FEV1/FVC",
    "spirometry_reproducible": "肺功能检查可重复性",
    "contraception_confirmed": "避孕要求确认",
    "informed_consent_confirmed": "知情同意确认",
    "visit_adherence_confirmed": "访视依从性确认",
    "prior_sun101": "既往 SUN-101 使用",
    "severe_comorbidity_concern": "严重合并症风险",
    "days_since_copd_exacerbation": "距 COPD 急性加重天数",
    "oxygen_hours_per_day": "每日氧疗时长",
    "days_since_respiratory_infection": "距呼吸道感染天数",
    "days_since_systemic_steroids": "距全身激素治疗天数",
    "other_significant_respiratory_disease": "其他重要呼吸系统疾病",
    "malignancy_within_5y": "5 年内恶性肿瘤",
    "bladder_outflow_obstruction_within_6m": "6 个月内膀胱流出道梗阻",
    "narrow_angle_glaucoma": "窄角型青光眼",
    "qtc_ms": "QTc 间期",
    "investigational_drug_within_30d": "30 天内试验药物使用",
    "study_drug_class_hypersensitivity": "同类药物超敏反应",
    "aerosol_medication_hypersensitivity": "吸入制剂超敏反应",
    "substance_abuse_within_3m": "3 个月内物质滥用",
    "psychiatric_completion_concern": "精神心理因素影响完成试验",
}
FIELD_LABELS_EN = {
    "age": "Age",
    "copd_diagnosis": "COPD diagnosis",
    "smoking_pack_years": "Smoking exposure (pack-years)",
    "post_bd_fev1_pct_predicted": "Post-BD FEV1, % predicted",
    "post_bd_fev1_liters": "Post-BD FEV1 volume",
    "post_bd_fev1_fvc": "Post-BD FEV1/FVC",
    "spirometry_reproducible": "Spirometry reproducibility",
    "contraception_confirmed": "Contraception confirmation",
    "informed_consent_confirmed": "Informed consent",
    "visit_adherence_confirmed": "Visit adherence",
    "prior_sun101": "Prior SUN-101 use",
    "severe_comorbidity_concern": "Severe comorbidity concern",
    "days_since_copd_exacerbation": "Days since COPD exacerbation",
    "oxygen_hours_per_day": "Daily oxygen therapy",
    "days_since_respiratory_infection": "Days since respiratory infection",
    "days_since_systemic_steroids": "Days since systemic steroids",
    "other_significant_respiratory_disease": "Other significant respiratory disease",
    "malignancy_within_5y": "Malignancy within 5 years",
    "bladder_outflow_obstruction_within_6m": "Bladder outflow obstruction within 6 months",
    "narrow_angle_glaucoma": "Narrow-angle glaucoma",
    "qtc_ms": "QTc interval",
    "investigational_drug_within_30d": "Investigational drug within 30 days",
    "study_drug_class_hypersensitivity": "Study-drug class hypersensitivity",
    "aerosol_medication_hypersensitivity": "Aerosol medication hypersensitivity",
    "substance_abuse_within_3m": "Substance abuse within 3 months",
    "psychiatric_completion_concern": "Psychiatric completion concern",
}
STATUS_LABELS_EN = {
    "eligible": "Rule-eligible",
    "ineligible": "Constraint not met",
    "missing_data": "Data unresolved",
    "needs_review": "Clinical review",
}


def kind_labels() -> dict[str, str]:
    return KIND_LABELS_EN if current_language() == "en" else KIND_LABELS_ZH


def execution_labels() -> dict[str, str]:
    return EXECUTION_LABELS_EN if current_language() == "en" else EXECUTION_LABELS_ZH


def operator_labels() -> dict[str, str]:
    return OPERATOR_LABELS_EN if current_language() == "en" else OPERATOR_LABELS_ZH


def field_labels() -> dict[str, str]:
    return FIELD_LABELS_EN if current_language() == "en" else FIELD_LABELS_ZH


def status_labels() -> dict[str, str]:
    return STATUS_LABELS_EN if current_language() == "en" else STATUS_LABELS_ZH


def criteria_to_frame(criteria: list[Criterion]) -> pd.DataFrame:
    rows = []
    for criterion in criteria:
        item = criterion.model_dump(mode="json")
        item["value"] = json.dumps(item.get("value"), ensure_ascii=False)
        item["applicability"] = json.dumps(item.get("applicability", {}), ensure_ascii=False)
        rows.append(item)
    return pd.DataFrame(rows)


def criteria_to_review_frame(criteria: list[Criterion]) -> pd.DataFrame:
    frame = criteria_to_frame(criteria)
    frame["kind"] = frame["kind"].map(kind_labels()).fillna(frame["kind"])
    frame["operator"] = frame["operator"].map(operator_labels()).fillna(frame["operator"])
    frame["execution_status"] = (
        frame["execution_status"].map(execution_labels()).fillna(frame["execution_status"])
    )
    frame["field"] = frame["field"].map(field_labels()).fillna(frame["field"])
    return frame


def criteria_from_frame(frame: pd.DataFrame) -> list[Criterion]:
    output: list[Criterion] = []
    for row_number, row in frame.iterrows():
        item = row.to_dict()
        try:
            item["value"] = json.loads(str(item.get("value", "null")))
            item["applicability"] = json.loads(str(item.get("applicability", "{}")))
            for nullable in ["field", "unit", "logic_group", "time_window_days"]:
                if pd.isna(item.get(nullable)) or item.get(nullable) == "":
                    item[nullable] = None
            output.append(Criterion.model_validate(item))
        except Exception as exc:
            raise ValueError(tr(
                f"第 {row_number + 1} 行格式无效：{exc}",
                f"Row {row_number + 1} is not a valid constraint: {exc}",
            )) from exc
    return output


def criteria_from_review_frame(frame: pd.DataFrame) -> list[Criterion]:
    normalized = frame.copy()
    normalized["kind"] = normalized["kind"].replace(
        {value: key for key, value in kind_labels().items()}
    )
    normalized["operator"] = normalized["operator"].replace(
        {value: key for key, value in operator_labels().items()}
    )
    normalized["execution_status"] = normalized["execution_status"].replace(
        {value: key for key, value in execution_labels().items()}
    )
    normalized["field"] = normalized["field"].replace(
        {value: key for key, value in field_labels().items()}
    )
    return criteria_from_frame(normalized)


def result_summary(result: Any) -> str:
    if result.overall_status == "eligible":
        return tr("全部可执行标准均通过", "All executable constraints passed")
    if result.overall_status == "ineligible":
        ids = result.failed_criteria
        suffix = ("、" if current_language() == "zh" else ", ").join(ids[:4])
        suffix += tr(" 等", " and others") if len(ids) > 4 else ""
        return tr(
            f"未满足 {len(ids)} 项标准：{suffix}",
            f"{len(ids)} constraints not met: {suffix}",
        )
    if result.overall_status == "missing_data":
        fields = sorted(
            {
                field_labels().get(item.field, item.field or tr("未定义字段", "Undefined field"))
                for item in result.evidences
                if item.status == "missing"
            }
        )
        joiner = "、" if current_language() == "zh" else ", "
        return tr("需要补充：", "Data needed: ") + joiner.join(fields[:3])
    return tr(
        f"{len(result.review_criteria)} 项标准需要研究者确认",
        f"{len(result.review_criteria)} constraints require clinical review",
    )


def display_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, bool):
        return tr("是", "Yes") if value else tr("否", "No")
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def evidence_message(item: Any) -> str:
    if current_language() != "en":
        return item.message
    if item.status == "pass":
        return f"Observed value {display_value(item.patient_value)} is consistent with the executable constraint."
    if item.status == "fail":
        return f"Observed value {display_value(item.patient_value)} does not satisfy the executable constraint."
    if item.status == "missing":
        return "A required field is unavailable; the record remains unresolved."
    if item.status == "review":
        return "This statement requires clinical judgement and is not auto-executed."
    return "The constraint does not apply to this candidate record."


def results_to_display_frame(results: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": result.patient_id,
                "overall_status": status_labels()[result.overall_status],
                "summary": result_summary(result),
                "failed_count": len(result.failed_criteria),
                "missing_count": len(result.missing_criteria),
                "review_count": len(result.review_criteria),
            }
            for result in results
        ]
    )


def criteria_json_bytes(criteria: list[Criterion]) -> bytes:
    return json.dumps(
        [item.model_dump(mode="json") for item in criteria],
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


def feishu_review_template(criteria: list[Criterion], trial_id: str) -> pd.DataFrame:
    """Build an importable review sheet without exposing credentials or patient data."""

    rows: list[dict[str, Any]] = []
    for criterion in criteria:
        row = criterion_to_feishu_fields(trial_id, criterion)
        row.update(
            {
                "审核状态": "需专家复核"
                if criterion.execution_status == "human_review"
                else "待审核",
                "审核人": "",
                "修改意见": "",
                "审核后指标": criterion.field or "",
                "审核后运算符": OPERATOR_LABELS_ZH.get(criterion.operator, criterion.operator),
                "审核后阈值": json.dumps(criterion.value, ensure_ascii=False),
                "审核后单位": criterion.unit or "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def feishu_settings() -> FeishuSettings:
    return FeishuSettings(
        app_id=str(read_setting("FEISHU_APP_ID", "") or ""),
        app_secret=str(read_setting("FEISHU_APP_SECRET", "") or ""),
        base_token=str(read_setting("FEISHU_BITABLE_APP_TOKEN", "") or ""),
        criteria_table_id=str(read_setting("FEISHU_CRITERIA_TABLE_ID", "") or ""),
        snapshot_table_id=str(read_setting("FEISHU_SNAPSHOT_TABLE_ID", "") or ""),
        validation_table_id=str(read_setting("FEISHU_VALIDATION_TABLE_ID", "") or ""),
        workspace_url=str(read_setting("FEISHU_BITABLE_URL", "") or ""),
    )


def render_feishu_review_panel() -> None:
    section_title(tr("飞书协同审核", "Feishu review workspace"))
    st.markdown(
        f"<div class='ts-boundary'><b>{escape(tr('同步边界：', 'Sync boundary:'))}</b>"
        f"{escape(tr('仅同步结构化标准和审核信息；不上传 PDF 正文，也不发送任何患者级数据。', 'Only structured constraints and review fields are synced. Protocol PDFs and row-level candidate data stay out of Feishu.'))}</div>",
        unsafe_allow_html=True,
    )
    settings = feishu_settings()
    if not bool_setting("ENABLE_FEISHU_SYNC", False):
        st.info(tr("飞书协同当前未启用。本地审核和仿真分析仍可正常使用。", "Feishu sync is disabled. Local review and simulation remain fully available."))
        return
    if not settings.configured:
        st.warning(tr("飞书协同已启用，但应用凭证或多维表格标识尚未填写完整。", "Feishu sync is enabled, but the app credentials or Base identifiers are incomplete."))
        return

    columns = st.columns(2)
    sync_clicked = columns[0].button(
        tr("同步至飞书审核", "Send constraints to Feishu"),
        use_container_width=True,
        help=tr("只更新系统生成字段，不覆盖审核状态、审核人和修改意见。", "Updates system-authored fields without overwriting reviewer status, owner or comments."),
    )
    pull_clicked = columns[1].button(
        tr("读取飞书审核结果", "Read reviewed fields"),
        use_container_width=True,
        help=tr("读取审核字段并生成修改差异，确认后才会更新当前规则。", "Loads reviewer fields and presents a diff. Current rules change only after confirmation."),
    )
    if settings.workspace_url:
        st.link_button(tr("打开飞书审核中心", "Open Feishu review workspace"), settings.workspace_url, use_container_width=True)

    if sync_clicked:
        try:
            with st.spinner(tr("正在同步结构化标准...", "Syncing structured constraints...")):
                with FeishuClient(settings) as client:
                    summary = client.sync_criteria(
                        st.session_state.source.identifier,
                        st.session_state.criteria,
                    )
            st.session_state.feishu_sync_note = tr(
                f"同步完成：新增 {summary.created} 条，更新 {summary.updated} 条，无变化 {summary.unchanged} 条。",
                f"Sync complete: {summary.created} created, {summary.updated} updated, {summary.unchanged} unchanged.",
            )
            st.success(st.session_state.feishu_sync_note)
        except FeishuError as exc:
            st.error(str(exc))

    if pull_clicked:
        try:
            with st.spinner(tr("正在读取医学审核结果...", "Reading clinical review fields...")):
                with FeishuClient(settings) as client:
                    records = client.list_records()
                trial_records = records_for_trial(
                    records,
                    st.session_state.source.identifier,
                )
                if trial_records:
                    reviewed, diffs = apply_reviewed_records(
                        st.session_state.criteria,
                        trial_records,
                    )
                else:
                    reviewed, diffs = None, []
            st.session_state.feishu_pending_criteria = reviewed
            st.session_state.feishu_review_diffs = diffs
            if not trial_records:
                st.warning(tr(
                    "飞书中还没有当前试验的审核记录，请先完成同步。",
                    "No review records exist for this trial in Feishu. Sync the constraints first.",
                ))
            elif diffs:
                st.info(tr(f"读取完成，发现 {len(diffs)} 处待确认修改。", f"Review loaded with {len(diffs)} change(s) awaiting confirmation."))
            else:
                st.success(tr("读取完成，飞书审核值与当前规则没有差异。", "Review loaded; no differences from the current constraints."))
        except FeishuError as exc:
            st.error(str(exc))

    pending = st.session_state.feishu_pending_criteria
    diffs = st.session_state.feishu_review_diffs
    if pending is not None and diffs:
        st.dataframe(pd.DataFrame(diffs), width="stretch", hide_index=True)
        confirm, cancel = st.columns(2)
        if confirm.button(tr("确认采用飞书审核结果", "Accept reviewed changes"), type="primary", use_container_width=True):
            st.session_state.criteria = pending
            st.session_state.results = None
            st.session_state.feishu_pending_criteria = None
            st.session_state.feishu_review_diffs = []
            record_workspace_event(tr("已应用协作审核结果", "Collaboration review applied"))
            st.success(tr("飞书审核结果已应用到当前规则。", "Reviewed changes are now applied to the current constraint set."))
            st.rerun()
        if cancel.button(tr("暂不采用", "Keep current rules"), use_container_width=True):
            st.session_state.feishu_pending_criteria = None
            st.session_state.feishu_review_diffs = []
            st.rerun()

    if st.session_state.feishu_sync_note:
        st.caption(st.session_state.feishu_sync_note)


def page_header(title: str, subtitle: str, kicker: str) -> None:
    st.markdown(
        f"""
        <div class="ts-page-header">
            <div class="ts-kicker">{escape(kicker)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_active_study() -> bool:
    """Render a consistent formal empty state until a protocol is imported."""

    if has_active_study():
        return True
    with st.container(key="no_active_study"):
        st.markdown(
            f"<div class='ts-empty-mark'>01</div>"
            f"<h2>{escape(tr('尚未建立研究工作区', 'No study workspace is active'))}</h2>"
            f"<p>{escape(tr('请先导入 NCT 研究、标准原文或文字型 PDF。方案确认后，审核、协作和分析模块会自动承接当前研究。', 'Import an NCT study, eligibility text or a searchable PDF. Review, collaboration and analysis modules will then use that study workspace.'))}</p>",
            unsafe_allow_html=True,
        )
        action, history, spacer = st.columns([1.15, 1.05, 3.4])
        action.button(
            tr("导入研究方案", "Import protocol"),
            key="empty_state_import",
            type="primary",
            use_container_width=True,
            on_click=go_to,
            args=("试验 / PDF 导入",),
        )
        history.button(
            tr("查看研究历史", "View history"),
            key="empty_state_history",
            use_container_width=True,
            on_click=go_to,
            args=("历史记录",),
        )
    return False


def section_title(title: str) -> None:
    st.markdown(f"<div class='ts-section-title'>{escape(title)}</div>", unsafe_allow_html=True)


def insight_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"<div class='ts-insight'><div class='ts-insight-label'>{escape(label)}</div>"
        f"<div class='ts-insight-value'>{escape(value)}</div>"
        f"<div class='ts-insight-note'>{escape(note)}</div></div>",
        unsafe_allow_html=True,
    )


def style_figure(fig: Any, *, height: int = 360) -> Any:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=28, r=24, t=28, b=28),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Microsoft YaHei, sans-serif", size=12, color="#415361"),
        title=dict(text=""),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#162736", font_color="#FFFFFF"),
    )
    fig.update_xaxes(title=None, gridcolor="#E8ECEF", zeroline=False)
    fig.update_yaxes(title=None, gridcolor="#E8ECEF", zeroline=False)
    return fig


def source_summary() -> None:
    source: TrialSource = st.session_state.source
    identifier = escape(source.identifier)
    title = escape(source.title)
    reference = escape(source.source_reference)
    source_type_label = {
        "clinicaltrials": "ClinicalTrials.gov",
        "pdf": tr("PDF 方案", "PDF protocol"),
        "text": tr("录入原文", "Entered source text"),
        "demo": tr("公开来源", "Public source"),
    }.get(source.source_type, source.source_type)
    status = str(source.metadata.get("overall_status") or tr("已导入", "Imported"))
    st.markdown(
        f"""
        <div class="ts-study-card">
            <div>
                <div class="ts-study-id">{identifier}</div>
                <div class="ts-study-title">{title}</div>
                <div class="ts-study-meta">{escape(tr('来源', 'Source'))}: {reference}</div>
            </div>
            <div class="ts-study-tags">
                <span class="ts-tag">{escape(source_type_label)}</span>
                <span class="ts-status ok">{escape(status)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_strip(active_step: int) -> None:
    names = (
        ["Protocol", "Rule review", "Team sign-off", "Cohort lab", "Decision view"]
        if current_language() == "en"
        else ["方案导入", "标准审核", "协作确认", "约束仿真", "决策评估"]
    )
    parts = []
    for number, name in enumerate(names, start=1):
        state = "done" if number < active_step else "active" if number == active_step else ""
        parts.append(
            f"<div class='ts-step {state}'><div class='ts-step-number'>0{number}</div>"
            f"<div class='ts-step-name'>{name}</div></div>"
        )
    st.markdown(f"<div class='ts-steps'>{''.join(parts)}</div>", unsafe_allow_html=True)


def go_to(page: str) -> None:
    st.session_state.navigation = page
    st.session_state.scroll_to_top = True


def scroll_to_top_if_requested() -> None:
    if not st.session_state.get("scroll_to_top"):
        return
    st.html(
        """
        <script>
        window.parent.scrollTo({top: 0, left: 0, behavior: 'instant'});
        const main = window.parent.document.querySelector('section.main');
        if (main) main.scrollTo({top: 0, left: 0, behavior: 'instant'});
        </script>
        """,
        unsafe_allow_javascript=True,
    )
    st.session_state.scroll_to_top = False


def set_language(language: str) -> None:
    st.session_state.language = language


def choose_import_method(method: str) -> None:
    st.session_state.import_method = method


def task_row(
    number: str,
    title: str,
    hint: str,
    status: str,
    page: str,
    action: str,
    *,
    active: bool = False,
) -> None:
    state = "active" if active else "ready"
    with st.container(border=True, key=f"task_{number}_{state}"):
        number_col, text_col, status_col, action_col = st.columns(
            [0.45, 4.5, 1.25, 1.25], vertical_alignment="center"
        )
        number_col.markdown(f"<div class='ts-task-number'>{number}</div>", unsafe_allow_html=True)
        text_col.markdown(
            f"<div class='ts-task-title'>{escape(title)}</div>"
            f"<div class='ts-task-hint'>{escape(hint)}</div>",
            unsafe_allow_html=True,
        )
        status_class = "current" if active else "complete"
        status_col.markdown(
            f"<span class='ts-task-status {status_class}'>{escape(status)}</span>",
            unsafe_allow_html=True,
        )
        action_col.button(
            action,
            key=f"task_action_{number}",
            type="primary" if active else "secondary",
            use_container_width=True,
            on_click=go_to,
            args=(page,),
        )


def page_home() -> None:
    """Public-facing product introduction, separate from the active study workspace."""

    with st.container(key="landing_hero"):
        copy_col, flow_col = st.columns([1.16, 0.84], gap="large", vertical_alignment="center")
        with copy_col:
            st.markdown(
                f"""
                <div class="ts-landing-kicker">{escape(tr('临床招募可行性工作空间', 'CLINICAL RECRUITMENT FEASIBILITY'))}</div>
                <h1>{escape(tr('把试验方案约束，转成可审核的招募判断', 'Turn protocol constraints into reviewable recruitment decisions'))}</h1>
                <p>{escape(tr('TrialScope 将方案导入、标准审核、队列评估和情景比较放进同一条工作流，让医学、临床开发和运营团队基于同一份证据讨论招募可行性。', 'TrialScope brings protocol intake, constraint review, cohort evaluation and scenario comparison into one workspace, so clinical development, medical and operations teams can work from the same evidence.'))}</p>
                """,
                unsafe_allow_html=True,
            )
            primary, secondary, spacer = st.columns([1.18, 1.08, 1.35], gap="small")
            primary.button(
                tr("导入研究方案", "Import protocol"),
                key="landing_open_workspace_primary",
                type="primary",
                use_container_width=True,
                on_click=go_to,
                args=("试验 / PDF 导入",),
            )
            secondary.button(
                tr("查看工作区", "View workspace"),
                key="landing_import_protocol",
                use_container_width=True,
                on_click=go_to,
                args=("研究工作台",),
            )
        with flow_col:
            st.markdown(
                f"""
                <div class="ts-landing-flow">
                    <div class="ts-flow-head"><span>{escape(tr('公开案例', 'PUBLIC CASE'))}</span><b>NCT02347774 · GOLDEN-4</b></div>
                    <div class="ts-flow-row"><span>01</span><div><b>{escape(tr('方案进入', 'Protocol intake'))}</b><small>{escape(tr('NCT、原文或文字型 PDF', 'NCT, source text or searchable PDF'))}</small></div><em>{escape(tr('已就绪', 'Ready'))}</em></div>
                    <div class="ts-flow-row"><span>02</span><div><b>{escape(tr('标准审核', 'Constraint review'))}</b><small>{escape(tr('原文、字段、阈值和时间窗', 'Source, field, threshold and time window'))}</small></div><em>27</em></div>
                    <div class="ts-flow-row"><span>03</span><div><b>{escape(tr('飞书协同审核', 'Feishu collaborative review'))}</b><small>{escape(tr('负责人、状态、意见和版本留痕', 'Owner, status, comments and version trail'))}</small></div><em>{escape(tr('可回读', 'Synced'))}</em></div>
                    <div class="ts-flow-row"><span>04</span><div><b>{escape(tr('队列评估', 'Cohort evaluation'))}</b><small>{escape(tr('逐条证据与主要限制因素', 'Evidence trail and leading constraints'))}</small></div><em>500</em></div>
                    <div class="ts-flow-row"><span>05</span><div><b>{escape(tr('情景比较', 'Scenario comparison'))}</b><small>{escape(tr('规模变化与人群构成权衡', 'Scale and composition trade-offs'))}</small></div><em>{escape(tr('可比较', 'Compare'))}</em></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="ts-landing-section-head">
            <span>{escape(tr('统一的判断依据', 'ONE DECISION RECORD'))}</span>
            <h2>{escape(tr('从方案文字到可追溯判断', 'From protocol text to traceable decisions'))}</h2>
            <p>{escape(tr('减少原文、表格和临时讨论之间的来回切换，让每一步都能回到方案出处。', 'Keep source text, reviewed constraints and feasibility evidence connected throughout the workflow.'))}</p>
        </div>
        <div class="ts-capability-grid">
            <article><span>01</span><h3>{escape(tr('结构化方案约束', 'Structured constraints'))}</h3><p>{escape(tr('保留标准原文，同时整理字段、运算符、阈值、单位和时间窗。', 'Retain the source statement while organizing fields, operators, thresholds, units and time windows.'))}</p></article>
            <article><span>02</span><h3>{escape(tr('医学审核与飞书协作', 'Clinical and Feishu review'))}</h3><p>{escape(tr('区分可执行规则与人工判断项，将负责人、审核状态、修改意见和版本同步到飞书。', 'Separate executable rules from clinical judgement and synchronize ownership, review status, comments and versions to Feishu.'))}</p></article>
            <article><span>03</span><h3>{escape(tr('队列约束分析', 'Cohort constraint analysis'))}</h3><p>{escape(tr('查看筛选路径、主要排除因素、缺失信息和每位候选者的证据链。', 'Inspect the screening path, leading blockers, missing information and candidate-level evidence.'))}</p></article>
            <article><span>04</span><h3>{escape(tr('方案情景比较', 'Protocol scenario comparison'))}</h3><p>{escape(tr('比较参数变化前后的候选规模和人群构成，为跨团队讨论提供量化依据。', 'Compare candidate scale and cohort composition before and after parameter changes.'))}</p></article>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="ts-landing-workflow">
            <div class="ts-workflow-copy">
                <span>{escape(tr('标准工作流', 'WORKFLOW'))}</span>
                <h2>{escape(tr('五个阶段完成一次可行性评估', 'Five stages for a feasibility assessment'))}</h2>
                <p>{escape(tr('每个阶段都有明确的输入、输出和人工确认点；历史工作区会保留最近进度。', 'Each stage has a clear input, output and review point. Workspace history retains the latest progress.'))}</p>
            </div>
            <div class="ts-workflow-steps">
                <div><b>01</b><span>{escape(tr('导入方案', 'Import'))}</span><small>{escape(tr('核对标准原文', 'Confirm source text'))}</small></div>
                <div><b>02</b><span>{escape(tr('结构化标准', 'Structure'))}</span><small>{escape(tr('确认可执行规则', 'Author constraints'))}</small></div>
                <div><b>03</b><span>{escape(tr('飞书审核', 'Sign off'))}</span><small>{escape(tr('跨团队确认留痕', 'Review in Feishu'))}</small></div>
                <div><b>04</b><span>{escape(tr('评估队列', 'Evaluate'))}</span><small>{escape(tr('定位限制因素', 'Identify blockers'))}</small></div>
                <div><b>05</b><span>{escape(tr('比较情景', 'Compare'))}</span><small>{escape(tr('讨论规模与构成', 'Discuss trade-offs'))}</small></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="ts-landing-section-head compact">
            <span>{escape(tr('跨职能协作', 'CROSS-FUNCTIONAL REVIEW'))}</span>
            <h2>{escape(tr('不同角色，共用同一条证据链', 'Different roles, one evidence trail'))}</h2>
        </div>
        <div class="ts-role-grid">
            <article><h3>{escape(tr('临床开发', 'Clinical development'))}</h3><p>{escape(tr('识别对候选规模影响最大的方案约束，准备可行性讨论。', 'Identify the protocol constraints with the greatest impact on candidate scale.'))}</p></article>
            <article><h3>{escape(tr('医学团队', 'Medical'))}</h3><p>{escape(tr('审核结构化标准，明确哪些判断必须保留人工复核。', 'Review structured criteria and preserve decisions that require clinical judgement.'))}</p></article>
            <article><h3>{escape(tr('临床运营', 'Clinical operations'))}</h3><p>{escape(tr('查看信息缺口、工作负担和情景变化，形成执行层面的反馈。', 'Review information gaps, workload and scenario changes for operational feedback.'))}</p></article>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evidence = [
        ("3", tr("种方案输入方式", "protocol input routes")),
        ("27", tr("条审核基准规则", "reviewed reference constraints")),
        ("4", tr("类队列判断状态", "cohort decision states")),
        ("100%", tr("标准来源可追溯", "source traceability")),
    ]
    st.markdown(
        f"<div class='ts-case-evidence-label'>{escape(tr('GOLDEN-4 公开案例的系统输出', 'SYSTEM OUTPUT FOR THE GOLDEN-4 PUBLIC CASE'))}</div>",
        unsafe_allow_html=True,
    )
    evidence_cols = st.columns(4, gap="small")
    for column, (value, label) in zip(evidence_cols, evidence):
        column.markdown(
            f"<div class='ts-evidence-item'><b>{escape(value)}</b><span>{escape(label)}</span></div>",
            unsafe_allow_html=True,
        )

    with st.container(key="landing_final_cta"):
        text_col, action_col = st.columns([3.6, 1.1], vertical_alignment="center")
        text_col.markdown(
            f"<h2>{escape(tr('建立一个干净的研究工作区', 'Start a clean study workspace'))}</h2>"
            f"<p>{escape(tr('从 NCT 编号、标准原文或文字型 PDF 开始，系统不会自动带入任何案例数据。', 'Begin with an NCT ID, eligibility text or searchable PDF. No case data is preloaded into the workspace.'))}</p>",
            unsafe_allow_html=True,
        )
        action_col.button(
            tr("导入研究方案", "Import protocol"),
            key="landing_open_workspace_final",
            type="primary",
            use_container_width=True,
            on_click=go_to,
            args=("试验 / PDF 导入",),
        )


def page_workspace() -> None:
    if not has_active_study():
        page_header(
            tr("研究工作台", "Study workspace"),
            tr(
                "研究建立后，这里将集中呈现方案约束、队列状态、主要限制因素和人群构成。",
                "Once a study is created, this workspace will consolidate protocol constraints, cohort status, leading blockers and cohort composition.",
            ),
            tr("研究总览", "STUDY OVERVIEW"),
        )
        require_active_study()
        return
    if not st.session_state.criteria:
        page_header(
            tr("研究工作台", "Study workspace"),
            tr("方案已经进入工作区，下一步是生成并审核结构化标准。", "The protocol is in the workspace. Generate and review its structured constraints next."),
            tr("研究进度", "STUDY PROGRESS"),
        )
        source_summary()
        with st.container(key="workspace_stage_state"):
            st.markdown(
                f"<span>01 / 04</span><h2>{escape(tr('方案原文已就绪', 'Protocol source ready'))}</h2>"
                f"<p>{escape(tr('标准审核、协作确认、队列评估和情景分析将在完成结构化后依次开放。', 'Rule review, collaboration, cohort evaluation and scenario analysis will open as the study progresses.'))}</p>",
                unsafe_allow_html=True,
            )
            st.button(
                tr("进入标准审核", "Open rule review"),
                key="workspace_to_review",
                type="primary",
                on_click=go_to,
                args=("标准解析",),
            )
        return
    if st.session_state.patients.empty:
        page_header(
            tr("研究工作台", "Study workspace"),
            tr("结构化标准已进入当前研究，导入候选队列后即可运行约束评估。", "Structured constraints are ready. Import a cohort to begin constraint evaluation."),
            tr("研究进度", "STUDY PROGRESS"),
        )
        source_summary()
        with st.container(key="workspace_stage_state"):
            st.markdown(
                f"<span>02 / 04</span><h2>{escape(tr('等待候选队列', 'Cohort data required'))}</h2>"
                f"<p>{escape(tr(f'当前研究已有 {len(st.session_state.criteria)} 条结构化标准。系统不会自动带入案例队列。', f'{len(st.session_state.criteria)} structured constraints are ready. The system does not preload a case cohort.'))}</p>",
                unsafe_allow_html=True,
            )
            st.button(
                tr("导入候选队列", "Import cohort"),
                key="workspace_to_cohort",
                type="primary",
                on_click=go_to,
                args=("患者预筛",),
            )
        return
    if st.session_state.results is None:
        page_header(
            tr("研究工作台", "Study workspace"),
            tr("方案标准与候选队列均已就绪，等待运行规则评估。", "The constraints and cohort are ready for rule evaluation."),
            tr("研究进度", "STUDY PROGRESS"),
        )
        source_summary()
        with st.container(key="workspace_stage_state"):
            st.markdown(
                f"<span>03 / 04</span><h2>{escape(tr('队列等待评估', 'Cohort ready for evaluation'))}</h2>"
                f"<p>{escape(tr(f'已载入 {len(st.session_state.patients)} 条候选记录，尚未生成判断结果。', f'{len(st.session_state.patients)} cohort rows are loaded; no results have been generated yet.'))}</p>",
                unsafe_allow_html=True,
            )
            st.button(
                tr("运行队列评估", "Run cohort evaluation"),
                key="workspace_run_cohort",
                type="primary",
                on_click=go_to,
                args=("患者预筛",),
            )
        return
    results = st.session_state.results or []
    patients = st.session_state.patients
    criteria = st.session_state.criteria
    counts = Counter(item.overall_status for item in results)
    potential = counts.get("eligible", 0) + counts.get("needs_review", 0)
    unresolved = counts.get("missing_data", 0) + counts.get("needs_review", 0)
    eligible_rate = counts.get("eligible", 0) / len(results) * 100 if results else 0

    st.markdown(
        f"""
        <div class="ts-dashboard-heading">
            <div>
                <div class="ts-kicker">{escape(tr('临床开发工作台', 'CLINICAL DEVELOPMENT WORKSPACE'))}</div>
                <h1>{escape(tr('招募可行性总览', 'Recruitment feasibility overview'))}</h1>
                <p>{escape(tr('集中查看方案约束、候选规模、信息负担与人群构成。', 'Monitor protocol constraints, candidate scale, information burden and cohort composition.'))}</p>
            </div>
            <span class="ts-system-state">{escape(tr('规则集已就绪', 'RULE SET READY'))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    source_summary()

    cols = st.columns(5)
    metrics = [
        (tr("方案约束", "Protocol constraints"), str(len(criteria)), tr("已结构化", "Structured")),
        (tr("评估队列", "Evaluated cohort"), str(len(patients)), tr("已导入记录", "Imported records")),
        (tr("规则符合", "Rule-eligible"), str(counts.get("eligible", 0)), f"{eligible_rate:.1f}%"),
        (tr("潜在候选", "Potential candidates"), str(potential), tr("含待医学复核", "Includes clinical review")),
        (tr("待处理", "Open workload"), str(unresolved), tr("信息补充或复核", "Data or review needed")),
    ]
    for column, (label, value, help_text) in zip(cols, metrics):
        column.metric(label, value, help=help_text)

    dashboard_status_labels = {
        "eligible": tr("规则符合", "Rule-eligible"),
        "ineligible": tr("未满足约束", "Constraint not met"),
        "missing_data": tr("信息待补", "Data unresolved"),
        "needs_review": tr("医学复核", "Clinical review"),
    }
    status_frame = pd.DataFrame(
        [
            {"status": status, "label": dashboard_status_labels[status], "count": counts.get(status, 0)}
            for status in ["eligible", "ineligible", "missing_data", "needs_review"]
        ]
    )
    funnel = build_funnel(patients, criteria)
    if current_language() == "en":
        funnel["stage"] = funnel["stage"].replace({
            "候选队列": "Evaluated cohort",
            "其余可执行标准": "Remaining executable constraints",
            "规则符合或待复核": "Eligible or review",
        })

    section_title(tr("队列运行概况", "Cohort operations"))
    chart_left, chart_right = st.columns([0.82, 1.18], gap="medium")
    with chart_left:
        with st.container(border=True, key="dashboard_status_panel"):
            st.markdown(f"<div class='ts-panel-title'>{escape(tr('队列状态分布', 'Cohort status'))}</div>", unsafe_allow_html=True)
            status_fig = px.pie(
                status_frame,
                names="label",
                values="count",
                hole=0.64,
                color="status",
                color_discrete_map={
                    "eligible": "#2F7A73",
                    "ineligible": "#7A8995",
                    "missing_data": "#B78232",
                    "needs_review": "#3F78A2",
                },
            )
            status_fig.update_traces(textposition="outside", textinfo="percent+label", sort=False)
            status_fig.add_annotation(
                text=f"<b>{counts.get('eligible', 0)}</b><br>{escape(tr('规则符合', 'eligible'))}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=15, color="#1B2733"),
            )
            st.plotly_chart(style_figure(status_fig, height=330), width="stretch", config={"displayModeBar": False})
    with chart_right:
        with st.container(border=True, key="dashboard_funnel_panel"):
            st.markdown(f"<div class='ts-panel-title'>{escape(tr('候选队列约束路径', 'Constraint path'))}</div>", unsafe_allow_html=True)
            funnel_fig = px.funnel(funnel, x="count", y="stage", text="count")
            funnel_fig.update_traces(marker_color="#326F98", textfont_color="#F4F8FB")
            st.plotly_chart(style_figure(funnel_fig, height=330), width="stretch", config={"displayModeBar": False})

    blockers = blocker_counts(results, criteria)
    criterion_map = {item.criterion_id: item for item in criteria}
    chart_left, chart_right = st.columns([1.08, 0.92], gap="medium")
    with chart_left:
        with st.container(border=True, key="dashboard_blockers_panel"):
            st.markdown(f"<div class='ts-panel-title'>{escape(tr('主要限制标准', 'Leading constraints'))}</div>", unsafe_allow_html=True)
            top_blockers = blockers.head(7).copy()
            if top_blockers.empty:
                st.info(tr("当前没有明确的未通过标准。", "No failed constraint is currently recorded."))
            else:
                top_blockers["label"] = [
                    f"{criterion_id} · {field_labels().get(criterion_map.get(criterion_id).field if criterion_map.get(criterion_id) else None, criterion_id)}"
                    for criterion_id in top_blockers["criterion_id"]
                ]
                blocker_fig = px.bar(
                    top_blockers.sort_values("count"),
                    x="count",
                    y="label",
                    orientation="h",
                    text="count",
                )
                blocker_fig.update_traces(marker_color="#326F98", textposition="outside", cliponaxis=False)
                st.plotly_chart(style_figure(blocker_fig, height=350), width="stretch", config={"displayModeBar": False})
    with chart_right:
        with st.container(border=True, key="dashboard_representation_panel"):
            st.markdown(f"<div class='ts-panel-title'>{escape(tr('人群构成对照', 'Cohort composition'))}</div>", unsafe_allow_html=True)
            representation = representation_table(patients, results)
            representation = representation[representation["metric"] != "平均年龄"].copy()
            if representation.empty:
                st.info(tr("当前队列没有可用的人群构成字段。", "No cohort-composition fields are available."))
            else:
                if current_language() == "en":
                    representation["group"] = representation["group"].replace({
                        "候选队列": "Evaluated cohort",
                        "规则符合或待复核": "Eligible or review",
                    })
                    representation["metric"] = representation["metric"].replace({
                        "女性占比": "Female",
                        "65岁及以上占比": "Age 65+",
                        "重度疾病占比": "Severe disease",
                    })
                composition_fig = px.bar(
                    representation,
                    x="metric",
                    y="value",
                    color="group",
                    barmode="group",
                    text_auto=".1f",
                    color_discrete_sequence=["#A7B5C0", "#2F7A73"],
                )
                composition_fig.update_yaxes(ticksuffix="%")
                st.plotly_chart(style_figure(composition_fig, height=350), width="stretch", config={"displayModeBar": False})

    section_title(tr("关键标准明细", "Constraint register"))
    if not blockers.empty:
        detail = blockers.head(8).copy()
        detail[tr("指标", "Field")] = [
            field_labels().get(criterion_map.get(item).field if criterion_map.get(item) else None, item)
            for item in detail["criterion_id"]
        ]
        detail = detail.rename(columns={
            "criterion_id": tr("标准编号", "Constraint ID"),
            "count": tr("未通过记录", "Failed records"),
            "criterion": tr("方案原文", "Protocol statement"),
        })
        detail.insert(0, tr("排序", "Rank"), range(1, len(detail) + 1))
        st.dataframe(
            detail[[tr("排序", "Rank"), tr("标准编号", "Constraint ID"), tr("指标", "Field"), tr("未通过记录", "Failed records"), tr("方案原文", "Protocol statement")]],
            width="stretch",
            hide_index=True,
            height=310,
        )

    st.markdown(
        f"<div class='ts-system-footnote'>{escape(tr('数据范围：当前研究方案与本会话导入的候选队列。结果用于方案评估和协作审核，不用于诊断或自动入组。', 'Data scope: the active protocol and cohort imported in this session. Results support protocol assessment and review; they are not used for diagnosis or automatic enrolment.'))}</div>",
        unsafe_allow_html=True,
    )


def page_history() -> None:
    page_header(
        tr("研究历史", "Workspace history"),
        tr(
            "回到已处理过的方案、审核状态和分析阶段，无需重新导入或重复设置。",
            "Return to prior protocols, review states and analysis stages without rebuilding the workflow.",
        ),
        tr("工作区记录", "WORKSPACE RECORDS"),
    )

    history = list(st.session_state.get("history_workspaces", []))
    scope_col, action_col = st.columns([4.1, 1.2], vertical_alignment="center")
    scope_col.markdown(
        f"<div class='ts-history-scope'><b>{escape(tr('仅保存在当前浏览器', 'Stored in this browser only'))}</b>"
        f"<span>{escape(tr('保存方案、审核标准、分析摘要和操作时间；不保存 PDF 原文件或候选人明细。', 'Stores protocol metadata, reviewed constraints, aggregate analysis and activity time. PDF files and candidate-level rows are excluded.'))}</span></div>",
        unsafe_allow_html=True,
    )
    if action_col.button(
        tr("保存当前研究", "Save current study"),
        type="primary",
        use_container_width=True,
        disabled=not has_active_study(),
    ):
        record_workspace_event(tr("手动保存工作区", "Workspace saved"))
        st.success(tr("当前进度已保存。", "Current progress saved."))
        history = list(st.session_state.history_workspaces)

    if not history:
        with st.container(key="history_empty_state"):
            st.markdown(
                f"<h3>{escape(tr('还没有已保存的研究', 'No saved studies yet'))}</h3>"
                f"<p>{escape(tr('导入方案或保存当前研究后，会在这里形成可恢复的工作区。', 'Import a protocol or save the active study to create a restorable workspace.'))}</p>",
                unsafe_allow_html=True,
            )
            first, second, spacer = st.columns([1.1, 1.1, 3.4])
            first.button(
                tr("导入方案", "Import protocol"),
                type="primary",
                use_container_width=True,
                on_click=go_to,
                args=("试验 / PDF 导入",),
            )
            second.button(
                tr("打开工作台", "Open workspace"),
                use_container_width=True,
                on_click=go_to,
                args=("研究工作台",),
            )
        return

    total_events = sum(len(item.get("events") or []) for item in history)
    reviewed = sum(int(item.get("criterion_count") or 0) > 0 for item in history)
    metrics = st.columns(3)
    metrics[0].metric(tr("研究工作区", "Studies"), len(history))
    metrics[1].metric(tr("已有审核标准", "With reviewed constraints"), reviewed)
    metrics[2].metric(tr("保留操作记录", "Recorded actions"), total_events)

    section_title(tr("最近研究", "Recent studies"))
    for record in history:
        workspace_id = str(record.get("workspace_id", ""))
        summary = dict(record.get("result_summary") or {})
        updated_at = str(record.get("updated_at", ""))
        try:
            display_time = datetime.fromisoformat(updated_at).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            display_time = updated_at
        with st.container(border=True, key=f"history_record_{workspace_id}"):
            copy_col, restore_col, delete_col = st.columns(
                [4.7, 1.05, 0.85], vertical_alignment="center"
            )
            copy_col.markdown(
                f"<div class='ts-history-record'><span>{escape(str(record.get('identifier', 'STUDY')))}</span>"
                f"<h3>{escape(str(record.get('title') or tr('未命名研究', 'Untitled study')))}</h3>"
                f"<p>{escape(tr('最近操作', 'Last action'))}: {escape(str(record.get('last_action', '—')))} · {escape(display_time)}</p></div>",
                unsafe_allow_html=True,
            )
            if restore_col.button(
                tr("恢复并继续", "Restore"),
                key=f"restore_history_{workspace_id}",
                type="primary",
                use_container_width=True,
            ):
                restore_history_workspace(workspace_id)
                st.rerun()
            if delete_col.button(
                tr("删除", "Delete"),
                key=f"delete_history_{workspace_id}",
                use_container_width=True,
            ):
                delete_history_workspace(workspace_id)
                st.rerun()

            facts = st.columns(4)
            facts[0].markdown(
                f"<div class='ts-history-fact'><b>{int(record.get('criterion_count') or 0)}</b><span>{escape(tr('条标准', 'constraints'))}</span></div>",
                unsafe_allow_html=True,
            )
            facts[1].markdown(
                f"<div class='ts-history-fact'><b>{int(summary.get('total') or 0)}</b><span>{escape(tr('条队列记录', 'cohort records'))}</span></div>",
                unsafe_allow_html=True,
            )
            facts[2].markdown(
                f"<div class='ts-history-fact'><b>{int(summary.get('eligible') or 0)}</b><span>{escape(tr('规则符合', 'rule-eligible'))}</span></div>",
                unsafe_allow_html=True,
            )
            facts[3].markdown(
                f"<div class='ts-history-fact'><b>{len(record.get('events') or [])}</b><span>{escape(tr('次操作', 'actions'))}</span></div>",
                unsafe_allow_html=True,
            )

            with st.expander(tr("查看操作时间线", "View activity timeline")):
                events = list(record.get("events") or [])
                for event in reversed(events):
                    event_at = str(event.get("at", ""))
                    try:
                        event_time = datetime.fromisoformat(event_at).strftime("%m-%d %H:%M")
                    except ValueError:
                        event_time = event_at
                    detail = str(event.get("detail") or "")
                    st.markdown(
                        f"<div class='ts-history-event'><time>{escape(event_time)}</time>"
                        f"<b>{escape(str(event.get('action', '')))}</b>"
                        f"<span>{escape(detail)}</span></div>",
                        unsafe_allow_html=True,
                    )


def page_import() -> None:
    page_header(
        tr("导入试验方案", "Start with the source protocol"),
        tr(
            "选择公开试验、粘贴标准原文或上传文字型 PDF；确认原文后再生成待审核规则。",
            "Load a public study, paste eligibility text or use a searchable PDF. Source text stays visible before any rule is generated.",
        ),
        tr("01 · 方案导入", "01 · PROTOCOL SOURCE"),
    )
    section_title(tr("选择方案来源", "Choose one protocol source"))
    st.caption(tr(
        "每次只处理一种输入，确认原文后才进入结构化审核。",
        "Each run uses one source. Review the extracted text before moving into rule authoring.",
    ))
    source_options = [
        ("NCT 编号", tr("公开试验", "NCT record"), tr("从 ClinicalTrials.gov 获取标准", "Fetch public eligibility text")),
        ("粘贴文本", tr("标准原文", "Paste text"), tr("适合已有 Word 或网页文本", "For copied protocol sections")),
        ("上传 PDF", tr("研究方案", "Searchable PDF"), tr("支持可检索文字的方案文件", "For protocols with searchable text")),
    ]
    source_columns = st.columns(3)
    for index, (method_name, title, hint) in enumerate(source_options):
        is_selected = st.session_state.import_method == method_name
        state = "selected" if is_selected else "idle"
        with source_columns[index]:
            with st.container(border=True, key=f"source_card_{index}_{state}"):
                st.markdown(
                    f"<div class='ts-source-title'>{escape(title)}</div>"
                    f"<div class='ts-source-hint'>{escape(hint)}</div>",
                    unsafe_allow_html=True,
                )
                st.button(
                    tr("已选择", "Selected") if is_selected else tr("选择此来源", "Use this source"),
                    key=f"choose_source_{index}",
                    disabled=is_selected,
                    use_container_width=True,
                    on_click=choose_import_method,
                    args=(method_name,),
                )

    method = st.session_state.import_method
    method_display = {
        "NCT 编号": "ClinicalTrials.gov",
        "粘贴文本": tr("粘贴标准原文", "Pasted eligibility text"),
        "上传 PDF": tr("文字型 PDF", "Searchable PDF"),
    }.get(method, method)
    st.markdown(
        f"<div class='ts-selection-label'>{escape(tr('当前选择', 'Active source'))}: {escape(method_display)}</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True, key="source_input_panel"):
        if method == "NCT 编号":
            st.markdown(tr("**输入公开试验编号**", "**Enter a public trial identifier**"))
            nct_id = st.text_input(
                tr("ClinicalTrials.gov NCT 编号", "ClinicalTrials.gov NCT ID"),
                value="",
                placeholder="NCT00000000",
                help=tr("输入完整 NCT 编号。", "Enter the complete NCT identifier."),
            )
            if st.button(tr("获取试验标准", "Fetch eligibility criteria"), type="primary", use_container_width=True):
                with st.spinner(tr("正在读取 ClinicalTrials.gov...", "Reading ClinicalTrials.gov...")):
                    try:
                        set_source(fetch_nct_study(nct_id))
                        st.success(tr("试验记录与入排标准已导入。请在下方核对原文。", "Trial record loaded. Review the eligibility source below."))
                    except SourceError as exc:
                        st.error(str(exc))
        elif method == "粘贴文本":
            st.markdown(tr("**粘贴入组与排除标准**", "**Paste the eligibility section**"))
            pasted = st.text_area(
                tr("标准原文", "Source criteria"),
                height=240,
                placeholder="Inclusion Criteria: ...\n\nExclusion Criteria: ...",
            )
            if st.button(tr("载入这段原文", "Load this source text"), type="primary", use_container_width=True):
                try:
                    set_source(source_from_text(pasted))
                    st.success(tr("文本已载入。请在下方核对原文。", "Text loaded. Check the source before continuing."))
                except SourceError as exc:
                    st.error(str(exc))
        else:
            st.markdown(tr("**上传可搜索的文字型 PDF**", "**Upload a searchable, text-based PDF**"))
            uploaded = st.file_uploader(tr("选择 PDF 文件", "Choose a PDF"), type=["pdf"], accept_multiple_files=False)
            st.caption(tr(
                "限制：20 MB、200 页；首版不支持扫描件 OCR。文件只在当前会话内存中处理。",
                "Limit: 20 MB and 200 pages. Scanned documents are flagged rather than guessed. Processing stays in session memory.",
            ))
            if uploaded and st.button(tr("提取入排标准", "Locate eligibility section"), type="primary", use_container_width=True):
                try:
                    extraction = extract_searchable_pdf(uploaded.getvalue(), uploaded.name)
                    source = TrialSource(
                        source_type="pdf",
                        identifier=uploaded.name,
                        title=Path(uploaded.name).stem,
                        source_reference=f"uploaded-pdf:{uploaded.name}",
                        criteria_text=extraction.criteria_text,
                        metadata={"page_count": extraction.page_count, "section_found": extraction.section_found},
                    )
                    set_source(source)
                    if extraction.warning:
                        st.warning(extraction.warning)
                    else:
                        st.success(tr(
                            f"已从 {extraction.page_count} 页 PDF 中定位入排标准章节。",
                            f"Eligibility text located in a {extraction.page_count}-page PDF.",
                        ))
                except (SourceError, PDFScannedError) as exc:
                    st.error(str(exc))

    if not has_active_study():
        return

    section_title(tr("核对方案原文", "Review the evidence source"))
    source_summary()
    left, right = st.columns([2.15, 1], gap="large")
    with left:
        edited_text = st.text_area(
            tr("用于生成规则的入排标准原文", "Eligibility text used to author constraints"),
            value=st.session_state.criteria_text,
            key="criteria_editor",
            height=360,
            help=tr("可以删除目录、页眉等无关内容；仅确认后的文本会用于自动解析。", "Remove headers or unrelated appendices. Only confirmed text is sent for semantic extraction."),
        )
        if st.button(tr("确认原文并进入标准审核", "Confirm source and review constraints"), type="primary", use_container_width=True):
            if len(edited_text.strip()) < 30:
                st.error(tr("文本过短，无法解析。", "The source text is too short to parse reliably."))
            else:
                st.session_state.criteria_text = edited_text.strip()
                go_to("标准解析")
                st.rerun()
    with right:
        st.markdown(
            f"<div class='ts-insight'><div class='ts-insight-label'>{escape(tr('提交前检查', 'SOURCE CHECK'))}</div>"
            f"<div class='ts-insight-value'>{escape(tr('原文可人工修订', 'Human-editable source'))}</div>"
            f"<div class='ts-insight-note'>{escape(tr('保留完整的入组与排除章节；删除目录、页眉和无关附录。', 'Keep complete inclusion and exclusion sections; remove navigation and unrelated appendices.'))}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='ts-boundary'><b>{escape(tr('文件处理：', 'File handling:'))}</b>"
            f"{escape(tr('上传内容仅在当前会话内存中处理，不写入仓库或数据库。扫描型 PDF 不进行 OCR。', 'Uploads stay in session memory and are not written to the repository or a database. OCR is intentionally out of scope.'))}</div>",
            unsafe_allow_html=True,
        )


def page_parse() -> None:
    page_header(
        tr("方案约束审核", "Author and review executable constraints"),
        tr(
            "逐条核对方案原文、结构化条件与执行方式；只有人工确认后的规则才进入候选人群仿真。",
            "Validate how each source statement becomes a rule. Clinical judgement stays explicit and every accepted constraint remains traceable.",
        ),
        tr("02 · 标准审核", "02 · RULE REVIEW"),
    )
    if not require_active_study():
        return
    source_summary()
    api_key = str(read_setting("DEEPSEEK_API_KEY", "") or "")
    model = str(read_setting("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL))
    live_enabled = bool_setting("ENABLE_LIVE_LLM", True)
    remaining = max(0, MAX_LIVE_CALLS_PER_SESSION - st.session_state.live_calls)
    required_chunks = len(split_for_llm(st.session_state.criteria_text))
    inclusion_count = sum(item.kind == "inclusion" for item in st.session_state.criteria)
    exclusion_count = sum(item.kind == "exclusion" for item in st.session_state.criteria)
    review_count = sum(item.execution_status == "human_review" for item in st.session_state.criteria)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("入组标准", "Inclusion"), inclusion_count)
    c2.metric(tr("排除标准", "Exclusion"), exclusion_count)
    c3.metric(tr("需要人工判断", "Clinical judgement"), review_count)
    traceable = sum(bool(item.source_text and item.source_reference) for item in st.session_state.criteria)
    coverage = traceable / len(st.session_state.criteria) * 100 if st.session_state.criteria else 0
    c4.metric(tr("原文追溯率", "Source traceability"), f"{coverage:.0f}%")

    section_title(tr("生成待审标准", "Generate draft constraints"))
    with st.container(border=True, key="semantic_extraction_controls"):
        if st.button(
            tr("生成或重新生成待审标准", "Generate or refresh draft constraints"),
            use_container_width=True,
            disabled=not api_key or not live_enabled or required_chunks > remaining,
        ):
            with st.spinner(tr("正在提取字段、阈值和时间窗...", "Extracting fields, thresholds and time windows...")):
                try:
                    outcome = parse_with_deepseek(
                        st.session_state.criteria_text,
                        api_key=api_key,
                        source_reference=st.session_state.source.source_reference,
                        model=model,
                    )
                    if not outcome.from_cache:
                        st.session_state.live_calls += outcome.chunk_count
                    st.session_state.criteria = outcome.criteria
                    st.session_state.results = None
                    st.session_state.last_parse_note = tr(
                        f"{'命中缓存' if outcome.from_cache else '实时解析完成'}：{outcome.model}，{outcome.chunk_count} 个文本块。",
                        f"{'Cached result' if outcome.from_cache else 'Live extraction complete'}: {outcome.model}, {outcome.chunk_count} text block(s).",
                    )
                    record_workspace_event(
                        tr("结构化标准已生成", "Structured constraints generated"),
                        outcome.model,
                    )
                    st.success(st.session_state.last_parse_note)
                except LLMParseError as exc:
                    st.error(str(exc))

    if not api_key:
        st.info(tr("当前未配置结构化解析服务，请由系统管理员在部署配置中启用。", "The structured extraction service is not configured. Ask the system administrator to enable it in deployment settings."))
    elif not live_enabled:
        st.warning(tr("自动解析服务当前已关闭。", "Live semantic extraction is disabled."))
    elif required_chunks > remaining:
        st.warning(tr(f"当前文本需要 {required_chunks} 次调用，已超过本会话剩余额度 {remaining} 次。", f"This source needs {required_chunks} calls; only {remaining} remain in the session."))
    with st.expander(tr("查看解析记录", "Extraction log"), expanded=False):
        st.caption(st.session_state.last_parse_note)
        st.caption(tr(f"解析模型：{model} · 本会话剩余额度：{remaining} 次", f"Model: {model} · Session calls remaining: {remaining}"))

    if not st.session_state.criteria:
        st.warning(tr("当前研究还没有结构化标准。完成解析后即可逐条审核。", "This study has no structured constraints yet. Generate them to begin line-by-line review."))
        return

    section_title(tr("逐条审核", "Review every constraint"))
    st.markdown(
        f"<div class='ts-action-guide'><strong>{escape(tr('当前任务：确认规则可以按方案原文执行', 'Decision gate: confirm that each rule reflects the source'))}</strong>"
        f"<span>{escape(tr('可直接修改表格；重点核对阈值、单位、时间窗和“人工确认”项。', 'Edit cells directly. Focus on thresholds, units, time windows and judgement-only criteria.'))}</span></div>",
        unsafe_allow_html=True,
    )
    frame = criteria_to_review_frame(st.session_state.criteria)
    with st.form("criteria_review_form", border=True):
        editable = st.data_editor(
            frame,
            width="stretch",
            height=520,
            hide_index=True,
            num_rows="fixed",
            disabled=["criterion_id"],
            column_order=[
                "criterion_id",
                "kind",
                "source_text",
                "field",
                "operator",
                "value",
                "unit",
                "time_window_days",
                "execution_status",
                "confidence",
            ],
            column_config={
                "criterion_id": st.column_config.TextColumn(tr("编号", "ID"), width="small"),
                "kind": st.column_config.SelectboxColumn(tr("类型", "Type"), options=list(kind_labels().values()), width="small"),
                "operator": st.column_config.SelectboxColumn(
                    tr("判断条件", "Operator"),
                    options=list(operator_labels().values()),
                    width="medium",
                ),
                "execution_status": st.column_config.SelectboxColumn(
                    tr("执行方式", "Execution"), options=list(execution_labels().values()), width="medium"
                ),
                "confidence": st.column_config.ProgressColumn(
                    tr("结构化置信度", "Extraction confidence"), min_value=0.0, max_value=1.0, format="%.2f", width="medium"
                ),
                "source_text": st.column_config.TextColumn(tr("标准原文", "Source statement"), width="large"),
                "field": st.column_config.TextColumn(tr("结构化字段", "Structured field"), width="large"),
                "value": st.column_config.TextColumn(tr("阈值", "Value"), width="medium"),
                "unit": st.column_config.TextColumn(tr("单位", "Unit"), width="small"),
                "time_window_days": st.column_config.NumberColumn(tr("时间窗（天）", "Window (days)"), width="small"),
            },
        )
        save_review = st.form_submit_button(
            tr("保存审核并进入协作确认", "Save review and continue to team sign-off"),
            type="primary",
            use_container_width=True,
        )
    if save_review:
        try:
            st.session_state.criteria = criteria_from_review_frame(editable)
            st.session_state.results = None
            go_to("协作审核")
            record_workspace_event(tr("标准审核已保存", "Constraint review saved"))
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.caption(tr("下载内容为当前已保存版本；未保存修改不会进入导出文件。", "Exports use the last saved version; unsaved table edits are excluded."))
    d1, d2 = st.columns(2)
    d1.download_button(
        tr("导出标准 JSON", "Export constraints as JSON"),
        data=criteria_json_bytes(st.session_state.criteria),
        file_name="trialscope_criteria.json",
        mime="application/json",
        use_container_width=True,
    )
    d2.download_button(
        tr("导出标准 CSV", "Export constraints as CSV"),
        data=criteria_to_frame(st.session_state.criteria).to_csv(index=False).encode("utf-8-sig"),
        file_name="trialscope_criteria.csv",
        mime="text/csv",
        use_container_width=True,
    )


def page_collaboration() -> None:
    page_header(
        tr("协作审核中心", "Cross-functional review workspace"),
        tr(
            "把方案约束交给医学与运营人员确认，在飞书中保留审核状态、修改意见和版本记录。",
            "Route constraints to medical and operations reviewers. Feishu provides ownership, comments and an auditable hand-off before simulation.",
        ),
        tr("03 · 协作确认", "03 · TEAM SIGN-OFF"),
    )
    if not require_active_study():
        return
    if not st.session_state.criteria:
        st.warning(tr("请先完成标准解析与本地审核。", "Complete local rule review before team sign-off."))
        st.button(tr("返回标准审核", "Back to rule review"), type="primary", on_click=go_to, args=("标准解析",))
        return

    settings = feishu_settings()
    enabled = bool_setting("ENABLE_FEISHU_SYNC", False)
    configured = settings.configured
    status_columns = st.columns(3)
    with status_columns[0]:
        insight_card(tr("待协作标准", "Constraints for review"), str(len(st.session_state.criteria)), tr("只同步结构化标准与审核字段", "Only structured and review fields are synced"))
    with status_columns[1]:
        human_review_count = sum(
            item.execution_status == "human_review" for item in st.session_state.criteria
        )
        insight_card(tr("需人工判断", "Judgement-only"), str(human_review_count), tr("模型不执行主观医学判断", "Subjective decisions stay with reviewers"))
    with status_columns[2]:
        insight_card(
            tr("飞书连接", "Feishu connection"),
            tr("已就绪", "Ready") if enabled and configured else tr("待配置", "Optional"),
            tr("不影响本地审核与规则计算", "Local rule review remains available"),
        )

    section_title(tr("协作流程", "Controlled review hand-off"))
    handoff = (
        """
        <div class="ts-handoff">
            <div><span>1</span><b>Send</b><p>System-authored fields sync without overwriting reviewer input.</p></div>
            <div><span>2</span><b>Review</b><p>Medical reviewers confirm, revise or escalate each constraint.</p></div>
            <div><span>3</span><b>Compare</b><p>The app reads review fields and presents a before/after diff.</p></div>
            <div><span>4</span><b>Accept</b><p>Only an explicit confirmation updates the executable rule set.</p></div>
        </div>
        """
        if current_language() == "en"
        else """
        <div class="ts-handoff">
            <div><span>1</span><b>同步标准</b><p>系统字段写入飞书，不覆盖已有审核意见。</p></div>
            <div><span>2</span><b>专业确认</b><p>医学人员标记已确认、需修改或需专家复核。</p></div>
            <div><span>3</span><b>读取差异</b><p>只读取审核字段，先展示修改前后差异。</p></div>
            <div><span>4</span><b>人工采用</b><p>用户确认后，审核结果才进入规则引擎。</p></div>
        </div>
        """
    )
    st.markdown(
        handoff,
        unsafe_allow_html=True,
    )

    template = feishu_review_template(
        st.session_state.criteria,
        st.session_state.source.identifier,
    )
    if enabled and configured:
        render_feishu_review_panel()
    else:
        st.markdown(
            f"<div class='ts-connection-empty'><div class='ts-connection-title'>{escape(tr('飞书尚未连接', 'Feishu is not connected'))}</div>"
            f"<p>{escape(tr('可以先下载审核模板或跳过此步骤；完成自建应用授权后会显示同步和读取按钮。', 'Download the review template or continue offline. Sync controls appear once the Feishu app is authorised.'))}</p></div>",
            unsafe_allow_html=True,
        )
        action_columns = st.columns(2)
        action_columns[0].download_button(
            tr("下载飞书审核模板 CSV", "Download Feishu review template"),
            data=template.to_csv(index=False).encode("utf-8-sig"),
            file_name="trialscope_feishu_review_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        with action_columns[1]:
            with st.expander(tr("查看接入所需配置", "Connection settings"), expanded=False):
                st.code(
                    "ENABLE_FEISHU_SYNC = true\n"
                    "FEISHU_APP_ID = \"cli_xxx\"\n"
                    "FEISHU_APP_SECRET = \"...\"\n"
                    "FEISHU_BITABLE_APP_TOKEN = \"bascxxx\"\n"
                    "FEISHU_CRITERIA_TABLE_ID = \"tblxxx\"\n"
                    "FEISHU_SNAPSHOT_TABLE_ID = \"tblxxx\"\n"
                    "FEISHU_VALIDATION_TABLE_ID = \"tblxxx\"",
                    language="toml",
                )

    section_title(tr("待审核数据预览", "Review payload preview"))
    preview_columns = [
        "标准编号",
        "类型",
        "标准原文",
        "结构化指标",
        "运算符",
        "阈值",
        "执行方式",
        "审核状态",
        "修改意见",
    ]
    preview = template[preview_columns].copy()
    if current_language() == "en":
        preview = preview.rename(columns={
            "标准编号": "Constraint ID",
            "类型": "Type",
            "标准原文": "Source statement",
            "结构化指标": "Structured field",
            "运算符": "Operator",
            "阈值": "Value",
            "执行方式": "Execution",
            "审核状态": "Review status",
            "修改意见": "Reviewer comment",
        })
    st.dataframe(
        preview,
        width="stretch",
        height=390,
        hide_index=True,
        column_config={} if current_language() == "en" else {
            "标准编号": st.column_config.TextColumn(width="small"),
            "类型": st.column_config.TextColumn(width="small"),
            "标准原文": st.column_config.TextColumn(width="large"),
            "结构化指标": st.column_config.TextColumn(width="medium"),
            "修改意见": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(tr("预览不包含候选者明细；飞书审核不是队列评估的强制前置条件。", "The preview contains no candidate-level data. Feishu sign-off is optional for cohort evaluation."))
    st.button(
        tr("继续运行方案约束仿真", "Continue to cohort laboratory"),
        type="primary",
        use_container_width=True,
        on_click=go_to,
        args=("患者预筛",),
    )


def page_screening() -> None:
    page_header(
        tr("候选队列评估", "Cohort evaluation"),
        tr(
            "导入当前研究的候选队列并执行已审核规则，逐条保留数据值、标准依据和判断结果。",
            "Import the active study cohort and execute reviewed constraints while retaining values, protocol evidence and row-level outcomes.",
        ),
        tr("04 · 队列评估", "04 · COHORT EVALUATION"),
    )
    if not require_active_study():
        return
    if not st.session_state.criteria:
        st.warning(tr("请先完成方案约束审核。", "Review the protocol constraints before running the cohort lab."))
        return

    required_fields = sorted(
        {
            field
            for criterion in st.session_state.criteria
            for field in ([criterion.field] if criterion.field else []) + list(criterion.applicability)
        }
    )
    section_title(tr("候选队列数据", "Cohort data"))
    with st.container(border=True, key="cohort_import_panel"):
        info_col, download_col = st.columns([3.4, 1], vertical_alignment="center")
        info_col.markdown(
            f"<div class='ts-cohort-requirements'><b>{escape(tr('CSV 数据要求', 'CSV requirements'))}</b>"
            f"<span>{escape(tr('必须包含唯一 patient_id；缺少的规则字段会被标记为信息不足，不会静默排除。', 'A unique patient_id is required. Missing rule fields are marked unresolved rather than silently excluded.'))}</span>"
            f"<small>{escape(tr('当前规则字段：', 'Current rule fields:'))} {escape(', '.join(required_fields) if required_fields else '—')}</small></div>",
            unsafe_allow_html=True,
        )
        template = pd.DataFrame(columns=["patient_id", *required_fields])
        download_col.download_button(
            tr("下载空白模板", "Download template"),
            data=template.to_csv(index=False).encode("utf-8-sig"),
            file_name="trialscope_cohort_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        uploaded_cohort = st.file_uploader(
            tr("上传候选队列 CSV", "Upload cohort CSV"),
            type=["csv"],
            accept_multiple_files=False,
            key="cohort_upload",
        )
        st.caption(tr(
            "仅可使用经过授权且已去标识化的数据，不要上传姓名、证件号、联系方式等直接身份信息。",
            "Use only authorized, de-identified data. Do not upload names, government identifiers, contact details or other direct identifiers.",
        ))
        if uploaded_cohort and st.button(
            tr("载入候选队列", "Load cohort"),
            key="load_cohort_button",
            type="primary",
        ):
            try:
                cohort = pd.read_csv(uploaded_cohort)
                if cohort.empty:
                    raise ValueError(tr("CSV 中没有数据记录。", "The CSV contains no data rows."))
                if "patient_id" not in cohort.columns:
                    raise ValueError(tr("CSV 必须包含 patient_id 列。", "The CSV must contain a patient_id column."))
                cohort["patient_id"] = cohort["patient_id"].astype(str).str.strip()
                if (cohort["patient_id"] == "").any() or cohort["patient_id"].duplicated().any():
                    raise ValueError(tr("patient_id 不能为空且必须唯一。", "patient_id values must be non-empty and unique."))
                if len(cohort) > 50000:
                    raise ValueError(tr("单次队列最多支持 50,000 条记录。", "A cohort may contain at most 50,000 rows."))
                st.session_state.patients = cohort
                st.session_state.cohort_file_name = uploaded_cohort.name
                st.session_state.results = None
                st.session_state.scenario_comparison = None
                st.session_state.scenario_results = None
                missing_columns = [field for field in required_fields if field not in cohort.columns]
                record_workspace_event(
                    tr("候选队列已导入", "Cohort imported"),
                    f"{uploaded_cohort.name} · {len(cohort)} rows",
                )
                if missing_columns:
                    st.warning(tr(
                        f"已载入 {len(cohort)} 条记录；缺少 {len(missing_columns)} 个规则字段，相关结果将标记为信息不足。",
                        f"Loaded {len(cohort)} rows. {len(missing_columns)} rule fields are absent and will be marked unresolved.",
                    ))
                else:
                    st.success(tr(f"已载入 {len(cohort)} 条候选记录。", f"Loaded {len(cohort)} cohort rows."))
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(str(exc))

    if st.session_state.patients.empty:
        st.info(tr("当前研究尚未导入候选队列。", "No cohort has been imported for this study."))
        return

    st.caption(tr(
        f"当前队列：{st.session_state.cohort_file_name or '会话数据'} · {len(st.session_state.patients)} 条记录",
        f"Active cohort: {st.session_state.cohort_file_name or 'session data'} · {len(st.session_state.patients)} rows",
    ))
    st.markdown(
        f"<div class='ts-action-guide'><strong>{escape(tr('当前任务：运行已审核规则', 'Current task: run reviewed constraints'))}</strong>"
        f"<span>{escape(tr('执行后先检查总体状态，再通过下方记录进入逐条证据审计。', 'Run the evaluation, review overall states, then open row-level evidence below.'))}</span></div>",
        unsafe_allow_html=True,
    )
    if st.button(tr("运行方案约束仿真", "Run cohort simulation"), type="primary"):
        with st.spinner(tr("正在执行确定性规则...", "Executing deterministic constraints...")):
            st.session_state.results = match_dataframe(
                st.session_state.patients, st.session_state.criteria
            )
            st.session_state.scenario_comparison = None
            record_workspace_event(tr("队列评估已运行", "Cohort evaluation run"))
        st.success(tr("基线约束仿真完成。", "Baseline cohort simulation complete."))
    if st.session_state.results is None:
        st.info(tr("队列已就绪，点击上方按钮运行评估。", "The cohort is ready. Run the evaluation to generate results."))
        return
    results = st.session_state.results
    counts = Counter(item.overall_status for item in results)
    columns = st.columns(4)
    for column, status in zip(columns, ["eligible", "ineligible", "missing_data", "needs_review"]):
        column.metric(status_labels()[status], counts.get(status, 0))

    section_title(tr("基线队列状态", "Baseline cohort states"))
    raw_frame = results_dataframe(results)
    display_frame = results_to_display_frame(results)
    status_filter = st.multiselect(
        tr("状态筛选", "Filter states"),
        options=list(status_labels()),
        default=list(status_labels()),
        format_func=lambda item: status_labels()[item],
    )
    selected_labels = {status_labels()[item] for item in status_filter}
    filtered_frame = display_frame[
        display_frame["overall_status"].isin(selected_labels)
    ].reset_index(drop=True)
    st.caption(tr("点击一行可以查看逐条规则证据。", "Select a row to audit every rule outcome."))
    selection_event = st.dataframe(
        filtered_frame,
        width="stretch",
        height=390,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="patient_results_table",
        column_config={
            "patient_id": st.column_config.TextColumn(tr("候选者编号", "Candidate ID"), width="small"),
            "overall_status": st.column_config.TextColumn(tr("队列状态", "Cohort state"), width="small"),
            "summary": st.column_config.TextColumn(tr("判断摘要", "Evidence summary"), width="large"),
            "failed_count": st.column_config.NumberColumn(tr("未满足", "Failed"), width="small"),
            "missing_count": st.column_config.NumberColumn(tr("缺失", "Missing"), width="small"),
            "review_count": st.column_config.NumberColumn(tr("待确认", "Review"), width="small"),
        },
    )
    st.download_button(
        tr("导出完整仿真结果 CSV", "Export simulation results"),
        data=raw_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="trialscope_patient_results.csv",
        mime="text/csv",
    )

    section_title(tr("逐条规则证据", "Row-level evidence audit"))
    selection = getattr(selection_event, "selection", None)
    selected_rows = getattr(selection, "rows", []) if selection is not None else []
    if selected_rows and selected_rows[0] < len(filtered_frame):
        st.session_state.selected_patient_id = str(
            filtered_frame.iloc[selected_rows[0]]["patient_id"]
        )
    available_ids = raw_frame["patient_id"].astype(str).tolist()
    if st.session_state.selected_patient_id not in available_ids:
        st.session_state.selected_patient_id = available_ids[0]
    patient_id = st.session_state.selected_patient_id
    result = next(item for item in results if item.patient_id == patient_id)
    st.markdown(
        f"<div class='ts-next-step'><strong>{escape(status_labels()[result.overall_status])}</strong><br>"
        f"{escape(result_summary(result))}</div>",
        unsafe_allow_html=True,
    )
    patient_row = st.session_state.patients[
        st.session_state.patients["patient_id"].astype(str) == patient_id
    ]
    with st.expander(tr("查看候选记录原始字段", "Inspect candidate source fields"), expanded=False):
        patient_view = patient_row.T.rename(columns={patient_row.index[0]: "值"}).reset_index()
        field_col, value_col = tr("字段", "Field"), tr("值", "Value")
        patient_view.columns = [field_col, value_col]
        patient_view[field_col] = patient_view[field_col].map(field_labels()).fillna(patient_view[field_col])
        patient_view[value_col] = patient_view[value_col].map(display_value)
        st.dataframe(patient_view, width="stretch", hide_index=True)
    evidence_status = {
        "pass": tr("通过", "Pass"),
        "fail": tr("未通过", "Fail"),
        "missing": tr("信息缺失", "Missing"),
        "review": tr("人工确认", "Clinical review"),
        "not_applicable": tr("不适用", "Not applicable"),
    }
    evidence_columns = {
        "id": tr("编号", "Constraint"),
        "result": tr("结果", "Outcome"),
        "field": tr("字段", "Field"),
        "observed": tr("患者值", "Observed"),
        "expected": tr("标准值", "Constraint value"),
        "reason": tr("判定说明", "Evidence note"),
        "source": tr("方案原文", "Source statement"),
    }
    evidence_frame = pd.DataFrame(
        [
            {
                evidence_columns["id"]: item.criterion_id,
                evidence_columns["result"]: evidence_status[item.status],
                evidence_columns["field"]: field_labels().get(item.field, item.field or "—"),
                evidence_columns["observed"]: display_value(item.patient_value),
                evidence_columns["expected"]: display_value(item.expected),
                evidence_columns["reason"]: evidence_message(item),
                evidence_columns["source"]: item.source_text,
            }
            for item in result.evidences
        ]
    )
    st.dataframe(
        evidence_frame,
        width="stretch",
        height=430,
        hide_index=True,
        column_config={
            evidence_columns["id"]: st.column_config.TextColumn(width="small"),
            evidence_columns["result"]: st.column_config.TextColumn(width="small"),
            evidence_columns["field"]: st.column_config.TextColumn(width="medium"),
            evidence_columns["reason"]: st.column_config.TextColumn(width="large"),
            evidence_columns["source"]: st.column_config.TextColumn(width="large"),
        },
    )


def page_analysis() -> None:
    page_header(
        tr("方案情景分析", "Protocol scenario analysis"),
        tr(
            "从候选规模、边际约束、数据负担和人群代表性四个角度比较方案情景，不自动给出放宽或收紧建议。",
            "Compare protocol scenarios across candidate scale, marginal constraint impact, information burden and cohort representation—without auto-recommending a protocol change.",
        ),
        tr("05 · 决策评估", "05 · DECISION VIEW"),
    )
    if not require_active_study():
        return
    if not st.session_state.results:
        st.warning(tr("请先运行方案约束仿真。", "Run the cohort laboratory before opening the decision view."))
        return
    patients = st.session_state.patients
    criteria = st.session_state.criteria
    results = st.session_state.results
    counts = Counter(item.overall_status for item in results)
    blockers = blocker_counts(results, criteria)
    missing = missing_field_counts(results)
    potential = counts.get("eligible", 0) + counts.get("needs_review", 0)
    eligible_rate = counts.get("eligible", 0) / len(results) * 100 if results else 0
    criterion_map = {item.criterion_id: item for item in criteria}
    if not blockers.empty:
        top_blocker_id = str(blockers.iloc[0]["criterion_id"])
        top_blocker_count = int(blockers.iloc[0]["count"])
        top_criterion = criterion_map.get(top_blocker_id)
        top_blocker_name = field_labels().get(
            top_criterion.field if top_criterion else None,
            top_blocker_id,
        )
    else:
        top_blocker_name, top_blocker_count = tr("暂无", "None"), 0

    summary_columns = st.columns(4)
    with summary_columns[0]:
        insight_card(tr("基线候选规模", "Baseline candidate scale"), str(counts.get("eligible", 0)), tr(f"占导入队列 {eligible_rate:.1f}%", f"{eligible_rate:.1f}% of the imported cohort"))
    with summary_columns[1]:
        insight_card(tr("潜在候选规模", "Potential candidate scale"), str(potential), tr("基线候选与待复核合计", "Rule-eligible plus clinical review"))
    with summary_columns[2]:
        insight_card(tr("首要约束", "Leading constraint"), top_blocker_name, tr(f"关联 {top_blocker_count} 次未通过", f"Linked to {top_blocker_count} failed records"))
    with summary_columns[3]:
        insight_card(
            tr("信息与审核负担", "Unresolved workload"),
            str(counts.get('missing_data', 0) + counts.get('needs_review', 0)),
            tr("缺失信息与医学判断合计", "Missing-data and judgement cases"),
        )

    section_title(tr("基线方案画像", "Baseline protocol profile"))
    funnel_tab, representation_tab, completeness_tab = st.tabs(
        [tr("约束路径", "Constraint path"), tr("人群代表性", "Cohort representation"), tr("数据完整性", "Information burden")]
    )
    with funnel_tab:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown(tr("**候选者约束路径**", "**How constraints narrow the cohort**"))
            funnel = build_funnel(patients, criteria)
            if current_language() == "en":
                funnel["stage"] = funnel["stage"].replace({
                    "候选队列": "Imported cohort",
                    "其余可执行标准": "Remaining executable constraints",
                    "规则符合或待复核": "Rule-eligible or clinical review",
                })
            fig = px.funnel(funnel, x="count", y="stage")
            fig.update_traces(marker_color="#2F7A73", textfont_color="#F4F8FB")
            st.plotly_chart(
                style_figure(fig, height=390),
                width="stretch",
                config={"displayModeBar": False},
            )
        with right:
            st.markdown(tr("**主要未通过标准**", "**Most frequent failed constraints**"))
            top_blockers = blockers.head(7).copy()
            if not top_blockers.empty:
                labels = []
                for criterion_id in top_blockers["criterion_id"]:
                    criterion = criterion_map.get(criterion_id)
                    field_label = field_labels().get(
                        criterion.field if criterion else None,
                        criterion_id,
                    )
                    labels.append(f"{criterion_id} · {field_label}")
                top_blockers["label"] = labels
                fig = px.bar(
                    top_blockers.sort_values("count"),
                    x="count",
                    y="label",
                    orientation="h",
                    hover_data={"criterion": True, "label": False},
                )
                fig.update_traces(marker_color="#B75B55")
                st.plotly_chart(
                    style_figure(fig, height=390),
                    width="stretch",
                    config={"displayModeBar": False},
                )
    with representation_tab:
        st.markdown(tr("**导入队列与基线候选人群对比**", "**Imported cohort vs baseline candidate cohort**"))
        representation = representation_table(patients, results)
        if representation.empty:
            st.info(tr("当前队列未提供 age、sex 或 disease_severity 字段，暂不展示人群构成。", "The cohort does not include age, sex or disease_severity, so composition metrics are unavailable."))
        else:
            if current_language() == "en":
                representation["group"] = representation["group"].replace({
                    "候选队列": "Imported cohort",
                    "规则符合或待复核": "Rule-eligible or clinical review",
                })
                representation["metric"] = representation["metric"].replace({
                    "平均年龄": "Mean age",
                    "女性占比": "Female (%)",
                    "65岁及以上占比": "Age 65+ (%)",
                    "重度疾病占比": "Severe disease (%)",
                })
            fig = px.bar(
                representation,
                x="metric",
                y="value",
                color="group",
                barmode="group",
                color_discrete_sequence=["#A7B5C0", "#2F7A73"],
                labels={"metric": tr("指标", "Measure"), "value": tr("数值", "Value"), "group": tr("人群", "Cohort")},
            )
            st.plotly_chart(
                style_figure(fig, height=390),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption(tr("比例指标单位为%，平均年龄单位为岁；人群构成只代表当前导入队列。", "Percentage measures use percentage points; mean age is in years. Composition reflects only the imported cohort."))
    with completeness_tab:
        st.markdown(tr("**影响规则执行的主要缺失字段**", "**Missing fields that block deterministic execution**"))
        top_missing = missing.head(8).copy()
        if top_missing.empty:
            st.success(tr("当前候选队列没有影响判断的字段缺失。", "No missing field currently blocks execution."))
        else:
            top_missing["label"] = top_missing["field"].map(
                lambda item: field_labels().get(item, item)
            )
            fig = px.bar(
                top_missing.sort_values("count"),
                x="count",
                y="label",
                orientation="h",
            )
            fig.update_traces(marker_color="#B78232")
            st.plotly_chart(
                style_figure(fig, height=360),
                width="stretch",
                config={"displayModeBar": False},
            )

    section_title(tr("单项标准边际影响", "Marginal impact by constraint"))
    st.markdown(
        f"<div class='ts-boundary'><b>{escape(tr('反事实解释：', 'Counterfactual lens:'))}</b>"
        f"{escape(tr('系统逐项暂不执行一条可计算标准，重新运行同一队列，观察基线候选规模和人群构成变化。该结果不是删除标准的建议。', 'The same cohort is re-run while one executable constraint is omitted. The result estimates marginal impact; it is not advice to remove a clinical criterion.'))}</div>",
        unsafe_allow_html=True,
    )
    criteria_payload = json.dumps(
        [item.model_dump(mode="json") for item in criteria],
        ensure_ascii=False,
        sort_keys=True,
    )
    with st.spinner(tr("正在计算逐项反事实影响...", "Running one-at-a-time counterfactuals...")):
        marginal = cached_marginal_impact(patients, criteria_payload)
    if not marginal.empty:
        marginal["field_label"] = marginal["field"].map(
            lambda item: field_labels().get(item, item or tr("未结构化", "Unstructured"))
        )
        marginal["label"] = marginal["criterion_id"] + " · " + marginal["field_label"]
        top_marginal = marginal.iloc[0]
        marginal_left, marginal_right = st.columns([1.15, 1], gap="large")
        with marginal_left:
            top_chart = marginal.head(8).sort_values("eligible_change")
            fig = px.bar(
                top_chart,
                x="eligible_change",
                y="label",
                orientation="h",
                labels={"eligible_change": tr("基线候选规模变化", "Change in candidate scale")},
            )
            fig.update_traces(marker_color="#2F7A73")
            st.plotly_chart(style_figure(fig, height=390), width="stretch", config={"displayModeBar": False})
        with marginal_right:
            insight_card(
                tr("边际影响最大", "Largest marginal effect"),
                str(top_marginal["label"]),
                tr(
                    f"暂不执行该标准时，基线候选规模变化 {int(top_marginal['eligible_change']):+d}",
                    f"Candidate scale changes by {int(top_marginal['eligible_change']):+d} when omitted",
                ),
            )
            st.markdown(
                f"<div class='ts-tradeoff-note'><b>{escape(tr('人群构成变化', 'Cohort mix shift'))}</b>"
                f"<span>{escape(tr('平均年龄', 'Mean age'))} {float(top_marginal['mean_age_change']):+.1f}; "
                f"{escape(tr('女性占比', 'Female share'))} {float(top_marginal['female_pct_change']):+.1f} pp; "
                f"{escape(tr('65岁及以上', 'Age 65+'))} {float(top_marginal['older_pct_change']):+.1f} pp</span></div>",
                unsafe_allow_html=True,
            )
        with st.expander(tr("查看全部标准的边际影响", "Inspect all marginal-impact estimates"), expanded=False):
            marginal_display = marginal[[
                "criterion_id", "field_label", "eligible_baseline", "eligible_without",
                "eligible_change", "missing_change", "mean_age_change", "female_pct_change",
            ]].rename(columns={
                "criterion_id": tr("编号", "ID"),
                "field_label": tr("指标", "Field"),
                "eligible_baseline": tr("基线候选", "Baseline"),
                "eligible_without": tr("暂不执行时", "When omitted"),
                "eligible_change": tr("规模变化", "Scale change"),
                "missing_change": tr("信息不足变化", "Missing-data change"),
                "mean_age_change": tr("平均年龄变化", "Mean-age change"),
                "female_pct_change": tr("女性占比变化(pp)", "Female-share change (pp)"),
            })
            st.dataframe(marginal_display, width="stretch", hide_index=True)

    section_title(tr("多目标情景权衡", "Multi-objective scenario trade-off"))
    st.markdown(
        f"<div class='ts-boundary'><b>{escape(tr('安全提示：', 'Decision boundary:'))}</b>"
        f"{escape(tr('参数调整只展示当前导入队列的变化，不构成临床试验方案修改建议。', 'Scenario controls expose trade-offs in the imported cohort; they do not recommend a protocol amendment.'))}</div>",
        unsafe_allow_html=True,
    )
    adjustable = [
        item
        for item in criteria
        if item.execution_status == "automated"
        and item.field
        and item.operator in {"lt", "lte", "gt", "gte", "within_days"}
        and isinstance(item.value, (int, float))
        and not isinstance(item.value, bool)
    ]
    if not adjustable:
        st.info(tr("当前研究没有可进行数值情景调整的标准。", "This study has no numeric constraints available for scenario adjustment."))
    else:
        with st.expander(tr("调整一项方案参数", "Adjust one protocol parameter"), expanded=False):
            st.caption(tr(
                "从当前研究的已审核数值标准中选择一项，比较阈值变化前后的队列结果。",
                "Select one reviewed numeric constraint from the active study and compare cohort outcomes before and after the threshold change.",
            ))
            selected_id = st.selectbox(
                tr("选择标准", "Select constraint"),
                options=[item.criterion_id for item in adjustable],
                format_func=lambda criterion_id: next(
                    f"{item.criterion_id} · {field_labels().get(item.field, item.field)} · {item.operator} {item.value}{(' ' + item.unit) if item.unit else ''}"
                    for item in adjustable
                    if item.criterion_id == criterion_id
                ),
                key="scenario_criterion_id",
            )
            selected = next(item for item in adjustable if item.criterion_id == selected_id)
            baseline_value = float(selected.value)
            step = max(abs(baseline_value) * 0.05, 0.1)
            scenario_value = st.number_input(
                tr("情景阈值", "Scenario threshold"),
                value=baseline_value,
                step=step,
                key=f"scenario_value_{selected_id}",
                help=selected.source_text,
            )
            st.caption(
                tr(
                    f"基线：{selected.operator} {selected.value}{(' ' + selected.unit) if selected.unit else ''}",
                    f"Baseline: {selected.operator} {selected.value}{(' ' + selected.unit) if selected.unit else ''}",
                )
            )

            if st.button(tr("运行情景权衡", "Run scenario trade-off"), type="primary"):
                scenario_parameters = {
                    "criterion_id": selected.criterion_id,
                    "field": selected.field,
                    "operator": selected.operator,
                    "baseline": selected.value,
                    "scenario": scenario_value,
                    "unit": selected.unit or "",
                }
                scenario_criteria = apply_scenario(criteria, {selected.criterion_id: scenario_value})
                comparison, _, scenario_results = scenario_comparison(
                    patients, criteria, scenario_criteria
                )
                st.session_state.scenario_comparison = comparison
                st.session_state.scenario_results = scenario_results
                st.session_state.scenario_parameters = scenario_parameters
                st.session_state.scenario_snapshot_key = (
                    f"{st.session_state.source.identifier}:scenario:"
                    f"{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d%H%M%S')}"
                )
                record_workspace_event(tr("情景分析已运行", "Scenario analysis run"), selected.criterion_id)

    if st.session_state.scenario_comparison is not None:
        comparison = st.session_state.scenario_comparison
        eligible_row = comparison[comparison["status"] == "eligible"].iloc[0]
        tradeoff = scenario_tradeoff(
            patients,
            results,
            st.session_state.scenario_results,
        )
        tradeoff_index = tradeoff.set_index("metric")
        scenario_metrics = st.columns(3)
        scale = tradeoff_index.loc["eligible_count"]
        representation_gap = tradeoff_index.loc["representation_gap"]
        missing_burden = tradeoff_index.loc["missing_data_count"]
        representation_available = any(
            field in patients.columns for field in ["age", "sex", "disease_severity"]
        )
        scenario_metrics[0].metric(
            tr("候选规模", "Candidate scale"),
            int(scale["scenario"]),
            delta=int(scale["change"]),
            help=tr("严格通过全部可执行规则的候选记录", "Candidate records that pass every executable constraint"),
        )
        scenario_metrics[1].metric(
            tr("代表性差距", "Representation gap"),
            f"{float(representation_gap['scenario']):.1f} pp" if representation_available else "—",
            delta=f"{float(representation_gap['change']):+.1f} pp" if representation_available else None,
            delta_color="inverse",
            help=tr("根据当前队列中可用的年龄、性别和疾病程度字段计算；越低越接近完整队列", "Calculated from available age, sex and disease-severity fields; lower is closer to the full cohort"),
        )
        scenario_metrics[2].metric(
            tr("信息不足", "Data-unresolved"),
            int(missing_burden["scenario"]),
            delta=int(missing_burden["change"]),
            delta_color="inverse",
            help=tr("因必要字段缺失而无法完成规则执行的记录", "Records that cannot be resolved because a required field is missing"),
        )
        tradeoff_labels = {
            "eligible_count": tr("候选规模", "Candidate scale"),
            "potential_count": tr("候选及待复核", "Candidate + clinical review"),
            "representation_gap": tr("代表性差距(pp)", "Representation gap (pp)"),
            "missing_data_count": tr("信息不足", "Data-unresolved"),
            "review_count": tr("医学复核", "Clinical review"),
        }
        tradeoff_display = tradeoff.copy()
        tradeoff_display["metric"] = tradeoff_display["metric"].map(tradeoff_labels)
        tradeoff_display = tradeoff_display.rename(columns={
            "metric": tr("决策维度", "Decision dimension"),
            "baseline": tr("基线", "Baseline"),
            "scenario": tr("情景", "Scenario"),
            "change": tr("变化", "Change"),
        })
        st.dataframe(tradeoff_display, width="stretch", hide_index=True)
        comparison = comparison.copy()
        comparison["label"] = comparison["status"].map(status_labels())
        baseline_label = tr("基线", "Baseline")
        scenario_label = tr("情景", "Scenario")
        chart_comparison = comparison.rename(
            columns={"baseline": baseline_label, "scenario": scenario_label}
        )
        fig = px.bar(
            chart_comparison,
            x="label",
            y=[baseline_label, scenario_label],
            barmode="group",
            labels={"label": tr("结果", "Cohort state"), "value": tr("人数", "Records"), "variable": tr("方案", "Run")},
            color_discrete_sequence=["#A7B5C0", "#2F7A73"],
        )
        st.plotly_chart(
            style_figure(fig, height=350),
            width="stretch",
            config={"displayModeBar": False},
        )

        settings = feishu_settings()
        if (
            bool_setting("ENABLE_FEISHU_SYNC", False)
            and settings.configured
            and settings.snapshot_table_id
        ):
            with st.expander(tr("保存本次情景快照", "Save this scenario snapshot"), expanded=False):
                st.caption(tr("只保存汇总人数、调整参数和审核意见，不同步候选者明细。", "Only aggregate counts, parameters and review notes are synced—never candidate-level rows."))
                scenario_name = st.selectbox(
                    tr("情景名称", "Scenario label"),
                    ["自定义", "适度放宽", "适度收紧", "基线"],
                    format_func=lambda value: {
                        "自定义": tr("自定义", "Custom"),
                        "适度放宽": tr("适度放宽", "Moderate expansion"),
                        "适度收紧": tr("适度收紧", "Moderate restriction"),
                        "基线": tr("基线", "Baseline"),
                    }[value],
                    key="feishu_scenario_name",
                )
                scenario_note = st.text_area(
                    tr("医学或统计审核意见（可稍后在飞书补充）", "Medical or statistical review note (can be completed in Feishu)"),
                    key="feishu_scenario_note",
                )
                if st.button(
                    tr("保存到飞书方案快照", "Save snapshot to Feishu"),
                    use_container_width=True,
                    key="save_feishu_scenario",
                ):
                    scenario_counts = Counter(
                        item.overall_status for item in st.session_state.scenario_results
                    )
                    snapshot_fields = {
                        "快照键": st.session_state.scenario_snapshot_key,
                        "试验编号": st.session_state.source.identifier,
                        "分析版本": 1,
                        "情景名称": scenario_name,
                        "候选记录数": len(patients),
                        "模拟符合人数": scenario_counts.get("eligible", 0),
                        "不符合人数": scenario_counts.get("ineligible", 0),
                        "信息不足人数": scenario_counts.get("missing_data", 0),
                        "人工复核人数": scenario_counts.get("needs_review", 0),
                        "主要排除原因": f"{top_blocker_name}（基线影响 {top_blocker_count} 人）",
                        "调整参数": json.dumps(
                            st.session_state.scenario_parameters,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "人数变化": int(eligible_row["change"]),
                        "代表性变化": "当前队列结果；人群构成变化需结合页面图表复核",
                        "医学统计意见": scenario_note.strip(),
                        "应用链接": feishu_url_value(
                            "https://trialscopeai.streamlit.app/",
                            "TrialScopeAI",
                        ),
                    }
                    try:
                        with st.spinner(tr("正在保存方案快照...", "Saving scenario snapshot...")):
                            with FeishuClient(settings) as client:
                                action = client.upsert_record(
                                    settings.snapshot_table_id,
                                    "快照键",
                                    snapshot_fields,
                                )
                        st.success(tr(
                            f"已在飞书方案快照表中{'新建' if action == 'created' else '更新'}本次汇总。",
                            f"Scenario summary {'created' if action == 'created' else 'updated'} in Feishu.",
                        ))
                    except FeishuError as exc:
                        st.error(str(exc))

    report = build_markdown_report(
        st.session_state.source.title,
        patients,
        results,
        criteria,
        language=current_language(),
    )
    st.download_button(
        tr("导出决策摘要 Markdown", "Export decision brief"),
        data=report.encode("utf-8"),
        file_name="trialscope_recruitment_report.md",
        mime="text/markdown",
    )


def page_validation() -> None:
    page_header(
        tr("审计与数据治理", "Audit and data governance"),
        tr(
            "集中检查当前研究的标准来源、执行方式、人工复核项与数据处理边界。",
            "Review constraint provenance, execution mode, clinical review items and data-handling boundaries for the active study.",
        ),
        tr("研究治理", "STUDY GOVERNANCE"),
    )
    if not require_active_study():
        return
    criteria = st.session_state.criteria
    traceable = sum(bool(item.source_text and item.source_reference) for item in criteria)
    automated = sum(item.execution_status == "automated" for item in criteria)
    human_review = sum(item.execution_status == "human_review" for item in criteria)
    columns = st.columns(4)
    metrics = [
        (tr("当前标准", "Current constraints"), str(len(criteria)), tr("当前研究工作区", "Active study workspace")),
        (tr("规则执行", "Rule-executable"), str(automated), tr("由确定性规则引擎运行", "Executed by the deterministic rule engine")),
        (tr("人工复核", "Clinical review"), str(human_review), tr("保留医学或运营判断", "Reserved for medical or operational judgement")),
        (tr("来源追溯", "Source traceability"), f"{traceable}/{len(criteria)}", tr("保留标准原文与来源", "Source statement and reference retained")),
    ]
    for column, (label, value, note) in zip(columns, metrics):
        with column:
            insight_card(label, value, note)

    section_title(tr("标准审计清单", "Constraint audit register"))
    if criteria:
        audit_frame = pd.DataFrame(
            [
                {
                    tr("标准编号", "Constraint ID"): item.criterion_id,
                    tr("类型", "Type"): kind_labels().get(item.kind, item.kind),
                    tr("执行方式", "Execution"): execution_labels().get(item.execution_status, item.execution_status),
                    tr("来源", "Source reference"): item.source_reference,
                    tr("置信度", "Confidence"): round(item.confidence, 2),
                    tr("备注", "Note"): item.note,
                }
                for item in criteria
            ]
        )
        st.dataframe(audit_frame, width="stretch", hide_index=True, height=360)
    else:
        st.info(tr("当前研究尚未生成结构化标准。", "The active study does not yet have structured constraints."))

    section_title(tr("数据处理原则", "Data-handling principles"))
    boundary_columns = st.columns(3)
    governance = [
        (
            tr("方案文件", "Protocol files"),
            tr("上传文件仅在当前会话内存中处理，不保存 PDF 原文件。", "Uploaded files are processed in session memory; original PDFs are not retained."),
        ),
        (
            tr("候选数据", "Candidate data"),
            tr("当前系统不接收真实患者身份信息，结果不用于自动入组。", "The current system does not accept patient identifiers and does not automate enrolment."),
        ),
        (
            tr("人工责任", "Human accountability"),
            tr("主观标准和情景变化必须由医学、统计或伦理人员复核。", "Judgement-only criteria and scenario changes require medical, statistical or ethics review."),
        ),
    ]
    for column, (title, note) in zip(boundary_columns, governance):
        column.markdown(
            f"<div class='ts-boundary-card'><b>{escape(title)}</b><p>{escape(note)}</p></div>",
            unsafe_allow_html=True,
        )


def top_navigation() -> str:
    page = st.session_state.navigation
    source: TrialSource = st.session_state.source
    active_study = has_active_study()
    study_label = (
        f"{source.identifier} · {source.title}"
        if active_study
        else tr("尚未选择研究 · 请先导入方案", "NO ACTIVE STUDY · IMPORT A PROTOCOL")
    )
    with st.container(key="application_header"):
        brand_col, study_col, language_col = st.columns(
            [2.5, 5.2, 1.25], vertical_alignment="center"
        )
        brand_col.markdown(
            "<div class='ts-top-brand'><span>TS</span><div><b>TrialScope</b>"
            f"<small>{escape(tr('临床招募可行性工作台', 'Clinical Feasibility Workspace'))}</small></div></div>",
            unsafe_allow_html=True,
        )
        study_col.markdown(
            f"<div class='ts-top-study'><span>{escape(tr('当前研究', 'ACTIVE STUDY'))}</span>"
            f"<b>{escape(study_label)}</b></div>",
            unsafe_allow_html=True,
        )
        with language_col:
            with st.container(key="language_switch"):
                language_columns = st.columns(2, gap="small")
                for index, (language, label) in enumerate([("zh", "中文"), ("en", "EN")]):
                    state = "active" if current_language() == language else "idle"
                    with language_columns[index]:
                        with st.container(key=f"language_option_{language}_{state}"):
                            st.button(
                                label,
                                key=f"language_button_{language}",
                                use_container_width=True,
                                on_click=set_language,
                                args=(language,),
                            )

    nav_items = [
        ("项目说明", tr("首页", "Home")),
        ("研究工作台", tr("工作台", "Workspace")),
        ("试验 / PDF 导入", tr("方案导入", "Protocol")),
        ("标准解析", tr("标准审核", "Rule review")),
        ("协作审核", tr("协作中心", "Collaboration")),
        ("患者预筛", tr("队列评估", "Cohort evaluation")),
        ("招募分析", tr("情景分析", "Scenario analysis")),
        ("历史记录", tr("历史记录", "History")),
        ("验证证据", tr("审计治理", "Governance")),
    ]
    with st.container(key="top_navigation"):
        columns = st.columns([0.62, 0.76, 0.88, 0.9, 1.0, 1.0, 1.0, 0.82, 0.78], gap="small")
        for index, (page_name, label) in enumerate(nav_items):
            state = "active" if page_name == page else "idle"
            with columns[index]:
                with st.container(key=f"top_nav_{index}_{state}"):
                    st.button(
                        label,
                        key=f"top_nav_button_{index}",
                        use_container_width=True,
                        on_click=go_to,
                        args=(page_name,),
                    )
    return page


init_state()
browser_storage = hydrate_browser_history()
current_page = top_navigation()
scroll_to_top_if_requested()

if current_page == "项目说明":
    page_home()
elif current_page == "研究工作台":
    page_workspace()
elif current_page == "试验 / PDF 导入":
    page_import()
elif current_page == "标准解析":
    page_parse()
elif current_page == "协作审核":
    page_collaboration()
elif current_page == "患者预筛":
    page_screening()
elif current_page == "招募分析":
    page_analysis()
elif current_page == "历史记录":
    page_history()
else:
    page_validation()

flush_browser_history(browser_storage)
