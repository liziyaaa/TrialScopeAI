"""TrialScope clinical recruitment feasibility workspace."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, time as datetime_time
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st

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
    DATA_DIR,
    DEEPSEEK_DEFAULT_MODEL,
    MAX_LIVE_CALLS_PER_SESSION,
    OUTPUT_DIR,
)
from src.llm_parser import (
    LLMParseError,
    load_cached_demo_criteria,
    parse_with_deepseek,
    split_for_llm,
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


@st.cache_data
def load_demo_source() -> TrialSource:
    payload = json.loads((DATA_DIR / "golden4_trial.json").read_text(encoding="utf-8"))
    return TrialSource.model_validate(payload)


@st.cache_data
def load_patients() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "synthetic_patients.csv")


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
    demo_source = load_demo_source()
    defaults = {
        "language": "zh",
        "navigation": "项目说明",
        "import_method": "预置研究",
        "selected_patient_id": None,
        "source": demo_source,
        "criteria_text": demo_source.criteria_text,
        "criteria": load_cached_demo_criteria(),
        "patients": load_patients(),
        "results": None,
        "live_calls": 0,
        "scenario_comparison": None,
        "scenario_results": None,
        "scenario_parameters": {},
        "scenario_snapshot_key": "",
        "feishu_pending_criteria": None,
        "feishu_review_diffs": [],
        "feishu_sync_note": "",
        "last_parse_note": "已载入 GOLDEN-4 结构化标准。",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("import_method") == "内置演示":
        st.session_state.import_method = "预置研究"


def set_source(source: TrialSource, criteria_text: str | None = None) -> None:
    text = criteria_text or source.criteria_text
    st.session_state.source = source
    st.session_state.criteria_text = text
    st.session_state.criteria_editor = text
    st.session_state.results = None
    st.session_state.scenario_comparison = None
    st.session_state.scenario_results = None
    st.session_state.scenario_parameters = {}
    st.session_state.scenario_snapshot_key = ""
    if source.identifier == "NCT02347774":
        st.session_state.criteria = load_cached_demo_criteria()
        st.session_state.last_parse_note = tr(
            "该案例可直接使用审核后的缓存标准，也可重新调用 DeepSeek。",
            "The reviewed reference rules are ready; live semantic extraction remains optional.",
        )
    else:
        st.session_state.criteria = []
        st.session_state.last_parse_note = tr(
            "请进入“标准解析”步骤生成结构化标准。",
            "Continue to rule review to generate structured constraints.",
        )


def current_language() -> str:
    return str(st.session_state.get("language", "zh"))


def tr(zh: str, en: str) -> str:
    return en if current_language() == "en" else zh


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
    return "The constraint does not apply to this synthetic candidate."


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
            st.success(tr("飞书审核结果已应用到当前规则。", "Reviewed changes are now applied to the current constraint set."))
            st.rerun()
        if cancel.button(tr("暂不采用", "Keep current rules"), use_container_width=True):
            st.session_state.feishu_pending_criteria = None
            st.session_state.feishu_review_diffs = []
            st.rerun()

    if st.session_state.feishu_sync_note:
        st.caption(st.session_state.feishu_sync_note)


def ensure_results() -> None:
    if st.session_state.results is None and st.session_state.criteria:
        st.session_state.results = match_dataframe(
            st.session_state.patients, st.session_state.criteria
        )


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
    st.markdown(
        f"""
        <div class="ts-study-card">
            <div>
                <div class="ts-study-id">{identifier}</div>
                <div class="ts-study-title">{title}</div>
                <div class="ts-study-meta">{escape(tr('来源', 'Source'))}: {reference}</div>
            </div>
            <div class="ts-study-tags">
                <span class="ts-tag">{escape(tr('Ⅲ期', 'Phase III'))}</span>
                <span class="ts-tag">COPD</span>
                <span class="ts-status ok">{escape(tr('公开方案', 'Public protocol'))}</span>
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


SCENARIO_PRESETS = {
    "baseline": {
        "scenario_age_min": 40,
        "scenario_pack_years": 10.0,
        "scenario_fev1_pct": 80.0,
        "scenario_fev1_liters": 0.7,
        "scenario_ratio": 0.70,
        "scenario_oxygen": 12.0,
        "scenario_exacerbation_days": 42,
        "scenario_infection_days": 42,
    },
    "expansion": {
        "scenario_age_min": 35,
        "scenario_pack_years": 5.0,
        "scenario_fev1_pct": 85.0,
        "scenario_fev1_liters": 0.6,
        "scenario_ratio": 0.72,
        "scenario_oxygen": 16.0,
        "scenario_exacerbation_days": 28,
        "scenario_infection_days": 28,
    },
    "focused": {
        "scenario_age_min": 45,
        "scenario_pack_years": 20.0,
        "scenario_fev1_pct": 70.0,
        "scenario_fev1_liters": 0.8,
        "scenario_ratio": 0.65,
        "scenario_oxygen": 8.0,
        "scenario_exacerbation_days": 56,
        "scenario_infection_days": 56,
    },
}


def load_scenario_preset(preset: str) -> None:
    st.session_state.update(SCENARIO_PRESETS[preset])
    st.session_state.scenario_comparison = None
    st.session_state.scenario_results = None
    st.session_state.scenario_parameters = {}
    st.session_state.scenario_snapshot_key = ""


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
    ensure_results()
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
        (tr("评估队列", "Evaluated cohort"), str(len(patients)), tr("合成记录", "Synthetic records")),
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
    funnel["stage"] = funnel["stage"].replace({"模拟符合或待复核": "规则符合或待复核"})
    if current_language() == "en":
        funnel["stage"] = funnel["stage"].replace({
            "候选队列": "Evaluated cohort",
            "年龄与诊断": "Age and diagnosis",
            "吸烟史": "Smoking exposure",
            "肺功能": "Pulmonary function",
            "近期事件": "Recent events",
            "其他可执行排除项": "Other exclusions",
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
                    "eligible": "#20744A",
                    "ineligible": "#71808C",
                    "missing_data": "#D28B25",
                    "needs_review": "#2878B8",
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
            funnel_fig.update_traces(marker_color="#2A6F97", textfont_color="#FFFFFF")
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
                blocker_fig.update_traces(marker_color="#2A6F97", textposition="outside", cliponaxis=False)
                st.plotly_chart(style_figure(blocker_fig, height=350), width="stretch", config={"displayModeBar": False})
    with chart_right:
        with st.container(border=True, key="dashboard_representation_panel"):
            st.markdown(f"<div class='ts-panel-title'>{escape(tr('人群构成对照', 'Cohort composition'))}</div>", unsafe_allow_html=True)
            representation = representation_table(patients, results)
            representation = representation[representation["metric"] != "平均年龄"].copy()
            if current_language() == "en":
                representation["group"] = representation["group"].replace({
                    "候选队列": "Evaluated cohort",
                    "模拟符合或待复核": "Eligible or review",
                })
                representation["metric"] = representation["metric"].replace({
                    "女性占比": "Female",
                    "65岁及以上占比": "Age 65+",
                    "重度COPD占比": "Severe COPD",
                })
            composition_fig = px.bar(
                representation,
                x="metric",
                y="value",
                color="group",
                barmode="group",
                text_auto=".1f",
                color_discrete_sequence=["#A7B4BE", "#2D7773"],
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
        f"<div class='ts-system-footnote'>{escape(tr('数据范围：公开试验方案与合成候选队列。页面结果用于方案评估和协作审核，不用于诊断或自动入组。', 'Data scope: public protocol plus a synthetic cohort. Results support protocol assessment and review; they are not used for diagnosis or automatic enrolment.'))}</div>",
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
        ("预置研究", "GOLDEN-4", tr("预置公开研究，可直接建立工作区", "A prepared public study for immediate use")),
        ("NCT 编号", tr("公开试验", "NCT record"), tr("从 ClinicalTrials.gov 获取标准", "Fetch public eligibility text")),
        ("粘贴文本", tr("标准原文", "Paste text"), tr("适合已有 Word 或网页文本", "For copied protocol sections")),
        ("上传 PDF", tr("研究方案", "Searchable PDF"), tr("首版不处理扫描图像", "Text PDFs only; no OCR guessing")),
    ]
    source_columns = st.columns(4)
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
        "预置研究": tr("预置公开研究", "Prepared public study"),
        "NCT 编号": "ClinicalTrials.gov",
        "粘贴文本": tr("粘贴标准原文", "Pasted eligibility text"),
        "上传 PDF": tr("文字型 PDF", "Searchable PDF"),
    }.get(method, method)
    st.markdown(
        f"<div class='ts-selection-label'>{escape(tr('当前选择', 'Active source'))}: {escape(method_display)}</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True, key="source_input_panel"):
        if method == "预置研究":
            left, right = st.columns([2, 1])
            with left:
                st.markdown("**GOLDEN-4（NCT02347774）**")
                st.caption(tr(
                    "COPD Ⅲ期公开试验，覆盖年龄、吸烟史、肺功能、用药和时间窗。",
                    "A public Phase III COPD study with numeric, medication and time-window constraints.",
                ))
                if st.button(tr("载入 GOLDEN-4", "Load GOLDEN-4"), type="primary", use_container_width=True):
                    set_source(load_demo_source())
                    st.success(tr("研究已载入，请在下方核对原文。", "Study loaded. Review the source text below."))
            with right:
                pdf_path = OUTPUT_DIR / "pdf" / "golden4_demo_protocol.pdf"
                st.download_button(
                    tr("下载研究方案 PDF", "Download protocol PDF"),
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        elif method == "NCT 编号":
            st.markdown(tr("**输入公开试验编号**", "**Enter a public trial identifier**"))
            nct_id = st.text_input(
                tr("ClinicalTrials.gov NCT 编号", "ClinicalTrials.gov NCT ID"),
                value="NCT02347774",
                help="格式示例：NCT02347774",
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

    section_title(tr("语义提取来源", "Semantic extraction controls"))
    left, right = st.columns(2)
    with left:
        if st.button(
            tr("重新生成待审标准", "Regenerate draft constraints"),
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
                    st.success(st.session_state.last_parse_note)
                except LLMParseError as exc:
                    st.error(str(exc))
    with right:
        if st.button(tr("恢复 GOLDEN-4 审核基准", "Restore reviewed reference rules"), use_container_width=True):
            st.session_state.criteria = load_cached_demo_criteria()
            st.session_state.results = None
            st.session_state.last_parse_note = tr("已载入 27 条审核标准。", "Loaded 27 reviewed constraints.")
            st.success(st.session_state.last_parse_note)

    if not api_key:
        st.info(tr("当前未配置自动解析服务；已审核的 GOLDEN-4 标准和规则分析功能仍可使用。", "Live extraction is not configured. The reviewed reference rules and all decision analyses remain available."))
    elif not live_enabled:
        st.warning(tr("自动解析服务当前已关闭。", "Live semantic extraction is disabled."))
    elif required_chunks > remaining:
        st.warning(tr(f"当前文本需要 {required_chunks} 次调用，已超过本会话剩余额度 {remaining} 次。", f"This source needs {required_chunks} calls; only {remaining} remain in the session."))
    with st.expander(tr("查看解析记录", "Extraction log"), expanded=False):
        st.caption(st.session_state.last_parse_note)
        st.caption(tr(f"解析模型：{model} · 本会话剩余额度：{remaining} 次", f"Model: {model} · Session calls remaining: {remaining}"))

    if not st.session_state.criteria:
        st.warning(tr("还没有结构化标准。请配置 API 后解析，或载入 GOLDEN-4 缓存。", "No structured constraints are available. Configure live extraction or restore the GOLDEN-4 reference rules."))
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
        tr("候选人群约束仿真", "Cohort laboratory"),
        tr(
            "在固定合成队列中执行已审核规则，为后续边际影响和情景权衡提供可复现的计算底座。",
            "Execute signed-off constraints against a fixed synthetic cohort. This is the computational layer behind marginal-impact and scenario analysis—not a patient enrolment tool.",
        ),
        tr("04 · 约束仿真", "04 · COHORT LAB"),
    )
    if not st.session_state.criteria:
        st.warning(tr("请先完成方案约束审核。", "Review the protocol constraints before running the cohort lab."))
        return
    st.markdown(
        f"<div class='ts-action-guide'><strong>{escape(tr('当前任务：建立可复现的基线队列', 'Current task: establish a reproducible baseline'))}</strong>"
        f"<span>{escape(tr('先运行已确认规则，再检查总体状态；逐条证据放在下方作为审计入口。', 'Run the signed-off rules first. Row-level evidence remains available below for audit and debugging.'))}</span></div>",
        unsafe_allow_html=True,
    )
    if st.button(tr("运行方案约束仿真", "Run cohort simulation"), type="primary"):
        with st.spinner(tr("正在执行确定性规则...", "Executing deterministic constraints...")):
            st.session_state.results = match_dataframe(
                st.session_state.patients, st.session_state.criteria
            )
            st.session_state.scenario_comparison = None
        st.success(tr("基线约束仿真完成。", "Baseline cohort simulation complete."))
    ensure_results()
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
    st.caption(tr("点击一行可以查看逐条规则证据；这些合成记录只用于验证计算逻辑。", "Select a row to audit every rule outcome. These synthetic records exist only to validate the decision logic."))
    selection_event = st.dataframe(
        filtered_frame,
        width="stretch",
        height=390,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="patient_results_table",
        column_config={
            "patient_id": st.column_config.TextColumn(tr("候选者编号", "Synthetic ID"), width="small"),
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
    with st.expander(tr("查看合成候选者原始字段", "Inspect synthetic source fields"), expanded=False):
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
    ensure_results()
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
        insight_card(tr("基线候选规模", "Baseline candidate scale"), str(counts.get("eligible", 0)), tr(f"占合成队列 {eligible_rate:.1f}%", f"{eligible_rate:.1f}% of the synthetic cohort"))
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
                    "候选队列": "Synthetic cohort",
                    "年龄与诊断": "Age and diagnosis",
                    "吸烟史": "Smoking exposure",
                    "肺功能": "Pulmonary function",
                    "近期事件": "Recent events",
                    "其他可执行排除项": "Other executable exclusions",
                    "模拟符合或待复核": "Rule-eligible or clinical review",
                })
            fig = px.funnel(funnel, x="count", y="stage")
            fig.update_traces(marker_color="#2D7773", textfont_color="#FFFFFF")
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
                fig.update_traces(marker_color="#B84A45")
                st.plotly_chart(
                    style_figure(fig, height=390),
                    width="stretch",
                    config={"displayModeBar": False},
                )
    with representation_tab:
        st.markdown(tr("**候选队列与基线候选人群对比**", "**Full synthetic cohort vs baseline candidate cohort**"))
        representation = representation_table(patients, results)
        if current_language() == "en":
            representation["group"] = representation["group"].replace({
                "候选队列": "Synthetic cohort",
                "模拟符合或待复核": "Rule-eligible or clinical review",
            })
            representation["metric"] = representation["metric"].replace({
                "平均年龄": "Mean age",
                "女性占比": "Female (%)",
                "65岁及以上占比": "Age 65+ (%)",
                "重度COPD占比": "Severe COPD (%)",
            })
        fig = px.bar(
            representation,
            x="metric",
            y="value",
            color="group",
            barmode="group",
            color_discrete_sequence=["#9BAAB4", "#2D7773"],
            labels={"metric": tr("指标", "Measure"), "value": tr("数值", "Value"), "group": tr("人群", "Cohort")},
        )
        st.plotly_chart(
            style_figure(fig, height=390),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(tr("比例指标单位为%，平均年龄单位为岁。合成数据不代表真实疾病人群分布。", "Percentage measures use percentage points; mean age is in years. Synthetic data do not estimate a real disease population."))
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
            fig.update_traces(marker_color="#A66B16")
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
            fig.update_traces(marker_color="#2D7773")
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
        f"{escape(tr('参数调整只展示合成队列变化，不构成临床试验方案修改建议。', 'Scenario controls expose trade-offs in the synthetic cohort; they do not recommend a protocol amendment.'))}</div>",
        unsafe_allow_html=True,
    )
    with st.expander(tr("调整方案参数", "Configure a protocol scenario"), expanded=False):
        st.caption(tr("可先载入一个对照示例，再逐项调整。预设仅用于展示权衡，不代表推荐方向。", "Start from an illustrative preset, then edit individual values. Presets demonstrate trade-offs and do not imply a preferred protocol."))
        preset_columns = st.columns(3)
        preset_columns[0].button(
            tr("恢复原方案基线", "Restore protocol baseline"),
            on_click=load_scenario_preset,
            args=("baseline",),
            use_container_width=True,
            key="scenario_preset_baseline",
        )
        preset_columns[1].button(
            tr("候选池扩展示例", "Illustrative expansion"),
            on_click=load_scenario_preset,
            args=("expansion",),
            use_container_width=True,
            key="scenario_preset_expansion",
        )
        preset_columns[2].button(
            tr("聚焦人群示例", "Illustrative focus"),
            on_click=load_scenario_preset,
            args=("focused",),
            use_container_width=True,
            key="scenario_preset_focused",
        )
        row1 = st.columns(4)
        age_min = row1[0].number_input(tr("最低年龄", "Minimum age"), 18, 90, 40, key="scenario_age_min")
        pack_years = row1[1].number_input(tr("最低吸烟包年", "Minimum pack-years"), 0.0, 100.0, 10.0, 1.0, key="scenario_pack_years")
        fev1_pct = row1[2].number_input(tr("FEV1 %预计值上限", "Maximum FEV1 % predicted"), 20.0, 120.0, 80.0, 1.0, key="scenario_fev1_pct")
        fev1_liters = row1[3].number_input(tr("FEV1 容量下限（L）", "Minimum FEV1 volume (L)"), 0.1, 5.0, 0.7, 0.1, key="scenario_fev1_liters")
        row2 = st.columns(4)
        ratio = row2[0].number_input(tr("FEV1/FVC 上限", "Maximum FEV1/FVC"), 0.3, 1.0, 0.7, 0.01, key="scenario_ratio")
        oxygen = row2[1].number_input(tr("每日氧疗上限（小时）", "Maximum oxygen hours/day"), 0.0, 24.0, 12.0, 1.0, key="scenario_oxygen")
        exacerbation_days = row2[2].number_input(tr("急性加重窗口（天）", "Exacerbation window (days)"), 1, 365, 42, key="scenario_exacerbation_days")
        infection_days = row2[3].number_input(tr("感染窗口（天）", "Infection window (days)"), 1, 365, 42, key="scenario_infection_days")

        if st.button(tr("运行情景权衡", "Run scenario trade-off"), type="primary"):
            scenario_parameters = {
                "最低年龄": age_min,
                "最低吸烟包年": pack_years,
                "FEV1预计值上限": fev1_pct,
                "FEV1容量下限_L": fev1_liters,
                "FEV1_FVC上限": ratio,
                "每日氧疗上限_小时": oxygen,
                "急性加重窗口_天": exacerbation_days,
                "感染窗口_天": infection_days,
            }
            scenario_criteria = apply_scenario(
                criteria,
                {
                    "I01": age_min,
                    "I03": pack_years,
                    "I04": fev1_pct,
                    "I05": fev1_liters,
                    "I06": ratio,
                    "E04": oxygen,
                    "E03": exacerbation_days,
                    "E05": infection_days,
                },
            )
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
        scenario_metrics[0].metric(
            tr("候选规模", "Candidate scale"),
            int(scale["scenario"]),
            delta=int(scale["change"]),
            help=tr("严格通过全部可执行规则的合成候选者", "Synthetic records that pass every executable constraint"),
        )
        scenario_metrics[1].metric(
            tr("代表性差距", "Representation gap"),
            f"{float(representation_gap['scenario']):.1f} pp",
            delta=f"{float(representation_gap['change']):+.1f} pp",
            delta_color="inverse",
            help=tr("女性、65岁以上和重度COPD占比相对完整队列的平均绝对差；越低越接近", "Mean absolute gap from the full cohort across female, age 65+ and severe COPD shares; lower is closer"),
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
            color_discrete_sequence=["#9BAAB4", "#2D7773"],
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
                        "合成候选人数": len(patients),
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
                        "代表性变化": "合成队列结果；人群构成变化需结合页面图表复核",
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
        tr("验证证据与适用边界", "Evidence register and decision boundaries"),
        tr(
            "把已经完成的工程验证、正在采集的业务证据和不能外推的结论分开呈现。",
            "Separate verified engineering behaviour from pending business evidence and conclusions outside the system's current scope.",
        ),
        tr("系统质量", "SYSTEM QUALITY"),
    )
    criteria = st.session_state.criteria
    traceable = sum(bool(item.source_text and item.source_reference) for item in criteria)
    columns = st.columns(4)
    metrics = [
        (tr("审核标准", "Reviewed constraints"), str(len(criteria)), tr("GOLDEN-4 人工校核版本", "Clinically reviewed GOLDEN-4 reference")),
        (tr("边界案例", "Boundary cases"), "50", tr("含等值、缺失、时间窗和多重失败", "Thresholds, missingness, windows and multiple failures")),
        (tr("自动化测试", "Automated checks"), "54", tr("本地与 GitHub Actions 使用同一套测试", "Same suite locally and in GitHub Actions")),
        (tr("来源追溯", "Source traceability"), f"{traceable}/{len(criteria)}", tr("标准原文与公开来源均保留", "Source statement and public reference retained")),
    ]
    for column, (label, value, note) in zip(columns, metrics):
        with column:
            insight_card(label, value, note)

    section_title(tr("证据状态", "What is verified—and what is not"))
    keys = {
        "item": tr("验证项目", "Evidence item"),
        "status": tr("状态", "Status"),
        "evidence": tr("当前证据", "Current evidence"),
        "claim": tr("能支持的结论", "Supported claim"),
    }
    evidence_rows = [
        {
            keys["item"]: tr("规则引擎边界案例", "Rule-engine boundary cases"),
            keys["status"]: tr("已完成", "Verified"),
            keys["evidence"]: tr("50 例人工设定预期结果", "50 cases with manually specified expected outcomes"),
            keys["claim"]: tr("受支持运算符和判定优先级可重复执行", "Supported operators and precedence rules execute reproducibly"),
        },
        {
            keys["item"]: tr("离线完整工作流", "Offline end-to-end workflow"),
            keys["status"]: tr("已完成", "Verified"),
            keys["evidence"]: tr("无 API Key 时仍可载入、审核、仿真和分析", "Protocol, review, simulation and analysis work without an API key"),
            keys["claim"]: tr("评审现场不依赖外部模型服务", "The reviewer path does not depend on a live model service"),
        },
        {
            keys["item"]: tr("标准来源追溯", "Constraint traceability"),
            keys["status"]: tr("已完成", "Verified"),
            keys["evidence"]: tr(f"{traceable}/{len(criteria)} 条保留原文和来源", f"{traceable}/{len(criteria)} constraints retain source text and reference"),
            keys["claim"]: tr("每条计算结果可以回到方案依据", "Every calculated outcome can be traced to protocol evidence"),
        },
        {
            keys["item"]: tr("多方案结构化准确性", "Cross-protocol extraction accuracy"),
            keys["status"]: tr("待采集", "Pending"),
            keys["evidence"]: tr("计划使用公开呼吸系统试验建立金标准", "A public respiratory-trial gold set is planned"),
            keys["claim"]: tr("完成后报告字段级准确率与错误类型", "Will support field-level accuracy and error analysis once measured"),
        },
        {
            keys["item"]: tr("医学人员效率对照", "Reviewer efficiency comparison"),
            keys["status"]: tr("待采集", "Pending"),
            keys["evidence"]: tr("计划比较人工录入与工具辅助审核耗时", "Manual authoring will be compared with assisted review"),
            keys["claim"]: tr("完成后只报告实测时间与遗漏数", "Only measured time and omissions will be reported"),
        },
        {
            keys["item"]: tr("飞书协作回读", "Feishu review round-trip"),
            keys["status"]: tr("待配置", "Not configured") if not feishu_settings().configured else tr("可验证", "Ready to verify"),
            keys["evidence"]: tr("同步不会覆盖审核字段，读取后需人工确认差异", "Sync preserves reviewer fields and requires confirmation before applying a diff"),
            keys["claim"]: tr("审核意见进入规则前有明确控制点", "A visible control gate exists before reviewer changes reach the rule engine"),
        },
    ]
    st.dataframe(
        pd.DataFrame(evidence_rows),
        width="stretch",
        hide_index=True,
        column_config={
            keys["item"]: st.column_config.TextColumn(width="medium"),
            keys["status"]: st.column_config.TextColumn(width="small"),
            keys["evidence"]: st.column_config.TextColumn(width="large"),
            keys["claim"]: st.column_config.TextColumn(width="large"),
        },
    )
    st.info(tr("多方案准确性和效率对照尚未完成，因此当前页面不展示推测性的提效百分比。", "Cross-protocol accuracy and reviewer-efficiency studies are still pending, so no speculative productivity claim is shown."))

    section_title(tr("效率验证记录模板", "Reviewer study template"))
    st.caption(tr("建议由医学专业同学分别完成纯人工和工具辅助任务，记录真实耗时、遗漏和修改数量。", "Medical reviewers should complete manual and assisted tasks while recording time, omissions and corrections."))
    validation_template = pd.DataFrame(
        columns=[
            "参与者编号",
            "专业背景",
            "公开试验编号",
            "处理方式",
            "开始时间",
            "结束时间",
            "总耗时_分钟",
            "标准总数",
            "遗漏数",
            "人工修改字段数",
            "备注",
        ]
    )
    st.download_button(
        tr("下载效率验证记录模板 CSV", "Download reviewer-study template"),
        data=validation_template.to_csv(index=False).encode("utf-8-sig"),
        file_name="trialscope_validation_log.csv",
        mime="text/csv",
    )

    settings = feishu_settings()
    if (
        bool_setting("ENABLE_FEISHU_SYNC", False)
        and settings.configured
        and settings.validation_table_id
    ):
        with st.expander(tr("记录一次真实验证", "Log a completed validation run"), expanded=False):
            st.caption(tr("请只填写实际完成的测试；尚未测得的数据不要用估计值代替。", "Record completed work only; leave unmeasured outcomes blank instead of estimating them."))
            with st.form("feishu_validation_form"):
                form_row_1 = st.columns(3)
                validation_type_options = ["规则边界", "完整路径", "提取准确性", "工作效率"]
                validation_type = form_row_1[0].selectbox(
                    tr("测试类型", "Study type"),
                    validation_type_options,
                    format_func=lambda value: {
                        "规则边界": tr("规则边界", "Rule boundaries"),
                        "完整路径": tr("完整路径", "End-to-end workflow"),
                        "提取准确性": tr("提取准确性", "Extraction accuracy"),
                        "工作效率": tr("工作效率", "Reviewer efficiency"),
                    }[value],
                )
                tester = form_row_1[1].text_input(tr("测试人员", "Reviewer"))
                test_date = form_row_1[2].date_input(tr("测试日期", "Study date"))
                form_row_2 = st.columns(4)
                total_criteria = form_row_2[0].number_input(
                    tr("标准总数", "Total constraints"), min_value=0, value=len(criteria), step=1
                )
                extracted_count = form_row_2[1].number_input(
                    tr("自动提取数", "Automatically extracted"), min_value=0, value=0, step=1
                )
                modified_count = form_row_2[2].number_input(
                    tr("人工修改数", "Reviewer corrections"), min_value=0, value=0, step=1
                )
                omitted_count = form_row_2[3].number_input(
                    tr("遗漏数", "Omissions"), min_value=0, value=0, step=1
                )
                form_row_3 = st.columns(4)
                manual_minutes = form_row_3[0].number_input(
                    tr("人工耗时（分钟）", "Manual time (min)"), min_value=0.0, value=0.0, step=0.5
                )
                assisted_minutes = form_row_3[1].number_input(
                    tr("辅助耗时（分钟）", "Assisted time (min)"), min_value=0.0, value=0.0, step=0.5
                )
                field_f1 = form_row_3[2].number_input(
                    tr("字段 F1", "Field-level F1"), min_value=0.0, max_value=1.0, value=0.0, step=0.01
                )
                traceability_rate = form_row_3[3].number_input(
                    tr("来源追溯率", "Traceability rate"), min_value=0.0, max_value=1.0, value=0.0, step=0.01
                )
                version = st.text_input(tr("版本", "Version"), value="v1")
                validation_note = st.text_area(tr("备注", "Notes"))
                submit_validation = st.form_submit_button(
                    tr("保存到飞书验证记录", "Save validation record to Feishu"),
                    use_container_width=True,
                )
            if submit_validation:
                normalized_tester = tester.strip() or "未署名"
                validation_key = (
                    f"{st.session_state.source.identifier}:{validation_type}:"
                    f"{test_date.isoformat()}:{normalized_tester}"
                )
                test_datetime = datetime.combine(
                    test_date,
                    datetime_time.min,
                    tzinfo=ZoneInfo("Asia/Shanghai"),
                )
                validation_fields = {
                    "验证键": validation_key,
                    "试验编号": st.session_state.source.identifier,
                    "测试类型": validation_type,
                    "标准总数": int(total_criteria),
                    "自动提取数": int(extracted_count),
                    "人工修改数": int(modified_count),
                    "遗漏数": int(omitted_count),
                    "人工耗时分钟": float(manual_minutes),
                    "辅助耗时分钟": float(assisted_minutes),
                    "字段F1": float(field_f1),
                    "来源追溯率": float(traceability_rate),
                    "测试人员": normalized_tester,
                    "版本": version.strip() or "v1",
                    "备注": validation_note.strip(),
                    "测试日期": int(test_datetime.timestamp() * 1000),
                }
                try:
                    with st.spinner(tr("正在保存验证记录...", "Saving validation record...")):
                        with FeishuClient(settings) as client:
                            action = client.upsert_record(
                                settings.validation_table_id,
                                "验证键",
                                validation_fields,
                            )
                    verb = "新建" if action == "created" else "更新"
                    st.success(
                        tr(
                            f"已在飞书验证记录表中{verb}本次数据。",
                            "The validation record has been saved to Feishu.",
                        )
                    )
                except FeishuError as exc:
                    st.error(str(exc))

    section_title(tr("当前结果不能支持的结论", "Conclusions outside the current system scope"))
    boundary_columns = st.columns(3)
    boundaries = [
        (tr("不代表真实入组率", "Not a real enrolment rate"), tr("500 名候选者为合成数据，只用于验证计算和展示流程。", "The 500 records are synthetic and validate only the workflow and calculations.")),
        (tr("不自动修改方案", "No automatic protocol amendment"), tr("情景比较用于讨论权衡，任何变更仍需医学、统计和伦理审核。", "Scenario analysis frames trade-offs; medical, statistical and ethics review remain required.")),
        (tr("不处理真实患者决策", "No patient-level clinical decision"), tr("系统不诊断、不自动入组，也不替代研究者判断。", "The system does not diagnose, enrol or replace investigator judgement.")),
    ]
    for column, (title, note) in zip(boundary_columns, boundaries):
        with column:
            st.markdown(
                f"<div class='ts-boundary-card'><b>{escape(title)}</b><p>{escape(note)}</p></div>",
                unsafe_allow_html=True,
            )


def top_navigation() -> str:
    page = st.session_state.navigation
    source: TrialSource = st.session_state.source
    study_name = (
        "GOLDEN-4 · Phase III COPD"
        if source.identifier == "NCT02347774"
        else source.title
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
            f"<b>{escape(source.identifier)} · {escape(study_name)}</b></div>",
            unsafe_allow_html=True,
        )
        with language_col:
            st.radio(
                "Language / 语言",
                options=["zh", "en"],
                format_func=lambda item: "中文" if item == "zh" else "EN",
                horizontal=True,
                key="language",
                label_visibility="collapsed",
            )

    nav_items = [
        ("项目说明", tr("总览", "Overview")),
        ("试验 / PDF 导入", tr("方案导入", "Protocol")),
        ("标准解析", tr("标准审核", "Rule review")),
        ("协作审核", tr("协作中心", "Collaboration")),
        ("患者预筛", tr("队列评估", "Cohort evaluation")),
        ("招募分析", tr("情景分析", "Scenario analysis")),
        ("验证证据", tr("质量控制", "Quality")),
    ]
    with st.container(key="top_navigation"):
        columns = st.columns([0.72, 0.92, 0.92, 1.02, 1.02, 1.02, 0.82], gap="small")
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
current_page = top_navigation()

if current_page == "项目说明":
    page_home()
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
else:
    page_validation()
