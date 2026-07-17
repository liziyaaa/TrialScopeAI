"""TrialScope clinical recruitment feasibility workspace."""

from __future__ import annotations

import json
import os
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
    STATUS_LABELS,
    apply_scenario,
    blocker_counts,
    build_funnel,
    build_markdown_report,
    missing_field_counts,
    representation_table,
    scenario_comparison,
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
    page_title="TrialScope | 招募可行性评估",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
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
        "navigation": "项目说明",
        "import_method": "内置演示",
        "selected_patient_id": None,
        "source": demo_source,
        "criteria_text": demo_source.criteria_text,
        "criteria": load_cached_demo_criteria(),
        "patients": load_patients(),
        "results": None,
        "live_calls": 0,
        "scenario_comparison": None,
        "scenario_results": None,
        "last_parse_note": "已载入医学审核的 GOLDEN-4 缓存标准。",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_source(source: TrialSource, criteria_text: str | None = None) -> None:
    text = criteria_text or source.criteria_text
    st.session_state.source = source
    st.session_state.criteria_text = text
    st.session_state.criteria_editor = text
    st.session_state.results = None
    st.session_state.scenario_comparison = None
    st.session_state.scenario_results = None
    if source.identifier == "NCT02347774":
        st.session_state.criteria = load_cached_demo_criteria()
        st.session_state.last_parse_note = "该案例可直接使用审核后的缓存标准，也可重新调用 DeepSeek。"
    else:
        st.session_state.criteria = []
        st.session_state.last_parse_note = "请进入“标准解析”步骤生成结构化标准。"


KIND_LABELS = {"inclusion": "入组", "exclusion": "排除"}
EXECUTION_LABELS = {"automated": "自动判断", "human_review": "人工确认"}
OPERATOR_LABELS = {
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
FIELD_LABELS = {
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
    frame["kind"] = frame["kind"].map(KIND_LABELS).fillna(frame["kind"])
    frame["operator"] = frame["operator"].map(OPERATOR_LABELS).fillna(frame["operator"])
    frame["execution_status"] = (
        frame["execution_status"].map(EXECUTION_LABELS).fillna(frame["execution_status"])
    )
    frame["field"] = frame["field"].map(FIELD_LABELS).fillna(frame["field"])
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
            raise ValueError(f"第 {row_number + 1} 行格式无效：{exc}") from exc
    return output


def criteria_from_review_frame(frame: pd.DataFrame) -> list[Criterion]:
    normalized = frame.copy()
    normalized["kind"] = normalized["kind"].replace({value: key for key, value in KIND_LABELS.items()})
    normalized["operator"] = normalized["operator"].replace(
        {value: key for key, value in OPERATOR_LABELS.items()}
    )
    normalized["execution_status"] = normalized["execution_status"].replace(
        {value: key for key, value in EXECUTION_LABELS.items()}
    )
    normalized["field"] = normalized["field"].replace(
        {value: key for key, value in FIELD_LABELS.items()}
    )
    return criteria_from_frame(normalized)


def result_summary(result: Any) -> str:
    if result.overall_status == "eligible":
        return "全部可执行标准均通过"
    if result.overall_status == "ineligible":
        ids = result.failed_criteria
        suffix = "、".join(ids[:4]) + (" 等" if len(ids) > 4 else "")
        return f"未满足 {len(ids)} 项标准：{suffix}"
    if result.overall_status == "missing_data":
        fields = sorted(
            {
                FIELD_LABELS.get(item.field, item.field or "未定义字段")
                for item in result.evidences
                if item.status == "missing"
            }
        )
        return "需要补充：" + "、".join(fields[:3])
    return f"{len(result.review_criteria)} 项标准需要研究者确认"


def display_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def results_to_display_frame(results: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": result.patient_id,
                "overall_status": STATUS_LABELS[result.overall_status],
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
                <div class="ts-study-meta">来源：{reference}</div>
            </div>
            <div class="ts-study-tags">
                <span class="ts-tag">Ⅲ期</span>
                <span class="ts-tag">COPD</span>
                <span class="ts-status ok">公开方案</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_strip(active_step: int) -> None:
    names = ["方案导入", "标准审核", "模拟预筛", "招募评估"]
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
    page_header(
        "GOLDEN-4 招募可行性评估",
        "基于公开试验方案与合成候选者，审核入排标准并定位可能影响招募的关键条件。",
        "项目概览",
    )
    source_summary()
    ensure_results()
    results = st.session_state.results or []
    counts = Counter(item.overall_status for item in results)
    eligible_rate = counts.get("eligible", 0) / len(results) * 100 if results else 0
    cols = st.columns(4)
    metrics = [
        ("入排标准", str(len(st.session_state.criteria)), "均保留方案原文"),
        ("合成候选者", str(len(st.session_state.patients)), "固定随机种子，可复现"),
        ("模拟符合率", f"{eligible_rate:.1f}%", "不包含待人工复核者"),
        ("待人工复核", str(counts.get("needs_review", 0)), "涉及主观或不可执行标准"),
    ]
    for column, (label, value, help_text) in zip(cols, metrics):
        column.metric(label, value, help=help_text)

    section_title("开始一次评估")
    st.caption("按顺序完成四项任务；右侧按钮是每一步的操作入口。当前建议先审核结构化标准。")
    task_row("01", "导入试验方案", "确认 NCT、粘贴文本或 PDF 中的入排标准原文", "已载入", "试验 / PDF 导入", "查看")
    task_row("02", "审核结构化标准", "核对字段、阈值、时间窗及需要人工判断的条件", "当前任务", "标准解析", "继续审核", active=True)
    task_row("03", "运行模拟预筛", "在 500 名合成候选者中执行已确认规则", "可运行", "患者预筛", "打开")
    task_row("04", "评估招募可行性", "查看筛减瓶颈、数据缺口、代表性与情景变化", "可查看", "招募分析", "打开")

    section_title("数据使用范围")
    st.markdown(
        "<div class='ts-insight'><div class='ts-insight-label'>当前演示数据</div>"
        "<div class='ts-insight-value'>0 条真实患者记录</div>"
        "<div class='ts-insight-note'>公开试验方案 + 500 名固定随机种子的合成候选者，结果可复现。</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='ts-boundary'><b>使用边界：</b>本工具用于试验设计与招募可行性讨论，不诊断、不自动入组，也不替代研究者、统计人员或伦理委员会。</div>",
        unsafe_allow_html=True,
    )


def page_import() -> None:
    page_header(
        "导入试验方案",
        "选择公开试验、粘贴标准原文或上传文字型 PDF；确认原文后再生成待审核规则。",
        "01 · 方案导入",
    )
    workflow_strip(1)
    section_title("选择方案来源")
    st.caption("先选择一种来源。每次只处理一种输入，原文确认后才会进入结构化审核。")
    source_options = [
        ("内置演示", "GOLDEN-4", "无需准备文件，适合直接体验"),
        ("NCT 编号", "公开试验", "从 ClinicalTrials.gov 获取标准"),
        ("粘贴文本", "标准原文", "适合已有 Word 或网页文本"),
        ("上传 PDF", "研究方案", "支持可搜索的文字型 PDF"),
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
                    "已选择" if is_selected else "选择此来源",
                    key=f"choose_source_{index}",
                    disabled=is_selected,
                    use_container_width=True,
                    on_click=choose_import_method,
                    args=(method_name,),
                )

    method = st.session_state.import_method
    st.markdown(f"<div class='ts-selection-label'>当前选择：{escape(method)}</div>", unsafe_allow_html=True)
    with st.container(border=True, key="source_input_panel"):
        if method == "内置演示":
            left, right = st.columns([2, 1])
            with left:
                st.markdown("**GOLDEN-4（NCT02347774）**")
                st.caption("COPD Ⅲ期公开试验，覆盖年龄、吸烟史、肺功能、用药和时间窗，适合完整演示。")
                if st.button("载入 GOLDEN-4 演示", type="primary", use_container_width=True):
                    set_source(load_demo_source())
                    st.success("演示案例已载入。请在下方核对原文。")
            with right:
                pdf_path = OUTPUT_DIR / "pdf" / "golden4_demo_protocol.pdf"
                st.download_button(
                    "下载演示 PDF",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        elif method == "NCT 编号":
            st.markdown("**输入公开试验编号**")
            nct_id = st.text_input(
                "ClinicalTrials.gov NCT 编号",
                value="NCT02347774",
                help="格式示例：NCT02347774",
            )
            if st.button("获取试验标准", type="primary", use_container_width=True):
                with st.spinner("正在读取 ClinicalTrials.gov..."):
                    try:
                        set_source(fetch_nct_study(nct_id))
                        st.success("试验记录与入排标准已导入。请在下方核对原文。")
                    except SourceError as exc:
                        st.error(str(exc))
        elif method == "粘贴文本":
            st.markdown("**粘贴入组与排除标准**")
            pasted = st.text_area(
                "标准原文",
                height=240,
                placeholder="Inclusion Criteria: ...\n\nExclusion Criteria: ...",
            )
            if st.button("载入这段原文", type="primary", use_container_width=True):
                try:
                    set_source(source_from_text(pasted))
                    st.success("文本已载入。请在下方核对原文。")
                except SourceError as exc:
                    st.error(str(exc))
        else:
            st.markdown("**上传可搜索的文字型 PDF**")
            uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"], accept_multiple_files=False)
            st.caption("限制：20 MB、200 页；首版不支持扫描件 OCR。文件只在当前会话内存中处理。")
            if uploaded and st.button("提取入排标准", type="primary", use_container_width=True):
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
                        st.success(f"已从 {extraction.page_count} 页 PDF 中定位入排标准章节。")
                except (SourceError, PDFScannedError) as exc:
                    st.error(str(exc))

    section_title("核对方案原文")
    source_summary()
    left, right = st.columns([2.15, 1], gap="large")
    with left:
        edited_text = st.text_area(
            "用于生成规则的入排标准原文",
            value=st.session_state.criteria_text,
            key="criteria_editor",
            height=360,
            help="可以删除目录、页眉等无关内容；仅确认后的文本会用于自动解析。",
        )
        if st.button("确认原文并进入标准审核", type="primary", use_container_width=True):
            if len(edited_text.strip()) < 30:
                st.error("文本过短，无法解析。")
            else:
                st.session_state.criteria_text = edited_text.strip()
                go_to("标准解析")
                st.rerun()
    with right:
        st.markdown(
            "<div class='ts-insight'><div class='ts-insight-label'>提交前检查</div>"
            "<div class='ts-insight-value'>原文可人工修订</div>"
            "<div class='ts-insight-note'>建议保留完整的入组与排除章节；删除目录、页眉和无关附录。</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='ts-boundary'><b>文件处理：</b>上传内容仅在当前会话内存中处理，不写入仓库或数据库。扫描型 PDF 不进行 OCR。</div>",
            unsafe_allow_html=True,
        )


def page_parse() -> None:
    page_header(
        "标准审核",
        "逐条核对方案原文、结构化条件与执行方式；只有人工确认后的规则才进入模拟预筛。",
        "02 · 标准审核",
    )
    workflow_strip(2)
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
    c1.metric("入组标准", inclusion_count)
    c2.metric("排除标准", exclusion_count)
    c3.metric("需要人工判断", review_count)
    traceable = sum(bool(item.source_text and item.source_reference) for item in st.session_state.criteria)
    coverage = traceable / len(st.session_state.criteria) * 100 if st.session_state.criteria else 0
    c4.metric("原文追溯率", f"{coverage:.0f}%")

    section_title("解析来源")
    left, right = st.columns(2)
    with left:
        if st.button(
            "重新生成待审标准",
            use_container_width=True,
            disabled=not api_key or not live_enabled or required_chunks > remaining,
        ):
            with st.spinner("正在提取字段、阈值和时间窗..."):
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
                    st.session_state.last_parse_note = (
                        f"{'命中缓存' if outcome.from_cache else '实时解析完成'}："
                        f"{outcome.model}，{outcome.chunk_count} 个文本块。"
                    )
                    st.success(st.session_state.last_parse_note)
                except LLMParseError as exc:
                    st.error(str(exc))
    with right:
        if st.button("恢复已审核的演示标准", use_container_width=True):
            st.session_state.criteria = load_cached_demo_criteria()
            st.session_state.results = None
            st.session_state.last_parse_note = "已载入 27 条审核标准。"
            st.success(st.session_state.last_parse_note)

    if not api_key:
        st.info("当前未配置自动解析服务；仍可使用已审核的演示标准和全部规则分析功能。")
    elif not live_enabled:
        st.warning("自动解析服务当前已关闭。")
    elif required_chunks > remaining:
        st.warning(f"当前文本需要 {required_chunks} 次调用，已超过本会话剩余额度 {remaining} 次。")
    with st.expander("查看解析记录", expanded=False):
        st.caption(st.session_state.last_parse_note)
        st.caption(f"解析模型：{model} · 本会话剩余额度：{remaining} 次")

    if not st.session_state.criteria:
        st.warning("还没有结构化标准。请配置 API 后解析，或载入 GOLDEN-4 缓存。")
        return

    section_title("逐条审核")
    st.markdown(
        "<div class='ts-action-guide'><strong>当前任务：确认规则可以按方案原文执行</strong>"
        "<span>点击表格单元格可修改；重点核对阈值、单位、时间窗和“人工确认”项。完成后使用表格下方的蓝色按钮。</span></div>",
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
                "criterion_id": st.column_config.TextColumn("编号", width="small"),
                "kind": st.column_config.SelectboxColumn("类型", options=list(KIND_LABELS.values()), width="small"),
                "operator": st.column_config.SelectboxColumn(
                    "判断条件",
                    options=list(OPERATOR_LABELS.values()),
                    width="medium",
                ),
                "execution_status": st.column_config.SelectboxColumn(
                    "执行方式", options=list(EXECUTION_LABELS.values()), width="medium"
                ),
                "confidence": st.column_config.ProgressColumn(
                    "结构化置信度", min_value=0.0, max_value=1.0, format="%.2f", width="medium"
                ),
                "source_text": st.column_config.TextColumn("标准原文", width="large"),
                "field": st.column_config.TextColumn("结构化字段", width="large"),
                "value": st.column_config.TextColumn("阈值", width="medium"),
                "unit": st.column_config.TextColumn("单位", width="small"),
                "time_window_days": st.column_config.NumberColumn("时间窗（天）", width="small"),
            },
        )
        save_review = st.form_submit_button(
            "保存审核并进入模拟预筛",
            type="primary",
            use_container_width=True,
        )
    if save_review:
        try:
            st.session_state.criteria = criteria_from_review_frame(editable)
            st.session_state.results = None
            go_to("患者预筛")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.caption("下载内容为当前已保存版本；表格中的未保存修改不会进入导出文件。")
    d1, d2 = st.columns(2)
    d1.download_button(
        "导出标准 JSON",
        data=criteria_json_bytes(st.session_state.criteria),
        file_name="trialscope_criteria.json",
        mime="application/json",
        use_container_width=True,
    )
    d2.download_button(
        "导出标准 CSV",
        data=criteria_to_frame(st.session_state.criteria).to_csv(index=False).encode("utf-8-sig"),
        file_name="trialscope_criteria.csv",
        mime="text/csv",
        use_container_width=True,
    )


def page_screening() -> None:
    page_header(
        "模拟预筛",
        "在 500 名合成候选者中执行已审核规则，并为每项判断保留患者值、条件和方案原文。",
        "03 · 模拟预筛",
    )
    workflow_strip(3)
    if not st.session_state.criteria:
        st.warning("请先完成标准解析。")
        return
    st.markdown(
        "<div class='ts-action-guide'><strong>当前任务：执行已审核规则</strong>"
        "<span>运行后先查看总体分布，再点击候选者结果表中的任意一行查看完整证据。</span></div>",
        unsafe_allow_html=True,
    )
    if st.button("运行模拟预筛", type="primary"):
        with st.spinner("正在执行确定性规则..."):
            st.session_state.results = match_dataframe(
                st.session_state.patients, st.session_state.criteria
            )
            st.session_state.scenario_comparison = None
        st.success("模拟预筛完成。")
    ensure_results()
    results = st.session_state.results
    counts = Counter(item.overall_status for item in results)
    columns = st.columns(4)
    for column, status in zip(columns, ["eligible", "ineligible", "missing_data", "needs_review"]):
        column.metric(STATUS_LABELS[status], counts.get(status, 0))

    section_title("候选者结果")
    raw_frame = results_dataframe(results)
    display_frame = results_to_display_frame(results)
    status_filter = st.multiselect(
        "状态筛选",
        options=list(STATUS_LABELS),
        default=list(STATUS_LABELS),
        format_func=lambda item: STATUS_LABELS[item],
    )
    selected_labels = {STATUS_LABELS[item] for item in status_filter}
    filtered_frame = display_frame[
        display_frame["overall_status"].isin(selected_labels)
    ].reset_index(drop=True)
    st.caption("操作提示：点击一行即可在下方打开该候选者的逐条判定证据。")
    selection_event = st.dataframe(
        filtered_frame,
        width="stretch",
        height=390,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="patient_results_table",
        column_config={
            "patient_id": st.column_config.TextColumn("候选者编号", width="small"),
            "overall_status": st.column_config.TextColumn("预筛状态", width="small"),
            "summary": st.column_config.TextColumn("判断摘要", width="large"),
            "failed_count": st.column_config.NumberColumn("未满足", width="small"),
            "missing_count": st.column_config.NumberColumn("缺失", width="small"),
            "review_count": st.column_config.NumberColumn("待确认", width="small"),
        },
    )
    st.download_button(
        "导出完整预筛结果 CSV",
        data=raw_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="trialscope_patient_results.csv",
        mime="text/csv",
    )

    section_title("候选者证据")
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
        f"<div class='ts-next-step'><strong>{escape(STATUS_LABELS[result.overall_status])}</strong><br>"
        f"{escape(result_summary(result))}</div>",
        unsafe_allow_html=True,
    )
    patient_row = st.session_state.patients[
        st.session_state.patients["patient_id"].astype(str) == patient_id
    ]
    with st.expander("查看合成候选者原始字段", expanded=False):
        patient_view = patient_row.T.rename(columns={patient_row.index[0]: "值"}).reset_index()
        patient_view.columns = ["字段", "值"]
        patient_view["字段"] = patient_view["字段"].map(FIELD_LABELS).fillna(patient_view["字段"])
        patient_view["值"] = patient_view["值"].map(display_value)
        st.dataframe(patient_view, width="stretch", hide_index=True)
    evidence_status = {
        "pass": "通过",
        "fail": "未通过",
        "missing": "信息缺失",
        "review": "人工确认",
        "not_applicable": "不适用",
    }
    evidence_frame = pd.DataFrame(
        [
            {
                "编号": item.criterion_id,
                "结果": evidence_status[item.status],
                "字段": FIELD_LABELS.get(item.field, item.field or "—"),
                "患者值": display_value(item.patient_value),
                "标准值": display_value(item.expected),
                "判定说明": item.message,
                "方案原文": item.source_text,
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
            "编号": st.column_config.TextColumn(width="small"),
            "结果": st.column_config.TextColumn(width="small"),
            "字段": st.column_config.TextColumn(width="medium"),
            "判定说明": st.column_config.TextColumn(width="large"),
            "方案原文": st.column_config.TextColumn(width="large"),
        },
    )


def page_analysis() -> None:
    page_header(
        "招募可行性评估",
        "识别候选人群的主要筛减环节、数据缺口和代表性变化，并比较不同参数情景。",
        "04 · 招募评估",
    )
    workflow_strip(4)
    ensure_results()
    if not st.session_state.results:
        st.warning("请先运行患者预筛。")
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
        top_blocker_name = FIELD_LABELS.get(
            top_criterion.field if top_criterion else None,
            top_blocker_id,
        )
    else:
        top_blocker_name, top_blocker_count = "暂无", 0

    summary_columns = st.columns(4)
    with summary_columns[0]:
        insight_card("模拟符合率", f"{eligible_rate:.1f}%", "不包含待人工复核者")
    with summary_columns[1]:
        insight_card("潜在候选者", f"{potential} 人", "模拟符合与待复核合计")
    with summary_columns[2]:
        insight_card("首要筛选瓶颈", top_blocker_name, f"影响 {top_blocker_count} 名候选者")
    with summary_columns[3]:
        insight_card(
            "待补充或复核",
            f"{counts.get('missing_data', 0) + counts.get('needs_review', 0)} 人",
            "优先补充信息并完成医学判断",
        )

    section_title("评估结果")
    funnel_tab, representation_tab, completeness_tab = st.tabs(
        ["招募瓶颈", "人群代表性", "数据完整性"]
    )
    with funnel_tab:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown("**候选者筛减路径**")
            funnel = build_funnel(patients, criteria)
            fig = px.funnel(funnel, x="count", y="stage")
            fig.update_traces(marker_color="#2D7773", textfont_color="#FFFFFF")
            st.plotly_chart(
                style_figure(fig, height=390),
                width="stretch",
                config={"displayModeBar": False},
            )
        with right:
            st.markdown("**主要未通过标准**")
            top_blockers = blockers.head(7).copy()
            if not top_blockers.empty:
                labels = []
                for criterion_id in top_blockers["criterion_id"]:
                    criterion = criterion_map.get(criterion_id)
                    field_label = FIELD_LABELS.get(
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
        st.markdown("**候选队列与潜在入组人群对比**")
        representation = representation_table(patients, results)
        fig = px.bar(
            representation,
            x="metric",
            y="value",
            color="group",
            barmode="group",
            color_discrete_sequence=["#9BAAB4", "#2D7773"],
            labels={"metric": "指标", "value": "数值", "group": "人群"},
        )
        st.plotly_chart(
            style_figure(fig, height=390),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption("比例指标单位为%，平均年龄单位为岁。合成数据不代表真实疾病人群分布。")
    with completeness_tab:
        st.markdown("**影响自动判断的主要缺失字段**")
        top_missing = missing.head(8).copy()
        if top_missing.empty:
            st.success("当前候选队列没有影响判断的字段缺失。")
        else:
            top_missing["label"] = top_missing["field"].map(
                lambda item: FIELD_LABELS.get(item, item)
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

    section_title("情景比较")
    st.markdown(
        "<div class='ts-boundary'><b>安全提示：</b>参数调整只展示合成队列变化，不构成临床试验方案修改建议。</div>",
        unsafe_allow_html=True,
    )
    with st.expander("调整模拟参数", expanded=False):
        row1 = st.columns(4)
        age_min = row1[0].number_input("最低年龄", 18, 90, 40)
        pack_years = row1[1].number_input("最低吸烟包年", 0.0, 100.0, 10.0, 1.0)
        fev1_pct = row1[2].number_input("FEV1 %预计值上限", 20.0, 120.0, 80.0, 1.0)
        fev1_liters = row1[3].number_input("FEV1 容量下限（L）", 0.1, 5.0, 0.7, 0.1)
        row2 = st.columns(4)
        ratio = row2[0].number_input("FEV1/FVC 上限", 0.3, 1.0, 0.7, 0.01)
        oxygen = row2[1].number_input("每日氧疗上限（小时）", 0.0, 24.0, 12.0, 1.0)
        exacerbation_days = row2[2].number_input("急性加重窗口（天）", 1, 365, 42)
        infection_days = row2[3].number_input("感染窗口（天）", 1, 365, 42)

        if st.button("运行情景比较", type="primary"):
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

    if st.session_state.scenario_comparison is not None:
        comparison = st.session_state.scenario_comparison
        eligible_row = comparison[comparison["status"] == "eligible"].iloc[0]
        scenario_metrics = st.columns(3)
        scenario_metrics[0].metric("基线模拟符合", int(eligible_row["baseline"]))
        scenario_metrics[1].metric("情景模拟符合", int(eligible_row["scenario"]))
        scenario_metrics[2].metric("人数变化", int(eligible_row["change"]), delta=int(eligible_row["change"]))
        display_comparison = comparison[["label", "baseline", "scenario", "change"]].rename(
            columns={"label": "结果", "baseline": "基线", "scenario": "情景", "change": "变化"}
        )
        st.dataframe(display_comparison, width="stretch", hide_index=True)
        fig = px.bar(
            comparison,
            x="label",
            y=["baseline", "scenario"],
            barmode="group",
            labels={"label": "结果", "value": "人数", "variable": "方案"},
            color_discrete_sequence=["#9BAAB4", "#2D7773"],
        )
        st.plotly_chart(
            style_figure(fig, height=350),
            width="stretch",
            config={"displayModeBar": False},
        )

    report = build_markdown_report(st.session_state.source.title, patients, results, criteria)
    st.download_button(
        "导出评估摘要 Markdown",
        data=report.encode("utf-8"),
        file_name="trialscope_recruitment_report.md",
        mime="text/markdown",
    )


def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div class='ts-brand'><div class='ts-brand-mark'>TS</div><div>"
            "<div class='ts-brand-name'>TrialScope</div>"
            "<div class='ts-brand-subtitle'>临床试验招募可行性评估</div></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='ts-nav-label'>工作流程</div>", unsafe_allow_html=True)
        nav_items = [
            ("项目说明", "项目概览"),
            ("试验 / PDF 导入", "01  方案导入"),
            ("标准解析", "02  标准审核"),
            ("患者预筛", "03  模拟预筛"),
            ("招募分析", "04  招募评估"),
        ]
        page = st.session_state.navigation
        for index, (page_name, label) in enumerate(nav_items):
            state = "active" if page_name == page else "idle"
            with st.container(key=f"sidebar_nav_{index}_{state}"):
                st.button(
                    label,
                    key=f"sidebar_nav_button_{index}",
                    use_container_width=True,
                    on_click=go_to,
                    args=(page_name,),
                )
        st.markdown(
            "<div class='ts-sidebar-help'>每一步都是可点击的操作入口；建议按 01–04 顺序完成。</div>",
            unsafe_allow_html=True,
        )
        source: TrialSource = st.session_state.source
        st.markdown(
            f"<div class='ts-sidebar-study'><div class='ts-nav-label'>当前研究</div>"
            f"<div class='ts-sidebar-id'>{escape(source.identifier)}</div>"
            f"<div class='ts-sidebar-title'>{escape(source.title)}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='ts-boundary'>仅使用公开试验与合成候选者。所有结果均为原型模拟。</div>",
            unsafe_allow_html=True,
        )
    return page


init_state()
current_page = sidebar()

if current_page == "项目说明":
    page_home()
elif current_page == "试验 / PDF 导入":
    page_import()
elif current_page == "标准解析":
    page_parse()
elif current_page == "患者预筛":
    page_screening()
else:
    page_analysis()
