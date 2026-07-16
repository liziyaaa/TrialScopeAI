"""TrialScopeAI Streamlit application."""

from __future__ import annotations

import json
import os
from collections import Counter
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
    status_counts,
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
    page_title="TrialScopeAI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .stApp { background: linear-gradient(180deg, #F7FAFC 0%, #FFFFFF 45%); }
    .hero {
        padding: 2.1rem 2.3rem;
        border-radius: 24px;
        background: linear-gradient(125deg, #073B4C 0%, #087F8C 65%, #11A5A8 100%);
        color: white;
        box-shadow: 0 16px 45px rgba(7, 59, 76, .18);
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; font-size: 2.55rem; margin: 0 0 .45rem 0; letter-spacing: -.03em; }
    .hero p { color: #E6FAFC; max-width: 780px; font-size: 1.08rem; line-height: 1.75; margin: 0; }
    .eyebrow { font-size: .78rem; letter-spacing: .13em; font-weight: 700; color: #A8EFF0; margin-bottom: .55rem; }
    .soft-card {
        border: 1px solid #D8E7EC;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        background: rgba(255,255,255,.92);
        min-height: 136px;
    }
    .soft-card h4 { margin: .15rem 0 .45rem 0; color: #0B5163; }
    .soft-card p { color: #52697A; line-height: 1.55; }
    .boundary {
        border-left: 4px solid #F4A261;
        background: #FFF8EE;
        color: #734D22;
        padding: .85rem 1rem;
        border-radius: 8px;
        margin: .6rem 0 1rem 0;
    }
    .source-chip {
        display: inline-block;
        padding: .28rem .65rem;
        border-radius: 999px;
        background: #DFF4F5;
        color: #0A6670;
        font-size: .8rem;
        font-weight: 650;
        margin-right: .35rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #DDE9ED;
        padding: .8rem 1rem;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(33, 70, 84, .05);
    }
    .footer-note { color: #718394; font-size: .82rem; line-height: 1.55; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


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
        "source": demo_source,
        "criteria_text": demo_source.criteria_text,
        "criteria_editor": demo_source.criteria_text,
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


def criteria_to_frame(criteria: list[Criterion]) -> pd.DataFrame:
    rows = []
    for criterion in criteria:
        item = criterion.model_dump(mode="json")
        item["value"] = json.dumps(item.get("value"), ensure_ascii=False)
        item["applicability"] = json.dumps(item.get("applicability", {}), ensure_ascii=False)
        rows.append(item)
    return pd.DataFrame(rows)


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


def hero(title: str, subtitle: str, eyebrow: str = "TRIALSCOPEAI") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_summary() -> None:
    source: TrialSource = st.session_state.source
    st.markdown(
        f"<span class='source-chip'>{source.source_type.upper()}</span>"
        f"<span class='source-chip'>{source.identifier}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{source.title}**")
    st.caption(f"来源：{source.source_reference}")


def page_home() -> None:
    hero(
        "让入排标准变得可执行",
        "将临床试验方案中的自然语言标准转为可审核规则，在合成患者队列中模拟筛选，定位招募瓶颈与人群代表性风险。",
        "AI + CLINICAL OPERATIONS + PREVENTIVE MEDICINE",
    )
    cols = st.columns(4)
    metrics = [
        ("真实公开方案", "NCT02347774", "ClinicalTrials.gov"),
        ("金标准规则", str(len(st.session_state.criteria)), "逐条保留原文证据"),
        ("合成候选患者", str(len(st.session_state.patients)), "固定种子，可复现"),
        ("真实患者数据", "0", "无隐私与授权风险"),
    ]
    for column, (label, value, help_text) in zip(cols, metrics):
        column.metric(label, value, help=help_text)

    st.subheader("一个可审计的完整闭环")
    cards = st.columns(5)
    steps = [
        ("01", "导入方案", "NCT 编号、粘贴文本或可搜索 PDF。"),
        ("02", "结构化", "DeepSeek 提取字段、阈值、单位和时间窗。"),
        ("03", "医学审核", "保留原文、置信度和主观标准提示。"),
        ("04", "模拟预筛", "确定性规则引擎输出逐患者证据。"),
        ("05", "招募洞察", "漏斗、排除原因、代表性和情景比较。"),
    ]
    for column, (number, title, text) in zip(cards, steps):
        column.markdown(
            f"<div class='soft-card'><span class='source-chip'>{number}</span>"
            f"<h4>{title}</h4><p>{text}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='boundary'><b>使用边界：</b>本产品是试验设计与招募可行性原型，不诊断、不自动入组、不替代研究者、统计人员或伦理委员会。</div>",
        unsafe_allow_html=True,
    )
    source_summary()


def page_import() -> None:
    hero("导入试验方案", "先提取并确认入排标准原文，再进入 AI 结构化；上传内容仅在当前会话内处理。", "STEP 1 / SOURCE")
    method = st.radio(
        "选择输入方式",
        ["内置演示", "NCT 编号", "粘贴文本", "上传 PDF"],
        horizontal=True,
    )
    if method == "内置演示":
        left, right = st.columns([2, 1])
        with left:
            st.info("GOLDEN-4 是 COPD Ⅲ期试验，适合展示年龄、吸烟史、肺功能和时间窗规则。")
            if st.button("重新载入 GOLDEN-4", type="primary", use_container_width=True):
                set_source(load_demo_source())
                st.success("演示案例已载入。")
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
        nct_id = st.text_input("ClinicalTrials.gov NCT 编号", value="NCT02347774")
        if st.button("获取公开试验", type="primary"):
            with st.spinner("正在读取 ClinicalTrials.gov..."):
                try:
                    set_source(fetch_nct_study(nct_id))
                    st.success("试验记录与入排标准已导入。")
                except SourceError as exc:
                    st.error(str(exc))
    elif method == "粘贴文本":
        pasted = st.text_area("粘贴入排标准", height=260, placeholder="Inclusion Criteria: ...")
        if st.button("使用这段文本", type="primary"):
            try:
                set_source(source_from_text(pasted))
                st.success("文本已载入。")
            except SourceError as exc:
                st.error(str(exc))
    else:
        uploaded = st.file_uploader("上传可搜索 PDF", type=["pdf"], accept_multiple_files=False)
        st.caption("限制：20 MB、200 页；首版不支持扫描件 OCR。文件不会写入磁盘。")
        if uploaded and st.button("提取 PDF 文本", type="primary"):
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

    st.divider()
    source_summary()
    edited_text = st.text_area(
        "确认用于 AI 解析的文本",
        key="criteria_editor",
        height=430,
        help="可以删除目录、页眉等无关内容；只有确认后的文本会发送给 DeepSeek。",
    )
    if st.button("确认文本", use_container_width=True):
        if len(edited_text.strip()) < 30:
            st.error("文本过短，无法解析。")
        else:
            st.session_state.criteria_text = edited_text.strip()
            st.success("文本已确认，请进入“标准解析”。")


def page_parse() -> None:
    hero("结构化与医学审核", "DeepSeek 负责语义提取，Pydantic 校验结构；最终规则由人工确认后才能进入筛选。", "STEP 2 / CRITERIA")
    source_summary()
    api_key = str(read_setting("DEEPSEEK_API_KEY", "") or "")
    model = str(read_setting("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL))
    live_enabled = bool_setting("ENABLE_LIVE_LLM", True)
    remaining = max(0, MAX_LIVE_CALLS_PER_SESSION - st.session_state.live_calls)
    required_chunks = len(split_for_llm(st.session_state.criteria_text))
    c1, c2, c3 = st.columns(3)
    c1.metric("当前结构化标准", len(st.session_state.criteria))
    c2.metric("本会话实时额度", remaining)
    traceable = sum(bool(item.source_text and item.source_reference) for item in st.session_state.criteria)
    coverage = traceable / len(st.session_state.criteria) * 100 if st.session_state.criteria else 0
    c3.metric("原文追溯率", f"{coverage:.0f}%")

    left, right = st.columns(2)
    with left:
        if st.button(
            "调用 DeepSeek 重新解析",
            type="primary",
            use_container_width=True,
            disabled=not api_key or not live_enabled or required_chunks > remaining,
        ):
            with st.spinner("DeepSeek 正在提取字段、阈值与时间窗..."):
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
        if st.button("载入审核后的 GOLDEN-4 缓存", use_container_width=True):
            st.session_state.criteria = load_cached_demo_criteria()
            st.session_state.results = None
            st.session_state.last_parse_note = "已载入 27 条审核标准。"
            st.success(st.session_state.last_parse_note)

    if not api_key:
        st.info("当前未配置新 DEEPSEEK_API_KEY；缓存演示和全部规则分析仍可使用。")
    elif not live_enabled:
        st.warning("实时解析已由 ENABLE_LIVE_LLM 关闭。")
    elif required_chunks > remaining:
        st.warning(f"当前文本需要 {required_chunks} 次调用，已超过本会话剩余额度 {remaining} 次。")
    st.caption(st.session_state.last_parse_note)

    if not st.session_state.criteria:
        st.warning("还没有结构化标准。请配置 API 后解析，或载入 GOLDEN-4 缓存。")
        return

    frame = criteria_to_frame(st.session_state.criteria)
    editable = st.data_editor(
        frame,
        width="stretch",
        height=560,
        hide_index=True,
        num_rows="fixed",
        disabled=["criterion_id", "source_reference"],
        column_config={
            "kind": st.column_config.SelectboxColumn("类型", options=["inclusion", "exclusion"]),
            "operator": st.column_config.SelectboxColumn(
                "运算符",
                options=["eq", "neq", "lt", "lte", "gt", "gte", "between", "in", "not_in", "is_true", "is_false", "within_days", "exists", "human_review"],
            ),
            "execution_status": st.column_config.SelectboxColumn("执行方式", options=["automated", "human_review"]),
            "confidence": st.column_config.ProgressColumn("置信度", min_value=0.0, max_value=1.0, format="%.2f"),
            "source_text": st.column_config.TextColumn("标准原文", width="large"),
            "value": st.column_config.TextColumn("JSON 阈值"),
            "applicability": st.column_config.TextColumn("适用条件 JSON"),
        },
    )
    if st.button("保存人工审核结果", type="primary"):
        try:
            st.session_state.criteria = criteria_from_frame(editable)
            st.session_state.results = None
            st.success("审核结果已保存到当前会话。")
        except ValueError as exc:
            st.error(str(exc))

    d1, d2 = st.columns(2)
    d1.download_button(
        "下载标准 JSON",
        data=criteria_json_bytes(st.session_state.criteria),
        file_name="trialscope_criteria.json",
        mime="application/json",
        use_container_width=True,
    )
    d2.download_button(
        "下载标准 CSV",
        data=criteria_to_frame(st.session_state.criteria).to_csv(index=False).encode("utf-8-sig"),
        file_name="trialscope_criteria.csv",
        mime="text/csv",
        use_container_width=True,
    )


def page_screening() -> None:
    hero("合成患者预筛", "规则引擎逐条执行审核后的标准，并保留患者值、阈值、标准原文和判定原因。", "STEP 3 / MATCHING")
    if not st.session_state.criteria:
        st.warning("请先完成标准解析。")
        return
    if st.button("运行 500 人模拟预筛", type="primary"):
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

    frame = results_dataframe(results)
    status_filter = st.multiselect(
        "筛选结果",
        options=list(STATUS_LABELS),
        default=list(STATUS_LABELS),
        format_func=lambda item: STATUS_LABELS[item],
    )
    st.dataframe(frame[frame["overall_status"].isin(status_filter)], width="stretch", hide_index=True)
    st.download_button(
        "下载患者筛选结果 CSV",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="trialscope_patient_results.csv",
        mime="text/csv",
    )

    st.subheader("逐患者证据链")
    patient_id = st.selectbox("选择患者", frame["patient_id"].tolist())
    result = next(item for item in results if item.patient_id == patient_id)
    patient_row = st.session_state.patients[
        st.session_state.patients["patient_id"].astype(str) == patient_id
    ]
    with st.expander("查看合成患者字段", expanded=False):
        st.dataframe(patient_row.T.rename(columns={patient_row.index[0]: "value"}), width="stretch")
    evidence_frame = pd.DataFrame(
        [
            {
                "criterion_id": item.criterion_id,
                "status": item.status,
                "field": item.field,
                "patient_value": item.patient_value,
                "expected": item.expected,
                "message": item.message,
                "source_text": item.source_text,
            }
            for item in result.evidences
        ]
    )
    st.dataframe(evidence_frame, width="stretch", hide_index=True)


def page_analysis() -> None:
    hero("招募可行性与情景模拟", "观察候选人群如何被逐层筛减，并比较参数调整前后的数量与代表性变化。", "STEP 4 / INSIGHT")
    ensure_results()
    if not st.session_state.results:
        st.warning("请先运行患者预筛。")
        return
    patients = st.session_state.patients
    criteria = st.session_state.criteria
    results = st.session_state.results

    left, right = st.columns(2)
    with left:
        funnel = build_funnel(patients, criteria)
        fig = px.funnel(funnel, x="count", y="stage", title="招募预筛漏斗")
        fig.update_traces(marker_color="#087F8C")
        st.plotly_chart(fig, width="stretch")
    with right:
        statuses = status_counts(results)
        fig = px.pie(
            statuses,
            values="count",
            names="label",
            hole=0.58,
            title="预筛结果构成",
            color="status",
            color_discrete_map={
                "eligible": "#2A9D8F",
                "ineligible": "#E76F51",
                "missing_data": "#E9C46A",
                "needs_review": "#4A7C9B",
            },
        )
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        blockers = blocker_counts(results, criteria).head(10)
        if not blockers.empty:
            fig = px.bar(
                blockers.sort_values("count"),
                x="count",
                y="criterion_id",
                orientation="h",
                hover_data=["criterion"],
                title="主要排除标准",
            )
            fig.update_traces(marker_color="#E76F51")
            st.plotly_chart(fig, width="stretch")
    with right:
        missing = missing_field_counts(results).head(10)
        if not missing.empty:
            fig = px.bar(
                missing.sort_values("count"),
                x="count",
                y="field",
                orientation="h",
                title="主要缺失字段",
            )
            fig.update_traces(marker_color="#E9A23B")
            st.plotly_chart(fig, width="stretch")

    representation = representation_table(patients, results)
    fig = px.bar(
        representation,
        x="metric",
        y="value",
        color="group",
        barmode="group",
        title="候选队列与模拟可入组人群的代表性对比",
        color_discrete_sequence=["#6C9EC1", "#087F8C"],
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("比例指标单位为%，平均年龄单位为岁。合成数据仅用于功能验证，不代表真实疾病人群分布。")

    st.subheader("What-if 情景模拟")
    st.markdown(
        "<div class='boundary'><b>安全提示：</b>参数调整只展示合成队列变化，不构成临床试验方案修改建议。</div>",
        unsafe_allow_html=True,
    )
    row1 = st.columns(4)
    age_min = row1[0].number_input("最低年龄", 18, 90, 40)
    pack_years = row1[1].number_input("最低吸烟包年", 0.0, 100.0, 10.0, 1.0)
    fev1_pct = row1[2].number_input("FEV1 %预测值上限", 20.0, 120.0, 80.0, 1.0)
    fev1_liters = row1[3].number_input("FEV1 容量下限 (L)", 0.1, 5.0, 0.7, 0.1)
    row2 = st.columns(4)
    ratio = row2[0].number_input("FEV1/FVC 上限", 0.3, 1.0, 0.7, 0.01)
    oxygen = row2[1].number_input("氧疗小时上限", 0.0, 24.0, 12.0, 1.0)
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
        comparison, _, scenario_results = scenario_comparison(patients, criteria, scenario_criteria)
        st.session_state.scenario_comparison = comparison
        st.session_state.scenario_results = scenario_results

    if st.session_state.scenario_comparison is not None:
        comparison = st.session_state.scenario_comparison
        st.dataframe(comparison, width="stretch", hide_index=True)
        fig = px.bar(
            comparison,
            x="label",
            y=["baseline", "scenario"],
            barmode="group",
            title="基线与情景结果对比",
            labels={"value": "人数", "variable": "方案"},
            color_discrete_sequence=["#6C9EC1", "#087F8C"],
        )
        st.plotly_chart(fig, width="stretch")

    report = build_markdown_report(st.session_state.source.title, patients, results, criteria)
    st.download_button(
        "下载分析摘要 Markdown",
        data=report.encode("utf-8"),
        file_name="trialscope_recruitment_report.md",
        mime="text/markdown",
    )


def sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🫁 TrialScopeAI")
        st.caption("临床试验招募可行性评估助手")
        page = st.radio(
            "工作流",
            ["项目说明", "试验 / PDF 导入", "标准解析", "患者预筛", "招募分析"],
            label_visibility="collapsed",
        )
        st.divider()
        source: TrialSource = st.session_state.source
        st.caption("当前案例")
        st.write(f"**{source.identifier}**")
        st.caption(source.title)
        st.divider()
        st.markdown(
            "<p class='footer-note'>仅使用公开试验与合成患者数据。所有输出均为原型模拟结果。</p>",
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
