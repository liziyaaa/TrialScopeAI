import json

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import DATA_DIR
from src.llm_parser import load_cached_demo_criteria
from src.models import TrialSource
from src.rules import match_dataframe


def prepare_public_case(app: AppTest, *, with_results: bool = True) -> None:
    """Populate test state explicitly; the production app never preloads this case."""

    source = TrialSource.model_validate(
        json.loads((DATA_DIR / "golden4_trial.json").read_text(encoding="utf-8"))
    )
    criteria = load_cached_demo_criteria()
    patients = pd.read_csv(DATA_DIR / "synthetic_patients.csv")
    app.session_state["source"] = source
    app.session_state["criteria_text"] = source.criteria_text
    app.session_state["criteria_editor"] = source.criteria_text
    app.session_state["criteria"] = criteria
    app.session_state["patients"] = patients
    app.session_state["cohort_file_name"] = "test-cohort.csv"
    app.session_state["results"] = match_dataframe(patients, criteria) if with_results else None


def test_app_starts_with_clean_workspace_and_public_case_only_on_homepage():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert any("TrialScope" in item.value for item in app.markdown)
    assert any("把试验方案约束，转成可审核的招募判断" in item.value for item in app.markdown)
    assert len(app.metric) == 0
    assert app.session_state["source"].identifier == ""
    assert app.session_state["criteria"] == []
    assert app.session_state["patients"].empty


def test_workspace_contains_operational_dashboard():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    prepare_public_case(app)
    app.session_state["navigation"] = "研究工作台"
    app.run()
    assert not app.exception
    assert len(app.metric) >= 4
    assert any("招募可行性总览" in item.value for item in app.markdown)


def test_screening_page_works_without_api_key():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    prepare_public_case(app, with_results=False)
    app.session_state["navigation"] = "患者预筛"
    app.run()
    download_labels = [item.label for item in app.get("download_button")]
    assert "01 · 完整混合队列" in download_labels
    assert "02 · 低风险队列" in download_labels
    assert "03 · 缺失字段队列" in download_labels
    assert "04 · 边界值队列" in download_labels
    next(item for item in app.button if item.label == "运行方案约束仿真").click().run()
    assert not app.exception
    values = [item.value for item in app.metric]
    assert sum(int(value) for value in values[:4]) == 500


def test_analysis_page_renders_offline():
    app = AppTest.from_file("app.py", default_timeout=40).run()
    prepare_public_case(app)
    app.session_state["navigation"] = "招募分析"
    app.run(timeout=40)
    assert not app.exception
    assert any(item.label == "运行情景权衡" for item in app.button)
    assert any("单项标准边际影响" in item.value for item in app.markdown)


def test_review_table_can_be_saved_without_losing_hidden_fields():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    prepare_public_case(app, with_results=False)
    app.session_state["navigation"] = "标准解析"
    app.run()
    next(item for item in app.button if item.label == "保存审核并进入协作确认").click().run()
    assert not app.exception
    assert len(app.session_state["criteria"]) == 27
    assert app.session_state["navigation"] == "协作审核"


def test_import_source_choices_are_explicit_buttons():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.session_state["navigation"] = "试验 / PDF 导入"
    app.run()
    labels = [item.label for item in app.button]
    assert labels.count("选择此来源") == 2
    assert "已选择" in labels
    assert not any("GOLDEN-4" in item.value for item in app.markdown)
    assert next(item for item in app.text_input if item.label == "ClinicalTrials.gov NCT 编号").value == ""


def test_collaboration_page_has_offline_fallback():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    prepare_public_case(app, with_results=False)
    app.session_state["navigation"] = "协作审核"
    app.run()
    assert not app.exception
    assert any(item.label == "下载飞书审核模板 CSV" for item in app.get("download_button"))
    assert any(item.label == "继续运行方案约束仿真" for item in app.button)


def test_governance_page_uses_active_study_audit_register():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    prepare_public_case(app, with_results=False)
    app.session_state["navigation"] = "验证证据"
    app.run()
    assert not app.exception
    markdown_values = [item.value for item in app.markdown]
    assert any("审计与数据治理" in value for value in markdown_values)
    assert any("标准审计清单" in value for value in markdown_values)


def test_english_interface_uses_adapted_decision_language():
    app = AppTest.from_file("app.py", default_timeout=40).run()
    app.session_state["language"] = "en"
    app.run(timeout=40)
    markdown_values = [item.value for item in app.markdown]
    assert any("Turn protocol constraints into reviewable recruitment decisions" in value for value in markdown_values)
    assert any("From protocol text to traceable decisions" in value for value in markdown_values)
    labels = [item.label for item in app.button]
    assert "Cohort evaluation" in labels
    assert "Scenario analysis" in labels


def test_global_navigation_is_top_level_and_sidebar_is_removed():
    app = AppTest.from_file("app.py", default_timeout=40).run()
    labels = [item.label for item in app.button]
    expected_navigation = [
        "首页",
        "工作台",
        "方案导入",
        "标准审核",
        "协作中心",
        "队列评估",
        "情景分析",
        "历史记录",
        "审计治理",
    ]
    assert all(label in labels for label in expected_navigation)
    assert labels.index("首页") < labels.index("审计治理")
    assert "中文" in labels and "EN" in labels
    assert not app.sidebar.button


def test_language_switch_uses_explicit_buttons():
    app = AppTest.from_file("app.py", default_timeout=40).run()
    next(item for item in app.button if item.label == "EN").click().run(timeout=40)
    assert not app.exception
    assert app.session_state["language"] == "en"
    assert any("Turn protocol constraints into reviewable recruitment decisions" in item.value for item in app.markdown)


def test_history_page_can_save_and_restore_current_workspace():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    prepare_public_case(app, with_results=False)
    app.session_state["navigation"] = "历史记录"
    app.run()
    next(item for item in app.button if item.label == "保存当前研究").click().run()
    assert not app.exception
    assert len(app.session_state["history_workspaces"]) == 1
    next(item for item in app.button if item.label == "恢复并继续").click().run()
    assert app.session_state["navigation"] == "研究工作台"
    assert len(app.session_state["criteria"]) == 27


def test_english_decision_page_renders_tradeoff_controls():
    app = AppTest.from_file("app.py", default_timeout=50).run()
    prepare_public_case(app)
    app.session_state["language"] = "en"
    app.session_state["navigation"] = "招募分析"
    app.run(timeout=50)
    assert not app.exception
    assert any(item.label == "Run scenario trade-off" for item in app.button)
    assert any("Marginal impact by constraint" in item.value for item in app.markdown)
