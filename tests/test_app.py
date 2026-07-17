from streamlit.testing.v1 import AppTest


def test_app_starts_offline_with_cached_demo():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert any("TrialScope" in item.value for item in app.markdown)
    assert len(app.metric) >= 4


def test_screening_page_works_without_api_key():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.session_state["navigation"] = "患者预筛"
    app.run()
    assert not app.exception
    values = [item.value for item in app.metric]
    assert sum(int(value) for value in values[:4]) == 500


def test_analysis_page_renders_offline():
    app = AppTest.from_file("app.py", default_timeout=40).run()
    app.session_state["navigation"] = "招募分析"
    app.run(timeout=40)
    assert not app.exception
    assert any(item.label == "运行情景比较" for item in app.button)


def test_review_table_can_be_saved_without_losing_hidden_fields():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.session_state["navigation"] = "标准解析"
    app.run()
    next(item for item in app.button if item.label == "保存审核并进入模拟预筛").click().run()
    assert not app.exception
    assert len(app.session_state["criteria"]) == 27


def test_import_source_choices_are_explicit_buttons():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.session_state["navigation"] = "试验 / PDF 导入"
    app.run()
    labels = [item.label for item in app.button]
    assert labels.count("选择此来源") == 3
    assert "已选择" in labels
    assert "确认原文并进入标准审核" in labels
