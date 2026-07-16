from streamlit.testing.v1 import AppTest


def test_app_starts_offline_with_cached_demo():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert any("TrialScopeAI" in item.value for item in app.markdown)
    assert len(app.metric) >= 4


def test_screening_page_works_without_api_key():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.radio[0].set_value("患者预筛").run()
    assert not app.exception
    values = [item.value for item in app.metric]
    assert sum(int(value) for value in values[:4]) == 500


def test_analysis_page_renders_offline():
    app = AppTest.from_file("app.py", default_timeout=40).run()
    app.radio[0].set_value("招募分析").run(timeout=40)
    assert not app.exception
    assert any(item.label == "运行情景比较" for item in app.button)
