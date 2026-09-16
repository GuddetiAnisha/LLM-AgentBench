from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_sample_and_run(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTBENCH_DB", str(tmp_path / "ui.sqlite"))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Explore sample demo").click().run()
    assert not app.exception
    assert any("DEMO / SCRIPTED" in w.value for w in app.warning)
    assert len(app.dataframe) >= 3
    next(n for n in app.number_input if n.label == "Tasks").set_value(4)
    next(n for n in app.number_input if n.label == "Repeats per task").set_value(1)
    next(b for b in app.button if b.label == "Run benchmark").click().run()
    assert not app.exception
    assert len(app.session_state.experiment["runs"]) == 8
    next(b for b in app.button if b.label == "Load experiment").click().run()
    assert not app.exception
    next(s for s in app.slider if s.label == "Low estimated cost").set_value(20).run()
    assert not app.exception
    assert any("unknown" in w.value for w in app.warning)
    next(n for n in app.number_input if n.label.startswith("Assumed compute cost")).set_value(1.0).run()
    assert not app.exception
    assert not any("unknown" in w.value for w in app.warning)
