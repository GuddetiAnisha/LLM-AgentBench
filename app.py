import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from agentbench.adapters import DemoAdapter, OllamaAdapter
from agentbench.analysis import WEIGHTS, export_csv, export_json, recommend, summarize
from agentbench.engine import benchmark
from agentbench.storage import Store

st.set_page_config(page_title="LLM-AgentBench", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background: #0b1120;}
[data-testid="stMetric"] {background: #162136; padding: 18px; border-radius: 12px; border: 1px solid #263653;}
h1 {letter-spacing: -1.5px;} .block-container {padding-top: 4.5rem;}
</style>""", unsafe_allow_html=True)
st.caption("ENGINEERING INTELLIGENCE / EXPERIMENT WORKBENCH")
st.title("LLM-AgentBench")
st.write("Measure agent accuracy, reliability and efficiency on reproducible synthetic geometry tasks.")
store = Store(os.environ.get("AGENTBENCH_DB", "data/experiments.sqlite"))

with st.sidebar:
    st.header("Experiment setup")
    mode = st.radio("Execution mode", ["Demo / scripted", "Ollama / real model"])
    if mode.startswith("Demo"):
        selected = st.multiselect("Demo profiles", ["demo-exact", "demo-imperfect"], default=["demo-exact", "demo-imperfect"])
        base_url = ""
    else:
        selected = [m.strip() for m in st.text_input("Installed model names (comma-separated)", "").split(",") if m.strip()]
        base_url = st.text_input("Ollama URL", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    count = st.number_input("Tasks", min_value=4, max_value=1000, value=24, step=4)
    repeats = st.number_input("Repeats per task", min_value=1, max_value=20, value=3)
    seed = st.number_input("Dataset seed", min_value=0, value=42)
    warmup = st.checkbox("Warm up each model (excluded from scores)")
    st.caption(f"{count * repeats * len(selected)} scored agent runs • up to 5 model calls each")
    run = st.button("Run benchmark", type="primary", use_container_width=True)
    sample = st.button("Explore sample demo", use_container_width=True)
    st.divider()
    st.caption("Software-only prototype. No CAD, physical measurements, or Volvo affiliation.")

if run:
    if not selected:
        st.error("Choose at least one model or profile.")
    elif len(set(selected)) != len(selected):
        st.error("Model names must be unique.")
    else:
        bar = st.progress(0.0, text="Running bounded agent trials…")
        adapters = [DemoAdapter(m) if mode.startswith("Demo") else OllamaAdapter(m, base_url) for m in selected]
        with st.spinner("Collecting measurements and evaluating answers…"):
            experiment = benchmark(adapters, count, repeats, seed, warmup=warmup, progress=bar.progress)
            store.save(experiment)
            st.session_state.experiment = experiment
        bar.empty()
        st.success("Experiment saved. Failed calls remain in the results.")
if sample:
    st.session_state.experiment = json.loads((Path(__file__).parent / "samples/demo_experiment.json").read_text(encoding="utf-8"))

history = store.history()
with st.expander("Experiment history", expanded=False):
    if history:
        names = {h["id"]: f"{h['created'][:19]} • {h['mode'].upper()} • {h['id'][:8]}" for h in history}
        chosen = st.selectbox("Saved experiment", list(names), format_func=names.get)
        if st.button("Load experiment"):
            st.session_state.experiment = store.load(chosen)
        st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
    else:
        st.caption("No saved experiments yet. Run a benchmark to begin.")

experiment = st.session_state.get("experiment")
if not experiment:
    st.info("Start with Explore sample demo, or run your own benchmark from the sidebar.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 01 · Generate\nSeeded dimension, point displacement, tolerance stack and clearance tasks.")
    c2.markdown("### 02 · Execute\nA bounded agent selects arithmetic tools and returns a structured answer.")
    c3.markdown("### 03 · Compare\nIndependent Python scoring, repeat reliability and explicit trade-offs.")
    st.stop()

if experiment["mode"] == "demo":
    st.warning("DEMO / SCRIPTED — these are software test profiles, not LLM benchmark results. Timings measure this demo execution only.")
else:
    st.info("OLLAMA RUN — measured model attempts on synthetic tasks. Resource metrics cover the Python client only.")
st.caption(f"Experiment {experiment['id']} · Dataset {experiment['dataset_sha256'][:12]} · {experiment['created']}")
summary = summarize(experiment)
runs = pd.DataFrame(experiment["runs"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Models / profiles", len(summary))
c2.metric("Scored attempts", len(runs))
c3.metric("Overall accuracy", f"{runs.correct.mean():.1%}")
c4.metric("Execution errors", int(runs.error.notna().sum()))
compare, recommend_tab, traces, methodology = st.tabs(["Model comparison", "Recommendation", "Tasks & agent traces", "Methodology & exports"])
with compare:
    st.subheader("Quality and operating trade-offs")
    a, b = st.columns(2)
    with a:
        st.caption("Accuracy and all-repeats-correct reliability (0–1)")
        st.bar_chart(summary.set_index("model")[["accuracy", "reliability"]], color=["#38bdf8", "#34d399"], stack=False)
    with b:
        st.caption("Mean and p95 end-to-end agent latency (seconds)")
        st.bar_chart(summary.set_index("model")[["latency_s", "p95_latency_s"]], stack=False)
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.caption("Client RAM and CPU exclude Ollama inference. Missing token counts and cost remain unknown, not zero. Reliability = fraction of tasks correct on every repeat.")
    st.subheader("Accuracy by task family")
    st.dataframe(runs.pivot_table(index="model", columns="kind", values="correct", aggfunc="mean"), use_container_width=True)
with recommend_tab:
    st.subheader("Choose what matters")
    st.write("Scores are relative to eligible models in this experiment. Adjust weights and constraints to see the trade-offs.")
    weights = {}
    cols = st.columns(3)
    labels = {"accuracy": "Accuracy", "reliability": "Repeat reliability", "latency_s": "Low latency", "client_peak_rss_mb": "Low client RAM", "estimated_cost_usd": "Low estimated cost"}
    for index, (key, default) in enumerate(WEIGHTS.items()):
        weights[key] = cols[index % 3].slider(labels[key], 0, 100, int(default*100), key=key)
    hourly = st.number_input("Assumed compute cost (USD/hour; 0 = unknown)", min_value=0.0, value=0.0, step=0.1)
    minimum = st.slider("Minimum accuracy", 0.0, 1.0, 0.0)
    max_latency = st.number_input("Maximum mean latency (seconds; 0 = no limit)", min_value=0.0, value=0.0)
    max_memory = st.number_input("Maximum client RAM (MB; 0 = no limit)", min_value=0.0, value=0.0)
    try:
        ranked = recommend(summarize(experiment, hourly or None), weights, minimum, max_latency or None, max_memory or None)
        if ranked.empty:
            st.warning("No models satisfy these constraints.")
        else:
            st.success(f"Highest weighted score: {ranked.iloc[0]['model']} ({ranked.iloc[0]['score']:.3f})")
            st.dataframe(ranked[["model", "score"] + [c for c in ranked if c.startswith("contribution_")]], hide_index=True, use_container_width=True)
    except ValueError as exc:
        st.warning(str(exc))
    st.caption("Cost is an optional runtime × hourly-rate estimate, not a bill or energy measurement. Client RAM cannot determine model hardware requirements.")
with traces:
    chosen_model = st.selectbox("Inspect model", experiment["config"]["models"])
    subset = runs[runs.model == chosen_model]
    st.dataframe(subset[["task_id", "kind", "repeat", "correct", "schema_valid", "latency_s", "error"]], hide_index=True, use_container_width=True)
    idx = st.selectbox("Inspect attempt", list(subset.index), format_func=lambda i: f"{runs.loc[i, 'task_id']} / repeat {runs.loc[i, 'repeat']}")
    record = experiment["runs"][idx]
    st.json({"task": next(t for t in experiment["tasks"] if t["id"] == record["task_id"]), "answer": record["answer"], "expected": record["expected"], "trace": record["trace"], "error": record["error"]})
with methodology:
    st.markdown("""**Strict evaluation.** A correct answer requires a finite numeric value, correct boolean decision and `mm` unit. Numeric tolerance: 1e-4 absolute or relative. Invalid outputs and failed calls score zero.

**Controlled comparison.** Every model sees the same seeded dataset, prompt, tool budget and repeat seeds. Jobs run sequentially in seeded shuffled order. Optional warmup is excluded. Temperature is zero; server reproducibility is not guaranteed.

**Measurement limits.** Peak RSS is sampled for this Python process at 20 ms intervals. CPU seconds cover this client process. Neither measures model RAM, GPU, server CPU or energy. Confidence intervals are descriptive Wilson intervals; repeated tasks are correlated. No automatic judgment of hidden reasoning, explanation quality or hallucination is claimed.

**Synthetic scope.** Tasks implement simple mathematical definitions, not GD&T compliance or validated production geometry assurance. Demo profiles are scripted and are never mixed with real models in a recommendation.""")
    st.json({"config": experiment["config"], "environment": experiment["environment"], "warmup_errors": experiment["warmup_errors"]})
    left, right = st.columns(2)
    left.download_button("Download complete JSON", export_json(experiment), f"{experiment['mode']}_{experiment['id']}.json", "application/json")
    right.download_button("Download attempts CSV", export_csv(experiment), f"{experiment['mode']}_{experiment['id']}.csv", "text/csv")
