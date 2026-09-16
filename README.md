# LLM-AgentBench

**A software-only experimental workbench for engineering AI agents.**

Compare local Ollama models on seeded geometry-assurance-inspired tasks using a bounded tool-calling agent, independent Python ground truth, persistent experiments and a Streamlit dashboard. Start without a model using clearly marked scripted demo profiles.

This independent portfolio project is inspired by the research question of choosing capable, reliable and efficient engineering agents. It is not affiliated with Volvo and is not a validated geometry assurance product. No CAD software, physical parts, proprietary datasets or paid APIs are required.

## Quick start — Windows PowerShell

Install Python 3.11 or newer (3.12 recommended). Open PowerShell in this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Click **Explore sample demo** for saved sample data or **Run benchmark** to generate fresh demo measurements. Using the virtual environment's Python directly avoids PowerShell activation policy issues.

Run a benchmark without the dashboard:

```powershell
.\.venv\Scripts\python.exe -m agentbench --mode demo --tasks 24 --repeats 3 --seed 42
.\.venv\Scripts\python.exe -m pytest -q
```

Experiments are stored in `data/experiments.sqlite`; the CLI also writes `data/latest.json` and `data/latest.csv`. The UI offers full JSON and per-attempt CSV downloads. No benchmark runs occur automatically when opening the dashboard.

## Real Ollama runs

Install [Ollama](https://ollama.com/download/windows), start it, and pull a model you have selected. Copy its exact name from `ollama list` into the dashboard's **Ollama / real model** mode. Models must support structured JSON responses; this application uses a text JSON action protocol rather than native vendor tool calls.

```powershell
ollama list
# Replace YOUR_INSTALLED_MODEL with a name from that list:
.\.venv\Scripts\python.exe -m agentbench --mode ollama --models YOUR_INSTALLED_MODEL --tasks 24 --repeats 3 --warmup
```

Use `--models MODEL_A MODEL_B` to compare models and `--base-url http://localhost:11434` to select the server. The dashboard also accepts `OLLAMA_BASE_URL`. No network call is made in demo mode. Ollama requests use temperature 0, a repeat seed, a 512-token generation limit, a 5-second connection timeout and a 60-second read timeout per call. Read timeouts are not a strict whole-experiment deadline. Each agent has at most five calls by default. A missing server/model, invalid JSON or step exhaustion is recorded as failure, **never silently replaced with a demo result**.

The integration follows Ollama's [chat API](https://docs.ollama.com/api/chat) and [usage metrics](https://docs.ollama.com/api/usage). Token totals sum reported counts over all agent steps; missing counts stay null. Hosted APIs and remote Ollama servers can be added, but this release includes no paid API adapter.

## What is implemented

- Dashboard: benchmark setup, comparison charts, task-family breakdowns, weighted recommendations, constraints, saved history, trace inspection and exports.
- Four deterministic synthetic task families with randomized inputs and fixed mathematical definitions.
- Allowlisted tools: subtraction, absolute value, Euclidean distance and sum of absolute values. Inputs are bounded; no arbitrary Python execution, shell access or `eval`.
- A bounded multi-step agent protocol with tool error feedback, token accounting, structured final answers and complete response/tool traces.
- Sequential benchmark execution in seeded shuffled order, optional excluded warmup, repeated tasks, shared prompts and dataset hashes.
- SQLite persistence, hardware/client environment metadata, CLI, Docker configuration and automated tests.

## Architecture

```text
Streamlit dashboard / Python CLI
              |
       benchmark engine -------- SQLite experiment store
              |
    seeded synthetic dataset
              |
     bounded agent loop -------- Ollama / scripted demo adapter
              |
    allowlisted arithmetic tools
              |
    independent reference scorer
              |
    metrics -> recommendation -> CSV / JSON
```

The reference evaluator is never supplied to the real model or exposed as a tool. Demo profiles independently use arithmetic tools; they do not call the evaluator. A separate HTTP backend would duplicate this local application's interface, so FastAPI is intentionally omitted.

## Dataset and evaluation contract

All distances are in millimetres. No external geometry standard is implied.

| Family | Reference value | Pass condition |
|---|---|---|
| Dimension | absolute measured minus nominal | value <= tolerance |
| Point displacement | Euclidean 3D distance | value <= tolerance |
| Worst-case stack | sum of absolute bilateral half-widths | value <= limit |
| Clearance | hole minus shaft, signed | value >= minimum |

Tolerance boundaries allow 1e-12 for floating-point representation. A final answer must have a finite numeric `value`, a true JSON boolean `passed`, and `unit: "mm"`. Numerical agreement uses absolute or relative tolerance 1e-4. Both numerical and pass/fail agreement are required for strict accuracy. Missing/invalid responses score zero. Explanations are retained for inspection but not automatically graded as reasoning quality.

Generation is balanced by cycling through the four families; choose a multiple of four. Seeds reproduce inputs exactly in this implementation. The dataset SHA-256, full tasks, version, settings and environment accompany each result. Reproducible prompts do not guarantee identical model outputs on different runtimes or hardware.

## Metrics and honest limitations

| Metric | Definition / interpretation |
|---|---|
| Accuracy | correct attempts / all attempts, including failures |
| Reliability | fraction of tasks correct on **every** repeat |
| Completion rate | fraction with a valid final answer schema |
| Latency | measured wall time for the full agent loop, including tool/API/error overhead |
| p95 latency | pandas 95th percentile of attempt latency |
| Throughput | 60 / mean latency; sequential agent tasks/minute, not generation tokens/sec |
| Tokens | server-reported input/output totals across agent calls, or unknown |
| Tool success | successful allowlisted tool calls / attempted tool calls |
| Client RAM | maximum observed RSS of the Python client process, sampled every 20 ms plus endpoints |
| Client CPU | process CPU seconds consumed during an attempt |
| Cost | optional mean latency × user-entered hourly USD rate / 3600 |

RAM includes the dashboard, Python interpreter, accumulated results and other client allocations. CPU may include other threads in the same process. These measurements **exclude the Ollama server, model RAM/VRAM, GPU and energy** and cannot establish whether a model fits on particular hardware. Sampling can miss short memory peaks. Server cold starts and shared machine load affect latency. Cost is an assumption-based compute estimate, not a bill; local inference is not claimed to be free. No fabricated token counts or resource values are substituted.

The displayed 95% Wilson accuracy bounds are descriptive; repeats of the same task are correlated, so they are not a rigorous independent-sample significance test. Reliability with one repeat equals accuracy. These simple arithmetic tasks are a small controlled benchmark, not evidence of production engineering competence, broad model intelligence, hallucination rate or hidden reasoning quality.

## Recommendation method

1. Filter by minimum accuracy, maximum mean latency and maximum **client** RAM.
2. Min-max normalize each weighted metric among eligible models; reverse latency, RAM and cost so smaller is better. Tied dimensions contribute equally (utility 1).
3. Normalize nonnegative user weights and sum their contributions. All-zero weights are rejected. A positively weighted unknown cost is rejected until a cost assumption is supplied.

The default weights are accuracy 45%, repeat reliability 25%, latency 20%, client RAM 10%, cost 0%. Contribution columns explain every score. A single eligible model scores 1; equal scores are ties, with model name used only for display order. Scores are relative to the candidate set, not universal rankings. Demo and real model results cannot be ranked together.

## Sample data provenance

`samples/demo_experiment.json` and `.csv` are generated by executing the scripted adapters on this development machine. **They are not real LLM results.** `demo-exact` performs the defined arithmetic; `demo-imperfect` deliberately flips selected decisions based on task index and repeat seed. The resulting quality patterns are scripted. Timings and client resources are actual measurements of that demo execution and will vary between machines. Demo tokens and cost are unknown.

Regenerate the sample explicitly:

```powershell
.\.venv\Scripts\python.exe -m agentbench --mode demo --tasks 24 --repeats 3 --seed 42 --export samples/demo_experiment
```

## Docker (optional)

```powershell
docker compose up --build
```

Open http://localhost:8501. Experiment storage uses a named Docker volume. Docker Desktop uses `host.docker.internal` to reach a host Ollama installation; Ollama must be reachable from the container for real runs. Demo needs no server. Container setup is supplied for portability; see `VERIFICATION.md` for which checks were actually run. This is a trusted local application, with no authentication or multi-user job queue; the Compose port binds to localhost.

## Research extension plan

For a thesis study: freeze model digests/quantization and Ollama version; record server hardware/drivers and power settings; define separate tuning and held-out datasets; pre-register weights; increase task diversity and sample counts; randomize/counterbalance model order over multiple sessions; monitor the inference server separately; add human-rated explanations with inter-rater agreement; bootstrap by task rather than treating repeated attempts as independent. This release records requested model names, not immutable model digests. Avoid unqualified cross-machine comparisons.

## Project layout

```text
app.py                 Streamlit dashboard
agentbench/tasks.py    generator + independent reference scoring
agentbench/tools.py    bounded arithmetic tools
agentbench/agent.py    JSON agent protocol
agentbench/adapters.py demo and Ollama adapters
agentbench/engine.py   experiments and client resource sampling
agentbench/analysis.py summaries, recommendations, export
agentbench/storage.py  SQLite persistence
agentbench/__main__.py command-line entry point
tests/                unit, integration, dashboard tests
samples/              explicitly labeled demo exports
```

## Troubleshooting

- **Cannot connect to Ollama:** check that it is running, the URL is correct and the model appears in `ollama list`. Inspect attempt errors in the dashboard.
- **Invalid outputs:** choose a model that follows JSON instructions; inspect agent traces. Such attempts are deliberately scored as failures.
- **Long runs:** reduce task count/repeats; the displayed run count can involve up to five model calls per attempt. Stop the Streamlit process to interrupt; only completed experiments are saved.
- **No eligible recommendation:** relax constraints. To weight cost, enter an hourly-rate assumption first.
- **Port already in use:** add `--server.port 8502` to the Streamlit command.

License: MIT. See `LICENSE`.
