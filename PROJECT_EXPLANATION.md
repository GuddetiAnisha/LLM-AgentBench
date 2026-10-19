# LLM-AgentBench — Project Explanation

## 1. What is this project?

LLM-AgentBench is a Python-based experimental workbench for studying how AI agents solve small engineering problems. The main idea is simple: two language models may both sound confident, but we need a fair way to check whether their answers are correct, consistent, and efficient.

The project gives models the same clearly defined tasks, records how they respond, and compares their answers with results calculated independently in Python. It then presents the findings in a Streamlit dashboard. The project also includes separate research prototypes for test planning, computing-resource allocation, safe red-blue agent simulations, and verification of explicit reasoning traces.

This is a software research and portfolio project. It uses synthetic data rather than real industrial measurements, and it is not a validated production geometry-assurance system.

## 2. Why I built it

Selecting an AI model should involve more than checking whether its answer looks convincing. A model might give the right answer once and fail when a similar question is repeated. Another model might be accurate but slow. A third might struggle to call tools correctly.

I wanted a controlled environment where these differences could be measured instead of guessed. LLM-AgentBench helps investigate the trade-offs between accuracy, repeat reliability, response time, and other observable measurements.

## 3. How the main benchmark works

The workflow has six stages:

1. **Generate tasks.** Python creates synthetic engineering questions from a fixed random seed, so the inputs can be reproduced.
2. **Choose an agent.** I can select a scripted demo profile or a real local language model served through Ollama.
3. **Execute the task.** The agent reads the question, may request an allowed arithmetic tool, and then returns a structured JSON answer.
4. **Calculate an independent reference.** A separate Python evaluator calculates the correct numeric value and pass/fail decision. The real model does not receive this reference as a tool.
5. **Evaluate the response.** The system checks the answer format, numeric agreement, and pass/fail decision. Invalid or failed attempts are counted rather than hidden.
6. **Compare and save.** Results are summarized, displayed in Streamlit, and saved for future inspection.

The agent can use only a small set of permitted tools, including subtraction, absolute value, Euclidean distance, and sums of absolute values. It cannot run arbitrary Python or shell commands through this tool interface. Each attempt has a bounded number of agent steps.

## 4. An example in everyday language

Suppose a manufactured component is supposed to measure 50 mm, but its measured length is 50.3 mm. The maximum allowed deviation is 0.2 mm.

The difference is |50.3 − 50| = 0.3 mm. Since 0.3 mm is larger than 0.2 mm, the correct decision is **fail**.

The benchmark asks the agent to solve a task like this. Its independent evaluator already knows the reference answer and checks whether the agent returned both the correct number and the correct decision. It does not assume that a confident explanation is proof of correctness.

## 5. What kinds of tasks are tested?

The main dataset contains four task families:

- **Dimension deviation:** Check how far a measurement is from its nominal dimension.
- **3D point displacement:** Calculate the straight-line distance between two positions.
- **Worst-case tolerance stack:** Add absolute bilateral tolerance half-widths and compare with a limit.
- **Clearance:** Calculate signed hole-minus-shaft clearance and compare it with a minimum.

All distances are expressed in millimetres. These exercises provide a controlled mathematical test, not a substitute for real CAD, GD&T, or industrial validation.

## 6. What is measured?

- **Accuracy:** The proportion of attempts with the correct numeric value and decision.
- **Repeat reliability:** How often a task is answered correctly on every repeat.
- **Completion rate:** How often the agent produces an answer in the required format.
- **Latency:** The wall-clock time taken by the full agent process, including tools and API overhead.
- **Tool success rate:** How many attempted permitted tool calls succeed.
- **Token usage:** Input and output token counts when reported by Ollama.
- **Client CPU and memory:** Resource use of the Python client process, **not** total model or Ollama server resource use.
- **Estimated cost:** An optional time-based calculation using a user-supplied hourly rate, not an actual bill.

The recommendation view lets a user assign weights to accuracy, repeat reliability, latency, client memory, and estimated cost. Its rankings are relative to the models included in that experiment.

## 7. Scripted demos versus real language models

The project offers two clearly different execution modes.

**Demo / scripted:** The profiles `demo-exact` and `demo-imperfect` are programmed behaviors that help test the experiment pipeline. They are **not LLMs**, and their results must not be presented as evidence of real model performance.

**Ollama / real model:** A locally installed Ollama model receives prompts and responds through the implemented JSON action protocol. Models must follow the requested format well enough to be evaluated. A missing model, failed connection, invalid response, or exhausted step limit is recorded as an unsuccessful attempt. Real Ollama benchmarking must be performed on the user's own installation; the repository's verification report does not claim completed real-model benchmarks.

## 8. Project components

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard to set up, inspect, and export experiments |
| `agentbench/tasks.py` | Synthetic task generation and independent correct answers |
| `agentbench/tools.py` | Safe, restricted arithmetic tools |
| `agentbench/agent.py` | Multi-step structured agent execution |
| `agentbench/adapters.py` | Scripted demo profiles and Ollama integration |
| `agentbench/engine.py` | Experiment execution and measurement |
| `agentbench/analysis.py` | Metrics, comparisons, recommendations, and exports |
| `agentbench/storage.py` | SQLite experiment persistence |
| `agentbench/__main__.py` | Command-line interface |
| `tests/` | Automated checks for different parts of the project |

## 9. Additional research modules

These are distinct experimental extensions, not features that all run automatically when the main benchmark starts.

### Test redundancy analysis — `agentbench/test_redundancy.py`

Different tests sometimes cover almost the same requirements. This module looks for overlap using requirement similarity and numeric outcome similarity. A greedy selection method then proposes a smaller set of tests while attempting to retain specified requirement coverage. It reports test count, coverage, estimated cost, and duration before and after selection. Greedy selection is a heuristic; it does not prove that the result is the smallest mathematically possible set.

### Agentic AI resource orchestration — `agentbench/resource_orchestrator.py`

This module models workloads that need CPU, memory, tool calls, and response-time targets. It compares static, rule-based, and predictive allocation policies across simulated edge and cloud resources. The results include modeled latency, completion, service-level violations, utilization, and estimated cost. It does not control a real cloud deployment or Kubernetes cluster.

### Safe red-blue evaluation — `agentbench/red_blue_sim.py`

This module represents attacker and defender decisions as abstract states and symbolic actions. It allows defensive behavior, detection, containment, recovery, and false positives to be studied in a controlled simulation. It does not perform real network attacks or contain exploit execution.

### ReasonTraceVerifier — `agentbench/reasoning_trace_verifier.py`

This extension looks at explicit, recorded reasoning steps rather than hidden model thoughts. Each step may contain a claim, context, tool result, and supporting evidence. The verifier categorizes claims as **SUPPORTED**, **CONTRADICTED**, or **UNRESOLVED**, using symbolic consistency checks and lightweight text similarity. It also reports measurements related to grounding, coherence, contradiction, unresolved claims, tool-evidence consistency, and final-answer support.

This is an early research prototype. It does not provide formal proof of an AI model's reasoning, and it does not access private internal chain-of-thought.

## 10. Run the project in VS Code on Windows

Install Git, Python 3.11 or newer (Python 3.12 recommended), and Visual Studio Code. In a PowerShell terminal, run:

```powershell
git clone https://github.com/GuddetiAnisha/LLM-AgentBench.git
cd LLM-AgentBench
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**. Choose **Explore sample demo** to inspect included data, or select **Demo / scripted** and **Run benchmark** to create a fresh experiment. A typical demo uses 24 tasks, three repeats, and seed 42.

To run a demo from the command line:

```powershell
.\.venv\Scripts\python.exe -m agentbench --mode demo --tasks 24 --repeats 3 --seed 42
```

For real models, install [Ollama](https://ollama.com/download/windows), download a compatible model, and check the exact installed name with `ollama list`. In the dashboard choose **Ollama / real model**, enter that name and the default server URL `http://localhost:11434`, then run a small experiment. The model's memory requirements depend on the model and computer.

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Experiments are stored in `data/experiments.sqlite`. The command-line benchmark also creates `data/latest.json` and `data/latest.csv`. The dashboard offers exports and a history view.

## 11. What I learned and what the project demonstrates

The project brings together reproducible synthetic data generation, restricted agent tool use, model integration, experiment tracking, automated evaluation, visual analytics, and tests. It shows how to make an AI evaluation more transparent: keep the reference answer separate, count failures fairly, use consistent tasks, and record what was measured.

It also shows the limits of those measurements. Good scores on simple arithmetic questions do not establish that a model can make safe production engineering decisions. Python client memory does not tell us how much GPU memory a model uses. A reasoning-trace similarity check is not a formal verification proof.

## 12. Possible future improvements

Future work could include larger held-out datasets, more realistic engineering tasks, verified model versions, independent inference-server resource measurements, stronger statistical analysis, human assessment of explanation quality, and more rigorous evaluation of evidence and reasoning traces.



For detailed implementation rules, limitations, and configuration options, see [README.md](README.md) and [VERIFICATION.md](VERIFICATION.md).
