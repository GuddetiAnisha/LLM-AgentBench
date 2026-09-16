import hashlib
import json
import platform
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version

import psutil

from . import __version__
from .agent import SYSTEM, run_agent
from .tasks import evaluate, generate_tasks


class ResourceMonitor:
    """Samples this Python client process, NOT the Ollama server or GPU."""
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.start_cpu = self.process.cpu_times()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        end = self.process.cpu_times()
        self.cpu_seconds = max(0, end.user + end.system - self.start_cpu.user - self.start_cpu.system)


def benchmark(adapters, count=24, repeats=3, seed=42, max_steps=5, warmup=False, progress=None):
    if not adapters or len({a.model for a in adapters}) != len(adapters):
        raise ValueError("Supply one or more uniquely named models")
    if len({a.provenance for a in adapters}) != 1:
        raise ValueError("Demo and real models must be run in separate experiments")
    if not 1 <= repeats <= 20 or not 1 <= max_steps <= 10:
        raise ValueError("Repeats must be 1–20 and steps 1–10")
    tasks = generate_tasks(count, seed)
    experiment = dict(id=str(uuid.uuid4()), created=datetime.now(timezone.utc).isoformat(),
        mode=adapters[0].provenance, version=__version__,
        config=dict(count=count, repeats=repeats, seed=seed, max_steps=max_steps, warmup=warmup,
                    models=[a.model for a in adapters], temperature=0, output_limit=512,
                    adapters=[dict(model=a.model, provenance=a.provenance,
                                   base_url=getattr(a, "base_url", None), read_timeout_s=getattr(a, "timeout", None)) for a in adapters]),
        environment=dict(os=platform.platform(), python=platform.python_version(), processor=platform.processor(),
                         logical_cpus=psutil.cpu_count(), host_ram_gb=psutil.virtual_memory().total / 1024**3,
                         packages={name: version(name) for name in ("streamlit", "pandas", "psutil", "requests")}),
        system_prompt=SYSTEM,
        dataset_sha256=hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
        tasks=tasks, runs=[], warmup_errors=[])
    if warmup:
        for adapter in adapters:
            result = run_agent(adapter, tasks[0], seed, max_steps)
            if result["error"]:
                experiment["warmup_errors"].append({"model": adapter.model, "error": result["error"]})
    jobs = [(adapter, task, repeat) for adapter in adapters for task in tasks for repeat in range(repeats)]
    random.Random(seed).shuffle(jobs)
    for index, (adapter, task, repeat) in enumerate(jobs):
        with ResourceMonitor() as resource:
            started = time.perf_counter()
            result = run_agent(adapter, task, seed + repeat, max_steps)
            elapsed = time.perf_counter() - started
        score = evaluate(task, result["answer"])
        experiment["runs"].append(dict(model=adapter.model, provenance=adapter.provenance,
            task_id=task["id"], kind=task["kind"], repeat=repeat, latency_s=elapsed,
            client_peak_rss_mb=resource.peak / 1024**2, client_cpu_s=resource.cpu_seconds,
            **result, **score))
        if progress:
            progress((index + 1) / len(jobs))
    return experiment
