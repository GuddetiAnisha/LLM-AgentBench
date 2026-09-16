import argparse
from pathlib import Path
from .adapters import DemoAdapter, OllamaAdapter
from .analysis import export_csv, export_json, summarize
from .engine import benchmark
from .storage import Store


def main():
    parser = argparse.ArgumentParser(description="LLM-AgentBench: synthetic engineering agent benchmarks")
    parser.add_argument("--mode", choices=["demo", "ollama"], default="demo")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--db", default="data/experiments.sqlite")
    parser.add_argument("--export", default="data/latest")
    args = parser.parse_args()
    if args.mode == "ollama" and not args.models:
        parser.error("--models must name installed Ollama models")
    models = args.models or ["demo-exact", "demo-imperfect"]
    adapters = [DemoAdapter(m) if args.mode == "demo" else OllamaAdapter(m, args.base_url) for m in models]
    result = benchmark(adapters, args.tasks, args.repeats, args.seed, warmup=args.warmup)
    Store(args.db).save(result)
    prefix = Path(args.export)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(export_json(result), encoding="utf-8")
    prefix.with_suffix(".csv").write_text(export_csv(result), encoding="utf-8")
    print(f"PROVENANCE: {result['mode'].upper()} | Experiment: {result['id']}")
    print(summarize(result).to_string(index=False))


if __name__ == "__main__":
    main()
