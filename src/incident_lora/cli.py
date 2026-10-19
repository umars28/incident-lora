from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import build_dataset, read_jsonl
from .scenarios import REPO_ROOT, load_actions, load_scenarios

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def cmd_data(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.scenarios)
    actions = load_actions(args.scenarios)
    manifest = build_dataset(scenarios, actions, Path(args.out), args.train_variants,
                             args.test_variants, args.seed)
    print(json.dumps({k: manifest[k] for k in ("counts", "heldout_scenarios", "leak_checks")},
                     indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    report = {}
    for name in ("train", "val", "test_seen", "test_heldout"):
        path = Path(args.data) / f"{name}.jsonl"
        if not path.exists():
            continue
        lengths = sorted(
            len(tok.apply_chat_template(
                r["messages"] + [{"role": "assistant", "content": r["target"]}],
                tokenize=True, return_dict=False))
            for r in read_jsonl(path)
        )
        report[name] = {"n": len(lengths), "min": lengths[0], "median": lengths[len(lengths) // 2],
                        "max": lengths[-1], "total": sum(lengths)}
    print(json.dumps(report, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    config = TrainConfig(
        base_model=args.base_model, rank=args.rank, alpha=args.alpha, dropout=args.dropout,
        learning_rate=args.lr, epochs=args.epochs, grad_accum=args.grad_accum,
        seed=args.seed, limit=args.limit,
        gradient_checkpointing=not args.no_checkpointing,
    )
    meta = train(config, Path(args.data), Path(args.out))
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}, indent=2))
    return 0


def _backend(args: argparse.Namespace, records: list[dict] | None = None):
    from . import backends

    if args.backend == "oracle":
        by_prompt = {r["messages"][1]["content"]: r["target"] for r in records or []}
        return backends.FakeBackend(lambda m: by_prompt[m[1]["content"]], name="oracle")
    if args.backend == "hf":
        return backends.HFBackend(args.model, adapter=args.adapter)
    if args.backend == "ollama":
        return backends.OllamaBackend(args.model)
    if args.backend == "openai":
        return backends.OpenAIBackend(args.base_url, args.model, api_key=args.api_key)
    raise SystemExit(f"unknown backend {args.backend}")


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import run_eval, write_result

    records = read_jsonl(Path(args.data) / f"{args.set}.jsonl")
    if args.limit:
        records = records[: args.limit]
    action_ids = {a["id"] for a in load_actions(args.scenarios)}
    backend = _backend(args, records)

    def progress(i: int, n: int, s) -> None:
        print(f"[{i}/{n}] {s.id} json={int(s.json_valid)} rc={int(s.root_cause_hit)} "
              f"cat={int(s.category_correct)} act={int(s.action_match)} {s.latency_s:.2f}s",
              file=sys.stderr, flush=True)

    result = run_eval(backend, records, action_ids, progress)
    extra = {"label": args.label or backend.name, "test_set": args.set,
             "model": args.model, "adapter": args.adapter}
    write_result(result, Path(args.out), extra)
    print(json.dumps({k: v for k, v in result["summary"].items() if k != "by_scenario"}, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .evaluate import markdown_table, per_scenario_table

    results_dir = Path(args.results)
    order = [line.strip() for line in args.order.split(",")] if args.order else None
    paths = sorted(results_dir.glob("eval-*.json"))
    results = [json.loads(p.read_text()) for p in paths]
    if order:
        rank = {name: i for i, name in enumerate(order)}
        results.sort(key=lambda r: (rank.get(r.get("label"), 99), r.get("test_set", "")))
    sections = ["## Overall\n", markdown_table(results)]
    for test_set in sorted({r.get("test_set") for r in results}):
        subset = [r for r in results if r.get("test_set") == test_set]
        sections += [f"\n## Per scenario: {test_set}\n", per_scenario_table(subset)]
    loads = [json.loads(p.read_text()) for p in sorted(results_dir.glob("loadtest-*.json"))]
    if loads:
        sections += [
            "\n## Load tests\n",
            "| server | model | concurrency | requests | ok | wall s | req/s | out tok/s | p50 s | p95 s |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for d in sorted(loads, key=lambda d: (d.get("server", ""), d.get("dtype", ""), d["concurrency"])):
            sections.append(
                f"| {d.get('server', 'vllm')} {d.get('dtype', '')} | {d['model']} | {d['concurrency']} | "
                f"{d['requests']} | {d['succeeded']} | {d['wall_s']} | {d['requests_per_s']} | "
                f"{d['output_tokens_per_s']} | {d['latency_p50_s']} | {d['latency_p95_s']} |"
            )
    text = "\n".join(sections) + "\n"
    (results_dir / "RESULTS.md").write_text(text)
    print(text)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .backends import OpenAIBackend

    records = read_jsonl(Path(args.data) / f"{args.set}.jsonl")
    record = next(r for r in records if r["id"] == args.id) if args.id else records[0]
    gen = OpenAIBackend(args.base_url, args.model, api_key=args.api_key).generate(record["messages"])
    print(json.dumps({"id": record["id"], "model": args.model, "latency_s": round(gen.latency_s, 3),
                      "output_tokens": gen.output_tokens, "output": gen.text,
                      "expected": record["answer"]}, indent=2))
    return 0


def cmd_loadtest(args: argparse.Namespace) -> int:
    import asyncio

    from .loadtest import run_load

    records = read_jsonl(Path(args.data) / f"{args.set}.jsonl")
    prompts = [r["messages"] for r in records]
    result = asyncio.run(run_load(args.base_url, args.model, prompts, args.concurrency,
                                  args.requests, args.max_tokens))
    result = {"server": args.server, "dtype": args.dtype, **result}
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import bench_step

    result = bench_step(args.model, args.seq_len, args.answer_len, not args.full_logits,
                        not args.no_checkpointing, args.repeats)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


def cmd_shim(args: argparse.Namespace) -> int:
    from .messages_shim import serve

    serve(args.upstream, args.host, args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ilora")
    parser.add_argument("--scenarios", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data")
    p.add_argument("--out", default=str(DATA_DIR))
    p.add_argument("--train-variants", type=int, default=24)
    p.add_argument("--test-variants", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261019)
    p.set_defaults(fn=cmd_data)

    p = sub.add_parser("stats")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default=str(DATA_DIR))
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("train")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-checkpointing", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("--backend", choices=["hf", "ollama", "openai", "oracle"], default="hf")
    p.add_argument("--model", default="oracle")
    p.add_argument("--adapter", default=None)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--api-key", default=None)
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--set", default="test_heldout")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("report")
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--order", default=None)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("ask")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="incident-lora")
    p.add_argument("--api-key", default=None)
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--set", default="test_heldout")
    p.add_argument("--id", default=None)
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("loadtest")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="incident-lora")
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--set", default="test_heldout")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--server", default="vllm")
    p.add_argument("--dtype", default="")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_loadtest)

    p = sub.add_parser("bench-step")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--seq-len", type=int, default=1607)
    p.add_argument("--answer-len", type=int, default=127)
    p.add_argument("--full-logits", action="store_true")
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("shim")
    p.add_argument("--upstream", default="http://localhost:8000")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8090)
    p.set_defaults(fn=cmd_shim)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
