from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .backends import pick_device
from .dataset import read_jsonl

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES))
    learning_rate: float = 2e-4
    epochs: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.05
    max_len: int = 4096
    seed: int = 0
    gradient_checkpointing: bool = True
    limit: int | None = None


def encode(tokenizer: Any, record: dict[str, Any], max_len: int) -> dict[str, list[int]]:
    prompt = tokenizer.apply_chat_template(
        record["messages"], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    full = tokenizer.apply_chat_template(
        record["messages"] + [{"role": "assistant", "content": record["target"]}],
        tokenize=True, return_dict=False,
    )
    if list(full[: len(prompt)]) != list(prompt):
        raise ValueError(f"chat template does not extend the prompt for {record['id']}")
    full = list(full)
    if len(full) > max_len:
        raise ValueError(f"{record['id']} is {len(full)} tokens, over max_len={max_len}")
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "labels": labels}


def answer_span(labels: list[int]) -> int:
    first = next(i for i, v in enumerate(labels) if v != -100)
    if any(v == -100 for v in labels[first:]):
        raise ValueError("answer tokens must be a contiguous suffix")
    return len(labels) - first


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def train(config: TrainConfig, data_dir: Path, out_dir: Path) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    train_records = read_jsonl(data_dir / "train.jsonl")
    val_records = read_jsonl(data_dir / "val.jsonl")
    if config.limit:
        train_records = train_records[: config.limit]
    train_set = [encode(tokenizer, r, config.max_len) for r in train_records]
    val_set = [encode(tokenizer, r, config.max_len) for r in val_records]

    model = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=torch.bfloat16)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.to(device)
    trainable, total = model.get_nb_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_set) / config.grad_accum)
    total_steps = steps_per_epoch * config.epochs
    warmup = max(1, int(total_steps * config.warmup_ratio))

    def tensors(example: dict[str, list[int]]) -> tuple[Any, Any]:
        ids = torch.tensor([example["input_ids"]], device=device)
        labels = torch.tensor([example["labels"]], device=device)
        return ids, labels

    def answer_loss(ids: Any, labels: Any) -> Any:
        keep = answer_span(labels[0].tolist()) + 1
        logits = model(input_ids=ids, logits_to_keep=keep).logits[0, :-1].float()
        return torch.nn.functional.cross_entropy(
            logits, labels[0, ids.shape[1] - keep + 1:], ignore_index=-100
        )

    def val_loss() -> float:
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for ex in val_set:
                ids, labels = tensors(ex)
                total_loss += float(answer_loss(ids, labels))
        model.train()
        return total_loss / max(1, len(val_set))

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log = log_path.open("w")
    history: list[dict[str, Any]] = []
    initial_val = val_loss()
    log.write(json.dumps({"step": 0, "val_loss": initial_val}) + "\n")
    log.flush()
    print(json.dumps({"epoch": 0, "val_loss": initial_val}), flush=True)
    history.append({"epoch": 0, "val_loss": initial_val})

    started = time.perf_counter()
    step = 0
    model.train()
    order = list(range(len(train_set)))
    tokens_seen = 0
    for epoch in range(config.epochs):
        random.Random(config.seed + epoch).shuffle(order)
        running = 0.0
        micro = 0
        for position, idx in enumerate(order):
            ids, labels = tensors(train_set[idx])
            tokens_seen += ids.shape[1]
            loss = answer_loss(ids, labels)
            (loss / config.grad_accum).backward()
            running += float(loss)
            micro += 1
            last = position == len(order) - 1
            if micro == config.grad_accum or last:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, total_steps, warmup, config.learning_rate)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if device == "mps":
                    torch.mps.empty_cache()
                entry = {
                    "step": step + 1,
                    "epoch": epoch + 1,
                    "train_loss": running / micro,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_s": round(time.perf_counter() - started, 1),
                }
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(json.dumps(entry), flush=True)
                step += 1
                running = 0.0
                micro = 0
        v = val_loss()
        history.append({"epoch": epoch + 1, "val_loss": v})
        log.write(json.dumps({"step": step, "epoch": epoch + 1, "val_loss": v}) + "\n")
        log.flush()
        print(json.dumps({"epoch": epoch + 1, "val_loss": v}), flush=True)
    if device == "mps":
        torch.mps.synchronize()
    wall = time.perf_counter() - started
    log.close()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    weights = sorted(adapter_dir.glob("*.safetensors"))
    meta = {
        "config": asdict(config),
        "device": device,
        "torch": torch.__version__,
        "train_examples": len(train_set),
        "val_examples": len(val_set),
        "train_tokens_per_epoch": sum(len(e["input_ids"]) for e in train_set),
        "train_tokens_seen": tokens_seen,
        "max_example_tokens": max(len(e["input_ids"]) for e in train_set),
        "optimizer_steps": step,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": round(100 * trainable / total, 4),
        "adapter_weights_bytes": sum(p.stat().st_size for p in weights),
        "adapter_dir_bytes": dir_size(adapter_dir),
        "wall_clock_s": round(wall, 1),
        "val_loss_history": history,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
