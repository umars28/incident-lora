from __future__ import annotations

import time
from typing import Any

from .backends import pick_device


def bench_step(model_id: str, seq_len: int, answer_len: int, answer_only: bool,
               checkpointing: bool, repeats: int = 3) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from .train import TARGET_MODULES

    device = pick_device()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    if checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=list(TARGET_MODULES),
                                             task_type="CAUSAL_LM"))
    model.to(device).train()
    ids = torch.randint(0, 1000, (1, seq_len), device=device)
    labels = ids.clone()
    labels[0, : seq_len - answer_len] = -100

    def sync() -> None:
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    timings = []
    for _ in range(repeats + 1):
        started = time.perf_counter()
        if answer_only:
            keep = answer_len + 1
            logits = model(input_ids=ids, logits_to_keep=keep).logits[0, :-1].float()
            loss = F.cross_entropy(logits, labels[0, seq_len - keep + 1:], ignore_index=-100)
        else:
            loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        sync()
        timings.append(round(time.perf_counter() - started, 3))
    peak = None
    if device == "mps":
        peak = torch.mps.driver_allocated_memory()
    elif device == "cuda":
        peak = torch.cuda.max_memory_allocated()
    return {
        "model": model_id,
        "device": device,
        "seq_len": seq_len,
        "answer_len": answer_len,
        "answer_only_logits": answer_only,
        "gradient_checkpointing": checkpointing,
        "warmup_s": timings[0],
        "step_s": timings[1:],
        "driver_bytes": peak,
    }
