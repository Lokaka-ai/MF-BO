"""Run a candidate and capture the fixed evaluator's actual return value."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time


def validate_model_contract(model, inputs, targets, vocab_size):
    """Reject future-token leakage and altered evaluation loss before scoring.

    Use the trained eager model to avoid a new compilation for the small probe.
    The caller supplies the same autocast context used by official validation.
    This is a correctness check, not an adversarial Python sandbox.
    """
    import torch
    model = getattr(model, "_orig_mod", model)
    with torch.no_grad():
        logits = model(inputs)
        if logits.shape != (*inputs.shape, vocab_size) or not torch.isfinite(logits).all():
            raise ValueError("Invalid next-token logits shape or non-finite logits")
        length = inputs.shape[1]
        if length < 2:
            raise ValueError("Causality probe needs at least two tokens")
        # A midpoint-only probe misses non-causal attention confined to chunks
        # that divide the midpoint (e.g. 256 within a 2048-token sequence).
        cuts = sorted({cut for cut in (1, length//2-1, length//2, length-1) if 0 < cut < length})
        prefix_checks = []
        for cut in cuts:
            changed = inputs.clone()
            changed[:, cut:] = (changed[:, cut:] + 1) % vocab_size
            changed_logits = model(changed)
            if changed_logits.shape != logits.shape or not torch.isfinite(changed_logits).all():
                raise ValueError("Invalid changed-input logits shape or non-finite logits")
            prefix = logits[:, :cut].float()
            other_prefix = changed_logits[:, :cut].float()
            difference = (prefix-other_prefix).abs().max().item()
            prefix_checks.append({"cut":cut, "max_difference":difference})
            if not torch.allclose(prefix, other_prefix, rtol=1e-3, atol=1e-3):
                raise ValueError(f"Non-causal model: future tokens changed earlier logits at cut {cut} (max difference {difference:.6g})")
        losses = model(inputs, targets, reduction="none").float().reshape(-1)
        expected = torch.nn.functional.cross_entropy(logits.float().reshape(-1, vocab_size),
                                                     targets.reshape(-1), reduction="none")
        if losses.shape != expected.shape or not torch.allclose(losses, expected, rtol=1e-3, atol=1e-3):
            raise ValueError("Evaluation loss must equal unmodified per-token cross entropy of target-free logits")
        return {"passed":True, "prefix_max_difference":max(c["max_difference"] for c in prefix_checks),
                "causality_probe_version":"multiple_prefix_cuts_v2", "prefix_checks":prefix_checks,
                "loss_max_difference":(losses-expected).abs().max().item(),
                "probe_tokens":inputs.numel()}


def main():
    trial = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(trial))
    # Affinity inherits the scheduler's allowed CPUs, including non-contiguous sets.
    cpus = sorted(os.sched_getaffinity(0))[:int(os.environ.get("AUTORESEARCH_CPUS", "8"))]
    os.sched_setaffinity(0, cpus)
    os.environ.setdefault("OMP_NUM_THREADS", str(len(cpus)))
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(min(len(cpus), 4)))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    import prepare

    assert torch.cuda.is_available(), "This task requires a CUDA GPU"
    assert torch.cuda.device_count() == 1, "Expose exactly one GPU with CUDA_VISIBLE_DEVICES"
    assert prepare.TIME_BUDGET == 300 and prepare.EVAL_TOKENS == 20_971_520
    assert prepare.MAX_SEQ_LEN == 2048 and prepare.VAL_SHARD == 6542
    torch.set_num_threads(len(cpus))
    original_evaluate = prepare.evaluate_bpb
    evaluation = []
    contract_checks = []

    def evaluate(model, tokenizer, batch_size):
        if batch_size <= 0 or prepare.EVAL_TOKENS % (batch_size * prepare.MAX_SEQ_LEN):
            raise ValueError("Batch size must divide the fixed validation token count")
        probe_loader = prepare.make_dataloader(tokenizer, 1, prepare.MAX_SEQ_LEN, "val")
        probe_x, probe_y, _ = next(probe_loader)
        check = validate_model_contract(model, probe_x.clone(), probe_y.clone(), tokenizer.get_vocab_size())
        contract_checks.append(check)
        del probe_loader, probe_x, probe_y
        print("Causality and cross-entropy checks passed: " + json.dumps(check), flush=True)
        print(f"Starting official validation: {prepare.EVAL_TOKENS:,} tokens", flush=True)
        score = original_evaluate(model, tokenizer, batch_size)
        if not math.isfinite(score) or score <= 0:
            raise ValueError("Invalid official validation BPB")
        evaluation.append(score)
        return score

    prepare.evaluate_bpb = evaluate
    started = time.monotonic()
    state = runpy.run_path(str(trial / "train.py"), run_name="__main__")
    if len(evaluation) != 1 or state["total_training_time"] < 300:
        raise ValueError("Candidate did not finish the training and validation protocol")
    model = getattr(state["model"], "_orig_mod", state["model"])
    torch.save({"model": model.state_dict(), "config": asdict(state["config"]),
                "val_bpb": evaluation[0]}, trial / "checkpoint.pt")
    result = {
        "valid": True, "val_bpb": evaluation[0], "training_seconds": state["total_training_time"],
        "max_step_seconds": state["max_step_seconds"], "num_steps": state["step"],
        "eval_tokens": prepare.EVAL_TOKENS, "peak_vram_mb": state["peak_vram_mb"],
        "total_seconds": time.monotonic() - started, "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device_batch_size": state["DEVICE_BATCH_SIZE"], "total_batch_size": state["TOTAL_BATCH_SIZE"],
        "model_config": asdict(state["config"]), "train_dtype": str(state["TRAIN_DTYPE"]),
        "model_contract_check": contract_checks[0],
        "num_params": state["num_params"], "total_tokens": state["total_tokens"],
        "code_sha256": hashlib.sha256((trial / "train.py").read_bytes()).hexdigest(),
    }
    temporary = trial / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(trial / "result.json")
    print("Saved official result and checkpoint", flush=True)


if __name__ == "__main__":
    main()
