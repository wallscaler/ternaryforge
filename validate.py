"""
TernaryForge Phase 0 — Minimum Viable Validation

The smallest possible test: load a pre-built 1.58-bit ternary model
and run inference on CPU. Measure speed, memory, and output quality.

Target model: Microsoft's BitNet b1.58 2B4T (2 billion params, ternary)
"""

import time
import sys
import os
import json
import psutil

# ── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "1bitLLM/bitnet_b1_58-3B"  # Smallest public ternary model
PROMPTS = [
    "The capital of France is",
    "Explain quantum computing in one sentence:",
    "def fibonacci(n):",
    "The three laws of thermodynamics are",
]
MAX_NEW_TOKENS = 50
RESULTS_DIR = "benchmarks/results"


def get_memory_mb():
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_validation():
    print("=" * 60)
    print("TernaryForge Phase 0 — Ternary Model Validation")
    print("=" * 60)

    # Record baseline memory
    mem_before = get_memory_mb()
    print(f"\nBaseline memory: {mem_before:.0f} MB")

    # ── Step 1: Load model ──────────────────────────────────────────
    print(f"\nLoading model: {MODEL_ID}")
    print("(First run will download ~1-2GB from HuggingFace)")

    load_start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # CPU needs float32
        device_map="cpu",
    )
    model.eval()

    load_time = time.time() - load_start
    mem_after_load = get_memory_mb()
    model_mem = mem_after_load - mem_before

    print(f"Load time: {load_time:.1f}s")
    print(f"Model memory: {model_mem:.0f} MB")
    print(f"Total memory: {mem_after_load:.0f} MB")

    # ── Step 2: Run inference ───────────────────────────────────────
    print(f"\nRunning {len(PROMPTS)} prompts, {MAX_NEW_TOKENS} tokens each...")
    print("-" * 60)

    results = []

    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        # Warmup (first run may be slower)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )

        # Timed generation
        mem_before_gen = get_memory_mb()
        start = time.time()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        elapsed = time.time() - start
        mem_after_gen = get_memory_mb()

        # Decode
        generated_ids = output[0][input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

        result = {
            "prompt": prompt,
            "generated": generated_text,
            "tokens": tokens_generated,
            "time_sec": round(elapsed, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "memory_mb": round(mem_after_gen, 0),
        }
        results.append(result)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {generated_text[:100]}...")
        print(f"Tokens: {tokens_generated} | Time: {elapsed:.2f}s | Speed: {tokens_per_sec:.1f} tok/s")

    # ── Step 3: Summary ─────────────────────────────────────────────
    avg_speed = sum(r["tokens_per_sec"] for r in results) / len(results)
    peak_mem = max(r["memory_mb"] for r in results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model:           {MODEL_ID}")
    print(f"Hardware:        CPU only (Apple M4, 24GB)")
    print(f"Model memory:    {model_mem:.0f} MB")
    print(f"Peak memory:     {peak_mem:.0f} MB")
    print(f"Avg speed:       {avg_speed:.1f} tokens/sec")
    print(f"Load time:       {load_time:.1f}s")
    print("=" * 60)

    # Verdict
    print("\nVERDICT:")
    if avg_speed >= 5:
        print("PASS — Ternary CPU inference is viable. Proceed to Phase 1.")
    elif avg_speed >= 1:
        print("MARGINAL — Works but slow. Needs bitnet.cpp optimized runtime.")
        print("Proceed to Phase 1 but prioritize inference optimization.")
    else:
        print("FAIL — Too slow for practical use on this hardware.")
        print("Investigate bitnet.cpp native runtime or different model.")

    # ── Save results ────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "model": MODEL_ID,
        "hardware": "Apple M4, 24GB RAM",
        "load_time_sec": round(load_time, 1),
        "model_memory_mb": round(model_mem, 0),
        "peak_memory_mb": round(peak_mem, 0),
        "avg_tokens_per_sec": round(avg_speed, 1),
        "results": results,
    }

    outpath = os.path.join(RESULTS_DIR, "phase0_validation.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull results saved to {outpath}")


if __name__ == "__main__":
    run_validation()
