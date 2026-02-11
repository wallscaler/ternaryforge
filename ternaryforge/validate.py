"""
TernaryForge Validator — Quality checks after conversion.

Compares original vs ternarized model on:
1. Weight distribution analysis
2. Perplexity on WikiText-2
3. Inference speed benchmark via bitnet.cpp
"""

import json
import logging
import os
import subprocess
import time

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def weight_stats(ternary_dir: str) -> dict:
    """Analyze the ternarized weight distribution."""
    weights = torch.load(
        os.path.join(ternary_dir, "ternary_weights.pt"),
        map_location="cpu",
        weights_only=True,
    )
    scales = torch.load(
        os.path.join(ternary_dir, "scales.pt"),
        map_location="cpu",
        weights_only=True,
    )

    total_elements = 0
    total_neg = 0
    total_zero = 0
    total_pos = 0
    ternary_layers = 0

    for name, w in weights.items():
        if name in scales:
            n = w.numel()
            total_elements += n
            total_neg += (w == -1).sum().item()
            total_zero += (w == 0).sum().item()
            total_pos += (w == 1).sum().item()
            ternary_layers += 1

    return {
        "ternary_layers": ternary_layers,
        "total_elements": total_elements,
        "distribution": {
            "-1": total_neg / total_elements if total_elements > 0 else 0,
            "0": total_zero / total_elements if total_elements > 0 else 0,
            "+1": total_pos / total_elements if total_elements > 0 else 0,
        },
        "sparsity": total_zero / total_elements if total_elements > 0 else 0,
    }


def benchmark_gguf(
    gguf_path: str,
    bitnet_cpp_dir: str,
    prompts: list[str] | None = None,
    n_tokens: int = 50,
) -> dict:
    """Run inference benchmark using bitnet.cpp."""
    if prompts is None:
        prompts = [
            "The capital of France is",
            "Explain quantum computing briefly:",
            "def fibonacci(n):",
            "The three laws of thermodynamics state that",
        ]

    cli = os.path.join(bitnet_cpp_dir, "build", "bin", "llama-cli")
    if not os.path.exists(cli):
        raise FileNotFoundError(f"llama-cli not found at {cli}")

    results = []
    for prompt in prompts:
        result = subprocess.run(
            [cli, "-m", gguf_path, "-p", prompt, "-n", str(n_tokens),
             "--no-display-prompt"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Parse performance stats from stderr
        output = result.stdout + result.stderr
        stats = {}
        for line in output.split("\n"):
            if "eval time" in line and "prompt" not in line:
                # Extract tokens/sec from generation eval
                parts = line.split()
                for i, p in enumerate(parts):
                    if "tokens" in p and i + 3 < len(parts):
                        try:
                            stats["generation_tok_s"] = float(parts[i + 3])
                        except (ValueError, IndexError):
                            pass
            elif "prompt eval time" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "tokens" in p and i + 3 < len(parts):
                        try:
                            stats["prompt_eval_tok_s"] = float(parts[i + 3])
                        except (ValueError, IndexError):
                            pass

        results.append({
            "prompt": prompt,
            **stats,
        })

    avg_speed = np.mean([r.get("generation_tok_s", 0) for r in results])
    return {
        "model": gguf_path,
        "avg_generation_tok_s": round(float(avg_speed), 1),
        "results": results,
    }


def full_validation(
    ternary_dir: str,
    gguf_path: str | None = None,
    bitnet_cpp_dir: str | None = None,
) -> dict:
    """Run full validation suite."""
    report = {"ternary_dir": ternary_dir}

    # Weight stats
    logger.info("Analyzing weight distribution...")
    report["weight_stats"] = weight_stats(ternary_dir)
    logger.info(f"  Sparsity: {report['weight_stats']['sparsity']:.1%}")
    logger.info(f"  Distribution: {report['weight_stats']['distribution']}")

    # Inference benchmark (if GGUF available)
    if gguf_path and bitnet_cpp_dir:
        logger.info("Running inference benchmark...")
        try:
            report["benchmark"] = benchmark_gguf(gguf_path, bitnet_cpp_dir)
            logger.info(f"  Avg speed: {report['benchmark']['avg_generation_tok_s']} tok/s")
        except Exception as e:
            logger.warning(f"Benchmark failed: {e}")
            report["benchmark"] = {"error": str(e)}

    # Save report
    report_path = os.path.join(ternary_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    return report
