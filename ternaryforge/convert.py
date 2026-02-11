"""
TernaryForge Converter — Stage 1 & 2

Takes any HuggingFace model and ternarizes its weights to {-1, 0, +1}.
Uses per-channel absmax scaling to preserve as much information as possible.

The key insight: naive sign() ternarization destroys magnitude info.
Per-channel scaling finds optimal thresholds per output channel,
keeping the relative importance of different channels intact.
"""

import os
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    model_id: str
    original_params: int
    ternary_params: int
    skipped_params: int  # embeddings, norms kept in fp16
    sparsity: float  # fraction of weights that are 0
    output_dir: str
    layer_stats: list = field(default_factory=list)
    elapsed_sec: float = 0.0


def ternarize_weight(
    weight: torch.Tensor,
    method: str = "absmax",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ternarize a weight tensor to {-1, 0, +1} with per-channel scaling.

    Returns:
        ternary: Tensor of {-1, 0, +1} with same shape
        scales: Per-channel scale factors (shape: [out_channels])
    """
    if method == "absmax":
        # Per-output-channel absmax scaling
        # For a weight matrix [out, in], scale per out channel
        if weight.dim() == 2:
            scales = weight.abs().mean(dim=1, keepdim=True)  # [out, 1]
        elif weight.dim() == 1:
            scales = weight.abs().mean().unsqueeze(0)
        else:
            # Flatten all but first dim
            flat = weight.view(weight.shape[0], -1)
            scales = flat.abs().mean(dim=1)
            for _ in range(weight.dim() - 1):
                scales = scales.unsqueeze(-1)

        # Avoid division by zero
        scales = scales.clamp(min=1e-8)

        # Scale, round, clamp to ternary
        ternary = (weight / scales).round().clamp(-1, 1)

        return ternary, scales.squeeze()

    elif method == "naive":
        # Just sign() — baseline for comparison
        ternary = torch.sign(weight)
        scales = weight.abs().mean(dim=-1) if weight.dim() > 1 else weight.abs().mean()
        return ternary, scales

    else:
        raise ValueError(f"Unknown ternarization method: {method}")


def should_ternarize(name: str) -> bool:
    """Decide whether a parameter should be ternarized or kept in fp16."""
    # Keep these in higher precision — they're small and critical
    skip_patterns = [
        "embed_tokens",
        "lm_head",
        "layernorm",
        "layer_norm",
        "norm",
        "rotary",
        "bias",
    ]
    name_lower = name.lower()
    return not any(p in name_lower for p in skip_patterns)


def convert_model(
    model_id: str,
    output_dir: str,
    method: str = "absmax",
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> ConversionResult:
    """
    Load a HuggingFace model, ternarize its weights, and save.

    Args:
        model_id: HuggingFace model ID or local path
        output_dir: Where to save the ternarized model
        method: Ternarization method ("absmax" or "naive")
        dtype: Load precision for original model
        device: Device to use for conversion
    """
    start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading model: {model_id}")
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    total_params = sum(p.numel() for p in model.state_dict().values())
    logger.info(f"Model loaded: {total_params:,} parameters (state_dict)")

    # Ternarize
    ternary_count = 0
    skipped_count = 0
    total_zeros = 0
    total_ternary_elements = 0
    layer_stats = []

    state_dict = model.state_dict()
    ternary_state_dict = {}
    scales_dict = {}

    for name, param in tqdm(state_dict.items(), desc="Ternarizing"):
        if should_ternarize(name) and param.dim() >= 2:
            ternary, scales = ternarize_weight(param.float(), method=method)

            # Stats
            n = param.numel()
            zeros = (ternary == 0).sum().item()
            ternary_count += n
            total_zeros += zeros
            total_ternary_elements += n

            layer_stats.append({
                "name": name,
                "shape": list(param.shape),
                "params": n,
                "sparsity": zeros / n,
                "scale_mean": scales.mean().item(),
                "scale_std": scales.std().item(),
            })

            ternary_state_dict[name] = ternary.to(torch.int8)
            scales_dict[name] = scales.half()
        else:
            skipped_count += param.numel()
            ternary_state_dict[name] = param.half()

    # Save
    logger.info(f"Saving ternarized model to {output_dir}")
    torch.save(ternary_state_dict, os.path.join(output_dir, "ternary_weights.pt"))
    torch.save(scales_dict, os.path.join(output_dir, "scales.pt"))
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Save conversion metadata
    sparsity = total_zeros / total_ternary_elements if total_ternary_elements > 0 else 0
    elapsed = time.time() - start

    result = ConversionResult(
        model_id=model_id,
        original_params=total_params,
        ternary_params=ternary_count,
        skipped_params=skipped_count,
        sparsity=sparsity,
        output_dir=output_dir,
        layer_stats=layer_stats,
        elapsed_sec=elapsed,
    )

    meta = {
        "model_id": model_id,
        "method": method,
        "original_params": total_params,
        "ternary_params": ternary_count,
        "skipped_params": skipped_count,
        "sparsity": round(sparsity, 4),
        "elapsed_sec": round(elapsed, 1),
        "architecture": config.architectures[0] if config.architectures else "unknown",
    }
    with open(os.path.join(output_dir, "conversion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"  Ternarized: {ternary_count:,} params ({ternary_count/total_params*100:.1f}%)")
    logger.info(f"  Kept FP16:  {skipped_count:,} params ({skipped_count/total_params*100:.1f}%)")
    logger.info(f"  Sparsity:   {sparsity:.1%} zeros")

    return result
