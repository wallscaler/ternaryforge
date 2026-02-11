"""
TernaryForge Calibrator — Fast Scale Optimization

After naive ternarization, scale factors need adjustment to minimize
reconstruction error. This module:

1. Runs one forward pass to capture input activations to each linear layer
2. For each layer, computes the reference output (original weight × input)
3. Grid searches over scale multipliers: (ternary × scale) × input vs reference
4. Picks the scale that minimizes MSE

This is O(1 forward pass + layers × factors × cheap_matmul) instead of
O(layers × factors × full_forward).
"""

import logging
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)


def get_calibration_data(
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 256,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
) -> list[torch.Tensor]:
    """Load and tokenize calibration samples."""
    logger.info(f"Loading calibration data: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    samples = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        if len(samples) >= n_samples:
            break
        samples.append(tokens[i : i + seq_len].unsqueeze(0))

    logger.info(f"Prepared {len(samples)} calibration samples of length {seq_len}")
    return samples


@torch.no_grad()
def calibrate_scales(
    model,
    ternary_weights: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    samples: list[torch.Tensor],
    n_samples: int = 32,
    n_steps: int = 21,
    scale_range: tuple[float, float] = (0.5, 3.0),
) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Fast per-layer scale optimization using cached activations.

    Strategy:
    1. Run calibration samples through the model ONCE
    2. For each linear layer, capture its input activations
    3. Compute reference output: original_weight @ input
    4. For each scale factor: compute ternary_output = (ternary * scale) @ input
    5. Pick the scale that minimizes MSE(ternary_output, reference_output)

    This avoids running the full model for each (layer, factor) combination.
    """
    start = time.time()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    cal_samples = [s.to(device) for s in samples[:n_samples]]
    scale_factors = torch.linspace(scale_range[0], scale_range[1], steps=n_steps)

    # Step 1: Capture input activations for each linear layer
    logger.info("Capturing layer activations...")
    layer_inputs = {}  # layer_name -> list of input tensors
    hooks = []

    # Map from state_dict key to module
    module_map = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sd_key = f"{name}.weight"
            for key in ternary_weights:
                if key.endswith(sd_key) or key == sd_key:
                    module_map[key] = module
                    break

    def make_hook(sd_key):
        def hook_fn(module, args, output):
            inp = args[0] if isinstance(args, tuple) else args
            if sd_key not in layer_inputs:
                layer_inputs[sd_key] = []
            # Store a small subset — mean across batch and most of seq dim
            # to keep memory manageable
            layer_inputs[sd_key].append(inp.detach().float())
        return hook_fn

    for sd_key, module in module_map.items():
        if sd_key in scales:
            hooks.append(module.register_forward_hook(make_hook(sd_key)))

    for sample in tqdm(cal_samples, desc="Collecting activations"):
        model(sample)

    for h in hooks:
        h.remove()

    # Step 2: Optimize scales using cached activations
    logger.info(f"Optimizing {len(layer_inputs)} layer scales...")
    optimized_scales = {k: v.clone() for k, v in scales.items()}
    stats = {"layers_optimized": 0, "improvements": []}

    for sd_key in tqdm(sorted(layer_inputs.keys()), desc="Calibrating"):
        if sd_key not in scales or sd_key not in module_map:
            continue

        module = module_map[sd_key]
        original_weight = module.weight.data.float()
        w_tern = ternary_weights[sd_key].float().to(device)
        base_scale = scales[sd_key].float().to(device)
        inputs = layer_inputs[sd_key]

        # Compute reference outputs: original_weight @ input
        ref_outputs = []
        for inp in inputs:
            ref_out = F.linear(inp, original_weight)
            ref_outputs.append(ref_out)

        # Grid search over scale factors
        best_mse = float("inf")
        best_factor = 1.0

        for factor in scale_factors:
            adjusted_scale = base_scale * factor.item()

            # Reconstruct weight
            if adjusted_scale.dim() == 0:
                recon = w_tern * adjusted_scale
            elif adjusted_scale.dim() == 1 and w_tern.dim() == 2:
                recon = w_tern * adjusted_scale.unsqueeze(1)
            else:
                recon = w_tern * adjusted_scale

            # Compute ternary outputs and MSE
            total_mse = 0.0
            for inp, ref_out in zip(inputs, ref_outputs):
                tern_out = F.linear(inp, recon)
                total_mse += F.mse_loss(tern_out, ref_out).item()

            avg_mse = total_mse / len(inputs)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_factor = factor.item()

        optimized_scales[sd_key] = scales[sd_key] * best_factor
        stats["layers_optimized"] += 1

        if best_factor != 1.0:
            stats["improvements"].append({
                "layer": sd_key,
                "factor": round(best_factor, 3),
                "mse": round(best_mse, 6),
            })

    # Free cached activations
    del layer_inputs

    stats["elapsed_sec"] = round(time.time() - start, 1)
    n_improved = len(stats["improvements"])
    logger.info(f"Calibration done in {stats['elapsed_sec']}s. "
                f"{n_improved}/{stats['layers_optimized']} layers adjusted.")

    return optimized_scales, stats


@torch.no_grad()
def compute_logit_divergence(
    model,
    ternary_weights: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    samples: list[torch.Tensor],
    max_samples: int = 16,
) -> dict:
    """
    Measure quality by comparing original vs ternary logits.

    Patches the model with ternary weights, runs samples, and computes
    KL divergence and cosine similarity against the original outputs.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Collect original logits
    orig_logits_list = []
    for sample in tqdm(samples[:max_samples], desc="Original forward"):
        out = model(sample.to(device))
        orig_logits_list.append(out.logits.float().cpu())

    # Patch weights with ternary * scale
    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            full_name = f"{name}.weight"
            for key in ternary_weights:
                if key.endswith(full_name) or key == full_name:
                    if key in scales:
                        original_weights[key] = module.weight.data.clone()
                        w_tern = ternary_weights[key].float().to(device)
                        s = scales[key].float().to(device)
                        if s.dim() == 0:
                            recon = w_tern * s
                        elif s.dim() == 1 and w_tern.dim() == 2:
                            recon = w_tern * s.unsqueeze(1)
                        else:
                            recon = w_tern * s
                        module.weight.data = recon.to(dtype)
                    break

    # Collect ternary logits
    kl_divs = []
    cosine_sims = []
    for i, sample in enumerate(tqdm(samples[:max_samples], desc="Ternary forward")):
        out = model(sample.to(device))
        tern_logits = out.logits.float().cpu()
        orig_logits = orig_logits_list[i]

        # KL divergence
        orig_probs = F.softmax(orig_logits, dim=-1)
        tern_log_probs = F.log_softmax(tern_logits, dim=-1)
        kl = F.kl_div(tern_log_probs, orig_probs, reduction="batchmean")
        kl_divs.append(kl.item())

        # Cosine similarity
        cos = F.cosine_similarity(
            orig_logits.view(-1, orig_logits.shape[-1]),
            tern_logits.view(-1, tern_logits.shape[-1]),
            dim=-1,
        )
        cosine_sims.append(cos.mean().item())

    # Restore original weights
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            full_name = f"{name}.weight"
            for key in original_weights:
                if key.endswith(full_name) or key == full_name:
                    module.weight.data = original_weights[key]
                    break

    return {
        "kl_divergence_mean": sum(kl_divs) / len(kl_divs),
        "kl_divergence_max": max(kl_divs),
        "cosine_similarity_mean": sum(cosine_sims) / len(cosine_sims),
        "cosine_similarity_min": min(cosine_sims),
        "n_samples": len(kl_divs),
    }
