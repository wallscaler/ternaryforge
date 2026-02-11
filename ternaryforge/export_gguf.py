"""
TernaryForge GGUF Exporter — Stage 4

Takes ternarized weights and exports them to GGUF format compatible
with bitnet.cpp.

Strategy: Write an F32 GGUF with ternary values, then use bitnet.cpp's
llama-quantize to pack to i2_s format. This leverages the existing
toolchain rather than reimplementing the bit-packing.
"""

import os
import json
import logging
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

# GGUF tensor name mapping: HuggingFace → GGUF
# Based on bitnet.cpp's convert-hf-to-gguf-bitnet.py
HF_TO_GGUF = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

# Per-layer mappings
LAYER_MAP = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}


def hf_name_to_gguf(hf_name: str) -> str | None:
    """Convert HuggingFace parameter name to GGUF tensor name."""
    if hf_name in HF_TO_GGUF:
        return HF_TO_GGUF[hf_name]

    # Try layer pattern: model.layers.{N}.{rest}
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])
        if rest in LAYER_MAP:
            return f"blk.{layer_idx}.{LAYER_MAP[rest]}"

    return None


def export_ternary_to_hf(
    ternary_dir: str,
    output_dir: str,
) -> str:
    """
    Reconstruct a HuggingFace-compatible model directory from ternarized weights.

    The ternarized weights (int8 {-1,0,1}) are multiplied by their scales
    and saved as float32 safetensors. This produces a model directory that
    bitnet.cpp's convert-hf-to-gguf-bitnet.py can consume.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading ternarized weights from {ternary_dir}")
    ternary_weights = torch.load(
        os.path.join(ternary_dir, "ternary_weights.pt"),
        map_location="cpu",
        weights_only=True,
    )
    scales = torch.load(
        os.path.join(ternary_dir, "scales.pt"),
        map_location="cpu",
        weights_only=True,
    )

    # Reconstruct float32 weights from ternary * scale
    reconstructed = {}
    for name, w in ternary_weights.items():
        if name in scales:
            s = scales[name]
            # Reconstruct: ternary * scale
            if s.dim() == 0:
                reconstructed[name] = (w.float() * s.float()).to(torch.float32)
            elif s.dim() == 1 and w.dim() == 2:
                reconstructed[name] = (w.float() * s.float().unsqueeze(1)).to(torch.float32)
            else:
                reconstructed[name] = (w.float() * s.float()).to(torch.float32)
        else:
            # Non-ternarized (embeddings, norms) — keep as-is
            reconstructed[name] = w.float()

    # Save as pytorch model
    torch.save(reconstructed, os.path.join(output_dir, "pytorch_model.bin"))

    # Copy config and tokenizer
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "tokenizer.model"]:
        src = os.path.join(ternary_dir, fname)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(output_dir, fname))

    # Patch config to mark as bitnet architecture
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        # Add bitnet markers
        config["model_type"] = "llama"  # bitnet.cpp expects llama-compatible
        config["quantization_config"] = {
            "quant_method": "ternary",
            "bits": 1.58,
            "source": "ternaryforge",
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    logger.info(f"Reconstructed model saved to {output_dir}")
    return output_dir


def export_to_gguf(
    ternary_dir: str,
    output_path: str,
    bitnet_cpp_dir: str | None = None,
) -> str:
    """
    Full export pipeline: ternary weights → GGUF file.

    Steps:
    1. Reconstruct float32 model from ternary + scales
    2. Use bitnet.cpp's converter to produce F32 GGUF
    3. Use llama-quantize to pack to i2_s

    Args:
        ternary_dir: Directory with ternary_weights.pt and scales.pt
        output_path: Where to write the final .gguf
        bitnet_cpp_dir: Path to bitnet.cpp repo (for converter and quantizer)
    """
    import subprocess

    if bitnet_cpp_dir is None:
        # Try to find it relative to this repo
        repo_root = Path(__file__).parent.parent
        bitnet_cpp_dir = str(repo_root / "bitnet-cpp")

    if not os.path.exists(bitnet_cpp_dir):
        raise FileNotFoundError(
            f"bitnet.cpp not found at {bitnet_cpp_dir}. "
            "Clone it with: git clone --recursive https://github.com/microsoft/BitNet.git bitnet-cpp"
        )

    # Step 1: Reconstruct HF-format model
    hf_dir = os.path.join(ternary_dir, "hf_reconstructed")
    export_ternary_to_hf(ternary_dir, hf_dir)

    # Step 2: Convert to F32 GGUF using bitnet.cpp's converter
    converter = os.path.join(bitnet_cpp_dir, "utils", "convert-hf-to-gguf-bitnet.py")
    if not os.path.exists(converter):
        raise FileNotFoundError(f"Converter not found at {converter}")

    f32_gguf = os.path.join(ternary_dir, "ggml-model-f32.gguf")
    logger.info("Converting to F32 GGUF...")
    result = subprocess.run(
        ["python", converter, hf_dir, "--outtype", "f32"],
        capture_output=True,
        text=True,
        cwd=bitnet_cpp_dir,
    )
    if result.returncode != 0:
        logger.error(f"GGUF conversion failed: {result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.stderr[:500]}")

    # Step 3: Quantize to i2_s using llama-quantize
    quantizer = os.path.join(bitnet_cpp_dir, "build", "bin", "llama-quantize")
    if not os.path.exists(quantizer):
        raise FileNotFoundError(
            f"llama-quantize not found at {quantizer}. "
            "Build bitnet.cpp first."
        )

    logger.info("Quantizing to i2_s...")
    result = subprocess.run(
        [quantizer, f32_gguf, output_path, "I2_S", "1"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Quantization failed: {result.stderr}")
        raise RuntimeError(f"Quantization failed: {result.stderr[:500]}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"GGUF exported: {output_path} ({size_mb:.1f} MB)")
    return output_path
