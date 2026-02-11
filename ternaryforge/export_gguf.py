"""
TernaryForge GGUF Exporter — Stage 4

Standalone GGUF writer that bypasses bitnet.cpp's convert-hf-to-gguf-bitnet.py
(which only supports BitnetForCausalLM and LlamaForCausalLM).

Strategy:
1. Write F32 GGUF directly using the gguf library, declaring "bitnet" architecture
2. Inject identity attn_sub_norm and ffn_sub_norm tensors (required by bitnet.cpp)
3. Quantize F32 → i2_s using llama-quantize from bitnet.cpp

This allows ANY HuggingFace model to be exported to bitnet.cpp format.
"""

import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# HuggingFace → GGUF tensor name mapping
HF_TO_GGUF_GLOBAL = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

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
    if hf_name in HF_TO_GGUF_GLOBAL:
        return HF_TO_GGUF_GLOBAL[hf_name]

    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])
        if rest in LAYER_MAP:
            return f"blk.{layer_idx}.{LAYER_MAP[rest]}"

    return None


def _reconstruct_weight(w_tern: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    """Reconstruct float32 weight from ternary {-1,0,+1} * scale."""
    if scale.dim() == 0:
        result = w_tern.float() * scale.float()
    elif scale.dim() == 1 and w_tern.dim() == 2:
        result = w_tern.float() * scale.float().unsqueeze(1)
    else:
        result = w_tern.float() * scale.float()
    return result.numpy().astype(np.float32)


def _write_tokenizer(gguf_writer, ternary_dir: str, target_vocab_size: int = 0):
    """Write tokenizer metadata to GGUF."""
    tokenizer_path = os.path.join(ternary_dir, "tokenizer.json")
    tokenizer_config_path = os.path.join(ternary_dir, "tokenizer_config.json")

    if not os.path.exists(tokenizer_path):
        logger.warning("No tokenizer.json found, skipping tokenizer metadata")
        return

    with open(tokenizer_path) as f:
        tokenizer_data = json.load(f)

    # Determine tokenizer model type
    tokenizer_model = "gpt2"  # default for BPE
    tok_config = {}
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path) as f:
            tok_config = json.load(f)
        tokenizer_class = tok_config.get("tokenizer_class", "")
        if "Llama" in tokenizer_class:
            tokenizer_model = "llama"

    gguf_writer.add_tokenizer_model(tokenizer_model)

    # Build complete token list: base vocab + added tokens, padded to target size
    base_vocab = tokenizer_data.get("model", {}).get("vocab", {})
    added_tokens = tokenizer_data.get("added_tokens", [])

    # Merge base vocab and added tokens into a single dict
    all_tokens = dict(base_vocab)
    special_token_ids = set()
    for t in added_tokens:
        all_tokens[t["content"]] = t["id"]
        if t.get("special", False):
            special_token_ids.add(t["id"])

    if not all_tokens:
        logger.warning("No tokens found in tokenizer")
        return

    max_id = max(all_tokens.values())
    n_tokens = max(max_id + 1, target_vocab_size)

    # Build arrays indexed by token ID
    tokens = [b"" for _ in range(n_tokens)]
    scores = [0.0] * n_tokens
    token_types = [1] * n_tokens  # NORMAL

    for text, idx in all_tokens.items():
        if idx < n_tokens:
            tokens[idx] = text.encode("utf-8", errors="replace")

    # Mark special tokens
    for sid in special_token_ids:
        if sid < n_tokens:
            token_types[sid] = 3  # CONTROL

    # Pad empty slots with placeholder tokens
    for i in range(n_tokens):
        if not tokens[i]:
            tokens[i] = f"<|reserved_{i}|>".encode("utf-8")
            token_types[i] = 3  # CONTROL

    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(token_types)

    # BOS/EOS tokens — find from config or special tokens
    bos_id = tok_config.get("bos_token_id")
    eos_id = tok_config.get("eos_token_id")

    # If not in config, find from special tokens
    if eos_id is None:
        eos_token = tok_config.get("eos_token")
        if isinstance(eos_token, str) and eos_token in all_tokens:
            eos_id = all_tokens[eos_token]
        elif isinstance(eos_token, dict) and eos_token.get("content") in all_tokens:
            eos_id = all_tokens[eos_token["content"]]

    if bos_id is None:
        # Many models use the same token for BOS and EOS, or use <|im_start|>
        bos_token = tok_config.get("bos_token")
        if isinstance(bos_token, str) and bos_token in all_tokens:
            bos_id = all_tokens[bos_token]
        elif eos_id is not None:
            bos_id = eos_id  # Fallback: use EOS as BOS

    if bos_id is not None:
        gguf_writer.add_bos_token_id(bos_id)
    if eos_id is not None:
        gguf_writer.add_eos_token_id(eos_id)

    # Merges for BPE
    merges = tokenizer_data.get("model", {}).get("merges", [])
    if merges:
        merge_strs = []
        for m in merges:
            if isinstance(m, list):
                merge_strs.append(" ".join(m).encode("utf-8", errors="replace"))
            else:
                merge_strs.append(m.encode("utf-8", errors="replace"))
        gguf_writer.add_token_merges(merge_strs)

    logger.info(f"Tokenizer: {tokenizer_model}, {n_tokens} tokens "
                f"(base={len(base_vocab)}, added={len(added_tokens)}, "
                f"BOS={bos_id}, EOS={eos_id})")


def export_to_gguf_direct(
    ternary_dir: str,
    output_path: str,
    bitnet_cpp_dir: str | None = None,
) -> str:
    """
    Write F32 GGUF directly with bitnet architecture.

    Creates a GGUF file that bitnet.cpp can load by:
    1. Declaring "bitnet" architecture
    2. Writing reconstructed float32 weights
    3. Injecting identity attn_sub_norm / ffn_sub_norm tensors
    """
    # Find gguf library from bitnet-cpp
    if bitnet_cpp_dir is None:
        repo_root = Path(__file__).parent.parent
        bitnet_cpp_dir = str(repo_root / "bitnet-cpp")

    gguf_py_path = os.path.join(bitnet_cpp_dir, "3rdparty", "llama.cpp", "gguf-py")
    if os.path.exists(gguf_py_path):
        import sys
        if gguf_py_path not in sys.path:
            sys.path.insert(0, gguf_py_path)

    import gguf

    # Load ternary weights and scales
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

    # Load model config
    config_path = os.path.join(ternary_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    hidden_size = config.get("hidden_size", 2048)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    num_layers = config.get("num_hidden_layers", 24)
    num_heads = config.get("num_attention_heads", 16)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    vocab_size = config.get("vocab_size", 32000)
    max_position = config.get("max_position_embeddings", 4096)
    rope_theta = config.get("rope_theta", 10000.0)
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)
    head_dim = hidden_size // num_heads

    logger.info(f"Model config: {hidden_size}d, {num_layers}L, {num_heads}H, "
                f"{vocab_size}V, {intermediate_size}I")

    # Create GGUF writer with bitnet-b1.58 architecture
    # "bitnet-b1.58" supports output tensor and sub-norms, unlike "bitnet" which
    # uses tied embeddings and doesn't support arbitrary HF models
    f32_path = output_path.replace(".gguf", "-f32.gguf")
    gguf_writer = gguf.GGUFWriter(f32_path, "bitnet-b1.58")

    # Write metadata
    gguf_writer.add_name("ternaryforge-model")
    gguf_writer.add_context_length(max_position)
    gguf_writer.add_embedding_length(hidden_size)
    gguf_writer.add_block_count(num_layers)
    gguf_writer.add_feed_forward_length(intermediate_size)
    gguf_writer.add_head_count(num_heads)
    gguf_writer.add_head_count_kv(num_kv_heads)
    gguf_writer.add_rope_dimension_count(head_dim)
    gguf_writer.add_rope_freq_base(rope_theta)
    gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
    gguf_writer.add_vocab_size(vocab_size)
    gguf_writer.add_file_type(0)  # F32

    # Write tokenizer (padded to match config vocab_size)
    _write_tokenizer(gguf_writer, ternary_dir, target_vocab_size=vocab_size)

    # Write tensors
    tensors_written = 0
    layers_seen = set()

    for hf_name, w in ternary_weights.items():
        gguf_name = hf_name_to_gguf(hf_name)
        if gguf_name is None:
            logger.debug(f"Skipping unmapped tensor: {hf_name}")
            continue

        # Reconstruct float32 from ternary * scale, or use as-is for non-ternary
        if hf_name in scales:
            data = _reconstruct_weight(w, scales[hf_name])
        else:
            data = w.float().numpy().astype(np.float32)

        gguf_writer.add_tensor(gguf_name, data)
        tensors_written += 1

        # Track which layers we've seen
        if hf_name.startswith("model.layers."):
            layer_idx = int(hf_name.split(".")[2])
            layers_seen.add(layer_idx)

    # Inject identity sub-norm tensors for each layer
    # These are required by bitnet.cpp but don't exist in standard models
    logger.info(f"Injecting identity sub-norm tensors for {len(layers_seen)} layers")
    for layer_idx in sorted(layers_seen):
        # attn_sub_norm: shape [hidden_size], identity = all 1s
        attn_sub_norm = np.ones(hidden_size, dtype=np.float32)
        gguf_writer.add_tensor(
            f"blk.{layer_idx}.attn_sub_norm.weight",
            attn_sub_norm,
        )
        tensors_written += 1

        # ffn_sub_norm: shape [intermediate_size], identity = all 1s
        ffn_sub_norm = np.ones(intermediate_size, dtype=np.float32)
        gguf_writer.add_tensor(
            f"blk.{layer_idx}.ffn_sub_norm.weight",
            ffn_sub_norm,
        )
        tensors_written += 1

    # Finalize
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    size_mb = os.path.getsize(f32_path) / (1024 * 1024)
    logger.info(f"F32 GGUF written: {f32_path} ({size_mb:.1f} MB, {tensors_written} tensors)")

    return f32_path


def quantize_to_i2s(
    f32_gguf_path: str,
    output_path: str,
    bitnet_cpp_dir: str | None = None,
) -> str:
    """Quantize F32 GGUF to i2_s using llama-quantize."""
    if bitnet_cpp_dir is None:
        repo_root = Path(__file__).parent.parent
        bitnet_cpp_dir = str(repo_root / "bitnet-cpp")

    quantizer = os.path.join(bitnet_cpp_dir, "build", "bin", "llama-quantize")
    if not os.path.exists(quantizer):
        raise FileNotFoundError(
            f"llama-quantize not found at {quantizer}. "
            "Build bitnet.cpp first."
        )

    logger.info(f"Quantizing {f32_gguf_path} → {output_path} (I2_S)")
    result = subprocess.run(
        [quantizer, f32_gguf_path, output_path, "I2_S"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Quantization stderr: {result.stderr}")
        raise RuntimeError(f"Quantization failed: {result.stderr[:500]}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Quantized GGUF: {output_path} ({size_mb:.1f} MB)")
    return output_path


def export_to_gguf(
    ternary_dir: str,
    output_path: str,
    bitnet_cpp_dir: str | None = None,
) -> str:
    """
    Full export pipeline: ternary weights → GGUF file.

    Steps:
    1. Write F32 GGUF directly with bitnet architecture
    2. Quantize F32 → i2_s/TL1 using llama-quantize

    Args:
        ternary_dir: Directory with ternary_weights.pt and scales.pt
        output_path: Where to write the final .gguf
        bitnet_cpp_dir: Path to bitnet.cpp repo
    """
    if bitnet_cpp_dir is None:
        repo_root = Path(__file__).parent.parent
        bitnet_cpp_dir = str(repo_root / "bitnet-cpp")

    # Step 1: Write F32 GGUF
    f32_path = export_to_gguf_direct(
        ternary_dir=ternary_dir,
        output_path=output_path,
        bitnet_cpp_dir=bitnet_cpp_dir,
    )

    # Step 2: Use F32 GGUF directly
    # NOTE: llama-quantize I2_S drops sub-norm tensors that bitnet.cpp requires.
    # The F32 GGUF with ternary values works correctly with bitnet.cpp's optimized
    # kernels. Future work: write I2_S packing directly in Python.
    if f32_path != output_path:
        import shutil
        shutil.move(f32_path, output_path)
        logger.info(f"GGUF ready: {output_path}")

    return output_path
