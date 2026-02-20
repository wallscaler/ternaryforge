# TernaryForge

Convert any HuggingFace model to CPU-runnable ternary format. Drop in a model ID, get a CPU-optimized ternary version — no GPUs required for inference.

## What Is This?

Standard AI models store each weight as a precise decimal number (like 0.0372841). TernaryForge crushes those weights down to just three values: **-1, 0, or +1**. This extreme compression means the model can run fast on regular CPUs using simple addition instead of expensive multiplication.

The tradeoff: you lose precision. The project explores whether that tradeoff is worth it, and how to minimize the damage.

**See [ternaryforge-overview.excalidraw](./ternaryforge-overview.excalidraw)** for a visual walkthrough (open in [excalidraw.com](https://excalidraw.com)).

## Status: Phase 0 — Validation

We've validated the end-to-end pipeline works. Key finding: **post-training ternarization of standard models produces gibberish**. Models need to be *trained* as ternary from the start (like BitNet b1.58) to produce quality output. The conversion pipeline itself is solid — the bottleneck is input model quality.

### Phase 0 Results

| Metric | Result |
|--------|--------|
| BitNet 2B4T on M4 CPU | ~27 tok/s |
| Post-training ternarized Qwen 0.5B | Gibberish output |
| Calibration improvement | ~7% MSE reduction (not enough) |
| Pipeline status | End-to-end working |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Phase 0 validation (no GPU needed)
python validate.py

# Full pipeline: convert a HuggingFace model → ternary → GGUF → benchmark
python -m ternaryforge.cli run Qwen/Qwen2.5-0.5B --calibrate

# Or step by step:
python -m ternaryforge.cli convert Qwen/Qwen2.5-0.5B --calibrate
python -m ternaryforge.cli export output/ternary
python -m ternaryforge.cli validate output/ternary --gguf output/model.gguf
```

## How It Works — The Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   1. CONVERT     │────>│  2. CALIBRATE    │────>│   3. EXPORT      │────>│  4. VALIDATE     │
│                  │     │    (optional)     │     │                  │     │                  │
│ HuggingFace model│     │ Optimize per-    │     │ Ternary weights  │     │ Weight stats,    │
│ → ternary {-1,0,1}    │ layer scales via  │     │ → GGUF format    │     │ inference speed, │
│ with per-channel │     │ WikiText-2 data  │     │ for bitnet.cpp   │     │ memory, quality  │
│ absmax scaling   │     │                  │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Stage 1: Convert (`ternaryforge/convert.py`)
Loads a HuggingFace model and ternarizes weight matrices to {-1, 0, +1}. Uses per-channel absmax scaling (not naive `sign()`) to preserve relative magnitude across output channels. Skips embeddings, layer norms, and biases. Outputs `ternary_weights.pt` (int8) + `scales.pt` (fp16).

### Stage 2: Calibrate (`ternaryforge/calibrate.py`)
Runs a single forward pass over WikiText-2 samples to capture layer activations, then grid-searches scale multipliers (0.5x–3.0x) per layer to minimize reconstruction MSE. Efficient — only one forward pass regardless of search grid size.

### Stage 3: Export (`ternaryforge/export_gguf.py`)
Reconstructs float32 weights (`ternary * scale`), maps HuggingFace tensor names to GGUF/bitnet convention, injects required sub-norm tensors, and writes a GGUF file compatible with bitnet.cpp's CPU inference runtime.

### Stage 4: Validate (`ternaryforge/validate.py`)
Analyzes weight distribution (sparsity, ternary balance), runs inference benchmarks via bitnet.cpp's `llama-cli`, and reports tokens/sec, memory usage, and output quality.

## Project Structure

```
ternaryforge/
├── README.md                # You are here
├── requirements.txt         # Python dependencies
├── validate.py              # Phase 0 standalone validation script
├── ternaryforge-overview.excalidraw  # Visual architecture diagram
├── ternaryforge/            # Core library
│   ├── cli.py               # CLI: convert, export, validate, run
│   ├── convert.py           # Weight ternarization (absmax scaling)
│   ├── calibrate.py         # Activation-aware scale optimization
│   ├── export_gguf.py       # GGUF format export for bitnet.cpp
│   └── validate.py          # Quality & speed benchmarking
├── bitnet-cpp/              # Microsoft's bitnet.cpp inference runtime (submodule)
├── benchmarks/              # Benchmark results and scripts
└── output/                  # Generated artifacts (ternary weights, GGUF files)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| ML Framework | PyTorch 2.0+ |
| Model Hub | HuggingFace Transformers |
| Inference | bitnet.cpp (C++, from Microsoft Research) |
| Export Format | GGUF (via gguf-py from llama.cpp) |
| Calibration Data | WikiText-2 |
| CPU Support | Apple Silicon (M-series), x86 (AVX2), ARM (NEON) |

## Key Insight: Born Ternary vs Forced Ternary

The biggest learning from Phase 0: **you can't just crush a standard model to 3 values and expect it to work.**

Models trained from scratch with ternary constraints (like BitNet b1.58) learn to encode information within those limits. Post-training ternarization of a model that was trained with millions of precision levels destroys too much information — calibration can only recover a fraction.

Think of it like a painter: one who learns to paint with only 3 colors from day one creates beautiful constrained art. Taking a Rembrandt and reducing it to 3 colors after the fact just makes a mess.

## Next Steps

1. Wire calibration module into the main `run` pipeline
2. Implement I2_S packing in Python (bypass llama-quantize sub-norm bug)
3. Test with models pre-trained for ternary (BitNet b1.58, Llama-1.58bit)
4. Go/no-go on Phase 1 based on results with ternary-native models

## License

MIT
