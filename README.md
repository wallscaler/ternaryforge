# TernaryForge

Convert any HuggingFace model to CPU-runnable ternary format.

Drop in a model ID, get a CPU-optimized ternary version. No GPUs required for inference.

## Status: Phase 0 — Validation

Currently validating that ternary inference on CPU is viable by benchmarking existing pre-built ternary models.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Phase 0 validation (no GPU needed)
python validate.py
```

## How It Works

1. **Convert** — Transform FP16/BF16 model weights to ternary {-1, 0, +1} format
2. **Calibrate** — Run calibration pass to minimize output divergence
3. **Serve** — Run inference on CPU using addition-only operations

## Phase 0 Goals

- [ ] Benchmark BitNet 2B4T on Apple M4 CPU
- [ ] Measure tokens/sec, memory usage, time-to-first-token
- [ ] Compare quality against original FP16 baseline
- [ ] Go/no-go decision for Phase 1

## Architecture

```
ternaryforge/
├── validate.py        # Phase 0 validation script
├── convert/           # Model conversion pipeline (Phase 1)
├── serve/             # CPU inference server (Phase 2)
└── benchmarks/        # Quality & speed benchmarks
```

## License

MIT
