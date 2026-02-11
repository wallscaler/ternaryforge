"""
TernaryForge CLI

Usage:
    ternaryforge convert <model_id> [--output <dir>] [--method absmax]
    ternaryforge export <ternary_dir> [--output <path>] [--bitnet-cpp <dir>]
    ternaryforge validate <ternary_dir> [--gguf <path>] [--bitnet-cpp <dir>]
    ternaryforge run <model_id> [--output <dir>]  # full pipeline
"""

import argparse
import logging
import os
import sys


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_convert(args):
    from ternaryforge.convert import convert_model

    result = convert_model(
        model_id=args.model,
        output_dir=args.output,
        method=args.method,
        calibrate=getattr(args, "calibrate", False),
        calibration_samples=getattr(args, "calibration_samples", 32),
    )
    print(f"\nConversion complete:")
    print(f"  Ternarized: {result.ternary_params:,} params")
    print(f"  Kept FP16:  {result.skipped_params:,} params")
    print(f"  Sparsity:   {result.sparsity:.1%}")
    print(f"  Time:       {result.elapsed_sec:.1f}s")
    print(f"  Output:     {result.output_dir}")


def cmd_export(args):
    from ternaryforge.export_gguf import export_to_gguf

    gguf_path = export_to_gguf(
        ternary_dir=args.ternary_dir,
        output_path=args.output,
        bitnet_cpp_dir=args.bitnet_cpp,
    )
    print(f"\nGGUF exported: {gguf_path}")
    size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")


def cmd_validate(args):
    from ternaryforge.validate import full_validation

    report = full_validation(
        ternary_dir=args.ternary_dir,
        gguf_path=args.gguf,
        bitnet_cpp_dir=args.bitnet_cpp,
    )
    print(f"\nValidation complete:")
    ws = report.get("weight_stats", {})
    print(f"  Sparsity: {ws.get('sparsity', 0):.1%}")
    dist = ws.get("distribution", {})
    print(f"  -1: {dist.get('-1', 0):.1%}  0: {dist.get('0', 0):.1%}  +1: {dist.get('+1', 0):.1%}")
    if "benchmark" in report and "avg_generation_tok_s" in report["benchmark"]:
        print(f"  Speed: {report['benchmark']['avg_generation_tok_s']} tok/s")


def cmd_run(args):
    """Full pipeline: convert → export → validate."""
    from ternaryforge.convert import convert_model
    from ternaryforge.export_gguf import export_to_gguf
    from ternaryforge.validate import full_validation

    output_dir = args.output or f"output/{args.model.replace('/', '_')}"
    ternary_dir = os.path.join(output_dir, "ternary")
    gguf_path = os.path.join(output_dir, "ggml-model-i2_s.gguf")

    print(f"TernaryForge: Converting {args.model}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Step 1: Convert
    print("\n[1/3] Ternarizing weights...")
    result = convert_model(
        model_id=args.model,
        output_dir=ternary_dir,
        method=args.method,
        calibrate=getattr(args, "calibrate", False),
        calibration_samples=getattr(args, "calibration_samples", 32),
    )
    print(f"  Done: {result.ternary_params:,} params ternarized, "
          f"{result.sparsity:.1%} sparsity")

    # Step 2: Export
    print("\n[2/3] Exporting to GGUF...")
    try:
        export_to_gguf(
            ternary_dir=ternary_dir,
            output_path=gguf_path,
            bitnet_cpp_dir=args.bitnet_cpp,
        )
        size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
        print(f"  Done: {gguf_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  Export failed: {e}")
        print("  (Ternary weights saved — export can be retried)")
        gguf_path = None

    # Step 3: Validate
    print("\n[3/3] Validating...")
    report = full_validation(
        ternary_dir=ternary_dir,
        gguf_path=gguf_path,
        bitnet_cpp_dir=args.bitnet_cpp,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Ternary weights: {ternary_dir}")
    if gguf_path and os.path.exists(gguf_path):
        print(f"  GGUF model:      {gguf_path}")
    print(f"  Report:          {ternary_dir}/validation_report.json")


def main():
    parser = argparse.ArgumentParser(
        description="TernaryForge: Convert any model to CPU-runnable ternary format",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # convert
    p_convert = subparsers.add_parser("convert", help="Ternarize model weights")
    p_convert.add_argument("model", help="HuggingFace model ID or local path")
    p_convert.add_argument("-o", "--output", default="output/ternary",
                           help="Output directory")
    p_convert.add_argument("-m", "--method", default="absmax",
                           choices=["absmax", "naive"],
                           help="Ternarization method")
    p_convert.add_argument("--calibrate", action="store_true",
                           help="Run calibration to optimize scales (slower but better quality)")
    p_convert.add_argument("--calibration-samples", type=int, default=32,
                           help="Number of calibration samples")

    # export
    p_export = subparsers.add_parser("export", help="Export to GGUF")
    p_export.add_argument("ternary_dir", help="Directory with ternary weights")
    p_export.add_argument("-o", "--output", default="output/model.gguf",
                          help="Output GGUF path")
    p_export.add_argument("--bitnet-cpp", default=None,
                          help="Path to bitnet.cpp repo")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate converted model")
    p_validate.add_argument("ternary_dir", help="Directory with ternary weights")
    p_validate.add_argument("--gguf", default=None, help="GGUF file to benchmark")
    p_validate.add_argument("--bitnet-cpp", default=None,
                            help="Path to bitnet.cpp repo")

    # run (full pipeline)
    p_run = subparsers.add_parser("run", help="Full pipeline: convert + export + validate")
    p_run.add_argument("model", help="HuggingFace model ID or local path")
    p_run.add_argument("-o", "--output", default=None, help="Output directory")
    p_run.add_argument("-m", "--method", default="absmax",
                       choices=["absmax", "naive"])
    p_run.add_argument("--calibrate", action="store_true",
                       help="Run calibration to optimize scales")
    p_run.add_argument("--calibration-samples", type=int, default=32,
                       help="Number of calibration samples")
    p_run.add_argument("--bitnet-cpp", default=None,
                       help="Path to bitnet.cpp repo")

    args = parser.parse_args()
    setup_logging(args.verbose)

    commands = {
        "convert": cmd_convert,
        "export": cmd_export,
        "validate": cmd_validate,
        "run": cmd_run,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
