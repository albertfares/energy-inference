import argparse
import sys
from pathlib import Path

# Allow running script without package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_inference.experiment_runner import (
    parse_experiments_csv,
    run_experiments_from_csv,
    summarize_config,
)
from energy_inference.pipeline import run_cpu_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark + feature extraction together and write one merged CSV."
    )
    parser.add_argument(
        "--experiments-csv",
        type=str,
        default=None,
        help="If set, run experiments from CSV and ignore single-run args below.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --experiments-csv, validate and print experiments without running.",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--experiment", type=str, default="cpu_full")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--backend", type=str, default="eager")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)

    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        choices=["model", "batch", "resolution", "precision"],
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["resnet18", "resnet50", "mobilenet_v3_large", "vit_b_16", "swin_t"],
    )
    parser.add_argument("--batches", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--resolutions", nargs="*", type=int, default=[224, 320, 384])
    parser.add_argument("--precisions", nargs="*", default=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--enable-energy", action="store_true", help="Enable INA3221 hardware power sampling."
    )

    args = parser.parse_args()
    if args.experiments_csv:
        if args.dry_run:
            parsed = parse_experiments_csv(args.experiments_csv)
            if not parsed:
                print("No enabled experiments found in CSV.")
                return
            print("Dry run (no benchmarks executed). Enabled experiments:")
            for row_number, cfg in parsed:
                print(f"- row={row_number} {summarize_config(cfg)}")
            return

        run_group_dir, completed = run_experiments_from_csv(args.experiments_csv)
        if not completed:
            print("No enabled experiments found in CSV.")
            return
        print(f"Run group directory: {run_group_dir}")
        print("Completed experiments:")
        for row_number, run_id, out_path in completed:
            print(f"- row={row_number} run_id={run_id} output={out_path}")
        return
    if not args.sweep:
        parser.error("--sweep is required when --experiments-csv is not provided.")

    out_path, run_id = run_cpu_sweep(
        mode="full",
        device=args.device,
        out=args.out,
        append=args.append,
        experiment=args.experiment,
        run_id=args.run_id,
        notes=args.notes,
        model=args.model,
        batch=args.batch,
        resolution=args.resolution,
        precision=args.precision,
        backend=args.backend,
        iters=args.iters,
        warmup=args.warmup,
        sweep=args.sweep,
        models=args.models,
        batches=args.batches,
        resolutions=args.resolutions,
        precisions=args.precisions,
        enable_energy=args.enable_energy,
    )
    print(f"Done. run_id={run_id} combined results saved to: {out_path}")


if __name__ == "__main__":
    main()
