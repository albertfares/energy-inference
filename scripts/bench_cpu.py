import argparse
import sys
from pathlib import Path

# Allow running script without package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_inference.pipeline import run_cpu_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU/CUDA inference benchmark sweep for vision models."
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--experiment", type=str, default="cpu_bench")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)

    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        choices=["model", "batch", "resolution"],
    )
    parser.add_argument("--models", nargs="*", default=["resnet18", "resnet50"])
    parser.add_argument("--batches", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--resolutions", nargs="*", type=int, default=[224, 320, 384])
    parser.add_argument("--enable-energy", action="store_true", help="Enable INA3221 hardware power sampling.")

    args = parser.parse_args()
    out_path, run_id = run_cpu_sweep(
        mode="bench",
        device=args.device,
        out=args.out,
        append=args.append,
        experiment=args.experiment,
        run_id=args.run_id,
        notes=args.notes,
        model=args.model,
        batch=args.batch,
        resolution=args.resolution,
        precision="fp32",
        backend="eager",
        iters=args.iters,
        warmup=args.warmup,
        sweep=args.sweep,
        models=args.models,
        batches=args.batches,
        resolutions=args.resolutions,
        enable_energy=args.enable_energy,
    )
    print(f"Done. run_id={run_id} results saved to: {out_path}")


if __name__ == "__main__":
    main()

