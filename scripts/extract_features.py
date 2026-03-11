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
        description="Extract model/config features (params, FLOPs, metadata) to CSV."
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--experiment", type=str, default="cpu_features")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--backend", type=str, default="eager")

    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        choices=["model", "batch", "resolution"],
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["resnet18", "resnet50", "mobilenet_v3_large", "vit_b_16", "swin_t"],
    )
    parser.add_argument("--batches", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--resolutions", nargs="*", type=int, default=[224, 320, 384])

    args = parser.parse_args()
    out_path, run_id = run_cpu_sweep(
        mode="features",
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
        iters=200,
        warmup=30,
        sweep=args.sweep,
        models=args.models,
        batches=args.batches,
        resolutions=args.resolutions,
    )
    print(f"Done. run_id={run_id} features saved to: {out_path}")


if __name__ == "__main__":
    main()

