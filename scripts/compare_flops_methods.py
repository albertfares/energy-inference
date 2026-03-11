import argparse
import csv
import sys
from pathlib import Path

import torch

# Allow running script without package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_inference.models import get_model


def _pct_diff(reference: float, other: float) -> float | str:
    if reference == 0:
        return ""
    return ((other - reference) / reference) * 100.0


@torch.no_grad()
def compare_once(
    *,
    model_name: str,
    batch: int,
    resolution: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    try:
        from thop import profile
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics-thop is not installed. Install with `pip install ultralytics-thop`."
        ) from exc
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "fvcore is not installed. Install with `pip install fvcore`."
        ) from exc

    model = get_model(model_name).to(device).eval()
    x = torch.randn(batch, 3, resolution, resolution, device=device)

    analysis = FlopCountAnalysis(model, x)
    analysis.unsupported_ops_warnings(False)
    fvcore_flops = float(analysis.total())

    # THOP returns MACs by default. Many papers/tools report FLOPs as 2 * MACs.
    thop_macs, _ = profile(model, inputs=(x,), verbose=False)
    thop_macs = float(thop_macs)
    thop_flops_x2 = thop_macs * 2.0

    return {
        "model": model_name,
        "batch": batch,
        "resolution": resolution,
        "fvcore_flops": fvcore_flops,
        "thop_macs": thop_macs,
        "thop_flops_x2": thop_flops_x2,
        "pct_diff_fvcore_vs_thop_macs": _pct_diff(fvcore_flops, thop_macs),
        "pct_diff_fvcore_vs_thop_flops_x2": _pct_diff(fvcore_flops, thop_flops_x2),
    }


def _write_csv(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "batch",
        "resolution",
        "fvcore_flops",
        "thop_macs",
        "thop_flops_x2",
        "pct_diff_fvcore_vs_thop_macs",
        "pct_diff_fvcore_vs_thop_flops_x2",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FLOPs accounting from fvcore and THOP for same model/input configs."
        )
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet50"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[224])
    parser.add_argument(
        "--out",
        type=str,
        default="results/flops/compare_fvcore_vs_thop.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but torch.cuda.is_available() is False")

    rows: list[dict[str, float | int | str]] = []
    for model_name in args.models:
        for batch in args.batches:
            for resolution in args.resolutions:
                row = compare_once(
                    model_name=model_name,
                    batch=batch,
                    resolution=resolution,
                    device=device,
                )
                rows.append(row)
                print(
                    f"{model_name} b={batch} r={resolution}: "
                    f"fvcore={row['fvcore_flops']:.0f}, "
                    f"thop_macs={row['thop_macs']:.0f}, "
                    f"diff={row['pct_diff_fvcore_vs_thop_macs']:.4f}%"
                )

    out_path = Path(args.out)
    _write_csv(rows, out_path)
    print(f"\nSaved comparison CSV to: {out_path}")


if __name__ == "__main__":
    main()
