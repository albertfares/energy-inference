import argparse
import csv
import os
import re
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
from tqdm import tqdm

# Allow running script without package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_inference.benchmarking import bench_once
from energy_inference.features import compute_flops, count_parameters, infer_model_family, infer_model_task
from energy_inference.io_utils import append_csv_row
from energy_inference.models import get_model
from energy_inference.tools.INA3221Sampler import INA3221Sampler


DEFAULT_MODELS = [
    "resnet18",
    "resnet50",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "googlenet",
    "shufflenet_v2_x1_0",
    "vgg16",
    "vit_b_16",
    "swin_t",
    "ssdlite",
    "yolo",
]
DEFAULT_BATCHES = [1, 2, 4]
DEFAULT_RESOLUTIONS = [160, 192, 224, 256, 320, 384, 448, 512, 576, 640] # use resolutions supported by camera and most used
DEFAULT_PRECISIONS = ["fp32", "fp16", "bf16"]

FIELDNAMES_FULL = [
    "run_id",
    "experiment",
    "notes",
    "timestamp",
    "device",
    "sweep_param",
    "model_family",
    "model",
    "model_task",
    "batch",
    "resolution",
    "precision",
    "backend",
    "iters",
    "warmup",
    "num_params",
    "macs_total",
    "flops_total",
    "flops_total_strict",
    "flops_per_sample",
    "unsupported_ops_count",
    "latency_ms",
    "fps",
    "energy_cpu_J",
    "energy_gpu_J",
    "energy_io_J",
    "power_total_W",
    "status",
    "error_msg",
]


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    now = datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{now.microsecond // 1000:03d}"


def _sanitize_slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return normalized.strip("_") or "default"


def _resolve_output_path(out: str | None, experiment: str, run_id: str) -> str:
    if out:
        return out
    exp_slug = _sanitize_slug(experiment)
    return str(Path("results") / "runs" / f"full_cartesian_{exp_slug}_{run_id}.csv")


def _load_existing_combos(
    csv_path: Path,
    *,
    rerun_failed: bool,
) -> set[tuple[str, int, int, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    combos: set[tuple[str, int, int, str]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                model = str(row.get("model", "")).strip()
                batch = int(row.get("batch", ""))
                resolution = int(row.get("resolution", ""))
                precision = str(row.get("precision", "")).strip()
                status = str(row.get("status", "")).strip().lower()
            except Exception:
                continue

            if not model or not precision:
                continue
            if rerun_failed and status == "failed":
                continue
            combos.add((model, batch, resolution, precision))
    return combos


def _infer_existing_run_id(csv_path: Path) -> str | None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = str(row.get("run_id", "")).strip()
            if value:
                return value
    return None


def _round_float_or_empty(value: object) -> float | str:
    return round(value, 4) if isinstance(value, float) else ""


def _compute_total_power_w(
    *,
    energy_dict: dict[str, float],
    latency_ms: float | str,
    iters: int,
) -> float | str:
    if not isinstance(latency_ms, float):
        return ""
    total_s = (latency_ms / 1000.0) * max(iters, 1)
    if total_s <= 0:
        return ""

    total_energy_j = 0.0
    has_energy = False
    for key in ("cpu", "gpu", "io"):
        value = energy_dict.get(key)
        if isinstance(value, float):
            total_energy_j += value
            has_energy = True
    if not has_energy:
        return ""
    return round(total_energy_j / total_s, 4)


@torch.no_grad()
def run_cartesian(
    *,
    device: str,
    out: str | None,
    experiment: str,
    run_id: str | None,
    notes: str,
    backend: str,
    iters: int,
    warmup: int,
    models: list[str],
    batches: list[int],
    resolutions: list[int],
    precisions: list[str],
    enable_energy: bool,
    keep_power_trace: bool,
    resume_csv: str | None,
    rerun_failed: bool,
) -> tuple[str, str]:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but torch.cuda.is_available() is False")

    if resume_csv:
        out_path = resume_csv
        resolved_run_id = run_id or _infer_existing_run_id(Path(out_path)) or _resolve_run_id(None)
    else:
        resolved_run_id = _resolve_run_id(run_id)
        out_path = _resolve_output_path(out, experiment, resolved_run_id)
    power_trace_path = Path(out_path.replace(".csv", "_power_trace.csv"))

    sampler = None
    if enable_energy:
        power_trace_path.parent.mkdir(parents=True, exist_ok=True)
        sampler = INA3221Sampler(
            exe_candidate="src/energy_inference/tools/sample_ina3221",
            hz=1000,
            power_csv=str(power_trace_path),
            hw="all",
        )

    model_cache: dict[str, torch.nn.Module] = {}
    all_combos = list(product(models, batches, resolutions, precisions))
    existing_combos = _load_existing_combos(
        Path(out_path),
        rerun_failed=rerun_failed,
    )
    pending_combos = [
        combo
        for combo in all_combos
        if combo not in existing_combos
    ]
    print(
        f"Total combos={len(all_combos)} existing={len(existing_combos)} pending={len(pending_combos)}"
    )
    for model_name, batch, resolution, precision in tqdm(
        pending_combos, desc="Running full cartesian benchmark", unit="cfg"
    ):
        if model_name not in model_cache:
            model_cache[model_name] = get_model(model_name).to(torch_device).eval()
        curr_model = model_cache[model_name]

        status = "ok"
        error_msg = ""
        num_params: int | str = ""
        macs_total: float | str = ""
        flops_total_strict: float | str = ""
        flops_total: float | str = ""
        flops_per_sample: float | str = ""
        unsupported_ops_count: int | str = ""
        latency_ms: float | str = ""
        fps: float | str = ""
        energy_dict: dict[str, float] = {}

        try:
            num_params = count_parameters(curr_model)
            flops_model = get_model(model_name).to(torch_device).eval()
            macs_total, flops_total_strict, unsupported_ops_count = compute_flops(
                model=flops_model,
                batch=batch,
                resolution=resolution,
                device=torch_device,
            )
            del flops_model
            flops_total = flops_total_strict
            flops_per_sample = flops_total_strict / max(batch, 1)

            latency_ms, fps, energy_dict = bench_once(
                model=curr_model,
                batch=batch,
                resolution=resolution,
                iters=iters,
                warmup=warmup,
                device=torch_device,
                precision=precision,
                sampler=sampler,
            )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_msg = str(exc)

        row = {
            "run_id": resolved_run_id,
            "experiment": experiment,
            "notes": notes,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device": str(torch_device),
            "sweep_param": "cartesian",
            "model_family": infer_model_family(model_name),
            "model": model_name,
            "model_task": infer_model_task(model_name),
            "batch": batch,
            "resolution": resolution,
            "precision": precision,
            "backend": backend,
            "iters": iters,
            "warmup": warmup,
            "num_params": num_params,
            "macs_total": _round_float_or_empty(macs_total),
            "flops_total": _round_float_or_empty(flops_total),
            "flops_total_strict": _round_float_or_empty(flops_total_strict),
            "flops_per_sample": _round_float_or_empty(flops_per_sample),
            "unsupported_ops_count": unsupported_ops_count,
            "latency_ms": _round_float_or_empty(latency_ms),
            "fps": _round_float_or_empty(fps),
            "energy_cpu_J": energy_dict.get("cpu", ""),
            "energy_gpu_J": energy_dict.get("gpu", ""),
            "energy_io_J": energy_dict.get("io", ""),
            "power_total_W": _compute_total_power_w(
                energy_dict=energy_dict,
                latency_ms=latency_ms,
                iters=iters,
            ),
            "status": status,
            "error_msg": error_msg,
        }
        append_csv_row(out_path, FIELDNAMES_FULL, row)

    if enable_energy and (not keep_power_trace) and power_trace_path.exists():
        os.remove(power_trace_path)

    return out_path, resolved_run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exhaustive full benchmark over all model/batch/resolution/precision combinations."
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="comprehensive_cartesian")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--notes", type=str, default="exhaustive cartesian benchmark")
    parser.add_argument("--backend", type=str, default="eager")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)

    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--batches", nargs="*", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--resolutions", nargs="*", type=int, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--precisions", nargs="*", default=DEFAULT_PRECISIONS)
    parser.add_argument(
        "--resume-csv",
        type=str,
        default=None,
        help="Resume into an existing cartesian CSV by skipping already present combinations.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="With --resume-csv, rerun combos previously marked as failed.",
    )

    parser.add_argument("--enable-energy", action="store_true")
    parser.add_argument(
        "--keep-power-trace",
        action="store_true",
        help="Keep *_power_trace.csv file (default deletes it after CSV is saved).",
    )
    args = parser.parse_args()

    out_path, run_id = run_cartesian(
        device=args.device,
        out=args.out,
        experiment=args.experiment,
        run_id=args.run_id,
        notes=args.notes,
        backend=args.backend,
        iters=args.iters,
        warmup=args.warmup,
        models=args.models,
        batches=args.batches,
        resolutions=args.resolutions,
        precisions=args.precisions,
        enable_energy=args.enable_energy,
        keep_power_trace=args.keep_power_trace,
        resume_csv=args.resume_csv,
        rerun_failed=args.rerun_failed,
    )
    print(f"Done. run_id={run_id} results saved to: {out_path}")


if __name__ == "__main__":
    main()
