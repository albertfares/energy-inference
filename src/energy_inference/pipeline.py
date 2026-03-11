import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from energy_inference.benchmarking import bench_once
from energy_inference.features import compute_flops, count_parameters, infer_model_family
from energy_inference.io_utils import append_csv_row
from energy_inference.models import get_model
from energy_inference.tools.INA3221Sampler import INA3221Sampler

RunMode = Literal["bench", "features", "full"]

FIELDNAMES_BY_MODE: dict[RunMode, list[str]] = {
    "bench": [
        "run_id",
        "experiment",
        "notes",
        "timestamp",
        "device",
        "sweep_param",
        "model",
        "batch",
        "resolution",
        "iters",
        "warmup",
        "latency_ms",
        "fps",
        "energy_cpu_J",
        "energy_gpu_J",
        "energy_io_J",
        "power_total_W",
    ],
    "features": [
        "run_id",
        "experiment",
        "notes",
        "timestamp",
        "device",
        "sweep_param",
        "model_family",
        "model",
        "batch",
        "resolution",
        "precision",
        "backend",
        "num_params",
        "flops_total",
        "flops_per_sample",
        "unsupported_ops_count",
        "status",
        "error_msg",
    ],
    "full": [
        "run_id",
        "experiment",
        "notes",
        "timestamp",
        "device",
        "sweep_param",
        "model_family",
        "model",
        "batch",
        "resolution",
        "precision",
        "backend",
        "iters",
        "warmup",
        "num_params",
        "flops_total",
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
    ],
}

DESC_PREFIX_BY_MODE: dict[RunMode, str] = {
    "bench": "Benchmarking",
    "features": "Extracting",
    "full": "Running",
}


def _sanitize_slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return normalized.strip("_") or "default"


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    now = datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{now.microsecond // 1000:03d}"


def _resolve_output_path(
    *,
    mode: RunMode,
    out: str | None,
    experiment: str,
    run_id: str,
    append: bool,
) -> str:
    if out:
        out_path = Path(out)
        if append:
            return str(out_path)
        if out_path.exists() and out_path.stat().st_size > 0:
            return str(out_path.with_name(f"{out_path.stem}_{run_id}{out_path.suffix}"))
        return str(out_path)

    exp_slug = _sanitize_slug(experiment)
    filename = f"{mode}_{exp_slug}_{run_id}.csv"
    return str(Path("results") / "runs" / filename)


def _append_run_index_row(
    *,
    run_id: str,
    mode: RunMode,
    out_path: str,
    device: str,
    sweep: str,
    experiment: str,
    notes: str,
) -> None:
    index_fieldnames = [
        "run_id",
        "timestamp",
        "mode",
        "output_path",
        "device",
        "sweep_param",
        "experiment",
        "notes",
    ]
    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "output_path": out_path,
        "device": device,
        "sweep_param": sweep,
        "experiment": experiment,
        "notes": notes,
    }
    append_csv_row("results/run_index.csv", index_fieldnames, row)


def _resolve_run_values(
    sweep: str,
    value: str | int,
    default_model: str,
    default_batch: int,
    default_resolution: int,
) -> tuple[str, int, int]:
    model_name = default_model
    curr_batch = default_batch
    curr_resolution = default_resolution

    if sweep == "model":
        model_name = str(value)
    elif sweep == "batch":
        curr_batch = int(value)
    else:
        curr_resolution = int(value)

    return model_name, curr_batch, curr_resolution


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


def _build_common_row(
    *,
    run_id: str,
    experiment: str,
    notes: str,
    torch_device: torch.device,
    sweep: str,
    model_name: str,
    curr_batch: int,
    curr_resolution: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "experiment": experiment,
        "notes": notes,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(torch_device),
        "sweep_param": sweep,
        "model": model_name,
        "batch": curr_batch,
        "resolution": curr_resolution,
    }


def get_sweep_values(
    sweep: str,
    models: list[str],
    batches: list[int],
    resolutions: list[int],
) -> list[str] | list[int]:
    if sweep == "model":
        return models
    if sweep == "batch":
        return batches
    return resolutions


@torch.no_grad()
def run_cpu_sweep(
    *,
    mode: RunMode,
    device: str,
    out: str | None,
    append: bool,
    experiment: str,
    run_id: str | None,
    notes: str,
    model: str,
    batch: int,
    resolution: int,
    precision: str,
    backend: str,
    iters: int,
    warmup: int,
    sweep: str,
    models: list[str],
    batches: list[int],
    resolutions: list[int],
    enable_energy: bool = False,
) -> tuple[str, str]:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but torch.cuda.is_available() is False")

    resolved_run_id = _resolve_run_id(run_id)
    out_path = _resolve_output_path(
        mode=mode,
        out=out,
        experiment=experiment,
        run_id=resolved_run_id,
        append=append,
    )
    _append_run_index_row(
        run_id=resolved_run_id,
        mode=mode,
        out_path=out_path,
        device=device,
        sweep=sweep,
        experiment=experiment,
        notes=notes,
    )

    sweep_values = get_sweep_values(sweep, models, batches, resolutions)
    fieldnames = FIELDNAMES_BY_MODE[mode]
    desc_prefix = DESC_PREFIX_BY_MODE[mode]

    model_cache: dict[str, torch.nn.Module] = {}
    
    sampler = None
    if enable_energy and mode in ("bench", "full"):
        power_csv_path = Path(out_path.replace(".csv", "_power_trace.csv"))
        power_csv_path.parent.mkdir(parents=True, exist_ok=True)
        sampler = INA3221Sampler(
            exe_candidate="src/energy_inference/tools/sample_ina3221",
            hz=1000,
            power_csv=str(power_csv_path),
            hw="all"
        )

    for value in tqdm(sweep_values, desc=f"{desc_prefix} {sweep}", unit="cfg"):
        model_name, curr_batch, curr_resolution = _resolve_run_values(
            sweep=sweep,
            value=value,
            default_model=model,
            default_batch=batch,
            default_resolution=resolution,
        )

        if model_name not in model_cache:
            model_cache[model_name] = get_model(model_name).to(torch_device).eval()
        curr_model = model_cache[model_name]

        base_row = _build_common_row(
            run_id=resolved_run_id,
            experiment=experiment,
            notes=notes,
            torch_device=torch_device,
            sweep=sweep,
            model_name=model_name,
            curr_batch=curr_batch,
            curr_resolution=curr_resolution,
        )

        if mode == "bench":
            latency_ms, fps, energy_dict = bench_once(
                model=curr_model,
                batch=curr_batch,
                resolution=curr_resolution,
                iters=iters,
                warmup=warmup,
                device=torch_device,
                sampler=sampler,
            )
            row = base_row | {
                "iters": iters,
                "warmup": warmup,
                "latency_ms": round(latency_ms, 4),
                "fps": round(fps, 4),
                "energy_cpu_J": energy_dict.get("cpu", ""),
                "energy_gpu_J": energy_dict.get("gpu", ""),
                "energy_io_J": energy_dict.get("io", ""),
                "power_total_W": _compute_total_power_w(
                    energy_dict=energy_dict,
                    latency_ms=latency_ms,
                    iters=iters,
                ),
            }
            append_csv_row(out_path, fieldnames, row)
            continue

        status = "ok"
        error_msg = ""
        num_params = ""
        flops_total = ""
        flops_per_sample = ""
        unsupported_ops_count = ""
        latency_ms = ""
        fps = ""
        energy_dict = {}

        try:
            num_params = count_parameters(curr_model)
            flops_total, unsupported_ops_count = compute_flops(
                model=curr_model,
                batch=curr_batch,
                resolution=curr_resolution,
                device=torch_device,
            )
            flops_per_sample = flops_total / max(curr_batch, 1)

            if mode == "full":
                latency_ms, fps, energy_dict = bench_once(
                    model=curr_model,
                    batch=curr_batch,
                    resolution=curr_resolution,
                    iters=iters,
                    warmup=warmup,
                    device=torch_device,
                    sampler=sampler,
                )
        except Exception as exc:  # Keep sweeps robust across model/config failures.
            status = "failed"
            error_msg = str(exc)

        feature_fields = {
            "model_family": infer_model_family(model_name),
            "precision": precision,
            "backend": backend,
            "num_params": num_params,
            "flops_total": _round_float_or_empty(flops_total),
            "flops_per_sample": _round_float_or_empty(flops_per_sample),
            "unsupported_ops_count": unsupported_ops_count,
            "status": status,
            "error_msg": error_msg,
        }

        if mode == "features":
            row = base_row | feature_fields
        else:
            row = base_row | feature_fields | {
                "iters": iters,
                "warmup": warmup,
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
            }

        append_csv_row(out_path, fieldnames, row)

    return out_path, resolved_run_id

