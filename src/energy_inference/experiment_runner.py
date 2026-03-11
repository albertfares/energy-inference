import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from energy_inference.pipeline import RunMode, run_cpu_sweep

VALID_MODES = {"bench", "features", "full"}
VALID_SWEEPS = {"model", "batch", "resolution"}


@dataclass
class ExperimentConfig:
    mode: RunMode = "full"
    device: str = "cpu"
    out: str | None = None
    append: bool = False
    experiment: str = "csv_experiment"
    run_id: str | None = None
    notes: str = ""
    model: str | None = "resnet18"
    batch: int = 1
    resolution: int = 224
    precision: str = "fp32"
    backend: str = "eager"
    iters: int = 200
    warmup: int = 30
    sweep: str = "model"
    models: list[str] | None = None
    batches: list[int] | None = None
    resolutions: list[int] | None = None
    enable_energy: bool = False


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value.strip())


def _parse_str(value: str | None, default: str) -> str:
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _parse_optional_str(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None or value.strip() == "":
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _enabled(row: dict[str, str]) -> bool:
    return _parse_bool(row.get("enabled"), default=True)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return normalized.strip("_") or "run"


def _build_config_from_row(row: dict[str, str]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    mode_str = _parse_str(row.get("mode"), cfg.mode)
    if mode_str not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode_str}'. Expected one of: {sorted(VALID_MODES)}")
    cfg.mode = mode_str  # type: ignore[assignment]
    cfg.device = _parse_str(row.get("device"), cfg.device)
    cfg.out = _parse_optional_str(row.get("out"))
    cfg.append = _parse_bool(row.get("append"), cfg.append)
    cfg.experiment = _parse_str(row.get("experiment"), cfg.experiment)
    cfg.run_id = _parse_optional_str(row.get("run_id"))
    cfg.notes = _parse_str(row.get("notes"), cfg.notes)
    cfg.model = _parse_optional_str(row.get("model")) or cfg.model
    cfg.batch = _parse_int(row.get("batch"), cfg.batch)
    cfg.resolution = _parse_int(row.get("resolution"), cfg.resolution)
    cfg.precision = _parse_str(row.get("precision"), cfg.precision)
    cfg.backend = _parse_str(row.get("backend"), cfg.backend)
    cfg.iters = _parse_int(row.get("iters"), cfg.iters)
    cfg.warmup = _parse_int(row.get("warmup"), cfg.warmup)
    cfg.sweep = _parse_str(row.get("sweep"), cfg.sweep)
    if cfg.sweep not in VALID_SWEEPS:
        raise ValueError(
            f"Invalid sweep '{cfg.sweep}'. Expected one of: {sorted(VALID_SWEEPS)}"
        )
    cfg.enable_energy = _parse_bool(row.get("enable_energy"), cfg.enable_energy)

    default_models = ["resnet18", "resnet50", "mobilenet_v3_large", "vit_b_16", "swin_t"]
    default_batches = [1, 2, 4, 8]
    default_resolutions = [224, 320, 384]
    cfg.models = _parse_str_list(row.get("models"), default_models)
    cfg.batches = _parse_int_list(row.get("batches"), default_batches)
    cfg.resolutions = _parse_int_list(row.get("resolutions"), default_resolutions)
    return cfg


def _resolve_effective_model(cfg: ExperimentConfig) -> str:
    """
    Resolve base model passed to pipeline.

    For `sweep=model`, the base model is only a fallback and the sweep list is used.
    For other sweeps, a concrete base model is required.
    """
    if cfg.model and cfg.model.strip():
        return cfg.model

    if cfg.sweep == "model" and cfg.models:
        return cfg.models[0]

    raise ValueError(
        "Missing model value. Set `model` in CSV for non-model sweeps "
        "or provide at least one value in `models`."
    )


def parse_experiments_csv(csv_path: str) -> list[tuple[int, ExperimentConfig]]:
    """Parse enabled rows in CSV into validated ExperimentConfig objects."""
    parsed: list[tuple[int, ExperimentConfig]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            if not _enabled(row):
                continue
            cfg = _build_config_from_row(row)
            parsed.append((row_number, cfg))
    return parsed


def summarize_config(cfg: ExperimentConfig) -> str:
    """Return a compact one-line summary for preview output."""
    if cfg.sweep == "model":
        swept_values = ",".join(cfg.models or [])
    elif cfg.sweep == "batch":
        swept_values = ",".join(str(v) for v in (cfg.batches or []))
    else:
        swept_values = ",".join(str(v) for v in (cfg.resolutions or []))

    base_model = cfg.model if cfg.model else "<auto>"
    return (
        f"mode={cfg.mode} device={cfg.device} sweep={cfg.sweep} "
        f"sweep_values=[{swept_values}] experiment={cfg.experiment} "
        f"model(base)={base_model} batch(base)={cfg.batch} "
        f"resolution(base)={cfg.resolution} iters={cfg.iters} warmup={cfg.warmup}"
    )


def _make_run_group_dir(csv_path: str) -> str:
    csv_stem = Path(csv_path).stem
    now = datetime.now()
    run_group_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{now.microsecond // 1000:03d}"
    group_name = f"{_slug(csv_stem)}_{run_group_id}"
    group_dir = Path("results") / "runs" / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    return str(group_dir)


def run_experiments_from_csv(csv_path: str) -> tuple[str, list[tuple[int, str, str]]]:
    """
    Run all enabled experiments from CSV.

    Returns:
        (run_group_dir, [ (row_number, run_id, out_path), ... ])
    """
    completed: list[tuple[int, str, str]] = []
    run_group_dir = _make_run_group_dir(csv_path)
    parsed = parse_experiments_csv(csv_path)
    for row_number, cfg in parsed:
        model_value = _resolve_effective_model(cfg)
        run_out = str(Path(run_group_dir) / f"row{row_number:03d}_{cfg.mode}_{_slug(cfg.experiment)}.csv")
        out_path, run_id = run_cpu_sweep(
            mode=cfg.mode,
            device=cfg.device,
            out=run_out,
            append=cfg.append,
            experiment=cfg.experiment,
            run_id=cfg.run_id,
            notes=cfg.notes,
            model=model_value,
            batch=cfg.batch,
            resolution=cfg.resolution,
            precision=cfg.precision,
            backend=cfg.backend,
            iters=cfg.iters,
            warmup=cfg.warmup,
            sweep=cfg.sweep,
            models=cfg.models or ["resnet18", "resnet50", "mobilenet_v3_large", "vit_b_16", "swin_t"],
            batches=cfg.batches or [1, 2, 4, 8],
            resolutions=cfg.resolutions or [224, 320, 384],
            enable_energy=cfg.enable_energy,
        )
        completed.append((row_number, run_id, out_path))

    return run_group_dir, completed

