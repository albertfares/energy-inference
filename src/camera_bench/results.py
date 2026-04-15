"""Results directory layout and I/O helpers."""
from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


FRAMES_CSV_FIELDS = [
    "frame_idx",
    "t_capture_start_ns", "t_capture_end_ns",
    "t_preprocess_start_ns", "t_preprocess_end_ns",
    "t_infer_start_ns", "t_infer_end_ns",
    "t_infer_fused_start_ns", "t_infer_fused_end_ns",
    "t_postprocess_start_ns", "t_postprocess_end_ns",
    "t_filter_start_ns", "t_filter_end_ns",
    "t_annotate_start_ns", "t_annotate_end_ns",
    "t_encode_start_ns", "t_encode_end_ns",
    "n_detections",
    "latency_total_ms",
    "fps_inst",
]

STAGE_ENERGY_CSV_FIELDS = [
    "stage", "rail", "energy_j", "mean_power_w", "share_pct",
]


class ResultsDir:
    """Manages a per-run results directory."""

    def __init__(self, base_dir: str | Path, run_name: str) -> None:
        self.run_dir = Path(base_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._frame_rows: list[dict] = []

    def path(self, filename: str) -> Path:
        return self.run_dir / filename

    def write_config(self, config: dict) -> None:
        with self.path("config.json").open("w") as f:
            json.dump(config, f, indent=2, default=_json_default)

    def append_frame(self, row: dict) -> None:
        self._frame_rows.append(row)

    def flush_frames_csv(self) -> None:
        with self.path("frames.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=FRAMES_CSV_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(self._frame_rows)

    def write_power_trace_csv(self, src_csv: str) -> None:
        import shutil
        shutil.copy2(src_csv, self.path("power_trace.csv"))

    def write_stage_energy_csv(self, rows: list[dict]) -> None:
        with self.path("stage_energy.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=STAGE_ENERGY_CSV_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)

    def write_summary(self, summary: dict) -> None:
        with self.path("summary.json").open("w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

    def write_log(self, text: str) -> None:
        with self.path("log.txt").open("w") as f:
            f.write(text)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def collect_env_info() -> dict[str, Any]:
    """Collect environment metadata for config.json."""
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass
    try:
        import torchvision
        info["torchvision_version"] = torchvision.__version__
    except Exception:
        pass
    try:
        import ultralytics
        info["ultralytics_version"] = ultralytics.__version__
    except Exception:
        pass
    try:
        jp = Path("/etc/nv_tegra_release")
        if jp.exists():
            info["jetpack_info"] = jp.read_text().strip().splitlines()[0]
    except Exception:
        pass
    return info


def make_run_name(
    model: str,
    width: int,
    height: int,
    precision: str,
    stream_mode: str,
    target_fps: int = 0,
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fps_tag = f"fps{target_fps}" if target_fps > 0 else "fpsmax"
    return f"{model}_{width}x{height}_{precision}_{stream_mode}_{fps_tag}_{ts}"
