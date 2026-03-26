import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_training_rows(data_dir: Path) -> pd.DataFrame:
    csv_files = sorted(
        p for p in data_dir.glob("*.csv") if "power_trace" not in p.name.lower()
    )
    if not csv_files:
        raise FileNotFoundError(f"No training CSVs found in: {data_dir}")

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "batch",
        "resolution",
        "macs_total",
        "flops_total",
        "flops_total_strict",
        "latency_ms",
        "fps",
        "energy_cpu_J",
        "energy_gpu_J",
        "energy_io_J",
        "power_total_W",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.copy()
    if "status" in ok.columns:
        ok = ok[ok["status"].astype(str) == "ok"].copy()

    energy_cols = [c for c in ("energy_cpu_J", "energy_gpu_J", "energy_io_J") if c in ok.columns]
    if energy_cols:
        ok["energy_total_J"] = ok[energy_cols].sum(axis=1, min_count=1)
    else:
        ok["energy_total_J"] = np.nan

    if "batch" in ok.columns:
        ok["energy_per_inf_J"] = ok["energy_total_J"] / ok["batch"]
    else:
        ok["energy_per_inf_J"] = np.nan

    return ok


def _ensure_out_dir(base_out_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / f"training_data_insights_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_metric_box_by_model(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    if metric not in df.columns or "model" not in df.columns:
        return
    plot_df = df[["model", metric]].dropna()
    if plot_df.empty:
        return

    models = sorted(plot_df["model"].astype(str).unique())
    values = [plot_df[plot_df["model"] == m][metric].values for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(values, labels=models, showfliers=False)
    ax.set_title(f"{metric} distribution by model")
    ax.set_xlabel("model")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    _save(fig, out_dir / f"box_{metric}_by_model.png")


def _plot_metric_bar_by_task(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    if metric not in df.columns or "model_task" not in df.columns:
        return
    g = df.groupby("model_task")[metric].median().dropna()
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g.index.astype(str), g.values)
    ax.set_title(f"Median {metric} by model_task")
    ax.set_xlabel("model_task")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, out_dir / f"bar_{metric}_by_model_task.png")


def _plot_batch_scaling(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    required = {"model", "batch", metric}
    if not required.issubset(df.columns):
        return
    if "sweep_param" in df.columns:
        allowed = {"batch", "cartesian"}
        plot_df = df[df["sweep_param"].astype(str).isin(allowed)][
            ["model", "batch", metric]
        ].dropna()
    else:
        plot_df = df[["model", "batch", metric]].dropna()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, g in plot_df.groupby("model"):
        agg = g.groupby("batch")[metric].median().sort_index()
        if len(agg) >= 2:
            ax.plot(agg.index.values, agg.values, marker="o", label=str(model_name))
    ax.set_title(f"{metric} vs batch (batch sweeps, median)")
    ax.set_xlabel("batch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir / f"line_{metric}_vs_batch.png")


def _plot_resolution_scaling(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    required = {"model", "resolution", metric}
    if not required.issubset(df.columns):
        return
    if "sweep_param" in df.columns:
        allowed = {"resolution", "cartesian"}
        plot_df = df[df["sweep_param"].astype(str).isin(allowed)][
            ["model", "resolution", metric]
        ].dropna()
    else:
        plot_df = df[["model", "resolution", metric]].dropna()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, g in plot_df.groupby("model"):
        agg = g.groupby("resolution")[metric].median().sort_index()
        if len(agg) >= 2:
            ax.plot(agg.index.values, agg.values, marker="o", label=str(model_name))
    ax.set_title(f"{metric} vs resolution (resolution sweeps, median)")
    ax.set_xlabel("resolution")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir / f"line_{metric}_vs_resolution.png")


def _plot_precision_bars(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    required = {"precision", metric}
    if not required.issubset(df.columns):
        return
    if "sweep_param" in df.columns:
        allowed = {"precision", "cartesian"}
        plot_df = df[df["sweep_param"].astype(str).isin(allowed)][
            ["precision", metric]
        ].dropna()
    else:
        plot_df = df[["precision", metric]].dropna()
    if plot_df.empty:
        return
    g = plot_df.groupby("precision")[metric].median()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g.index.astype(str), g.values)
    ax.set_title(f"Median {metric} by precision (precision sweeps)")
    ax.set_xlabel("precision")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, out_dir / f"bar_{metric}_by_precision.png")


def _plot_precision_lines(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    required = {"precision", "model", metric}
    if not required.issubset(df.columns):
        return
    if "sweep_param" in df.columns:
        allowed = {"precision", "cartesian"}
        plot_df = df[df["sweep_param"].astype(str).isin(allowed)][
            ["model", "precision", metric]
        ].dropna()
    else:
        plot_df = df[["model", "precision", metric]].dropna()
    if plot_df.empty:
        return

    precision_order = ["fp32", "fp16", "bf16"]
    available = [p for p in precision_order if p in set(plot_df["precision"].astype(str))]
    if len(available) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, g in plot_df.groupby("model"):
        agg = g.groupby("precision")[metric].median()
        x_labels = [p for p in available if p in agg.index]
        if len(x_labels) >= 2:
            y = [agg[p] for p in x_labels]
            ax.plot(x_labels, y, marker="o", label=str(model_name))
    ax.set_title(f"{metric} vs precision (precision sweeps, median)")
    ax.set_xlabel("precision")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir / f"line_{metric}_vs_precision.png")


def _plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        c
        for c in [
            "batch",
            "resolution",
            "macs_total",
            "flops_total",
            "latency_ms",
            "fps",
            "power_total_W",
            "energy_total_J",
            "energy_per_inf_J",
        ]
        if c in df.columns
    ]
    if len(cols) < 2:
        return

    corr_df = df[cols].dropna()
    if len(corr_df) < 4:
        return

    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("Correlation matrix (ok rows)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    _save(fig, out_dir / "corr_heatmap_ok_rows.png")


def _write_summary_tables(df_all: pd.DataFrame, df_ok: pd.DataFrame, out_dir: Path) -> None:
    overall = pd.DataFrame(
        [
            {
                "rows_total": len(df_all),
                "rows_ok": len(df_ok),
                "rows_failed": len(df_all) - len(df_ok),
            }
        ]
    )
    overall.to_csv(out_dir / "summary_overall.csv", index=False)

    metrics = [
        c
        for c in ["latency_ms", "fps", "energy_total_J", "energy_per_inf_J", "power_total_W"]
        if c in df_ok.columns
    ]

    if "model_task" in df_ok.columns and metrics:
        (
            df_ok.groupby("model_task")[metrics]
            .median()
            .round(6)
            .to_csv(out_dir / "summary_by_model_task_median.csv")
        )

    if "model" in df_ok.columns and metrics:
        (
            df_ok.groupby("model")[metrics]
            .median()
            .round(6)
            .to_csv(out_dir / "summary_by_model_median.csv")
        )

    if "precision" in df_ok.columns and metrics:
        (
            df_ok.groupby("precision")[metrics]
            .median()
            .round(6)
            .to_csv(out_dir / "summary_by_precision_median.csv")
        )

    if {"model", "status"}.issubset(df_all.columns):
        fail = (
            df_all.assign(failed=df_all["status"].astype(str) != "ok")
            .groupby("model")["failed"]
            .mean()
            .mul(100)
            .round(3)
            .rename("failure_rate_pct")
            .to_frame()
        )
        fail.to_csv(out_dir / "summary_failure_rate_by_model.csv")

    if {"model", "batch", "resolution", "precision", "latency_ms", "power_total_W", "energy_per_inf_J"}.issubset(
        df_ok.columns
    ):
        if "sweep_param" in df_ok.columns:
            allowed = {"precision", "cartesian"}
            p = df_ok[df_ok["sweep_param"].astype(str).isin(allowed)].copy()
        else:
            p = df_ok.copy()
        if not p.empty:
            idx = ["model", "batch", "resolution"]
            metrics_cmp = ["latency_ms", "power_total_W", "energy_per_inf_J"]
            wide = p.pivot_table(
                index=idx, columns="precision", values=metrics_cmp, aggfunc="median"
            )
            rows: list[dict[str, float]] = []
            for metric in metrics_cmp:
                for alt in ("fp16", "bf16"):
                    a = (metric, "fp32")
                    b = (metric, alt)
                    if a in wide.columns and b in wide.columns:
                        ratio = (wide[b] / wide[a] - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
                        if len(ratio) > 0:
                            rows.append(
                                {
                                    "metric": metric,
                                    "comparison": f"{alt}_vs_fp32",
                                    "median_delta_pct": float(np.median(ratio) * 100.0),
                                    "count": int(len(ratio)),
                                }
                            )
            if rows:
                pd.DataFrame(rows).to_csv(out_dir / "summary_precision_deltas.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summary tables and plots from data/training_data CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training_data",
        help="Directory containing training CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/plots",
        help="Base output directory for generated insight artifacts.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = _ensure_out_dir(Path(args.out_dir))

    df_all = _coerce_numeric(_load_training_rows(data_dir))
    df_ok = _prepare_ok_rows(df_all)

    _write_summary_tables(df_all, df_ok, out_dir)

    for metric in ("latency_ms", "fps", "energy_total_J", "energy_per_inf_J", "power_total_W"):
        _plot_metric_box_by_model(df_ok, metric, out_dir)
        _plot_metric_bar_by_task(df_ok, metric, out_dir)
        _plot_precision_bars(df_ok, metric, out_dir)
        _plot_precision_lines(df_ok, metric, out_dir)

    for metric in ("fps", "energy_per_inf_J", "power_total_W", "latency_ms"):
        _plot_batch_scaling(df_ok, metric, out_dir)
    for metric in ("latency_ms", "energy_per_inf_J", "power_total_W"):
        _plot_resolution_scaling(df_ok, metric, out_dir)

    _plot_correlation_heatmap(df_ok, out_dir)

    print(f"Saved training-data insights to: {out_dir}")


if __name__ == "__main__":
    main()
