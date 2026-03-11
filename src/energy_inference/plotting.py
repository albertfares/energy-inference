from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

VALID_SWEEPS = {"model", "batch", "resolution"}
SWEEP_TO_COLUMN = {
    "model": "model",
    "batch": "batch",
    "resolution": "resolution",
}


def load_results_csv(path: str) -> pd.DataFrame:
    """Load a benchmark results CSV into a DataFrame."""
    return pd.read_csv(path)


def infer_sweep_column(df: pd.DataFrame) -> tuple[str, str]:
    """
    Infer sweep mode and corresponding x-axis column.

    Returns:
        (sweep_param, x_column)
    """
    if "sweep_param" not in df.columns:
        raise ValueError("CSV is missing required column: sweep_param")

    sweep_values = df["sweep_param"].dropna().astype(str).unique()
    if len(sweep_values) != 1:
        raise ValueError(
            "Expected exactly one sweep_param value in file. "
            f"Found: {list(sweep_values)}"
        )

    sweep_param = sweep_values[0]
    if sweep_param not in VALID_SWEEPS:
        raise ValueError(f"Unsupported sweep_param: {sweep_param}")

    return sweep_param, SWEEP_TO_COLUMN[sweep_param]


def prepare_plot_dataframe(
    df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    include_failed: bool = False,
) -> pd.DataFrame:
    """Filter and sort rows for plotting."""
    if x_column not in df.columns:
        raise ValueError(f"CSV is missing x-axis column: {x_column}")
    if y_column not in df.columns:
        raise ValueError(f"CSV is missing y-axis column: {y_column}")

    plot_df = df.copy()

    if not include_failed and "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"].astype(str) == "ok"]

    plot_df = plot_df[[x_column, y_column]].dropna()
    if plot_df.empty:
        raise ValueError("No rows available to plot after filtering.")

    if x_column in {"batch", "resolution"}:
        plot_df[x_column] = pd.to_numeric(plot_df[x_column], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_column]).sort_values(by=x_column)
    else:
        plot_df[x_column] = plot_df[x_column].astype(str)

    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_column])
    if plot_df.empty:
        raise ValueError("No numeric values found for selected y-axis column.")

    return plot_df


def plot_sweep(
    df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Build a line plot for one swept variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[x_column], df[y_column], marker="o")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{y_column} vs {x_column}")
    fig.tight_layout()
    return fig, ax


def plot_latency_and_fps(
    df: pd.DataFrame,
    *,
    x_column: str,
    title: str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Build a dual-axis plot with latency_ms and fps."""
    required_columns = {x_column, "latency_ms", "fps"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            "DataFrame is missing required columns for dual plot: "
            f"{', '.join(missing)}"
        )

    fig, ax_left = plt.subplots(figsize=(8, 5))
    ax_right = ax_left.twinx()

    ax_left.plot(
        df[x_column],
        df["latency_ms"],
        marker="o",
        color="tab:blue",
        label="latency_ms",
    )
    ax_right.plot(
        df[x_column],
        df["fps"],
        marker="s",
        color="tab:orange",
        label="fps",
    )

    ax_left.set_xlabel(x_column)
    ax_left.set_ylabel("latency_ms", color="tab:blue")
    ax_right.set_ylabel("fps", color="tab:orange")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_right.tick_params(axis="y", labelcolor="tab:orange")
    ax_left.grid(True, alpha=0.3)

    if title:
        ax_left.set_title(title)
    else:
        ax_left.set_title(f"latency_ms and fps vs {x_column}")

    # Put one combined legend on the left axis.
    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(left_handles + right_handles, left_labels + right_labels)

    fig.tight_layout()
    return fig, (ax_left, ax_right)


def default_plot_path(csv_path: str, y_column: str) -> str:
    """Generate default output image path for a CSV run file."""
    src = Path(csv_path)
    out_dir = Path("results") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{src.stem}_{y_column}.png")


def list_run_csv_files(run_dir: str) -> list[str]:
    """List CSV result files inside one run-group directory."""
    directory = Path(run_dir)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    # Only include the per-row summary CSVs and skip auxiliary trace CSVs
    # such as power traces (e.g., "*_power_trace.csv"), which don't follow
    # the same schema and will break sweep inference/plotting.
    files = sorted(
        str(p)
        for p in directory.glob("*.csv")
        if "power_trace" not in p.stem
    )
    if not files:
        raise ValueError(f"No CSV files found in run directory: {run_dir}")
    return files


def default_plot_dir_for_run_group(run_dir: str) -> str:
    """Create and return default plot output directory for a run group."""
    run_name = Path(run_dir).name
    out_dir = Path("results") / "plots" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def save_figure(fig: plt.Figure, out_path: str, dpi: int = 150) -> str:
    """Save a matplotlib figure and return its output path."""
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    return str(output)

