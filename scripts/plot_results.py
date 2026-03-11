import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow running script without package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_inference.plotting import (
    default_plot_dir_for_run_group,
    default_plot_path,
    infer_sweep_column,
    list_run_csv_files,
    load_results_csv,
    plot_latency_and_fps,
    plot_sweep,
    prepare_plot_dataframe,
    save_figure,
)


def plot_energy_components(
    df,
    *,
    x_column: str,
    per_sample: bool = False,
    per_flop: bool = False,
    title: str | None = None,
):
    """Plot energy_cpu_J, energy_gpu_J, and energy_io_J on a shared axis.

    If per_sample is True, the caller is expected to have pre-normalized the
    energy columns (e.g., dividing by batch size) and labels will reflect
    'per inference' semantics.

    If per_flop is True, the caller is expected to have pre-normalized the
    energy columns by flops_total and labels will reflect 'per FLOP' semantics.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if per_sample and per_flop:
        raise ValueError("per_sample and per_flop cannot both be True.")

    if per_flop:
        unit_suffix = "per FLOP (J/FLOP)"
        title_suffix = "per FLOP"
    elif per_sample:
        unit_suffix = "per inference (J)"
        title_suffix = "per inference"
    else:
        unit_suffix = "(J)"
        title_suffix = ""
    series = [
        ("energy_cpu_J", f"CPU energy {unit_suffix}", "tab:green", "o"),
        ("energy_gpu_J", f"GPU energy {unit_suffix}", "tab:red", "s"),
        ("energy_io_J", f"IO energy {unit_suffix}", "tab:purple", "^"),
    ]

    any_plotted = False
    for column, label, color, marker in series:
        if column in df.columns:
            ax.plot(df[x_column], df[column], marker=marker, color=color, label=label)
            any_plotted = True

    if not any_plotted:
        raise ValueError(
            "No energy columns found to plot. Expected at least one of: "
            "energy_cpu_J, energy_gpu_J, energy_io_J."
        )

    ax.set_xlabel(x_column)
    ax.set_ylabel(f"Energy {unit_suffix}")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Energy {title_suffix} vs {x_column}".strip())
    ax.legend()
    fig.tight_layout()
    return fig, ax


def build_stacked_energy_figure(
    df,
    *,
    x_column: str,
    include_failed: bool = False,
    title: str | None = None,
):
    """Create a 3-row stacked figure:

    1) Energy (J) vs x
    2) Energy per inference (J) vs x  [requires batch]
    3) Energy per FLOP (J/FLOP) vs x  [requires flops_total]
    """
    plot_df = df.copy()

    # Filter to successful rows if requested.
    if not include_failed and "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"].astype(str) == "ok"]

    # Require x_column.
    if x_column not in plot_df.columns:
        raise ValueError(f"CSV is missing x-axis column: {x_column}")

    # Basic numeric coercions.
    if x_column in {"batch", "resolution"}:
        plot_df[x_column] = (
            plot_df[x_column].apply(pd.to_numeric, errors="coerce").dropna()
        )
    else:
        plot_df[x_column] = plot_df[x_column].astype(str)

    for col in ("energy_cpu_J", "energy_gpu_J", "energy_io_J"):
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # 1) Raw energy
    ax1 = axes[0]
    for col, label, color, marker in [
        ("energy_cpu_J", "CPU energy (J)", "tab:green", "o"),
        ("energy_gpu_J", "GPU energy (J)", "tab:red", "s"),
        ("energy_io_J", "IO energy (J)", "tab:purple", "^"),
    ]:
        if col in plot_df.columns:
            ax1.plot(plot_df[x_column], plot_df[col], marker=marker, color=color, label=label)
    ax1.set_ylabel("Energy (J)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title(f"Energy vs {x_column}")

    # 2) Energy per inference
    ax2 = axes[1]
    if "batch" in plot_df.columns:
        batch = pd.to_numeric(plot_df["batch"], errors="coerce")
        for col, label, color, marker in [
            ("energy_cpu_J", "CPU energy per inf (J)", "tab:green", "o"),
            ("energy_gpu_J", "GPU energy per inf (J)", "tab:red", "s"),
            ("energy_io_J", "IO energy per inf (J)", "tab:purple", "^"),
        ]:
            if col in plot_df.columns:
                ax2.plot(
                    plot_df[x_column],
                    plot_df[col] / batch,
                    marker=marker,
                    color=color,
                    label=label,
                )
        ax2.set_ylabel("Energy / inference (J)")
        ax2.legend()
    else:
        ax2.text(
            0.5,
            0.5,
            "No 'batch' column; per-inference energy unavailable.",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Energy per inference vs {x_column}")

    # 3) Energy per FLOP
    ax3 = axes[2]
    if "flops_total" in plot_df.columns:
        flops = pd.to_numeric(plot_df["flops_total"], errors="coerce")
        for col, label, color, marker in [
            ("energy_cpu_J", "CPU energy per FLOP (J/FLOP)", "tab:green", "o"),
            ("energy_gpu_J", "GPU energy per FLOP (J/FLOP)", "tab:red", "s"),
            ("energy_io_J", "IO energy per FLOP (J/FLOP)", "tab:purple", "^"),
        ]:
            if col in plot_df.columns:
                ax3.plot(
                    plot_df[x_column],
                    plot_df[col] / flops,
                    marker=marker,
                    color=color,
                    label=label,
                )
        ax3.set_ylabel("Energy / FLOP (J/FLOP)")
        ax3.legend()
    else:
        ax3.text(
            0.5,
            0.5,
            "No 'flops_total' column; per-FLOP energy unavailable.",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel(x_column)
    ax3.set_title(f"Energy per FLOP vs {x_column}")

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one metric from one CSV or all CSVs in a run-group directory."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Path to one results CSV.")
    input_group.add_argument(
        "--run-dir",
        type=str,
        help="Path to one run-group directory under results/runs containing per-row CSVs.",
    )
    parser.add_argument(
        "--y",
        type=str,
        default="latency_ms",
        help="Metric column to plot on y-axis (e.g., latency_ms, fps, flops_total).",
    )
    parser.add_argument(
        "--plot-latency-fps",
        action="store_true",
        help="Plot latency_ms and fps together using dual y-axes.",
    )
    parser.add_argument(
        "--plot-energy",
        action="store_true",
        help="Plot energy_cpu_J, energy_gpu_J, and energy_io_J together.",
    )
    parser.add_argument(
        "--per-sample-energy",
        action="store_true",
        help=(
            "When used with --plot-energy, divide energy_*_J columns by batch size "
            "to show energy per inference."
        ),
    )
    parser.add_argument(
        "--per-flop-energy",
        action="store_true",
        help=(
            "When used with --plot-energy, divide energy_*_J columns by flops_total "
            "to show energy per FLOP."
        ),
    )
    parser.add_argument(
        "--summary-energy",
        action="store_true",
        help=(
            "Produce a stacked summary figure with energy, energy per inference, "
            "and energy per FLOP vs sweep variable."
        ),
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a logarithmic scale for the x-axis.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic scale for the y-axis (where applicable).",
    )
    parser.add_argument("--output", type=str, default=None, help="Output image path.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory when using --run-dir (default: results/plots/<run_dir_name>).",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include rows with status != ok.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window.")
    parser.add_argument("--dpi", type=int, default=150)

    args = parser.parse_args()

    if sum(
        int(flag)
        for flag in (args.plot_latency_fps, args.plot_energy, args.summary_energy)
    ) > 1:
        raise ValueError(
            "Use at most one of --plot-latency-fps, --plot-energy, or --summary-energy."
        )

    if args.per_sample_energy and args.per_flop_energy:
        raise ValueError(
            "Cannot use --per-sample-energy and --per-flop-energy at the same time."
        )

    if args.per_sample_energy and not args.plot_energy:
        raise ValueError(
            "--per-sample-energy is only valid together with --plot-energy."
        )

    if args.per_flop_energy and not args.plot_energy:
        raise ValueError(
            "--per-flop-energy is only valid together with --plot-energy."
        )

    if args.input:
        df = load_results_csv(args.input)
        if args.per_sample_energy:
            if "batch" not in df.columns:
                raise ValueError(
                    "CSV is missing 'batch' column required for per-sample energy."
                )
            for col in ("energy_cpu_J", "energy_gpu_J", "energy_io_J"):
                if col in df.columns:
                    df[col] = df[col] / df["batch"]
        if args.per_flop_energy:
            if "flops_total" not in df.columns:
                raise ValueError(
                    "CSV is missing 'flops_total' column required for per-FLOP energy."
                )
            for col in ("energy_cpu_J", "energy_gpu_J", "energy_io_J"):
                if col in df.columns:
                    df[col] = df[col] / df["flops_total"]

        _, x_column = infer_sweep_column(df)
        if args.summary_energy:
            fig = build_stacked_energy_figure(
                df,
                x_column=x_column,
                include_failed=args.include_failed,
                title=args.title,
            )
            out_path = args.output or default_plot_path(args.input, "summary_energy")
        elif args.plot_latency_fps:
            latency_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="latency_ms",
                include_failed=args.include_failed,
            )
            fps_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="fps",
                include_failed=args.include_failed,
            )
            plot_df = latency_df.merge(fps_df, on=x_column, how="inner")
            if plot_df.empty:
                raise ValueError("No overlapping rows found for latency_ms and fps.")
            fig, axes = plot_latency_and_fps(
                plot_df, x_column=x_column, title=args.title
            )
            ax_left, ax_right = axes
            if args.log_x:
                ax_left.set_xscale("log")
                ax_right.set_xscale("log")
            if args.log_y:
                ax_left.set_yscale("log")
                ax_right.set_yscale("log")
            out_path = args.output or default_plot_path(args.input, "latency_fps")
        elif args.plot_energy:
            cpu_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="energy_cpu_J",
                include_failed=args.include_failed,
            )
            gpu_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="energy_gpu_J",
                include_failed=args.include_failed,
            )
            io_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="energy_io_J",
                include_failed=args.include_failed,
            )
            plot_df = cpu_df.merge(gpu_df, on=x_column, how="outer").merge(
                io_df, on=x_column, how="outer"
            )
            if plot_df.empty:
                raise ValueError("No rows available to plot energy columns.")
            fig, ax = plot_energy_components(
                plot_df,
                x_column=x_column,
                per_sample=args.per_sample_energy,
                per_flop=args.per_flop_energy,
                title=args.title,
            )
            if args.log_x:
                ax.set_xscale("log")
            if args.log_y:
                ax.set_yscale("log")
            if args.per_sample_energy:
                suffix = "energy_per_sample"
            elif args.per_flop_energy:
                suffix = "energy_per_flop"
            else:
                suffix = "energy"
            out_path = args.output or default_plot_path(args.input, suffix)
        else:
            plot_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column=args.y,
                include_failed=args.include_failed,
            )
            fig, ax = plot_sweep(
                plot_df, x_column=x_column, y_column=args.y, title=args.title
            )
            if args.log_x:
                ax.set_xscale("log")
            if args.log_y:
                ax.set_yscale("log")
            out_path = args.output or default_plot_path(args.input, args.y)
        saved_to = save_figure(fig, out_path, dpi=args.dpi)
        print(f"Plot saved to: {saved_to}")

        if args.show:
            plt.show()
        plt.close(fig)
        return

    csv_files = list_run_csv_files(args.run_dir)
    run_plot_dir = args.output_dir or default_plot_dir_for_run_group(args.run_dir)
    print(f"Run-group plot output directory: {run_plot_dir}")

    for csv_path in csv_files:
        df = load_results_csv(csv_path)
        if args.per_sample_energy:
            if "batch" not in df.columns:
                raise ValueError(
                    f"CSV '{csv_path}' is missing 'batch' column required for per-sample energy."
                )
            for col in ("energy_cpu_J", "energy_gpu_J", "energy_io_J"):
                if col in df.columns:
                    df[col] = df[col] / df["batch"]
        if args.per_flop_energy:
            if "flops_total" not in df.columns:
                raise ValueError(
                    f"CSV '{csv_path}' is missing 'flops_total' column required for per-FLOP energy."
                )
            for col in ("energy_cpu_J", "energy_gpu_J", "energy_io_J"):
                if col in df.columns:
                    df[col] = df[col] / df["flops_total"]

        _, x_column = infer_sweep_column(df)
        if args.summary_energy:
            fig = build_stacked_energy_figure(
                df,
                x_column=x_column,
                include_failed=args.include_failed,
                title=args.title,
            )
            # Apply log scaling to shared x / y if requested.
            axes = fig.get_axes()
            for ax in axes:
                if args.log_x:
                    ax.set_xscale("log")
                if args.log_y:
                    ax.set_yscale("log")
            suffix = "summary_energy"
        elif args.plot_latency_fps:
            latency_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="latency_ms",
                include_failed=args.include_failed,
            )
            fps_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="fps",
                include_failed=args.include_failed,
            )
            plot_df = latency_df.merge(fps_df, on=x_column, how="inner")
            if plot_df.empty:
                raise ValueError("No overlapping rows found for latency_ms and fps.")
            fig, axes = plot_latency_and_fps(
                plot_df, x_column=x_column, title=args.title
            )
            ax_left, ax_right = axes
            if args.log_x:
                ax_left.set_xscale("log")
                ax_right.set_xscale("log")
            if args.log_y:
                ax_left.set_yscale("log")
                ax_right.set_yscale("log")
            suffix = "latency_fps"
        elif args.plot_energy:
            cpu_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="energy_cpu_J",
                include_failed=args.include_failed,
            )
            gpu_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="energy_gpu_J",
                include_failed=args.include_failed,
            )
            io_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column="energy_io_J",
                include_failed=args.include_failed,
            )
            plot_df = cpu_df.merge(gpu_df, on=x_column, how="outer").merge(
                io_df, on=x_column, how="outer"
            )
            if plot_df.empty:
                raise ValueError("No rows available to plot energy columns.")
            fig, ax = plot_energy_components(
                plot_df,
                x_column=x_column,
                per_sample=args.per_sample_energy,
                per_flop=args.per_flop_energy,
                title=args.title,
            )
            if args.log_x:
                ax.set_xscale("log")
            if args.log_y:
                ax.set_yscale("log")
            if args.per_sample_energy:
                suffix = "energy_per_sample"
            elif args.per_flop_energy:
                suffix = "energy_per_flop"
            else:
                suffix = "energy"
        else:
            plot_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column=args.y,
                include_failed=args.include_failed,
            )
            fig, ax = plot_sweep(
                plot_df, x_column=x_column, y_column=args.y, title=args.title
            )
            if args.log_x:
                ax.set_xscale("log")
            if args.log_y:
                ax.set_yscale("log")
            suffix = args.y

        out_path = str(Path(run_plot_dir) / f"{Path(csv_path).stem}_{suffix}.png")
        saved_to = save_figure(fig, out_path, dpi=args.dpi)
        print(f"Plot saved to: {saved_to}")

        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()

