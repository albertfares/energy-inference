import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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


def plot_energy_components(df, *, x_column: str, title: str | None = None):
    """Plot energy_cpu_J, energy_gpu_J, and energy_io_J on a shared axis."""
    fig, ax = plt.subplots(figsize=(8, 5))

    series = [
        ("energy_cpu_J", "CPU energy (J)", "tab:green", "o"),
        ("energy_gpu_J", "GPU energy (J)", "tab:red", "s"),
        ("energy_io_J", "IO energy (J)", "tab:purple", "^"),
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
    ax.set_ylabel("Energy (J)")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Energy vs {x_column}")
    ax.legend()
    fig.tight_layout()
    return fig, ax


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

    if args.plot_latency_fps and args.plot_energy:
        raise ValueError("Cannot use --plot-latency-fps and --plot-energy together.")

    if args.input:
        df = load_results_csv(args.input)
        _, x_column = infer_sweep_column(df)
        if args.plot_latency_fps:
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
            fig, _ = plot_latency_and_fps(plot_df, x_column=x_column, title=args.title)
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
            fig, _ = plot_energy_components(
                plot_df,
                x_column=x_column,
                title=args.title,
            )
            out_path = args.output or default_plot_path(args.input, "energy")
        else:
            plot_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column=args.y,
                include_failed=args.include_failed,
            )
            fig, _ = plot_sweep(plot_df, x_column=x_column, y_column=args.y, title=args.title)
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
        _, x_column = infer_sweep_column(df)
        if args.plot_latency_fps:
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
            fig, _ = plot_latency_and_fps(plot_df, x_column=x_column, title=args.title)
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
            fig, _ = plot_energy_components(
                plot_df,
                x_column=x_column,
                title=args.title,
            )
            suffix = "energy"
        else:
            plot_df = prepare_plot_dataframe(
                df,
                x_column=x_column,
                y_column=args.y,
                include_failed=args.include_failed,
            )
            fig, _ = plot_sweep(plot_df, x_column=x_column, y_column=args.y, title=args.title)
            suffix = args.y

        out_path = str(Path(run_plot_dir) / f"{Path(csv_path).stem}_{suffix}.png")
        saved_to = save_figure(fig, out_path, dpi=args.dpi)
        print(f"Plot saved to: {saved_to}")

        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()

