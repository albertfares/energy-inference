import argparse
import sys
from pathlib import Path

# Allow running script without package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energy_inference.experiment_runner import (
    parse_experiments_csv,
    run_experiments_from_csv,
    summarize_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one or more experiments defined in a CSV file."
    )
    parser.add_argument(
        "--experiments-csv",
        type=str,
        required=True,
        help="Path to CSV describing experiments to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV and print enabled experiments without running benchmarks.",
    )
    args = parser.parse_args()

    if args.dry_run:
        parsed = parse_experiments_csv(args.experiments_csv)
        if not parsed:
            print("No enabled experiments found in CSV.")
            return
        print("Dry run (no benchmarks executed). Enabled experiments:")
        for row_number, cfg in parsed:
            print(f"- row={row_number} {summarize_config(cfg)}")
        return

    run_group_dir, completed = run_experiments_from_csv(args.experiments_csv)
    if not completed:
        print("No enabled experiments found in CSV.")
        return

    print(f"Run group directory: {run_group_dir}")
    print("Completed experiments:")
    for row_number, run_id, out_path in completed:
        print(f"- row={row_number} run_id={run_id} output={out_path}")


if __name__ == "__main__":
    main()

