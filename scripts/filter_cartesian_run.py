import argparse
import csv
from pathlib import Path


def _normalize_status(value: str) -> str:
    return value.strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a cartesian run CSV to rows with status=ok and batch in a target set."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input cartesian CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output filtered CSV.",
    )
    parser.add_argument(
        "--batches",
        nargs="*",
        type=int,
        default=[1, 2, 4],
        help="Batch sizes to keep (default: 1 2 4).",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="ok",
        help='Status value to keep (default: "ok").',
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    keep_batches = set(args.batches)
    keep_status = _normalize_status(args.status)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    kept = 0
    total = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")
        if "status" not in reader.fieldnames or "batch" not in reader.fieldnames:
            raise ValueError("Input CSV must include 'status' and 'batch' columns.")

        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                total += 1
                status_value = _normalize_status(str(row.get("status", "")))
                try:
                    batch_value = int(str(row.get("batch", "")).strip())
                except ValueError:
                    continue

                if status_value == keep_status and batch_value in keep_batches:
                    writer.writerow(row)
                    kept += 1

    print(
        f"Filtered rows: kept={kept} / total={total} "
        f"(status={keep_status}, batches={sorted(keep_batches)})"
    )
    print(f"Saved filtered CSV to: {output_path}")


if __name__ == "__main__":
    main()
