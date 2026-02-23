import csv
import os


def append_csv_row(csv_path: str, fieldnames: list[str], row: dict) -> None:
    """Append a row to CSV and create header if file does not exist."""
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    file_has_content = file_exists and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_has_content:
            writer.writeheader()
        writer.writerow(row)

