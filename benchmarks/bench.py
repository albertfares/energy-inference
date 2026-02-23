"""Legacy wrapper for the CPU benchmark entrypoint.

Prefer running: `python scripts/bench_cpu.py ...`
"""

import runpy
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target = project_root / "scripts" / "bench_cpu.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()