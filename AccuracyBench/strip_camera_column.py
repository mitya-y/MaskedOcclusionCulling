#!/usr/bin/env python3
"""Drop `camera` column from accbench_batch_results.csv → accbench_batch_results_without_camera.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "accbench_batch_results.csv",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "accbench_batch_results_without_camera.csv",
    )
    args = p.parse_args()

    with args.input.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise SystemExit("empty CSV")
        fields = [c for c in reader.fieldnames if c != "camera"]
        rows = [{k: r[k] for k in fields} for r in reader]

    with args.output.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
