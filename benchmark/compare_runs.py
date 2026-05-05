#!/usr/bin/env python3
"""Compare all CanItEdit benchmark runs side-by-side."""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent / "runs"


def main():
    run_dirs = sorted(
        d for d in RUNS_DIR.iterdir()
        if d.is_dir() and list(d.glob("*.results.json.gz"))
    )

    if not run_dirs:
        print("No completed runs found in runs/")
        print("(A run needs .results.json.gz files from Docker eval)")
        sys.exit(1)

    # Run pass_k.py on all dirs at once — it outputs one CSV line per dir
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "pass_k.py")] + [str(d) for d in run_dirs],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print("pass_k.py failed:", result.stderr, file=sys.stderr)
        sys.exit(1)

    lines = result.stdout.strip().split("\n")
    if len(lines) < 2:
        print("No results found.")
        sys.exit(1)

    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:]]

    # Pick the columns we care about
    cols = ["Name", "Pass@k", "Estimate", "NumProblems", "MinCompletions",
            "ExcessCode", "MeanMedianCoverage"]
    col_idx = []
    display_headers = []
    for c in cols:
        if c in header:
            col_idx.append(header.index(c))
            label = {"Estimate": "Pass@1 (%)", "MeanMedianCoverage": "Coverage",
                     "MinCompletions": "Completions", "NumProblems": "Problems"}.get(c, c)
            display_headers.append(label)

    display_rows = []
    for row in rows:
        display_rows.append([row[i] for i in col_idx])

    # Sort by Pass@1 descending
    estimate_pos = cols.index("Estimate")
    display_rows.sort(key=lambda r: float(r[estimate_pos]), reverse=True)

    # Calculate column widths
    all_rows = [display_headers] + display_rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(display_headers))]

    # Print table
    def fmt_row(row):
        return "  ".join(val.ljust(widths[i]) for i, val in enumerate(row))

    print()
    print("=" * (sum(widths) + 2 * (len(widths) - 1)))
    print("  CanItEdit Benchmark Comparison")
    print("=" * (sum(widths) + 2 * (len(widths) - 1)))
    print()
    print(fmt_row(display_headers))
    print("  ".join("-" * w for w in widths))
    for row in display_rows:
        print(fmt_row(row))
    print()

    # Also save CSV
    csv_path = RUNS_DIR / "comparison.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(display_headers) + "\n")
        for row in display_rows:
            f.write(",".join(row) + "\n")
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
