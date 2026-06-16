"""

Reads one or more global CSV files (semicolon-separated) and produces a
summary with:
  - Overall win/loss counts and rates
  - Loss breakdown by failReason (DragonEscaped, Timer)
  - Mean of each numeric metric, computed only over episodes where that
    metric is meaningful (e.g. timeToGrabKey excluded on DragonEscaped losses)

Usage:
    python summarize_global.py global.csv
    python summarize_global.py run1/global.csv run2/global.csv

Output:
    - Prints the summary to stdout
    - Saves it next to the first input file, e.g.:
        global.csv  →  global_summary.csv
"""

import sys
import pathlib
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

SEPARATOR = ";"

# Columns that are identifiers, not metrics to average.
NON_METRIC_COLS = {"episode", "win", "failReason"}

# Columns that are only meaningful when the key was grabbed.
# On DragonEscaped losses the key is never grabbed, so these are 0 as a
# sentinel — exclude those rows before averaging.
KEY_COLS = {"timeToGrabKey", "timeFromKeyGrabToEscape"}


# ── Load data ─────────────────────────────────────────────────────────────────


def load_files(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep=SEPARATOR)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Summarize ─────────────────────────────────────────────────────────────────


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    total = len(df)
    wins = df["win"].sum()
    losses = total - wins

    # ── Win/loss overview ──────────────────────────────────────────────────
    overview = pd.DataFrame(
        [
            {
                "total_episodes": total,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / total, 4),
                "loss_rate": round(losses / total, 4),
            }
        ]
    )

    # ── Loss breakdown by failReason ───────────────────────────────────────
    fail_counts = (
        df[df["win"] == 0]["failReason"]
        .value_counts()
        .rename_axis("failReason")
        .reset_index(name="count")
    )
    fail_counts["pct_of_losses"] = (fail_counts["count"] / losses).round(4)
    fail_counts["pct_of_total"] = (fail_counts["count"] / total).round(4)

    # ── Metric means ──────────────────────────────────────────────────────
    numeric_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in NON_METRIC_COLS
    ]

    means = {}
    for col in numeric_cols:
        if col in KEY_COLS:
            # Only average over episodes where the key was actually grabbed
            # (i.e. timeToGrabKey > 0, which also filters DragonEscaped zeros)
            valid = df[df["timeToGrabKey"] > 0][col]
        else:
            valid = df[col]
        means[f"{col}_mean"] = round(valid.mean(), 4)

    metric_means = pd.DataFrame([means])

    return {
        "overview": overview,
        "fail_breakdown": fail_counts,
        "metric_means": metric_means,
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_global.py <file1.csv> [file2.csv ...]")
        sys.exit(1)

    input_paths = sys.argv[1:]
    print(f"Loading {len(input_paths)} file(s): {', '.join(input_paths)}\n")

    df = load_files(input_paths)
    print(f"  → {len(df)} episodes loaded\n")

    results = summarize(df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("── Overview ─────────────────────────────────────────────────────")
    print(results["overview"].to_string(index=False))
    print()
    print("── Loss breakdown ───────────────────────────────────────────────")
    print(results["fail_breakdown"].to_string(index=False))
    print()
    print("── Metric means ─────────────────────────────────────────────────")
    print(results["metric_means"].to_string(index=False))
    print()

    # Save: one sheet-like CSV with sections separated by a blank row
    first = pathlib.Path(input_paths[0])
    output_file = first.parent / f"{first.stem}_summary{first.suffix}"

    with open(output_file, "w") as f:
        f.write("OVERVIEW\n")
        results["overview"].to_csv(f, index=False)
        f.write("\nLOSS BREAKDOWN\n")
        results["fail_breakdown"].to_csv(f, index=False)
        f.write("\nMETRIC MEANS\n")
        results["metric_means"].to_csv(f, index=False)

    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
