"""

Reads one or more global CSV files (semicolon-separated) and produces a
summary with:
  - Overall win/loss counts and rates
  - Loss breakdown by failReason (DragonEscaped, Timer)
  - Mean of each numeric metric, computed only over episodes where that
    metric is meaningful (e.g. timeToGrabKey excluded on DragonEscaped losses)
  - Phase time percentages: how much of episodeDuration each phase represents
    on average, computed only over episodes where the key was grabbed
    (wins + Timer losses; DragonEscaped excluded)

Usage:
    python cumulative_global_stats.py global.csv
    python cumulative_global_stats.py run1/global.csv run2/global.csv

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

# The three sequential phases that partition episodeDuration.
# timeToGrabKey starts after all dragons are killed, so the three phases
# are non-overlapping and should sum to ~episodeDuration on complete runs.
PHASE_COLS = ["timeToKillAllDragons", "timeToGrabKey", "timeFromKeyGrabToEscape"]


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

    # ── Phase time percentages ─────────────────────────────────────────────
    # Restricted to episodes where the key was grabbed (timeToGrabKey > 0),
    # i.e. wins and Timer losses. DragonEscaped episodes are excluded because
    # two of the three phases have sentinel-zero values there.
    #
    # We compute mean(phase / episodeDuration) per episode rather than
    # mean(phase) / mean(episodeDuration) to avoid bias from outlier durations.
    key_grabbed = df[df["timeToGrabKey"] > 0].copy()
    n_key_grabbed = len(key_grabbed)

    available_phases = [c for c in PHASE_COLS if c in key_grabbed.columns]
    phase_rows = []
    for col in available_phases:
        per_episode_pct = key_grabbed[col] / key_grabbed["episodeDuration"]
        phase_rows.append(
            {
                "phase": col,
                "episodes_used": n_key_grabbed,
                "mean_seconds": round(key_grabbed[col].mean(), 4),
                "mean_pct_of_duration": round(per_episode_pct.mean(), 4),
            }
        )

    phase_pcts = pd.DataFrame(phase_rows)

    return {
        "overview": overview,
        "fail_breakdown": fail_counts,
        "metric_means": metric_means,
        "phase_pcts": phase_pcts,
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python cumulative_global_stats.py <file1.csv> [file2.csv ...]")
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
    print("── Phase time percentages ───────────────────────────────────────")
    print("   (episodes where key was grabbed: wins + Timer losses)")
    print(results["phase_pcts"].to_string(index=False))
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
        f.write("\nPHASE TIME PERCENTAGES (key grabbed episodes only)\n")
        results["phase_pcts"].to_csv(f, index=False)

    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
