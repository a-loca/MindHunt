"""
cumulative_by_personality.py

Reads one or more agent CSV files (semicolon-separated) and computes the
mean and standard deviation of every numeric metric grouped by personality,
across all episodes and all input files.

Usage:
    python cumulative_by_personality.py agents.csv
    python cumulative_by_personality.py ep1.csv ep2.csv ep3.csv
    python cumulative_by_personality.py *.csv

Output:
    - Prints results in a  mean (± std)  format per personality
    - Saves it next to the first input file, e.g.:
        agents.csv  →  agents_cumulative.csv
"""

import sys
import pathlib
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

SEPARATOR = ";"

# Columns that are identifiers, not metrics to be averaged.
NON_METRIC_COLS = {"episode", "personality"}

# Columns that use -1 as a sentinel meaning "not applicable".
# Those values are replaced with NaN before computing mean/std.
SENTINEL_COLS = {"timeToFindKey"}


# ── Load data ─────────────────────────────────────────────────────────────────


def load_files(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep=SEPARATOR)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.columns = df.columns.str.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Compute mean and std ──────────────────────────────────────────────────────


def compute_stats(df: pd.DataFrame):
    numeric_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in NON_METRIC_COLS
    ]

    df = df.copy()
    for col in SENTINEL_COLS:
        if col in df.columns:
            df[col] = df[col].replace(-1, float("nan")).astype(float)

    grouped = df.groupby("personality", sort=True)[numeric_cols]
    means = grouped.mean()
    stds = grouped.std()

    # Interleave mean/std columns in the CSV: col_mean, col_std, ...
    frames = []
    for col in numeric_cols:
        frames.append(means[col].rename(f"{col}_mean"))
        frames.append(stds[col].rename(f"{col}_std"))

    result = pd.concat(frames, axis=1).reset_index()
    result = result.round(3)
    return result, numeric_cols


# ── Compute share metrics ─────────────────────────────────────────────────────


def compute_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per personality:
      - keyPickupPct   : % of episodes in which that personality held the key
      - killSharePct   : % of all dragon kills across all personalities
      - hitSharePct    : % of all hits inflicted across all personalities
    """
    by_p = df.groupby("personality", sort=True)

    total_episodes = df["episode"].nunique()
    total_kills = df["dragonsKilled"].sum()
    total_hits = df["hitsInflicted"].sum()

    shares = pd.DataFrame(
        {
            "personality": by_p["hasKey"].sum().index,
            "keyPickupPct": (by_p["hasKey"].sum() / total_episodes * 100)
            .round(1)
            .values,
            "killSharePct": (by_p["dragonsKilled"].sum() / total_kills * 100)
            .round(1)
            .values,
            "hitSharePct": (by_p["hitsInflicted"].sum() / total_hits * 100)
            .round(1)
            .values,
        }
    )
    return shares


# ── Pretty print ──────────────────────────────────────────────────────────────


def print_pretty(result: pd.DataFrame, numeric_cols: list[str]) -> None:
    personalities = result["personality"].tolist()

    col_w = 35  # metric name column width
    cell_w = 22  # per-personality cell width

    header = f"{'metric':<{col_w}}" + "".join(f"{p:^{cell_w}}" for p in personalities)
    sep = "─" * len(header)

    print(sep)
    print(header)
    print(sep)

    for col in numeric_cols:
        row_str = f"{col:<{col_w}}"
        for _, row in result.iterrows():
            mean = row[f"{col}_mean"]
            std = row[f"{col}_std"]
            cell = f"{mean:.3f} (± {std:.3f})"
            row_str += f"{cell:^{cell_w}}"
        print(row_str)

    print(sep)


def print_shares(shares: pd.DataFrame) -> None:
    personalities = shares["personality"].tolist()
    share_cols = ["keyPickupPct", "killSharePct", "hitSharePct"]
    labels = {
        "keyPickupPct": "key pickup %",
        "killSharePct": "kill share %",
        "hitSharePct": "hit share %",
    }

    col_w = 35
    cell_w = 22

    header = f"{'metric':<{col_w}}" + "".join(f"{p:^{cell_w}}" for p in personalities)
    sep = "─" * len(header)

    print(sep)
    print(header)
    print(sep)
    for col in share_cols:
        row_str = f"{labels[col]:<{col_w}}"
        for _, row in shares.iterrows():
            cell = f"{row[col]:.1f}%"
            row_str += f"{cell:^{cell_w}}"
        print(row_str)
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python cumulative_by_personality.py <file1.csv> [file2.csv ...]")
        sys.exit(1)

    input_paths = sys.argv[1:]
    print(f"Loading {len(input_paths)} file(s): {', '.join(input_paths)}")

    df = load_files(input_paths)
    print(
        f"  → {len(df)} rows, {df['episode'].nunique()} unique episodes, "
        f"{df['personality'].nunique()} personalities\n"
    )

    result, numeric_cols = compute_stats(df)
    shares = compute_shares(df)

    print_pretty(result, numeric_cols)
    print()
    print("── Share metrics ────────────────────────────────────────────────────")
    print_shares(shares)

    # Merge shares into result for saving
    combined = result.merge(shares, on="personality")
    first = pathlib.Path(input_paths[0])
    output_file = first.parent / f"{first.stem}_cumulative{first.suffix}"
    combined.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
