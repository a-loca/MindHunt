from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle"
)

PRESETS = {
    "cumulative": "Environment/Cumulative Reward",
    "group_cumulative": "Environment/Group Cumulative Reward",
    "episode_length": "Environment/Episode Length",
}

parser = argparse.ArgumentParser()
parser.add_argument("run_folder", help="Path to the ML-Agents run folder")

tag_group = parser.add_mutually_exclusive_group(required=True)
tag_group.add_argument(
    "--cumulative", action="store_true", help="Plot cumulative reward per run"
)
tag_group.add_argument(
    "--group-cumulative",
    action="store_true",
    help="Plot group cumulative reward (single series)",
)
tag_group.add_argument(
    "--episode-length", action="store_true", help="Plot episode length (single series)"
)
tag_group.add_argument("--tag", help="Custom TensorBoard scalar tag")
tag_group.add_argument(
    "--all",
    action="store_true",
    help="Plot all presets (cumulative, group-cumulative, episode-length)",
)

parser.add_argument(
    "--smooth",
    type=float,
    default=0.92,
    help="EMA smoothing factor (0=none, <1=more smooth)",
)
parser.add_argument(
    "--summary",
    action="store_true",
    help="Print training time and final value per run for each plotted tag",
)
parser.add_argument(
    "--max-gap",
    type=float,
    default=10,
    help="Gaps between logged points longer than this (minutes) are treated "
    "as a training pause (e.g. from --resume) and excluded from training time",
)
args = parser.parse_args()

if args.all:
    jobs = [
        (PRESETS["cumulative"], False),
        (PRESETS["group_cumulative"], True),
        (PRESETS["episode_length"], True),
    ]
elif args.cumulative:
    jobs = [(PRESETS["cumulative"], False)]
elif args.group_cumulative:
    jobs = [(PRESETS["group_cumulative"], True)]
elif args.episode_length:
    jobs = [(PRESETS["episode_length"], True)]
else:
    jobs = [(args.tag, False)]

RESULTS_DIR = Path(args.run_folder)
if not RESULTS_DIR.exists():
    raise FileNotFoundError(f"Run folder not found: {RESULTS_DIR}")


def ema_smooth(values, alpha):
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return np.array(smoothed)


def plot_series(ax, steps, values, label=None, annotate=True):
    smoothed = ema_smooth(values, args.smooth) if args.smooth > 0 else values
    (line,) = ax.plot(steps, smoothed, linewidth=2, label=label)
    if args.smooth > 0:
        ax.plot(steps, values, linewidth=0.6, alpha=0.2, color=line.get_color())
    if annotate:
        ax.annotate(
            f"{smoothed[-1]:.2f}",
            xy=(steps[-1], smoothed[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=line.get_color(),
        )


def format_duration(seconds):
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def active_training_time(events):
    wall_times = np.array([e.wall_time for e in events])
    if len(wall_times) < 2:
        return 0.0
    deltas = np.diff(wall_times)
    threshold = args.max_gap * 60  # minutes -> seconds
    paused = deltas[deltas > threshold]
    for gap in paused:
        print(f"  (excluded pause: {format_duration(gap)})")
    return deltas[deltas <= threshold].sum()


def print_summary(tag, single_series):
    print(f"\n--- {tag} ---")
    # Looks for events.out.tfevents.* files
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            # Reads all events.out.tfevents.* files in one
            # consecutive stream
            event_acc = EventAccumulator(str(run_dir))
            event_acc.Reload()
            if tag not in event_acc.Tags()["scalars"]:
                continue
            events = event_acc.Scalars(tag)
            if not events:
                continue
            final_value = events[-1].value
            if single_series:
                elapsed = active_training_time(events)
                print(
                    f"training time={format_duration(elapsed)}, "
                    f"final={final_value:.3f} (step {events[-1].step})"
                )
                break
            print(f"{run_dir.name}: final={final_value:.3f} (step {events[-1].step})")
        except Exception as e:
            print(f"Error in {run_dir.name}: {e}")


def make_plot(tag, single_series):
    file_name = tag.split("/")[1].replace(" ", "_")
    output_file = RESULTS_DIR / f"{file_name}.pdf"
    annotate = tag != PRESETS["episode_length"]

    fig, ax = plt.subplots(figsize=(10, 5))

    if single_series:
        # Find the first run directory that has the tag and plot it alone
        for run_dir in sorted(RESULTS_DIR.iterdir()):
            if not run_dir.is_dir():
                continue
            try:
                event_acc = EventAccumulator(str(run_dir))
                event_acc.Reload()
                if tag not in event_acc.Tags()["scalars"]:
                    continue
                events = event_acc.Scalars(tag)
                steps = np.array([e.step for e in events])
                values = np.array([e.value for e in events])
                plot_series(ax, steps, values, annotate=annotate)
                break
            except Exception as e:
                print(f"Error in {run_dir.name}: {e}")
    else:
        for run_dir in sorted(RESULTS_DIR.iterdir()):
            if not run_dir.is_dir():
                continue
            try:
                event_acc = EventAccumulator(str(run_dir))
                event_acc.Reload()
                if tag not in event_acc.Tags()["scalars"]:
                    continue
                events = event_acc.Scalars(tag)
                steps = np.array([e.step for e in events])
                values = np.array([e.value for e in events])
                plot_series(ax, steps, values, label=run_dir.name, annotate=annotate)
            except Exception as e:
                print(f"Error in {run_dir.name}: {e}")

    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel(tag.split("/")[-1], fontsize=12)
    ax.set_title(tag.split("/")[-1], fontsize=13, fontweight="bold")
    ax.margins(x=0.08)

    if not single_series:
        n = len(ax.get_lines())
        fontsize = 8 if n > 6 else 9
        ax.legend(fontsize=fontsize, loc="best", framealpha=0.7)

    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Saved to: {output_file}")


for tag, single_series in jobs:
    make_plot(tag, single_series)
    if args.summary and tag != PRESETS["episode_length"]:
        print_summary(tag, single_series)

plt.show()
