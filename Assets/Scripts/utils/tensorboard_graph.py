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


def plot_series(ax, steps, values, label=None):
    smoothed = ema_smooth(values, args.smooth) if args.smooth > 0 else values
    (line,) = ax.plot(steps, smoothed, linewidth=2, label=label)
    if args.smooth > 0:
        ax.plot(steps, values, linewidth=0.6, alpha=0.2, color=line.get_color())


def make_plot(tag, single_series):
    file_name = tag.split("/")[1].replace(" ", "_")
    output_file = RESULTS_DIR / f"{file_name}.pdf"

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
                plot_series(ax, steps, values)
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
                plot_series(ax, steps, values, label=run_dir.name)
            except Exception as e:
                print(f"Error in {run_dir.name}: {e}")

    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel(tag.split("/")[-1], fontsize=12)
    ax.set_title(tag.split("/")[-1], fontsize=13, fontweight="bold")

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

plt.show()
