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

parser.add_argument(
    "--smooth",
    type=float,
    default=0.92,
    help="EMA smoothing factor (0=none, <1=more smooth)",
)
args = parser.parse_args()

if args.cumulative:
    tag = PRESETS["cumulative"]
    single_series = False
elif args.group_cumulative:
    tag = PRESETS["group_cumulative"]
    single_series = True
elif args.episode_length:
    tag = PRESETS["episode_length"]
    single_series = True
else:
    tag = args.tag
    single_series = False

RESULTS_DIR = Path(args.run_folder)
if not RESULTS_DIR.exists():
    raise FileNotFoundError(f"Run folder not found: {RESULTS_DIR}")

OUTPUT_FILE = RESULTS_DIR / f"{tag.replace('/', '_')}.pdf"


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
    if n > 6:
        ax.legend(
            fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0
        )
    else:
        ax.legend(fontsize=9)

ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, bbox_inches="tight")
print(f"Saved to: {OUTPUT_FILE}")
plt.show()
