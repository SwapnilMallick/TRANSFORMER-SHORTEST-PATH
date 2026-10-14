"""
Plot Huber training loss vs. training iteration, one subplot per maze
environment, one colored line per alpha value in that environment's sweep.

Reads {env_dir}/alpha_*/{prefix}_metrics.csv (iter, loss columns) from the
existing online_cost_transformer.py --alpha-sweep results and saves a single
combined image with three side-by-side subplots (4room, 8room, decoy). The
same alpha value gets the same color across all three subplots.

Usage (run from v3/, where 4room/, 8room/, decoy/ already exist):
    python plot_loss.py [--out loss_by_alpha.png]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# (results subdirectory, filename prefix, subplot title)
ENV_DIRS = [
    ("4room", "four_room", "4-room maze"),
    ("8room", "eight_room", "8-room maze"),
    ("decoy", "decoy", "decoy maze"),
]


def load_loss(env_dir: str, prefix: str):
    """Returns {alpha_str: (iters, loss_values)}, sorted by alpha."""
    runs = {}
    for sub in sorted(glob.glob(os.path.join(env_dir, "alpha_*"))):
        alpha = os.path.basename(sub).split("alpha_")[-1]
        mc = os.path.join(sub, f"{prefix}_metrics.csv")
        if not os.path.exists(mc):
            continue
        iters, losses = [], []
        with open(mc) as f:
            for row in csv.DictReader(f):
                iters.append(int(row["iter"]))
                losses.append(float(row["loss"]))
        runs[alpha] = (iters, losses)
    return dict(sorted(runs.items(), key=lambda kv: float(kv[0])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="loss_by_alpha.png",
                   help="output image path (default: loss_by_alpha.png)")
    args = p.parse_args()

    # all alpha strings seen across every environment, so a given alpha gets
    # the same color on every subplot
    all_alphas = sorted({
        os.path.basename(sub).split("alpha_")[-1]
        for env_dir, _, _ in ENV_DIRS
        for sub in glob.glob(os.path.join(env_dir, "alpha_*"))
    }, key=float)
    cmap = matplotlib.colormaps["viridis"].resampled(max(len(all_alphas), 1))
    color_by_alpha = {a: cmap(i) for i, a in enumerate(all_alphas)}

    fig, axes = plt.subplots(1, len(ENV_DIRS), figsize=(6 * len(ENV_DIRS), 5))
    if len(ENV_DIRS) == 1:
        axes = [axes]

    for ax, (env_dir, prefix, title) in zip(axes, ENV_DIRS):
        runs = load_loss(env_dir, prefix)
        if not runs:
            ax.set_title(f"{title}\n(no data found in {env_dir}/)")
            continue
        for alpha, (iters, losses) in runs.items():
            ax.plot(iters, losses, marker="o", ms=3, lw=1.6,
                    color=color_by_alpha.get(alpha), label=f"alpha={alpha}")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel("Huber training loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="alpha")

    fig.suptitle("Huber training loss vs. training iteration, by alpha")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"[plot] saved -> {args.out}")


if __name__ == "__main__":
    main()
