"""
Plot Huber training loss vs. training iteration for one maze environment's
beta x alpha sweep: a single image with one subplot per mc_ema_beta value
(beta_0.2, 0.4, 0.6, 0.8), each holding one colored curve per alpha value
(alpha_0.00 .. alpha_0.10). The same alpha gets the same color in every
subplot.

Loss values are read from {env}/beta_*/alpha_*/{prefix}_metrics.csv (the
"loss" column -- identical to the per-iteration loss printed in
{prefix}_run.log). Output: plots_huber_loss/loss_{env}.png

Usage (run from new_maze_envs/):
    python plot_loss.py --env 4room
    python plot_loss.py --env decoy --out-dir plots_huber_loss
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

# results-folder name -> {prefix}_metrics.csv filename prefix
ENV_INFO = {
    "4room": "four_room",
    "8room": "eight_room",
    "decoy": "decoy",
    "bug_trap": "bug_trap",
    "perfect_maze": "perfect_maze",
    "spiral": "spiral",
}


def load_loss(run_dir: str, prefix: str):
    """Returns {alpha_str: (iters, loss_values)} for every alpha_* under
    run_dir, sorted by alpha."""
    runs = {}
    for sub in sorted(glob.glob(os.path.join(run_dir, "alpha_*"))):
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
    p.add_argument("--env", required=True, choices=list(ENV_INFO),
                   help="results-folder name (e.g. 4room, 8room, decoy, "
                        "bug_trap, perfect_maze, spiral)")
    p.add_argument("--root", default=".",
                   help="directory containing the {env}/ results folder (default: .)")
    p.add_argument("--out-dir", default="plots_huber_loss",
                   help="directory for the output image (default: plots_huber_loss)")
    p.add_argument("--out", default=None,
                   help="explicit output path; overrides --out-dir/loss_{env}.png")
    args = p.parse_args()

    prefix = ENV_INFO[args.env]
    env_root = os.path.join(args.root, args.env)
    beta_dirs = sorted(glob.glob(os.path.join(env_root, "beta_*")),
                       key=lambda d: float(os.path.basename(d).split("beta_")[-1]))
    if not beta_dirs:
        raise SystemExit(f"no beta_* subdirectories found under {env_root}/")

    # one color per alpha, shared across all beta subplots
    all_alphas = sorted({
        os.path.basename(sub).split("alpha_")[-1]
        for bdir in beta_dirs
        for sub in glob.glob(os.path.join(bdir, "alpha_*"))
    }, key=float)
    cmap = matplotlib.colormaps["viridis"].resampled(max(len(all_alphas), 1))
    color_by_alpha = {a: cmap(i) for i, a in enumerate(all_alphas)}

    n = len(beta_dirs)
    ncols = 2 if n > 2 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    flat = axes.flatten()

    handles_labels = None
    for ax, bdir in zip(flat, beta_dirs):
        beta = os.path.basename(bdir).split("beta_")[-1]
        runs = load_loss(bdir, prefix)
        if not runs:
            ax.set_title(f"beta={beta}\n(no data found in {bdir}/)")
            continue
        for alpha, (iters, losses) in runs.items():
            ax.plot(iters, losses, marker="o", ms=3, lw=1.6,
                    color=color_by_alpha.get(alpha), label=f"alpha={alpha}")
        ax.set_title(f"beta = {beta}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("Huber training loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    for ax in flat[n:]:                       # hide any unused axes
        ax.set_visible(False)

    if handles_labels:
        fig.legend(*handles_labels, title="alpha", fontsize=8,
                   loc="lower center", ncol=len(all_alphas))

    fig.suptitle(f"Huber training loss vs. training iteration -- "
                 f"{args.env} ({prefix}), by alpha, per beta")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    out = args.out or os.path.join(args.out_dir, f"loss_{args.env}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
