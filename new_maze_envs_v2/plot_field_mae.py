"""
Plot field cost-to-come MAE vs. training iteration for the whole
new_maze_envs_v2 alpha sweep: a single image with one subplot per maze
environment (4room, 8room, decoy, bug_trap, perfect_maze, spiral), each
holding one colored curve per alpha value (alpha_0.00 .. alpha_0.10). The
same alpha gets the same color in every subplot.

Field-MAE values are read from {env}/alpha_*/{prefix}_metrics.csv (the
"field_mae" column -- identical to the per-iteration "fieldMAE" printed in
{prefix}_run.log). The v2 sweep has no mc_ema_beta dimension, so there is a
single flat set of alpha_* runs per environment.
Output: plots_field_mae/field_mae_all_envs.png

Usage (run from new_maze_envs_v2/):
    python plot_field_mae.py
    python plot_field_mae.py --out-dir plots_field_mae
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


def load_field_mae(run_dir: str, prefix: str):
    """Returns {alpha_str: (iters, field_mae_values)} for every alpha_* under
    run_dir, sorted by alpha."""
    runs = {}
    for sub in sorted(glob.glob(os.path.join(run_dir, "alpha_*"))):
        alpha = os.path.basename(sub).split("alpha_")[-1]
        mc = os.path.join(sub, f"{prefix}_metrics.csv")
        if not os.path.exists(mc):
            continue
        iters, maes = [], []
        with open(mc) as f:
            for row in csv.DictReader(f):
                iters.append(int(row["iter"]))
                maes.append(float(row["field_mae"]))
        runs[alpha] = (iters, maes)
    return dict(sorted(runs.items(), key=lambda kv: float(kv[0])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".",
                   help="directory containing the six {env}/ results folders (default: .)")
    p.add_argument("--out-dir", default="plots_field_mae",
                   help="directory for the output image (default: plots_field_mae)")
    p.add_argument("--out", default=None,
                   help="explicit output path; overrides --out-dir/field_mae_all_envs.png")
    args = p.parse_args()

    # load every environment's alpha runs up front
    per_env = {}
    for env, prefix in ENV_INFO.items():
        per_env[env] = load_field_mae(os.path.join(args.root, env), prefix)
    if not any(per_env.values()):
        raise SystemExit(f"no {{env}}/alpha_*/*_metrics.csv found under {args.root}/")

    # one color per alpha, shared across all environment subplots
    all_alphas = sorted({a for runs in per_env.values() for a in runs}, key=float)
    cmap = matplotlib.colormaps["viridis"].resampled(max(len(all_alphas), 1))
    color_by_alpha = {a: cmap(i) for i, a in enumerate(all_alphas)}

    envs = list(ENV_INFO)
    n = len(envs)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows),
                             squeeze=False)
    flat = axes.flatten()

    handles_labels = None
    for ax, env in zip(flat, envs):
        prefix = ENV_INFO[env]
        runs = per_env[env]
        if not runs:
            ax.set_title(f"{env} ({prefix})\n(no data found)")
            ax.set_xlabel("iteration")
            ax.set_ylabel("field cost-to-come MAE")
            continue
        for alpha, (iters, maes) in runs.items():
            ax.plot(iters, maes, marker="o", ms=3, lw=1.6,
                    color=color_by_alpha.get(alpha), label=f"alpha={alpha}")
        ax.set_title(f"{env} ({prefix})")
        ax.set_xlabel("iteration")
        ax.set_ylabel("field cost-to-come MAE")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    for ax in flat[n:]:                       # hide any unused axes
        ax.set_visible(False)

    if handles_labels:
        fig.legend(*handles_labels, title="alpha", fontsize=8,
                   loc="lower center", ncol=len(all_alphas))

    fig.suptitle("Field cost-to-come MAE vs. training iteration, by alpha -- "
                 "per environment (new_maze_envs_v2)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    out = args.out or os.path.join(args.out_dir, "field_mae_all_envs.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
