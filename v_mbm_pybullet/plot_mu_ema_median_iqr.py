"""
Median mu_ema + interquartile box across ALL nodes in the graph, per
iteration, for every alpha in one maze environment's sweep -- answers "does
the whole graph, not just a hand-picked node, show the same spike-then-decline
shape," one image per alpha.

For each iteration, every node visited *in that iteration* contributes its
end-of-iteration mu_ema (the last value logged for that node that iteration --
same definition used throughout this analysis). The median and a box (Q1-Q3,
whiskers to 1.5xIQR) of that set are plotted against iteration, with a second
panel below showing how many nodes contributed to each iteration's statistic
(early iterations can rest on a thin sample, which matters for reading the
box honestly).

Reads {env}/alpha_*/{prefix}_mu_ema.csv for every alpha subdirectory found
under --env, and writes one {prefix}_alpha{X}_mu_ema_median_iqr.png per alpha
into --out-dir.

Usage:
    python plot_mu_ema_median_iqr.py --env 4room --out-dir mu_ema_median_iqr/4room
    python plot_mu_ema_median_iqr.py --env 8room --out-dir mu_ema_median_iqr/8room
    python plot_mu_ema_median_iqr.py --env decoy --out-dir mu_ema_median_iqr/decoy
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# maze environment (results directory name) -> {prefix}_mu_ema.csv filename prefix
ENV_INFO = {
    "4room": "four_room",
    "8room": "eight_room",
    "decoy": "decoy",
}


def load_mu_ema_log(csv_path):
    """Single pass over the mu_ema.csv log. Returns:
      by_iter: {iter: [end-of-iteration mu_ema, one per node touched that
                iteration]} -- repeated visits to the same node within an
                iteration collapse to that node's last (end-of-iteration) value.
      touched_by_iter: {iter: set(node ids touched that iteration)}
      first_iter: {node: the iteration of that node's first-ever logged
                   appearance in this file -- i.e. its first-tracked event,
                   NOT necessarily when the node was created in the graph
                   (brand-new graph nodes are never logged until later
                   re-approached; see load_mu_ema_log's docstring note above)}
    """
    last = {}
    first_iter = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            it = int(row["iter"])
            node = row["node"]
            last[(it, node)] = float(row["mu_ema"])
            if node not in first_iter:
                first_iter[node] = it

    by_iter = defaultdict(list)
    touched_by_iter = defaultdict(set)
    for (it, node), mu in last.items():
        by_iter[it].append(mu)
        touched_by_iter[it].add(node)
    return by_iter, touched_by_iter, first_iter


def plot_one_alpha(prefix, alpha, csv_path, out_dir):
    by_iter, touched_by_iter, first_iter = load_mu_ema_log(csv_path)
    iters = sorted(by_iter)
    data = [by_iter[i] for i in iters]
    medians = [float(np.median(v)) for v in data]

    # first-tracked (this is this node's first-ever mu_ema.csv row, anywhere)
    # vs. previously-tracked (already had at least one row in an earlier
    # iteration) -- see load_mu_ema_log for the precise definition/caveats.
    new_counts = [sum(1 for node in touched_by_iter[i] if first_iter[node] == i)
                  for i in iters]
    prev_counts = [len(touched_by_iter[i]) - n for i, n in zip(iters, new_counts)]

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax.boxplot(
        data, positions=iters, widths=0.6, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="#c44e52", alpha=0.35, edgecolor="#c44e52"),
        whiskerprops=dict(color="#c44e52"), capprops=dict(color="#c44e52"),
        medianprops=dict(color="black", linewidth=1.3),
    )
    ax.plot(iters, medians, color="#c44e52", lw=1.3, marker="o", ms=3,
            zorder=5, label="median mu_ema (connecting line)")
    ax.set_ylabel("mu_ema")
    ax.set_title(f"{prefix}, alpha={alpha}: median mu_ema + IQR box "
                 f"across all nodes, by iteration")
    ax.grid(alpha=0.25)
    box_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#c44e52", alpha=0.35,
                              edgecolor="#c44e52", label="IQR box (Q1-Q3), whiskers to 1.5xIQR")
    ax.legend(handles=[box_patch, ax.lines[-1]], fontsize=8)

    axn.bar(iters, prev_counts, color="#999999", alpha=0.85, width=0.7,
            label="previously-tracked")
    axn.bar(iters, new_counts, bottom=prev_counts, color="#3b3b3b", alpha=0.9,
            width=0.7, label="first-tracked")
    axn.set_ylabel("# nodes")
    axn.set_xlabel("iteration")
    axn.legend(fontsize=7, loc="upper right")
    # boxplot() installs a FixedFormatter on the shared x-axis that labels
    # ticks by draw order rather than value; MaxNLocator alone would inherit
    # that stale formatter and mislabel the (correctly positioned) ticks it
    # picks, so the formatter must be reset back to a plain numeric one too.
    axn.xaxis.set_major_locator(MaxNLocator(integer=True))
    axn.xaxis.set_major_formatter(ScalarFormatter())
    axn.grid(alpha=0.2, axis="y")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{prefix}_alpha{alpha}_mu_ema_median_iqr.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] saved -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, choices=list(ENV_INFO),
                   help="maze environment / results directory (4room, 8room, decoy)")
    p.add_argument("--out-dir", required=True,
                   help="directory to save the plots into; created automatically if missing")
    args = p.parse_args()

    prefix = ENV_INFO[args.env]
    subs = sorted(glob.glob(os.path.join(args.env, "alpha_*")))
    if not subs:
        raise SystemExit(f"no alpha_* subdirectories found under {args.env}/")

    for sub in subs:
        alpha = os.path.basename(sub).split("alpha_")[-1]
        csv_path = os.path.join(sub, f"{prefix}_mu_ema.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        plot_one_alpha(prefix, alpha, csv_path, args.out_dir)


if __name__ == "__main__":
    main()
