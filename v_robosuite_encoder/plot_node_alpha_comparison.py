"""
Compare mu_ema for one node across the alpha sweep, one line per alpha, using
only the LAST (end-of-iteration) mu_ema value per iteration -- the value
actually carried forward into the next EMA update -- instead of every
individual visit, so the plot stays readable.

Iterations where a given alpha never visited the node are real gaps, not
missing data to interpolate over: the line is broken there (no segment drawn
across a gap), and a second panel below explicitly marks, per alpha per
iteration, whether the node was visited at all -- a colored square if
visited, a gray x if not -- so "not visited" is never confused with "visited
but flat" or silently smoothed over.

Single generalized replacement for the earlier one-script-per-node variants
(plot_node4_alpha_comparison.py, plot_node185_alpha_comparison.py, etc.).

Reads node{N}_*_alpha*.txt (visit/iter/std_raw/mu_ema rows, as produced by
save_node_values.py) from --text-dir.

Usage:
    python plot_node_alpha_comparison.py --node 274 --env "decoy maze" \\
        --text-dir decoy/node_274_mu_ema_plots/text_files \\
        --out-dir decoy/node_274_mu_ema_plots/plots
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")


def load_end_of_iter(path):
    """Returns {iter: last mu_ema logged in that iteration} for one file.
    Rows are already in chronological/visit order, so simply overwriting on
    each match keeps the last (end-of-iteration) value per iteration."""
    last = {}
    with open(path) as f:
        for line in f:
            m = ROW_RE.match(line)
            if not m:
                continue
            _, it, _, mu = m.groups()
            last[int(it)] = float(mu)
    return last


def alpha_from_filename(path):
    m = re.search(r"alpha([\d.]+)\.txt$", path)
    return m.group(1) if m else path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node", required=True, help="node id (e.g. 274)")
    p.add_argument("--env", required=True,
                   help="maze environment label shown in the title (e.g. 'decoy maze')")
    p.add_argument("--text-dir", required=True,
                   help="directory containing node{N}_*_alpha*.txt files")
    p.add_argument("--out-dir", required=True,
                   help="directory to save the plot into; created automatically if missing")
    p.add_argument("--out", default=None,
                   help="output image filename, without directory "
                        "(default: node{N}_alpha_comparison.png)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.text_dir, f"node{args.node}_*_alpha*.txt")))
    if not files:
        raise SystemExit(f"no node{args.node}_*_alpha*.txt files found in {args.text_dir}")

    runs = {alpha_from_filename(f): load_end_of_iter(f) for f in files}
    alphas = sorted(runs, key=float)
    max_iter = max(it for d in runs.values() for it in d)
    all_iters = list(range(max_iter + 1))

    cmap = matplotlib.colormaps["viridis"].resampled(len(alphas))
    color = {a: cmap(i) for i, a in enumerate(alphas)}

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    for a in alphas:
        d = runs[a]
        y = np.array([d.get(i, np.nan) for i in all_iters])
        ax.plot(all_iters, y, marker="o", ms=4, lw=1.6, color=color[a],
                label=f"alpha={a}")

    ax.set_ylabel("mu_ema (end of iteration)")
    ax.set_title(f"Node {args.node} ({args.env}): mu_ema vs. iteration, by alpha")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title="alpha", ncol=2)

    # visited / not-visited indicator strip
    for row, a in enumerate(alphas):
        d = runs[a]
        visited = [i for i in all_iters if i in d]
        missing = [i for i in all_iters if i not in d]
        axv.scatter(visited, [row] * len(visited), marker="s", s=45, color=color[a])
        axv.scatter(missing, [row] * len(missing), marker="x", s=35, color="lightgray")
    axv.set_yticks(range(len(alphas)))
    axv.set_yticklabels([f"alpha={a}" for a in alphas], fontsize=8)
    axv.set_xlabel("iteration")
    axv.set_ylabel("visited?")
    axv.xaxis.set_major_locator(MaxNLocator(integer=True))
    axv.grid(alpha=0.2, axis="x")
    axv.set_ylim(-0.7, len(alphas) - 0.3)

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = args.out or f"node{args.node}_alpha_comparison.png"
    out = os.path.join(args.out_dir, out_name)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
