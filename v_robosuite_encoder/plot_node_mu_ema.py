"""
Plot mu_ema (y-axis) vs. iteration (x-axis) from a saved node-values text
file (e.g. node4_decoy_alpha0.05.txt).

Every visit's mu_ema is plotted -- not raw_std, and not just end-of-iteration
values -- since mu_ema is the actual smoothed/operational quantity (the veto
rule only ever reads mu_ema, never raw_std). Multiple visits within the same
iteration are placed at (near) the same x position with a small jitter so
they don't perfectly overlap, and connected in visit order so both the
within-iteration blending and the across-iteration jumps are visible. The
last visit of each iteration -- the value actually carried forward into the
next iteration -- is highlighted with a larger marker and its own line.

--out-dir controls where the plot is saved (created automatically if it
doesn't exist yet, default: current directory); --out overrides just the
filename (default: <infile stem>_mu_ema.png), not the directory.

Usage:
    python plot_node_mu_ema.py node4_decoy_alpha0.05.txt [--out node4_mu_ema.png]
    python plot_node_mu_ema.py decoy/node_43_mu_ema_plots/text_files/node43_decoy_alpha0.05.txt \\
        --out-dir decoy/node_43_mu_ema_plots/plots
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")


def load(path):
    """Parses the 'visit iter std_raw mu_ema' rows out of the text file."""
    visits, iters, mu_ema = [], [], []
    with open(path) as f:
        for line in f:
            m = ROW_RE.match(line)
            if not m:
                continue
            visit, it, std_raw, mu = m.groups()
            visits.append(int(visit))
            iters.append(int(it))
            mu_ema.append(float(mu))
    return visits, iters, mu_ema


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("infile", help="node-values text file (e.g. node4_decoy_alpha0.05.txt)")
    p.add_argument("--out-dir", default=".",
                   help="directory to save the plot into; created automatically "
                        "if it doesn't exist (default: current directory)")
    p.add_argument("--out", default=None,
                   help="output image filename, without directory "
                        "(default: <infile stem>_mu_ema.png)")
    args = p.parse_args()

    visits, iters, mu_ema = load(args.infile)
    if not visits:
        raise SystemExit(f"no data rows parsed from {args.infile}")

    # small deterministic jitter for repeated visits within the same
    # iteration, purely so overlapping points are visible -- doesn't affect
    # any y-values or which point is "last" in an iteration
    jitter_x = []
    seen = {}
    for it in iters:
        k = seen.get(it, 0)
        jitter_x.append(it + 0.06 * k)
        seen[it] = k + 1

    # index of the last logged visit for each iteration -- the mu_ema value
    # actually carried forward into the next iteration's queries
    last_idx_per_iter = {}
    for i, it in enumerate(iters):
        last_idx_per_iter[it] = i
    end_idx = sorted(last_idx_per_iter.values())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(jitter_x, mu_ema, color="#4c72b0", lw=1.3, alpha=0.6,
            marker="o", ms=4, label="mu_ema (every visit)")
    ax.plot([jitter_x[i] for i in end_idx], [mu_ema[i] for i in end_idx],
            color="#c44e52", lw=1.8, ls="--", alpha=0.85, zorder=4)
    ax.scatter([jitter_x[i] for i in end_idx], [mu_ema[i] for i in end_idx],
               color="#c44e52", s=70, zorder=5, edgecolors="white",
               label="mu_ema (end of iteration)")

    ax.set_xlabel("iteration")
    ax.set_ylabel("mu_ema")
    ax.set_title(f"mu_ema vs. iteration -- {os.path.basename(args.infile)}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = args.out or (os.path.basename(args.infile).rsplit(".", 1)[0] + "_mu_ema.png")
    out = os.path.join(args.out_dir, out_name)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
