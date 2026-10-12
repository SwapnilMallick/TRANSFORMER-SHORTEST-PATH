"""
Summarize, per node, whether std_raw == mu_ema (node visited exactly once --
the EMA was only ever seeded, never blended) or std_raw != mu_ema at some
point (node visited more than once -- the EMA was actually blended at least
once). Saves one table per maze environment to a single text file.

A node's *first* logged visit always has std_raw == mu_ema by construction
(the EMA has nothing to blend with yet -- see predict_last_mc / mu_ema update
in online_cost_transformer.py). So classifying by "does this node have any
row where std_raw != mu_ema" is equivalent to "was this node visited more
than once" -- which is what this script reports, indirectly giving the
fraction of nodes revisited during collection, for each alpha value.

Reads {env_dir}/alpha_*/{prefix}_mu_ema.csv (node, std_raw, mu_ema columns).

Usage (run from v3/, where 4room/, 8room/, decoy/ already exist):
    python node_visit_summary.py [--out node_visit_summary.txt]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

# (results subdirectory, filename prefix, table title)
ENV_DIRS = [
    ("4room", "four_room", "4-room maze"),
    ("8room", "eight_room", "8-room maze"),
    ("decoy", "decoy", "decoy maze"),
]


def node_visit_counts(mu_ema_csv: str):
    """Returns (n_single_visit, n_multi_visit) nodes for one mu_ema.csv.

    A node's row count in the log equals its number of veto-time visits (one
    row is appended per visit -- see collect_trajectory's ema_log). A node
    with exactly one row was only ever seeded (std_raw == mu_ema on that row,
    by construction); a node with more than one row was actually blended on
    every row after the first (std_raw != mu_ema there)."""
    visits = defaultdict(int)
    with open(mu_ema_csv) as f:
        for row in csv.DictReader(f):
            visits[row["node"]] += 1

    n_single = sum(1 for v in visits.values() if v == 1)
    n_multi = sum(1 for v in visits.values() if v > 1)
    return n_single, n_multi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="node_visit_summary.txt",
                   help="output text file (default: node_visit_summary.txt)")
    args = p.parse_args()

    lines = []
    lines.append("Per-node visit summary from mu_ema.csv logs")
    lines.append("std_raw == mu_ema  ->  node visited exactly once (EMA only ever seeded)")
    lines.append("std_raw != mu_ema  ->  node visited more than once (EMA actually blended)")
    lines.append("=" * 78)

    for env_dir, prefix, title in ENV_DIRS:
        lines.append("")
        lines.append(f"{title} ({env_dir})")
        lines.append("-" * 78)
        lines.append(f"{'alpha':>6} {'single-visit':>13} {'multi-visit':>12} "
                      f"{'total nodes':>12} {'frac multi-visit':>17}")
        for sub in sorted(glob.glob(os.path.join(env_dir, "alpha_*"))):
            alpha = os.path.basename(sub).split("alpha_")[-1]
            mu_ema_csv = os.path.join(sub, f"{prefix}_mu_ema.csv")
            if not os.path.exists(mu_ema_csv):
                continue
            n_single, n_multi = node_visit_counts(mu_ema_csv)
            total = n_single + n_multi
            frac = n_multi / total if total else float("nan")
            lines.append(f"{alpha:>6} {n_single:>13} {n_multi:>12} "
                          f"{total:>12} {frac:>16.3f}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary] saved -> {args.out}")


if __name__ == "__main__":
    main()
