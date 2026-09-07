"""
Save std_raw/mu_ema values for one node, one per alpha-sweep run, as separate
text files -- e.g. node185_eight_room_alpha0.00.txt, ..._alpha0.05.txt.

Reads {env_dir}/alpha_*/{prefix}_mu_ema.csv, filters to the given node id,
and writes one text file per alpha subdirectory found, in the same
'visit iter std_raw mu_ema' format used throughout this analysis (visit is
the 0-indexed order the rows appear in the source CSV -- i.e. chronological
visit order for that node).

--out-dir controls where the files are written (created automatically if it
doesn't exist yet, default: current directory). The filename itself already
includes the node number and maze-environment prefix by default
(node{N}_{prefix}_alpha{X}.txt) -- --out-prefix only needs to be passed to
override that naming, not to relocate the files (use --out-dir for that).

Usage (run from v3/, where {env_dir}/alpha_*/ already exist):
    python save_node_values.py --node 185 --env-dir 8room --prefix eight_room
    python save_node_values.py --node 4   --env-dir decoy --prefix decoy
    python save_node_values.py --node 43  --env-dir decoy --prefix decoy \\
        --out-dir decoy/node_43_mu_ema_plots/text_files
"""

from __future__ import annotations

import argparse
import csv
import glob
import os


def save_node_file(node: str, mu_ema_csv: str, out_path: str):
    with open(mu_ema_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["node"] == node]

    with open(out_path, "w") as f:
        f.write(f"Node {node} values from {mu_ema_csv}\n")
        f.write(f"Total visits: {len(rows)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'visit':>5} {'iter':>4} {'std_raw':>10} {'mu_ema':>10}\n")
        for i, r in enumerate(rows):
            f.write(f"{i:>5} {r['iter']:>4} {float(r['std_raw']):>10.6f} "
                    f"{float(r['mu_ema']):>10.6f}\n")
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node", required=True, help="node id to extract (e.g. 185)")
    p.add_argument("--env-dir", required=True,
                   help="results directory containing alpha_* subdirs (e.g. 8room)")
    p.add_argument("--prefix", required=True,
                   help="{prefix}_mu_ema.csv filename prefix (e.g. eight_room)")
    p.add_argument("--out-dir", default=".",
                   help="directory to write the text files into; created "
                        "automatically if it doesn't exist (default: current directory)")
    p.add_argument("--out-prefix", default=None,
                   help="output filename prefix, without directory "
                        "(default: node{N}_{prefix}, i.e. node number + maze env)")
    args = p.parse_args()

    out_prefix = args.out_prefix or f"node{args.node}_{args.prefix}"
    os.makedirs(args.out_dir, exist_ok=True)

    subs = sorted(glob.glob(os.path.join(args.env_dir, "alpha_*")))
    if not subs:
        raise SystemExit(f"no alpha_* subdirectories found under {args.env_dir}/")

    for sub in subs:
        alpha = os.path.basename(sub).split("alpha_")[-1]
        mu_ema_csv = os.path.join(sub, f"{args.prefix}_mu_ema.csv")
        if not os.path.exists(mu_ema_csv):
            print(f"[skip] {mu_ema_csv} not found")
            continue
        out_path = os.path.join(args.out_dir, f"{out_prefix}_alpha{alpha}.txt")
        n = save_node_file(args.node, mu_ema_csv, out_path)
        print(f"saved -> {out_path}  ({n} rows)")


if __name__ == "__main__":
    main()
