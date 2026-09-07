"""Generate go_explore_vs_transformer.txt for the new_maze_envs sweep.

One table per maze environment. Columns = the four mc_ema_beta values. Each
beta column compares, for that beta:
  * path cost      -- Go-Explore (25 seeds)  vs  Transformer (over alpha runs)
  * nodes explored -- Go-Explore (25 seeds)  vs  Transformer (over alpha runs)
  * ratio = Transformer mean / Go-Explore mean  (<1 => Transformer smaller)

Transformer runs that never reached the goal (goal_reached_by_collection = False
in {prefix}_paths.json) are dropped before aggregating.

Go-Explore  : results_go_explore/{prefix}/_run_log.txt  (per-seed len= / nodes=)
Transformer : {folder}/beta_*/alpha_*/{prefix}_{paths.json,metrics.csv}
"""
import csv
import glob
import json
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))          # new_maze_envs/
OUT = os.path.join(ROOT, "go_explore_vs_transformer.txt")
GE_DIR = os.path.join(ROOT, "results_go_explore")

# (transformer folder, file prefix / go-explore folder)
ENVS = [
    ("4room", "four_room"),
    ("8room", "eight_room"),
    ("decoy", "decoy"),
    ("bug_trap", "bug_trap"),
    ("perfect_maze", "perfect_maze"),
    ("spiral", "spiral"),
]
BETAS = ["0.2", "0.4", "0.6", "0.8"]

SEED_RE = re.compile(r"seed\s+\d+:\s+reached\s+len=\s*([0-9.]+)\s+nodes=\s*(\d+)")


def mean_std(xs):
    """(mean, sample std) -- std is 0.0 for a single value, None for empty."""
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return st.mean(xs), st.stdev(xs)


def load_go_explore(prefix):
    """path costs and node counts over the 25 goal-reaching seeds."""
    costs, nodes = [], []
    with open(os.path.join(GE_DIR, prefix, "_run_log.txt")) as f:
        for line in f:
            m = SEED_RE.search(line)
            if m:
                costs.append(float(m.group(1)))
                nodes.append(int(m.group(2)))
    return costs, nodes


def load_transformer(folder, prefix, beta):
    """(costs, nodes) over the alpha runs for one (env, beta) that reached goal."""
    costs, nodes = [], []
    bdir = os.path.join(ROOT, folder, f"beta_{beta}")
    for adir in sorted(glob.glob(os.path.join(bdir, "alpha_*")),
                       key=lambda p: float(os.path.basename(p).split("_")[1])):
        with open(os.path.join(adir, f"{prefix}_paths.json")) as f:
            j = json.load(f)
        if not j["goal_reached_by_collection"]:
            continue
        costs.append(float(j["transformer_path"]["cost"]))
        with open(os.path.join(adir, f"{prefix}_metrics.csv")) as f:
            nodes.append(int(list(csv.DictReader(f))[-1]["nodes"]))
    return costs, nodes


def ms(mean, std, w=22):
    return f"{mean:8.4f} +/- {std:<9.4f}".ljust(w) if mean is not None else "n/a".ljust(w)


def ratio(tf_mean, ge_mean):
    if tf_mean is None or not ge_mean:
        return "n/a"
    return f"{tf_mean / ge_mean:.3f}x"


def main():
    L = []
    L.append("Go-Explore (25 seeds) vs. Transformer-guided search, by mc_ema_beta")
    L.append("Mean +/- sample std of path cost and nodes explored, per environment")
    L.append("=" * 100)
    L.append("")
    L.append("transformer_cost = cost of the transformer's own predicted shortest path (the path it")
    L.append("actually produces), not the roadmap's Dijkstra-optimal cost")
    L.append("Transformer stats aggregate the alpha_0.00..0.10 runs for that beta; runs whose")
    L.append("collection never reached the goal (goal_reached_by_collection = False) are excluded.")
    L.append("n = number of alpha runs kept for that beta (out of 11).")
    L.append("Go-Explore stats are over its 25 seeds and do not depend on beta (shown once per env).")
    L.append("ratio = Transformer mean / Go-Explore mean  (<1 means Transformer is smaller/better).")

    for folder, prefix in ENVS:
        ge_costs, ge_nodes = load_go_explore(prefix)
        gcm, gcs = mean_std(ge_costs)
        gnm, gns = mean_std(ge_nodes)

        L.append("")
        L.append(f"{folder} ({prefix})")
        L.append("-" * 100)
        L.append(f"Go-Explore   n={len(ge_costs):<3d}  "
                 f"path cost {gcm:.4f} +/- {gcs:.4f}    "
                 f"nodes explored {gnm:.2f} +/- {gns:.2f}")
        L.append("-" * 100)
        L.append(f"{'':28}" + "".join(f"beta_{b:<17}" for b in BETAS))

        rows = {"n": ["n (alpha runs kept)"],
                "tc": ["Transformer path cost"],
                "tcr": ["  ratio (Tf / GE)"],
                "tn": ["Transformer nodes expl."],
                "tnr": ["  ratio (Tf / GE)"]}
        for b in BETAS:
            tc, tn = load_transformer(folder, prefix, b)
            tcm, tcsd = mean_std(tc)
            tnm, tnsd = mean_std(tn)
            rows["n"].append(str(len(tc)).ljust(22))
            rows["tc"].append(ms(tcm, tcsd))
            rows["tcr"].append(ratio(tcm, gcm).ljust(22))
            rows["tn"].append(ms(tnm, tnsd))
            rows["tnr"].append(ratio(tnm, gnm).ljust(22))

        for key in ("n", "tc", "tcr", "tn", "tnr"):
            L.append(f"{rows[key][0]:28}" + "".join(rows[key][1:]))

    L.append("")
    L.append("=" * 100)
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
