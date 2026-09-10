"""Generate go_explore_vs_transformer.txt for the new_maze_envs_v2 sweep.

One block per maze environment comparing:
  * path cost      -- Go-Explore (25 seeds)  vs  Transformer (over alpha runs)
  * nodes explored -- Go-Explore (25 seeds)  vs  Transformer (over alpha runs)
  * ratio = Transformer mean / Go-Explore mean  (<1 => Transformer smaller)

The v2 sweep has no mc_ema_beta dimension (the veto uses the raw MC-dropout std),
so the Transformer stats aggregate the alpha_0.00..0.10 runs directly.
Transformer runs that never reached the goal (goal_reached_by_collection = False
in {prefix}_paths.json) are dropped before aggregating.

Go-Explore  : results_go_explore/{prefix}/_run_log.txt  (per-seed len= / nodes=)
Transformer : {folder}/alpha_*/{prefix}_{paths.json,metrics.csv}
"""
import csv
import glob
import json
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))          # new_maze_envs_v2/
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

W = 92  # rule width

SEED_RE = re.compile(r"seed\s+\d+:\s+reached\s+len=\s*([0-9.]+)\s+nodes=\s*(\d+)")


def mean_std(xs):
    """(mean, sample std) -- std is 0.0 for a single value, None for empty."""
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return st.mean(xs), st.stdev(xs)


def load_go_explore(prefix):
    """path costs and node counts over the goal-reaching seeds."""
    costs, nodes = [], []
    with open(os.path.join(GE_DIR, prefix, "_run_log.txt")) as f:
        for line in f:
            m = SEED_RE.search(line)
            if m:
                costs.append(float(m.group(1)))
                nodes.append(int(m.group(2)))
    return costs, nodes


def load_transformer(folder, prefix):
    """(costs, nodes) over the alpha runs for one env that reached the goal."""
    costs, nodes = [], []
    for adir in sorted(glob.glob(os.path.join(ROOT, folder, "alpha_*")),
                       key=lambda p: float(os.path.basename(p).split("_")[1])):
        with open(os.path.join(adir, f"{prefix}_paths.json")) as f:
            j = json.load(f)
        if not j["goal_reached_by_collection"]:
            continue
        costs.append(float(j["transformer_path"]["cost"]))
        with open(os.path.join(adir, f"{prefix}_metrics.csv")) as f:
            nodes.append(int(list(csv.DictReader(f))[-1]["nodes"]))
    return costs, nodes


def ms(mean, std):
    return f"{mean:9.4f} +/- {std:<9.4f}" if mean is not None else f"{'n/a':>23}"


def ratio(tf_mean, ge_mean):
    if tf_mean is None or not ge_mean:
        return "n/a"
    return f"{tf_mean / ge_mean:.3f}x"


def main():
    L = []
    L.append("Go-Explore (25 seeds) vs. Transformer-guided search, per environment")
    L.append("Mean +/- sample std of path cost and nodes explored")
    L.append("=" * W)
    L.append("")
    L.append("transformer_cost = cost of the transformer's own predicted shortest path (the path it")
    L.append("actually produces), not the roadmap's Dijkstra-optimal cost.")
    L.append("Transformer stats aggregate the alpha_0.00..0.10 runs; runs whose collection never")
    L.append("reached the goal (goal_reached_by_collection = False) are excluded.")
    L.append("n_tf = number of alpha runs kept (out of 11).  n_ge = Go-Explore seeds.")
    L.append("Go-Explore veto: none (raw RRT + Go-Explore).  Transformer veto: raw MC-dropout std,")
    L.append("mc_samples=50, no EMA;  per-run seed=42, n_iters=20, k_trajectories=8, goal_bias=0.")
    L.append("ratio = Transformer mean / Go-Explore mean  (<1 means Transformer is smaller/better).")

    hdr = f"{'metric':<24}{'Go-Explore':>25}{'Transformer':>25}{'ratio (Tf/GE)':>16}"

    for folder, prefix in ENVS:
        ge_costs, ge_nodes = load_go_explore(prefix)
        gcm, gcs = mean_std(ge_costs)
        gnm, gns = mean_std(ge_nodes)

        tc, tn = load_transformer(folder, prefix)
        tcm, tcsd = mean_std(tc)
        tnm, tnsd = mean_std(tn)

        L.append("")
        L.append(f"{folder} ({prefix})" if folder != prefix else folder)
        L.append("-" * W)
        L.append(hdr)
        L.append("-" * W)
        L.append(f"{'n (runs)':<24}{len(ge_costs):>25d}{len(tc):>25d}{'':>16}")
        L.append(f"{'path cost':<24}{ms(gcm, gcs):>25}{ms(tcm, tcsd):>25}{ratio(tcm, gcm):>16}")
        L.append(f"{'nodes explored':<24}{ms(gnm, gns):>25}{ms(tnm, tnsd):>25}{ratio(tnm, gnm):>16}")
        if not tc:
            L.append("(no Transformer alpha run reached the goal -- Transformer stats n/a for this env)")

    L.append("")
    L.append("=" * W)
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
