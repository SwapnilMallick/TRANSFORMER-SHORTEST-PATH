"""Generate dijkstra_vs_transformer_costs.txt for the new_maze_envs_v2 alpha sweep.

One table per maze environment, columns
    alpha goal_reached nodes_explored dijkstra_cost transformer_cost ratio dist_to_goal
The v2 sweep has no beta/EMA dimension (the veto uses the raw MC-dropout std),
so each env is a single flat per-alpha table.
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))   # new_maze_envs_v2/
OUT = os.path.join(ROOT, "dijkstra_vs_transformer_costs.txt")

# (folder, file-prefix)
ENVS = [
    ("4room", "four_room"),
    ("8room", "eight_room"),
    ("decoy", "decoy"),
    ("bug_trap", "bug_trap"),
    ("perfect_maze", "perfect_maze"),
    ("spiral", "spiral"),
]

W = 116  # rule width

HDR = (f"{'alpha':>5}  {'goal_reached':>12}  {'nodes_explored':>14}  "
       f"{'dijkstra_cost':>13}  {'transformer_cost':>16}  {'ratio':>7}  {'dist_to_goal':>12}")


def load_run(run_dir, prefix):
    with open(os.path.join(run_dir, f"{prefix}_paths.json")) as f:
        j = json.load(f)
    with open(os.path.join(run_dir, f"{prefix}_metrics.csv")) as f:
        nodes = int(list(csv.DictReader(f))[-1]["nodes"])
    dk = j["roadmap_dijkstra_path"]["cost"]
    tr = j["transformer_path"]["cost"]
    return {
        "reached": bool(j["goal_reached_by_collection"]),
        "nodes": nodes,
        "dk": dk,
        "tr": tr,
        "ratio": (tr / dk) if dk else float("nan"),
        "dist": j["roadmap_dijkstra_path"]["target_dist_from_goal"],
    }


def fmt_row(alpha_label, r):
    return (f"{alpha_label:>5}  {str(r['reached']):>12}  "
            f"{r['nodes']:>14d}  {r['dk']:>13.4f}  {r['tr']:>16.4f}  "
            f"{r['ratio']:>7.3f}  {r['dist']:>12.4f}")


def main():
    lines = []
    lines.append("Dijkstra roadmap optimum vs. Transformer-predicted path cost, per alpha")
    lines.append("Source: {env}_paths.json (costs) and {env}_metrics.csv (final node count)")
    lines.append("in each alpha_* subdirectory under new_maze_envs_v2/"
                 "{4room, 8room, decoy, bug_trap, perfect_maze, spiral}")
    lines.append("dijkstra_cost = roadmap_dijkstra_path.cost | transformer_cost = transformer_path.cost")
    lines.append("ratio = transformer_cost / dijkstra_cost (1.0 = transformer path matches Dijkstra optimum)")
    lines.append("nodes_explored = size of the roadmap graph at the end of the run (last row of metrics.csv)")
    lines.append("alpha = fixed uncertainty discount in the cost-to-come veto")
    lines.append("dist_to_goal = roadmap_dijkstra_path.target_dist_from_goal (nearest reached node -> true goal)")
    lines.append("Per-run config: seed=42, n_iters=20, k_trajectories=8, goal_bias=0 "
                 "(veto uncertainty = raw MC-dropout std, mc_samples=50; no EMA)")
    lines.append("=" * W)

    not_reached = []  # (env_label, alpha, dist)

    for folder, prefix in ENVS:
        label = folder if folder == prefix else f"{folder} ({prefix})"

        alpha_dirs = sorted(glob.glob(os.path.join(ROOT, folder, "alpha_*")),
                            key=lambda p: float(os.path.basename(p).split("_")[1]))

        # pre-scan for the per-env caveat line
        env_runs = [load_run(adir, prefix)["reached"] for adir in alpha_dirs]
        n_miss = env_runs.count(False)

        lines.append("")
        lines.append(label)
        if n_miss:
            lines.append(f"({n_miss}/{len(env_runs)} runs did NOT reach the goal - "
                         f"their rows compare costs to the nearest reached node; see NOTE at end)")
        lines.append("-" * W)
        lines.append(HDR)
        lines.append("-" * W)

        ratios_ok = []
        for adir in alpha_dirs:
            alpha_label = os.path.basename(adir).split("_")[1]
            r = load_run(adir, prefix)
            lines.append(fmt_row(alpha_label, r))
            if r["reached"]:
                ratios_ok.append(r["ratio"])
            else:
                not_reached.append((label, alpha_label, r["dist"]))

        lines.append("-" * W)
        if ratios_ok:
            summary = (f"mean ratio over {len(ratios_ok)}/{len(alpha_dirs)} "
                       f"goal-reached rows = {sum(ratios_ok) / len(ratios_ok):.3f}")
        else:
            summary = f"no goal-reached rows (0/{len(alpha_dirs)}) - mean ratio n/a"
        lines.append(summary)

    lines.append("")
    lines.append("=" * W)
    lines.append("NOTE on goal_reached = False rows:")
    lines.append("  goal_reached_by_collection = False means trajectory collection never got within")
    lines.append("  goal_radius (0.05) of the true goal. For those rows both costs are to the NEAREST")
    lines.append("  reached node (dist_to_goal away from the true goal), not to the goal itself, so their")
    lines.append("  ratio is not directly comparable to the goal-reaching rows.")
    lines.append("")
    if not_reached:
        by_env = {}
        for env_label, alpha, dist in not_reached:
            by_env.setdefault(env_label, []).append((alpha, dist))
        for env_label, rows in by_env.items():
            lines.append(f"  {env_label}: {len(rows)} such run(s)")
            for alpha, dist in rows:
                lines.append(f"      alpha={alpha}   nearest reached node {dist:.3f} from goal")
    else:
        lines.append("  (none)")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
