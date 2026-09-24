"""
Full online training loop for the robosuite Lift/Panda red-cube task,
mirroring online_cost_transformer.py's main() (Algorithm 1 / Go-Explore
reset-replay -- see that module's docstring) but driving RobosuiteTask /
collect_trajectory_robosuite / find_uncertain_state_robosuite instead of the
2D maze. This is the main-loop driver that didn't previously exist alongside
task.py / robosuite_env.py / collect_trajectory_robosuite.py.

Reuses train_transformer / field_mae / build_sequences / CostTransformer /
Graph unchanged from online_cost_transformer.py (all already generic in
token_dim and cfg.diagonal); only the collection step and the cost-diagonal
computation are robosuite-specific.

No cost-field GIF here -- online_cost_transformer.py's plot_results_gif is
2D-maze-specific (draws wall segments, a flat 2D scatter of node positions).
There IS a per-iteration uncertainty visualization
(save_uncertainty_heatmap_robosuite, below), but it's a 3D scatter + 3-panel
projection figure rather than the maze's grid-binned heatmap -- see that
function's docstring for why a direct 2D-grid port doesn't make sense here
(most of the workspace bounding box is kinematically unreachable by the arm's
IK, so marking "empty" cells as max-uncertainty, the maze's convention for
coverage gaps, would be misleading). Output per run: the run log, per-
iteration metrics CSV (same columns as the maze's {env}_metrics.csv), one
uncertainty PNG per iteration, and (unless --skip-gif) two end-of-run GIFs:
robosuite_transformer_path.gif (the learned path, multi-camera arm view) and
robosuite_path_comparison.gif/.png (3D plot of the Dijkstra vs transformer path).

Usage:
    python main_robosuite.py                       # default config
    python main_robosuite.py --iters 15 --k 8 --out robosuite_results
"""

from __future__ import annotations

import os
import csv
import time
from typing import Dict, Tuple

import numpy as np
import torch

from online_cost_transformer import (
    Config, Graph, CostTransformer, train_transformer, field_mae, predict_last_mc,
)
from collect_trajectory_robosuite import collect_trajectory_robosuite, find_uncertain_state_robosuite
from robosuite_env import world_diagonal
from task import RobosuiteTask
from record_path_robosuite import (
    extract_transformer_path_robosuite, record_transformer_path_gif,
    save_path_comparison_gif, draw_goal_sphere,
)


# --------------------------------------------------------------------------- #
# Per-iteration uncertainty visualization: 3D scatter + XY/XZ/YZ projections
# --------------------------------------------------------------------------- #
def save_uncertainty_heatmap_robosuite(model, graph, task, cfg, it, outdir, log=None):
    """3D scatter + 3-panel (XY, XZ, YZ) projection view of the transformer's
    MC-dropout uncertainty after this iteration's training, over every node
    currently in the roadmap.

    Unlike the maze's grid-binned heatmap (save_uncertainty_heatmap in
    online_cost_transformer.py), this only plots actual graph nodes rather
    than binning the full workspace bounding box into cells and marking
    "unexplored" ones at max uncertainty: much of that box is kinematically
    unreachable by the arm's IK (behind the shoulder, outside its radius,
    etc.), so treating an empty region as an uncertainty signal would be
    misleading here, unlike the maze where every point in [0,L]^2 is
    potentially reachable. Query is the same history-free, zero-action token
    style as the maze's heatmap: [x, y, z, 0, 0, 0] -> MC-dropout std.

    Saved as {outdir}/robosuite_uncertainty_iter{it:02d}.png -- an *extra*
    per-iteration output, same role as the maze's {env}_uncertainty_iter##.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
    except Exception as e:                                     # pragma: no cover
        if log:
            log(f"[uncert]  skipped: {e}")
        return None

    P = np.asarray(graph.pos, dtype=np.float32)
    toks = np.concatenate([P, np.zeros_like(P)], axis=1)        # [x, y, z, 0, 0, 0]
    stds = np.empty(len(P), dtype=np.float64)
    for i in range(len(P)):
        _, std_raw = predict_last_mc(model, [toks[i]], cfg)
        stds[i] = std_raw

    vmin, vmax = float(stds.min()), float(stds.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6

    start = np.asarray(task.eef_start, float)
    cube = np.asarray(task.cube_pos, float)
    eps = float(task.cfg.eps_dist)                # success radius around the cube
    cmap = "YlGnBu_r"

    fig = plt.figure(figsize=(13, 11))
    fig.subplots_adjust(left=0.06, right=0.86, top=0.90, bottom=0.07,
                        hspace=0.4, wspace=0.4)

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    sc = ax3d.scatter(P[:, 0], P[:, 1], P[:, 2], c=stds, cmap=cmap,
                      vmin=vmin, vmax=vmax, s=8)
    ax3d.scatter(*start, c="white", edgecolors="k", s=60, label="start")
    ax3d.scatter(*cube, marker="*", c="gold", edgecolors="k", s=180, label="cube")
    draw_goal_sphere(ax3d, cube, eps)
    # explicit limits + proportional box so the goal sphere is actually round
    lo = np.minimum(P.min(axis=0), cube - eps)
    hi = np.maximum(P.max(axis=0), cube + eps)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ax3d.set_xlim(lo[0], hi[0]); ax3d.set_ylim(lo[1], hi[1]); ax3d.set_zlim(lo[2], hi[2])
    ax3d.set_box_aspect(tuple(hi - lo))
    # 3D axes don't respect normal subplot spacing and their tick labels can
    # bleed into neighbouring panels at default density -- thin them out and
    # shrink the font instead of relying on layout alone.
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
    ax3d.tick_params(labelsize=7, pad=0)
    ax3d.set_xlabel("x", labelpad=8); ax3d.set_ylabel("y", labelpad=8)
    ax3d.set_zlabel("z", labelpad=8)
    ax3d.set_title("3D view", pad=12)
    from matplotlib.patches import Patch
    handles, labels = ax3d.get_legend_handles_labels()
    handles.append(Patch(facecolor="gold", alpha=0.4))
    labels.append(f"goal radius ({eps:.3f} m)")
    ax3d.legend(handles, labels, loc="upper left", fontsize=7)

    for pos, xl, yl, xi, yi in [(2, "x", "y", 0, 1), (3, "x", "z", 0, 2), (4, "y", "z", 1, 2)]:
        ax = fig.add_subplot(2, 2, pos)
        ax.scatter(P[:, xi], P[:, yi], c=stds, cmap=cmap, vmin=vmin, vmax=vmax, s=10)
        ax.scatter(start[xi], start[yi], c="white", edgecolors="k", s=60, zorder=5)
        ax.scatter(cube[xi], cube[yi], marker="*", c="gold", edgecolors="k", s=180, zorder=5)
        # the sphere's silhouette in this projection is a circle of the same radius
        ax.add_patch(plt.Circle((cube[xi], cube[yi]), eps, color="gold", alpha=0.3,
                                zorder=2))
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.set_title(f"{xl}{yl} projection")

    fig.colorbar(sc, ax=fig.get_axes(), shrink=0.7, pad=0.02,
                label="MC-dropout std  (epistemic uncertainty)")
    fig.suptitle(f"robosuite Lift/Panda: uncertainty  iter {it}")

    out = os.path.join(outdir, f"robosuite_uncertainty_iter{it:02d}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    if log:
        log(f"[uncert]  saved -> {out}")
    return out


def main(cfg: Config, outdir: str = "robosuite_results", record_gif: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    # independent of `rng` on purpose -- see collect_trajectory_robosuite's
    # module docstring for why the uncertainty scan uses its own stream here
    scan_rng = np.random.default_rng(cfg.seed + 1_000_003)

    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "robosuite_run.log")
    logf = open(log_path, "w")

    def log(msg=""):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    task = RobosuiteTask(seed=cfg.seed)     # same seed -> same cube position across runs
    cfg.diagonal = world_diagonal(task.world.cfg)
    cfg.max_len = max(cfg.max_len, cfg.max_steps)
    model = CostTransformer(cfg, token_dim=6)
    graph = Graph(task.eef_start, task.world.cfg.novelty_radius)
    node_q: Dict[int, np.ndarray] = {0: task.q_start.copy()}
    D = []
    rows = []

    log(f"task = robosuite Lift/Panda red-cube  eef_start={task.eef_start.round(3)}  "
        f"cube={task.cube_pos.round(3)}  eps_dist={task.cfg.eps_dist}  seed={cfg.seed}")
    log(f"horizon (max_steps) = {cfg.max_steps}  |  pos-emb rows (max_len) = "
        f"{cfg.max_len}  |  d_model = {cfg.d_model}")
    log(f"cost normalization: labels & veto costs divided by diagonal "
        f"(workspace bbox) = {cfg.diagonal:.4f}")
    log(f"reset-replay: each of the k trajectories/iter resets to the "
        f"highest-uncertainty existing state s* (mu >= alpha={cfg.alpha}, "
        f"else falls back to the true start), scanning "
        f"{cfg.uncertainty_scan_actions} candidate actions/node")
    log(f"veto: c <= c_hat + mu (raw MC-dropout std, no EMA)  |  mc_samples = "
        f"{cfg.mc_samples}")
    hdr = (f"{'iter':>4} {'nodes':>6} {'trans':>7} {'succ':>5} {'len':>5} "
           f"{'veto':>5} {'sfail':>6} {'wallfb':>7} {'reset':>6} "
           f"{'hits':>6} {'miss':>6} {'loss':>8} {'fieldMAE':>9} {'sec':>7}")
    log(hdr)

    t_run_start = time.time()
    first_reach = None       # set once, the first time any state lands within the goal radius
    for it in range(cfg.n_iters):
        t_iter_start = time.time()
        succ, vetoes, steps = 0, 0, 0
        sample_fails, wall_fallbacks, resets_to_start = 0, 0, 0
        cache_hits, cache_misses = 0, 0
        uncertainty_cache: Dict[int, Tuple[float, float]] = {}
        for traj in range(cfg.k_trajectories):
            g_cur, prev_cur = graph.dijkstra(0, return_prev=True)
            s_star, tok_hist, id_hist, cost_hist, n_hits, n_miss = find_uncertain_state_robosuite(
                model, graph, g_cur, prev_cur, cfg, task, scan_rng, node_q,
                uncertainty_cache)
            resets_to_start += int(s_star == 0)
            cache_hits += n_hits
            cache_misses += n_miss
            tokens, node_ids, reached, nv, nsf, nwf = collect_trajectory_robosuite(
                model, graph, cfg, task, rng, node_q,
                start_node=s_star, start_tokens=tok_hist,
                start_node_ids=id_hist, start_path_cost=cost_hist)
            D.append((tokens, node_ids, reached))
            succ += int(reached)
            if reached and first_reach is None:
                # a trajectory ends the instant it lands within the goal radius, so
                # the graph size right now IS the number of nodes explored so far
                first_reach = {"iter": it, "traj": traj + 1, "nodes": len(graph.pos),
                               "sec": time.time() - t_run_start}
                log(f"[goal]    first state within the goal radius "
                    f"({task.cfg.eps_dist:.3f} m of the cube) reached: iter {it}, "
                    f"trajectory {traj + 1}/{cfg.k_trajectories}  |  "
                    f"nodes explored: {first_reach['nodes']}  |  "
                    f"{first_reach['sec']:.1f}s into the run")
            vetoes += nv
            sample_fails += nsf
            wall_fallbacks += nwf
            steps += len(node_ids)

        g = graph.dijkstra(0)
        loss = train_transformer(model, D, g, cfg, rng)
        mae = field_mae(model, D, g, cfg)
        # per-iteration spatial uncertainty view of the *just-trained* model
        save_uncertainty_heatmap_robosuite(model, graph, task, cfg, it, outdir, log)

        n_trans = sum(len(t[1]) for t in D)
        avg_len = steps / cfg.k_trajectories
        iter_sec = time.time() - t_iter_start
        log(f"{it:>4} {len(graph.pos):>6} {n_trans:>7} "
            f"{succ:>2}/{cfg.k_trajectories:<2} {avg_len:>5.1f} {vetoes:>5} "
            f"{sample_fails:>6} {wall_fallbacks:>7} {resets_to_start:>6} "
            f"{cache_hits:>6} {cache_misses:>6} "
            f"{loss:>8.3f} {mae:>9.4f} {iter_sec:>7.1f}")
        rows.append({
            "iter": it, "nodes": len(graph.pos), "trans": n_trans,
            "succ": succ, "avg_len": round(avg_len, 3), "vetoes": vetoes,
            "sample_fails": sample_fails, "wall_fallbacks": wall_fallbacks,
            "resets_to_start": resets_to_start,
            "cache_hits": cache_hits, "cache_misses": cache_misses,
            "loss": round(loss, 6), "field_mae": round(mae, 6),
            "iter_sec": round(iter_sec, 3),
        })
    total_sec = time.time() - t_run_start

    csv_path = os.path.join(outdir, "robosuite_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    g = graph.dijkstra(0)
    log(f"\nField cost-to-come MAE (transformer vs Dijkstra, normalized units): "
        f"{field_mae(model, D, g, cfg):.4f}")
    if first_reach:
        log(f"Goal radius ({task.cfg.eps_dist:.3f} m) first reached at iter "
            f"{first_reach['iter']}, trajectory {first_reach['traj']}, after "
            f"exploring {first_reach['nodes']} nodes "
            f"(of {len(graph.pos)} by the end of the run)")
    else:
        log(f"Goal radius ({task.cfg.eps_dist:.3f} m) never reached in "
            f"{len(graph.pos)} explored nodes")

    t_nodes, q_path, t_reached, t_cost = extract_transformer_path_robosuite(
        model, graph, cfg, task, node_q)
    log(f"Transformer-learned path: {len(t_nodes)} nodes, cost={t_cost:.4f}, "
        f"reached_start={t_reached}")
    if record_gif:
        gif_path = os.path.join(outdir, "robosuite_transformer_path.gif")
        record_transformer_path_gif(task, q_path, gif_path, log=log)
        save_path_comparison_gif(
            graph, task, t_nodes,
            os.path.join(outdir, "robosuite_path_comparison.gif"),
            os.path.join(outdir, "robosuite_path_comparison.png"), log=log)

    log(f"[metrics] saved -> {csv_path}")
    log(f"[log]     saved -> {log_path}")
    log(f"[time]    total run wall-clock: {total_sec:.1f}s")
    logf.close()
    task.close()
    return model, graph, g


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Online transformer cost-to-come training for the "
                    "robosuite Lift/Panda red-cube task.")
    p.add_argument("--iters", type=int, default=15, help="n_iters (default: 15)")
    p.add_argument("--k", type=int, default=8,
                   help="trajectories collected per iteration (default: 8)")
    p.add_argument("--max-steps", type=int, default=40, dest="max_steps",
                   help="trajectory horizon (default: 40)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="mu(s,a) threshold for picking the reset-replay "
                        "target s* (default: 0.05)")
    p.add_argument("--scan-actions", type=int, default=8, dest="scan_actions",
                   help="candidate actions probed per existing node when "
                        "searching for s* (default: 8)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--out", type=str, default="robosuite_results",
                   help="output directory (default: robosuite_results)")
    p.add_argument("--skip-gif", action="store_true", dest="skip_gif",
                   help="skip the end-of-run GIFs (robosuite_transformer_path.gif "
                        "and robosuite_path_comparison.gif/.png)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(max_steps=args.max_steps, n_iters=args.iters,
                k_trajectories=args.k, alpha=args.alpha,
                uncertainty_scan_actions=args.scan_actions, seed=args.seed)
    main(cfg, outdir=args.out, record_gif=not args.skip_gif)
