"""
Full online training loop for the MotionBenchMaker (MBM) Panda problems, PyBullet only. Adapted from
legacy_robosuite/main_robosuite.py; the algorithm (Go-Explore reset-replay to the most uncertain node, MC-dropout
veto, Dijkstra labels, retraining on the growing buffer) is unchanged -- see CLAUDE.md for what changed:

  * states are 7-D joint configurations q; actions are 7-D joint increments; no IK anywhere
  * images (DINOv2 over the configured cameras) are the transformer's ONLY state input; q supervises (labels,
    veto cost, merging, goal test)
  * the goal is a joint configuration, reached through an explicit exact-goal connection (--goal-connect)
  * PyBullet is both the collision checker (single source of truth) and the renderer
  * at the end, the roadmap-Dijkstra path, the transformer's greedy path and a like-for-like joint-space
    RRT-Connect baseline (same checker, same step size, 5 seeds) are compared in joint and EEF space

Usage:
    python main_mbm.py --scene table_pick --problem 0
    python main_mbm.py --scene-json scenes/panda/bookshelf_small/p000.json --iters 15 --k 8
    python main_mbm.py --scene table_pick --problem 0 --iters 2 --k 2 --max-steps 40 --n-calib 50   # smoke run
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Dict, Set, Tuple

import numpy as np
import torch

from collect_trajectory_mbm import (
    collect_trajectory_mbm, find_uncertain_state_mbm, state_feat,
)
from collision_mbm import PAIRS_PATH, load_self_collision_pairs
from image_encoder import EncoderConfig, ImageEncoder
from online_cost_transformer import (
    Config, Graph, ImageCostTransformer, dijkstra_path, field_mae, path_length, predict_last_mc,
    train_transformer,
)
from record_path_mbm import (
    extract_transformer_path_mbm, record_transformer_path_gif, save_path_comparison_gif,
)
from robot_mbm import N_DOF
from rrtconnect_mbm import run_rrtconnect_baseline
from scene_mbm import DEFAULT_SCENES_DIR, scene_path
from task_mbm import MBMTask, MBMTaskConfig


# --------------------------------------------------------------------------- #
# Per-iteration uncertainty visualization: 3D scatter + XY/XZ/YZ projections at node EEF positions
# --------------------------------------------------------------------------- #
def save_uncertainty_figure_mbm(model, graph, task, cfg, node_emb, node_eef, goal_nodes, it, outdir, log=None):
    """MC-dropout uncertainty of the just-trained model at every roadmap node, drawn at the node's END-EFFECTOR
    position (`node_eef`; plotting only -- the model is queried with each node's image embedding and a zero
    action: [emb, 0 x 7] -> std). The goal is a joint configuration, so it is marked by a star at
    fk_eef(q_goal); there is NO goal sphere (an EEF radius does not define the goal here)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:                                     # pragma: no cover
        if log:
            log(f"[uncert]  skipped: {e}")
        return None

    n = len(graph.pos)
    P = np.stack([node_eef[i] for i in range(n)]).astype(np.float32)
    zero_a = np.zeros(N_DOF, dtype=np.float32)
    stds = np.empty(n, dtype=np.float64)
    for i in range(n):
        tok = np.concatenate([state_feat(node_emb, i), zero_a]).astype(np.float32)
        _, stds[i] = predict_last_mc(model, [tok], cfg)
    vmin, vmax = float(stds.min()), float(stds.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6

    start = node_eef[0]
    goals = [task.fk_eef(g) for g in task.goals]
    cmap = "YlGnBu_r"
    fig = plt.figure(figsize=(13, 11.6))
    fig.subplots_adjust(left=0.06, right=0.86, top=0.90, bottom=0.10, hspace=0.4, wspace=0.4)

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    sc = ax3d.scatter(P[:, 0], P[:, 1], P[:, 2], c=stds, cmap=cmap, vmin=vmin, vmax=vmax, s=8)
    ax3d.scatter(*start, c="white", edgecolors="k", s=60, label="start")
    for k, gp in enumerate(goals):
        ax3d.scatter(*gp, marker="*", c="gold", edgecolors="k", s=180, label="goal (EEF of q_goal)" if k == 0 else None)
    lo = np.minimum(P.min(axis=0), np.min(goals, axis=0))
    hi = np.maximum(P.max(axis=0), np.max(goals, axis=0))
    pad = 0.05 * (hi - lo) + 1e-3
    lo, hi = lo - pad, hi + pad
    ax3d.set_xlim(lo[0], hi[0]); ax3d.set_ylim(lo[1], hi[1]); ax3d.set_zlim(lo[2], hi[2])
    ax3d.set_box_aspect(tuple(hi - lo))
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
    ax3d.tick_params(labelsize=7, pad=0)
    ax3d.set_xlabel("x", labelpad=8); ax3d.set_ylabel("y", labelpad=8); ax3d.set_zlabel("z", labelpad=8)
    ax3d.set_title("3D view (EEF positions)", pad=12)
    ax3d.legend(loc="upper left", fontsize=7)

    for pos, xl, yl, xi, yi in [(2, "x", "y", 0, 1), (3, "x", "z", 0, 2), (4, "y", "z", 1, 2)]:
        ax = fig.add_subplot(2, 2, pos)
        ax.scatter(P[:, xi], P[:, yi], c=stds, cmap=cmap, vmin=vmin, vmax=vmax, s=10)
        ax.scatter(start[xi], start[yi], c="white", edgecolors="k", s=60, zorder=5)
        for gp in goals:
            ax.scatter(gp[xi], gp[yi], marker="*", c="gold", edgecolors="k", s=180, zorder=5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.set_title(f"{xl}{yl} projection")

    fig.colorbar(sc, ax=fig.get_axes(), shrink=0.7, pad=0.02, label="MC-dropout std  (epistemic uncertainty)")
    fig.suptitle(f"{task.scene.name}: uncertainty  iter {it}  ({n} nodes, {len(goal_nodes)} goal nodes)")
    fig.text(0.5, 0.02, "Nodes are plotted at their end-effector position (forward kinematics of q). The goal is a "
             "joint configuration, not an EEF radius: the star marks fk_eef(q_goal) and no goal sphere is drawn.",
             ha="center", fontsize=8, wrap=True)
    out = os.path.join(outdir, f"mbm_uncertainty_iter{it:02d}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    if log:
        log(f"[uncert]  saved -> {out}")
    return out


def _pick_goal_node(graph, g, goal_nodes, task):
    """Reached goal node with the lowest cost-to-come, else the node nearest any goal in joint space."""
    finite_goal = [n for n in goal_nodes if math.isfinite(g[n])]
    if finite_goal:
        return min(finite_goal, key=lambda n: g[n]), True
    P = np.stack([np.asarray(p) for p in graph.pos])
    d = np.min([np.linalg.norm(P - q_goal, axis=1) for q_goal in task.goals], axis=0)
    d[~np.isfinite(np.asarray(g))] = np.inf
    return int(np.argmin(d)), False


def main(cfg: Config, task_cfg: MBMTaskConfig, scene_json: str, outdir: str, record_gif: bool = True,
         enc_cfg: EncoderConfig = None, token_fusion: str = "concat", rrt_seeds: int = 5):
    enc_cfg = enc_cfg or EncoderConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    scan_rng = np.random.default_rng(cfg.seed + 1_000_003)        # independent of exploration (deliberate)
    calib_rng = np.random.default_rng(cfg.seed + 2_000_003)       # third independent stream

    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "mbm_run.log")
    logf = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    task = MBMTask(scene_json, task_cfg)
    cfg.diagonal = task.diagonal
    cfg.max_len = max(cfg.max_len, cfg.max_steps)
    checker = task.checker

    torch.manual_seed(cfg.seed)                 # before the encoder and model are built
    encoder = ImageEncoder(enc_cfg)
    t_calib = time.time()
    encoder.calibrate(task, calib_rng)          # fits + freezes the standardization stats
    calib_sec = time.time() - t_calib
    stats_path = os.path.join(outdir, "encoder_stats.npz")
    encoder.save_stats(stats_path)
    task.renderer.set_q(task.q_start)
    node_emb: Dict[int, np.ndarray] = {0: encoder.embed(task.renderer)}
    node_eef: Dict[int, np.ndarray] = {0: task.fk_eef(task.q_start)}
    goal_nodes: Set[int] = set()

    torch.manual_seed(cfg.seed)
    model = ImageCostTransformer(
        cfg, encoder.n_views, encoder.emb_dim, action_dim=N_DOF,
        action_scale=1.0 / task_cfg.step_size,           # tokens keep raw radians; the model rescales
        view_dropout=enc_cfg.view_dropout, token_fusion=token_fusion).to(cfg.device)
    graph = Graph(task.q_start, task_cfg.novelty_radius)
    D = []
    rows = []

    bench = checker.benchmark(n=1000)
    n_pairs = len(load_self_collision_pairs(PAIRS_PATH)["checked"])
    sc = task.scene
    d_sg = float(np.linalg.norm(task.goals[0] - task.q_start))
    log(f"task = MBM Panda  scene={sc.scene} problem={sc.index}  seed={cfg.seed}  json={sc.path}")
    log(f"start-goal joint distance = {d_sg:.3f} rad  |  goals = {len(task.goals)}  |  "
        f"obstacles = {len(sc.obstacles)}  |  diagonal (|q_high - q_low|) = {cfg.diagonal:.3f} rad")
    log(f"collision checker = PyBullet '{checker.strategy}' strategy  |  "
        f"{bench['state_checks_per_sec']:.0f} state checks/s, {bench['motion_checks_per_sec']:.0f} motion checks/s "
        f"(random q valid {bench['valid_fraction_random_q']:.2f})  |  self-collision pairs checked = {n_pairs}")
    log(f"scaled params: step_size={task_cfg.step_size} rad  novelty_radius={task_cfg.novelty_radius:.4f} rad "
        f"(frac {task_cfg.novelty_frac})  goal_connect={task_cfg.goal_connect} rad  "
        f"motion_res={task_cfg.motion_res} rad  max_sample_tries={task_cfg.max_sample_tries}  "
        f"action_scale={1.0 / task_cfg.step_size:.3f}")
    log(f"image state: encoder={enc_cfg.model_name}  cameras={list(enc_cfg.cameras)}  "
        f"image_size={enc_cfg.image_size}  pool={enc_cfg.pool}  "
        f"encoder_device={encoder.device}  transformer_device={cfg.device}")
    log(f"token width = {encoder.n_views} views x {encoder.emb_dim} dims + {N_DOF} action = "
        f"{encoder.n_views * encoder.emb_dim + N_DOF}  |  view_dropout={enc_cfg.view_dropout}  |  "
        f"standardization fitted on {enc_cfg.n_calib} states in {calib_sec:.1f}s (saved -> {stats_path})")
    log(f"horizon (max_steps) = {cfg.max_steps}  |  pos-emb rows (max_len) = {cfg.max_len}  |  "
        f"d_model = {cfg.d_model}")
    n_params = sum(p.numel() for p in model.parameters())
    enc_width = model.cost_head.in_features
    log(f"token fusion = {token_fusion}  |  encoder width = {enc_width}  |  "
        f"dim_ff = {cfg.dim_ff * (1 if token_fusion == 'sum' else 2)}  |  transformer params = {n_params:,}")
    log(f"cost normalization: labels & veto costs divided by diagonal = {cfg.diagonal:.4f}")
    log(f"reset-replay: each of the k trajectories/iter resets to the highest-uncertainty existing state s* "
        f"(mu >= alpha={cfg.alpha}, else falls back to the true start), scanning "
        f"{cfg.uncertainty_scan_actions} candidate actions/node")
    log(f"veto: c <= c_hat + mu (raw MC-dropout std, no EMA)  |  mc_samples = {cfg.mc_samples}")
    hdr = (f"{'iter':>4} {'nodes':>6} {'trans':>7} {'succ':>5} {'len':>5} "
           f"{'veto':>5} {'sfail':>6} {'wallfb':>7} {'reset':>6} "
           f"{'hits':>6} {'miss':>6} {'loss':>8} {'fieldMAE':>9} {'sec':>7} "
           f"{'emb_s':>7} {'n_emb':>6} {'st_chk':>8} {'mo_chk':>7} {'goal':>4} {'merge':>6} "
           f"{'samp_s':>7} {'scan_s':>7} {'train_s':>7}")
    log(hdr)

    t_run_start = time.time()
    first_reach = None
    for it in range(cfg.n_iters):
        t_iter_start = time.time()
        emb_t0, emb_n0 = encoder.embed_time, encoder.embed_count
        sc0, mc0 = checker.n_state_checks, checker.n_motion_checks
        succ, vetoes, steps = 0, 0, 0
        sample_fails, wall_fallbacks, resets_to_start = 0, 0, 0
        cache_hits, cache_misses = 0, 0
        scan_sec = collect_sec = 0.0
        stats: dict = {}
        uncertainty_cache: Dict[int, Tuple[float, float]] = {}
        for traj in range(cfg.k_trajectories):
            g_cur, prev_cur = graph.dijkstra(0, return_prev=True)
            t0 = time.perf_counter()
            s_star, tok_hist, id_hist, cost_hist, n_hits, n_miss = find_uncertain_state_mbm(
                model, graph, g_cur, prev_cur, cfg, task, scan_rng, node_emb, uncertainty_cache)
            scan_sec += time.perf_counter() - t0
            resets_to_start += int(s_star == 0)
            cache_hits += n_hits
            cache_misses += n_miss
            t0 = time.perf_counter()
            tokens, node_ids, reached, nv, nsf, nwf = collect_trajectory_mbm(
                model, graph, cfg, task, rng, node_emb, node_eef, goal_nodes, encoder,
                start_node=s_star, start_tokens=tok_hist, start_node_ids=id_hist,
                start_path_cost=cost_hist, stats=stats)
            collect_sec += time.perf_counter() - t0
            D.append((tokens, node_ids, reached))
            succ += int(reached)
            if reached and first_reach is None:
                first_reach = {"iter": it, "traj": traj + 1, "nodes": len(graph.pos),
                               "sec": time.time() - t_run_start}
                log(f"[goal]    exact goal node first reached: iter {it}, trajectory {traj + 1}/"
                    f"{cfg.k_trajectories}  |  nodes explored: {first_reach['nodes']}  |  "
                    f"{first_reach['sec']:.1f}s into the run")
            vetoes += nv
            sample_fails += nsf
            wall_fallbacks += nwf
            steps += len(node_ids)

        t0 = time.perf_counter()
        g = graph.dijkstra(0)
        loss = train_transformer(model, D, g, cfg, rng)
        mae = field_mae(model, D, g, cfg)
        train_sec = time.perf_counter() - t0
        save_uncertainty_figure_mbm(model, graph, task, cfg, node_emb, node_eef, goal_nodes, it, outdir, log)

        n_trans = sum(len(t[1]) for t in D)
        avg_len = steps / cfg.k_trajectories
        iter_sec = time.time() - t_iter_start
        embed_sec = encoder.embed_time - emb_t0
        n_embedded = encoder.embed_count - emb_n0
        sample_sec = max(collect_sec - embed_sec, 0.0)   # sampling, collision checks, veto queries
        d_states, d_motions = checker.n_state_checks - sc0, checker.n_motion_checks - mc0
        merges = stats.get("merges", 0)
        log(f"{it:>4} {len(graph.pos):>6} {n_trans:>7} "
            f"{succ:>2}/{cfg.k_trajectories:<2} {avg_len:>5.1f} {vetoes:>5} "
            f"{sample_fails:>6} {wall_fallbacks:>7} {resets_to_start:>6} "
            f"{cache_hits:>6} {cache_misses:>6} "
            f"{loss:>8.3f} {mae:>9.4f} {iter_sec:>7.1f} "
            f"{embed_sec:>7.2f} {n_embedded:>6} {d_states:>8} {d_motions:>7} {len(goal_nodes):>4} {merges:>6} "
            f"{sample_sec:>7.2f} {scan_sec:>7.2f} {train_sec:>7.2f}")
        rows.append({
            "iter": it, "nodes": len(graph.pos), "trans": n_trans,
            "succ": succ, "avg_len": round(avg_len, 3), "vetoes": vetoes,
            "sample_fails": sample_fails, "wall_fallbacks": wall_fallbacks,
            "resets_to_start": resets_to_start,
            "cache_hits": cache_hits, "cache_misses": cache_misses,
            "loss": round(loss, 6), "field_mae": round(mae, 6),
            "iter_sec": round(iter_sec, 3),
            "embed_sec": round(embed_sec, 3), "n_embedded": n_embedded,
            "n_state_checks": d_states, "n_motion_checks": d_motions, "goal_nodes": len(goal_nodes),
            "sample_sec": round(sample_sec, 3), "scan_sec": round(scan_sec, 3), "train_sec": round(train_sec, 3),
            "merges": merges,
        })
    total_sec = time.time() - t_run_start

    csv_path = os.path.join(outdir, "mbm_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    g, prev = graph.dijkstra(0, return_prev=True)
    log(f"\nField cost-to-come MAE (transformer vs Dijkstra, normalized units): {field_mae(model, D, g, cfg):.4f}")
    n_nodes = len(graph.pos)
    assert len(node_emb) == len(node_eef) == n_nodes, (
        f"node bookkeeping out of sync: graph={n_nodes} node_emb={len(node_emb)} node_eef={len(node_eef)}")
    if first_reach:
        log(f"Exact goal node first reached at iter {first_reach['iter']}, trajectory {first_reach['traj']}, after "
            f"exploring {first_reach['nodes']} nodes (of {n_nodes} by the end of the run)")
    else:
        log(f"Goal never reached in {n_nodes} explored nodes (goal_connect={task_cfg.goal_connect} rad)")

    npz_path = os.path.join(outdir, "mbm_node_data.npz")
    np.savez_compressed(
        npz_path,
        q=np.stack(graph.pos).astype(np.float32),
        eef=np.stack([node_eef[i] for i in range(n_nodes)]).astype(np.float32),
        emb=np.stack([node_emb[i] for i in range(n_nodes)]).astype(np.float32),
        goal_nodes=np.array(sorted(goal_nodes), dtype=np.int64),
        g=np.asarray(g, dtype=np.float64))
    log(f"[nodes]   saved {n_nodes} nodes -> {npz_path}")

    # ------------------------------------------------------------------ final paths
    goal_node, goal_reached = _pick_goal_node(graph, g, goal_nodes, task)
    gap = min(float(np.linalg.norm(graph.pos[goal_node] - q_g)) for q_g in task.goals)
    d_nodes = dijkstra_path(graph, prev, goal_node)
    d_q = [graph.pos[i] for i in d_nodes]
    d_joint = path_length(graph, d_nodes)
    d_eef_pts = task.eef_polyline(d_q)
    d_eef = float(np.linalg.norm(np.diff(d_eef_pts, axis=0), axis=1).sum()) if len(d_eef_pts) > 1 else 0.0

    t_nodes, t_reached, t_joint = extract_transformer_path_mbm(model, graph, cfg, node_emb, goal_node)
    t_q = [graph.pos[i] for i in t_nodes]
    t_eef_pts = task.eef_polyline(t_q)
    t_eef = float(np.linalg.norm(np.diff(t_eef_pts, axis=0), axis=1).sum()) if len(t_eef_pts) > 1 else 0.0

    log("")
    log(f"Path target: node {goal_node} ({'exact goal node' if goal_reached else f'NOT the goal; nearest node, {gap:.3f} rad from q_goal'})")
    log(f"Roadmap Dijkstra path:   {len(d_nodes):>3} nodes  joint = {d_joint:.4f} rad  eef = {d_eef:.4f} m")
    log(f"Transformer greedy path: {len(t_nodes):>3} nodes  joint = {t_joint:.4f} rad  eef = {t_eef:.4f} m  "
        f"reached_start={t_reached}")
    ratio_td = t_joint / d_joint if d_joint > 0 else float("nan")
    log(f"transformer/Dijkstra:  joint {ratio_td:.3f}  eef {t_eef / d_eef if d_eef > 0 else float('nan'):.3f}")

    log(f"\nRRT-Connect baseline (same checker, step {task_cfg.step_size} rad, {rrt_seeds} seeds, "
        f"200 shortcut iterations):")
    rrt = run_rrtconnect_baseline(task, range(rrt_seeds), log=log)
    ok = [r for r in rrt if r["success"]]
    rrt_csv = os.path.join(outdir, "mbm_rrtconnect.csv")
    with open(rrt_csv, "w", newline="") as f:
        keys = ["seed", "success", "time_sec", "n_waypoints_raw", "n_waypoints", "joint_len_raw", "joint_len",
                "eef_len_raw", "eef_len", "n_state_checks", "n_motion_checks"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rrt)
    best = None
    if ok:
        r_joint = float(np.mean([r["joint_len"] for r in ok]))
        r_eef = float(np.mean([r["eef_len"] for r in ok]))
        best = min(ok, key=lambda r: r["joint_len"])
        log(f"RRT-Connect: {len(ok)}/{len(rrt)} success | mean time {np.mean([r['time_sec'] for r in ok]):.2f}s | "
            f"joint = {r_joint:.4f} rad (best {best['joint_len']:.4f}) | eef = {r_eef:.4f} m | "
            f"mean checks {np.mean([r['n_state_checks'] for r in ok]):.0f} states / "
            f"{np.mean([r['n_motion_checks'] for r in ok]):.0f} motions")
        caveat = "" if goal_reached else f"  (NOT like-for-like: the Dijkstra path stops {gap:.3f} rad short of q_goal)"
        log(f"Dijkstra/RRT-Connect:  joint {d_joint / r_joint:.3f}  eef {d_eef / r_eef:.3f}{caveat}")
        log(f"transformer/RRT-Connect: joint {t_joint / r_joint:.3f}  eef {t_eef / r_eef:.3f}{caveat}")
    else:
        log("RRT-Connect: no successful seed; ratios unavailable")

    if record_gif:
        try:
            record_transformer_path_gif(task, t_q, os.path.join(outdir, "mbm_transformer_path.gif"), log=log)
            all_eef = np.stack([node_eef[i] for i in range(n_nodes)])
            paths = {"Dijkstra": d_eef_pts, "transformer": t_eef_pts}
            lengths = {"Dijkstra": d_joint, "transformer": t_joint}
            if best is not None:
                paths["RRT-Connect"] = task.eef_polyline(best["path"])
                lengths["RRT-Connect"] = best["joint_len"]
            title = (f"{sc.scene} p{sc.index}: Dijkstra vs transformer vs RRT-Connect (EEF space)\n"
                     f"transformer/Dijkstra = {ratio_td:.3f}; "
                     + ("both paths end at the goal node" if goal_reached
                        else f"roadmap paths end {gap:.2f} rad short of q_goal"))
            save_path_comparison_gif(paths, lengths, all_eef, node_eef[0], task.fk_eef(task.goals[0]), title,
                                     os.path.join(outdir, "mbm_path_comparison.gif"),
                                     os.path.join(outdir, "mbm_path_comparison.png"), log=log)
        except Exception as e:                                  # recording must never lose the run's results
            log(f"[record]  FAILED: {type(e).__name__}: {e}")

    log(f"[metrics] saved -> {csv_path}")
    log(f"[log]     saved -> {log_path}")
    log(f"[time]    total run wall-clock: {total_sec:.1f}s  |  checker totals: "
        f"{checker.n_state_checks} state / {checker.n_motion_checks} motion checks")
    logf.close()
    task.close()
    return model, graph, g


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Online transformer cost-to-come training on MotionBenchMaker Panda problems (PyBullet).")
    p.add_argument("--scene-json", default=None, dest="scene_json", help="path to one exported problem JSON")
    p.add_argument("--scenes-dir", default=str(DEFAULT_SCENES_DIR), dest="scenes_dir",
                   help="directory of exported scenes (<dir>/<scene>/pNNN.json; default: scenes/panda)")
    p.add_argument("--scene", default="table_pick", help="scene name (default: table_pick)")
    p.add_argument("--problem", type=int, default=0, help="0-based problem index (default: 0)")
    p.add_argument("--step-size", type=float, default=0.15, dest="step_size", help="max joint step, rad (default 0.15)")
    p.add_argument("--novelty-frac", type=float, default=0.25, dest="novelty_frac",
                   help="merge radius = frac * step (default 0.25)")
    p.add_argument("--goal-connect", type=float, default=0.5, dest="goal_connect",
                   help="joint distance (rad) within which an exact goal connection is attempted (default 0.5)")
    p.add_argument("--motion-res", type=float, default=0.02, dest="motion_res",
                   help="joint-space interpolation resolution of motion checks, rad (default 0.02)")
    p.add_argument("--iters", type=int, default=15, help="n_iters (default: 15)")
    p.add_argument("--k", type=int, default=8, help="trajectories collected per iteration (default: 8)")
    p.add_argument("--max-steps", type=int, default=100, dest="max_steps", help="trajectory horizon (default: 100)")
    p.add_argument("--alpha", type=float, default=0.05, help="mu threshold for picking s* (default: 0.05)")
    p.add_argument("--scan-actions", type=int, default=8, dest="scan_actions",
                   help="candidate actions probed per existing node when searching for s* (default: 8)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--out", type=str, default=None, help="output dir (default: mbm_results/<scene>_p<idx>)")
    p.add_argument("--skip-gif", action="store_true", dest="skip_gif", help="skip the end-of-run GIFs")
    p.add_argument("--rrt-seeds", type=int, default=5, dest="rrt_seeds", help="RRT-Connect baseline seeds (default 5)")
    p.add_argument("--encoder", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    p.add_argument("--cameras", default="front,side,wrist",
                   help="comma-separated camera names among front,side,top,iso,wrist (default: front,side,wrist)")
    p.add_argument("--image-size", type=int, default=224, dest="image_size",
                   help="render/encode resolution, multiple of 14 (default: 224)")
    p.add_argument("--pool", default="cls", choices=["cls", "cls_mean"])
    p.add_argument("--view-dropout", type=float, default=0.0, dest="view_dropout")
    p.add_argument("--token-fusion", default="concat", choices=["concat", "sum", "wide_sum"], dest="token_fusion")
    p.add_argument("--n-calib", type=int, default=200, dest="n_calib",
                   help="states used to fit the embedding standardization (default: 200)")
    p.add_argument("--device", default="auto",
                   help="ENCODER device: auto (cuda if available else cpu) | cpu | mps | cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    js = args.scene_json or str(scene_path(args.scenes_dir, args.scene, args.problem))
    if args.out is None:
        import json
        with open(js) as f:
            d = json.load(f)
        args.out = os.path.join("mbm_results", f"{d['scene']}_p{d['index']}")
    cfg = Config(max_steps=args.max_steps, n_iters=args.iters, k_trajectories=args.k, alpha=args.alpha,
                 uncertainty_scan_actions=args.scan_actions, seed=args.seed)
    task_cfg = MBMTaskConfig(step_size=args.step_size, novelty_frac=args.novelty_frac,
                             goal_connect=args.goal_connect, motion_res=args.motion_res)
    enc_cfg = EncoderConfig(
        model_name=args.encoder, cameras=tuple(c.strip() for c in args.cameras.split(",") if c.strip()),
        image_size=args.image_size, pool=args.pool, device=args.device, n_calib=args.n_calib,
        view_dropout=args.view_dropout)
    main(cfg, task_cfg, js, args.out, record_gif=not args.skip_gif, enc_cfg=enc_cfg,
         token_fusion=args.token_fusion, rrt_seeds=args.rrt_seeds)
