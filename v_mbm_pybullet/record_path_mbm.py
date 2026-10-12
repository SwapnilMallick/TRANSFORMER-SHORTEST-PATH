"""
Path extraction and recording for the MBM port (adapted from legacy_robosuite/record_path_robosuite.py, which
cannot be imported here: it pulls in robosuite/mujoco).

  * extract_transformer_path_mbm: greedy descent on the trained model's predicted cost field, queried with each
    node's cached image embedding + a zero action; online_cost_transformer.transformer_path is reused unchanged
    (an undirected roadmap walk is correct: arm motions are reversible).
  * record_transformer_path_gif: multi-camera GIF (front, side, iso, wrist) of a joint-space path rendered with
    the task's PyBulletRenderer. Joint interpolation between waypoints is COSMETIC (smooth video only).
  * save_path_comparison_gif: 3D matplotlib comparison of Dijkstra vs transformer vs RRT-Connect paths in EEF
    space. The EEF arrays are passed in (node_eef / task.eef_polyline), not read from graph positions, since the
    graph lives in joint space. The goal is a joint configuration, so it is drawn as a star at fk_eef(q_goal)
    with no goal sphere.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from collect_trajectory_mbm import state_feat
from online_cost_transformer import Graph, predict_cost_field, transformer_path, path_length
from robot_mbm import N_DOF

GIF_CAMERAS = ("front", "side", "iso", "wrist")


def extract_transformer_path_mbm(model, graph: Graph, cfg, node_emb: Dict[int, np.ndarray], target_node: int):
    """Greedy descent from `target_node` back to the start (node 0). Returns (node_ids start -> target,
    reached_start, joint_cost)."""
    feats = np.stack([state_feat(node_emb, i) for i in range(len(graph.pos))])
    chat = predict_cost_field(model, graph, cfg, state_feats=feats, action_dim=N_DOF)
    t_nodes, t_reached = transformer_path(model, graph, cfg, target_node, chat=chat)
    return t_nodes, t_reached, path_length(graph, t_nodes)


def _lerp_q_path(q_path: Sequence[np.ndarray], interp_steps: int) -> List[np.ndarray]:
    if len(q_path) < 2:
        return list(q_path)
    out = [q_path[0]]
    for q0, q1 in zip(q_path[:-1], q_path[1:]):
        for k in range(1, interp_steps + 1):
            out.append(q0 + (q1 - q0) * (k / interp_steps))
    return out


def _quantize_frames_shared_palette(frames: List[np.ndarray], colors: int = 256):
    """Remap every frame onto ONE shared palette sampled across the clip (per-frame palettes shift colours)."""
    from PIL import Image
    sample_stride = max(1, len(frames) // 24)
    composite = np.concatenate(frames[::sample_stride], axis=0)
    palette_img = Image.fromarray(composite).convert("RGB").quantize(colors=colors, dither=Image.Dither.NONE)
    return [Image.fromarray(f).convert("RGB").quantize(palette=palette_img, dither=Image.Dither.NONE)
            for f in frames]


def _label_frame(arr: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 7 * len(text) + 6, 16], fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return np.array(img)


def record_transformer_path_gif(task, q_path: Sequence[np.ndarray], out_path: str, cameras=GIF_CAMERAS,
                                size: int = 256, interp_steps: int = 6, max_frames: int = 300,
                                frame_ms: int = 120, hold_ms: int = 1500, log=None):
    """Renders q_path (start -> target) from several cameras side by side into one animated GIF."""
    if len(q_path) < 1:
        if log:
            log("[video]   skipped: empty path")
        return None
    interp_steps = max(1, min(interp_steps, max_frames // max(1, len(q_path) - 1)))
    renderer = task.renderer
    frames = []
    for q in _lerp_q_path(list(q_path), interp_steps):
        renderer.set_q(q)
        views = [_label_frame(renderer.render(c, size, size), c) for c in cameras]
        frames.append(np.concatenate(views, axis=1))
    renderer.set_q(task.q_start)
    images = _quantize_frames_shared_palette(frames)
    durations = [frame_ms] * (len(images) - 1) + [hold_ms]
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=durations, loop=0, disposal=2)
    if log:
        log(f"[video]   saved -> {out_path}  ({len(frames)} frames, cameras={list(cameras)})")
    return out_path


def _point_along(P: np.ndarray, cum: np.ndarray, s: float):
    """Position after travelling arc length `s` along polyline P, and the trail drawn so far."""
    if len(P) < 2:
        return P[0], P[:1]
    s = min(max(s, 0.0), cum[-1])
    i = min(max(int(np.searchsorted(cum, s, side="right")) - 1, 0), len(P) - 2)
    seg = cum[i + 1] - cum[i]
    frac = 0.0 if seg <= 1e-12 else min(max((s - cum[i]) / seg, 0.0), 1.0)
    pos = P[i] + frac * (P[i + 1] - P[i])
    return pos, np.vstack([P[: i + 1], pos])


_STYLES = {"Dijkstra": dict(color="crimson", lw=2.6, ls="-"),
           "transformer": dict(color="deepskyblue", lw=2.0, ls="--"),
           "RRT-Connect": dict(color="forestgreen", lw=2.0, ls=":")}


def save_path_comparison_gif(paths: Dict[str, np.ndarray], lengths: Dict[str, float], all_eef: np.ndarray,
                             start_eef: np.ndarray, goal_eef: np.ndarray, title: str, out_gif: str,
                             out_png: str = None, n_frames: int = 60, frame_ms: int = 80, hold_ms: int = 2500,
                             log=None):
    """Animated 3D comparison of EEF-space paths. `paths` maps a name in _STYLES to an (N, 3) EEF polyline;
    `lengths` gives the joint-space length shown in the legend. Every path advances by the same FRACTION of its
    own arc length per frame, so all leave together and arrive together. Explored nodes are drawn faintly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    paths = {k: np.asarray(v, float) for k, v in paths.items() if v is not None and len(v) >= 1}
    if not paths:
        if log:
            log("[paths]   comparison skipped: no paths")
        return None
    cum = {k: np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
           for k, P in paths.items()}
    pts = np.vstack([all_eef, start_eef[None], goal_eef[None]] + list(paths.values()))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.05 * (hi - lo) + 1e-3
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(7.5, 6.5))
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.86)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(all_eef[:, 0], all_eef[:, 1], all_eef[:, 2], c="0.7", s=3, alpha=0.3, depthshade=False)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))
    ax.view_init(elev=30, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xlabel("x", labelpad=8); ax.set_ylabel("y", labelpad=8); ax.set_zlabel("z", labelpad=8)
    ax.scatter(*start_eef, c="white", edgecolors="k", s=70, label="start (EEF)")
    ax.scatter(*goal_eef, marker="*", c="gold", edgecolors="k", s=220, label="goal (EEF of q_goal)")
    trails, dots = {}, {}
    for k in paths:
        st = _STYLES[k]
        (trails[k],) = ax.plot([], [], [], color=st["color"], lw=st["lw"], ls=st["ls"],
                               label=f"{k} (joint len={lengths.get(k, float('nan')):.3f} rad)")
        (dots[k],) = ax.plot([], [], [], "o", color=st["color"], ms=7, mec="k")
    ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(title, fontsize=9)

    frames = []
    for t in np.linspace(0.0, 1.0, n_frames):
        for k, P in paths.items():
            pos, trail = _point_along(P, cum[k], t * cum[k][-1])
            trails[k].set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
            dots[k].set_data_3d([pos[0]], [pos[1]], [pos[2]])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    if out_png:
        fig.savefig(out_png, dpi=110)
    plt.close(fig)
    images = _quantize_frames_shared_palette(frames)
    durations = [frame_ms] * (len(images) - 1) + [hold_ms]
    images[0].save(out_gif, save_all=True, append_images=images[1:], duration=durations, loop=0, disposal=2)
    if log:
        log(f"[paths]   saved -> {out_gif}" + (f" and {out_png}" if out_png else ""))
    return out_gif
