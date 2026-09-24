"""
Records the final transformer-learned path (greedy descent on the trained
model's predicted cost-to-come field -- online_cost_transformer.transformer_path,
reused completely unchanged since it's already environment-agnostic: it only
touches graph.pos/model/cfg, never anything maze-specific) as a side-by-side
multi-camera GIF: agentview, frontview, and a custom free camera framing the
whole reachable workspace from an angled overhead POV, similar in spirit to
the per-iteration uncertainty plot's 3D scatter default viewing angle.

The path is a sequence of GRAPH NODES (nearest-to-cube -> start, reversed to
start -> nearest-to-cube for playback); node_q gives each node's actual joint
configuration -- see collect_trajectory_robosuite.py's module docstring for
why a node's EEF position alone doesn't pin down its joint config.
Consecutive waypoints' joint configs are linearly interpolated purely for a
visually smooth video (matching RobosuiteWorld.collides' substep
convention) -- there's no physics here, so the interpolated in-between
configs are cosmetic only, not new plan states.

Rendering uses a FRESH, rendering-enabled RobosuiteIK instance: the training
run's own task.ik was constructed with rendering disabled and robosuite
can't reconfigure that after the fact. The fresh instance is built with the
SAME frozen cube placement as `task` (via task.py's _fixed_cube_initializer)
so the recorded cube position matches the training run's.

save_path_comparison_gif is a separate, matplotlib-only output (no MuJoCo
rendering): a single 3D plot animating the roadmap-Dijkstra path and the
transformer's greedy path side by side, with the final frame held. Both
paths are advanced by the same FRACTION of their own total length per frame
(not the same number of nodes), so they start together and arrive together
even though they generally differ in length and node count.

Usage: called from main_robosuite.py at the end of training -- there's
nothing to record standalone (no trained model/graph/node_q to draw a path
from) without first running that training loop, e.g.:
    python main_robosuite.py --iters 3 --k 4 --max-steps 20
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import mujoco

from online_cost_transformer import (
    Graph, predict_cost_field, transformer_path, path_length, dijkstra_path,
)
from ik_solver import RobosuiteIK
from task import RobosuiteTask, _fixed_cube_initializer


def extract_transformer_path_robosuite(model, graph: Graph, cfg, task: RobosuiteTask,
                                       node_q: Dict[int, np.ndarray]):
    """Greedy descent on the trained model's predicted cost field, from the
    node nearest the cube back to the start -- the robosuite counterpart of
    online_cost_transformer.extract_and_save_paths's transformer_path call.
    predict_cost_field / transformer_path / path_length need no robosuite-
    specific adaptation at all: they only touch graph.pos/model/cfg, which
    are already 3D-native here.

    Returns (node_ids, q_path, reached, cost) -- node_ids/q_path run
    start -> nearest-to-cube (playback order)."""
    goal_node = graph.nearest(task.cube_pos)[0]
    chat = predict_cost_field(model, graph, cfg)
    t_nodes, t_reached = transformer_path(model, graph, cfg, goal_node, chat=chat)
    q_path = [node_q[n] for n in t_nodes]
    cost = path_length(graph, t_nodes)
    return t_nodes, q_path, t_reached, cost


def _lerp_q_path(q_path: Sequence[np.ndarray], interp_steps: int) -> List[np.ndarray]:
    """Linear joint-space interpolation between consecutive waypoints, purely
    for a smooth-looking video -- see module docstring."""
    if len(q_path) < 2:
        return list(q_path)
    out = [q_path[0]]
    for q0, q1 in zip(q_path[:-1], q_path[1:]):
        for k in range(1, interp_steps + 1):
            out.append(q0 + (q1 - q0) * (k / interp_steps))
    return out


def _quantize_frames_shared_palette(frames: List[np.ndarray], colors: int = 256):
    """Remaps every frame onto ONE shared 256-color palette built from a
    sample spanning the whole sequence, instead of letting each frame pick
    its own palette independently (PIL's GIF-saving default). Per-frame
    independent quantization is why the same real-world color -- the
    Franka's silver plastic, the cube's red -- can shift between frames (a
    different palette is chosen each time), and can crush subtle shading
    gradients (e.g. the cube's darker faces) into a nearby grey that isn't
    actually there. A single shared palette, informed by colors from across
    the whole clip rather than just one frame, keeps color consistent."""
    from PIL import Image
    sample_stride = max(1, len(frames) // 24)
    composite = np.concatenate(frames[::sample_stride], axis=0)
    palette_img = Image.fromarray(composite).convert("RGB").quantize(
        colors=colors, dither=Image.Dither.NONE)
    # dither=NONE: this is a rendered scene (mostly flat/smooth color
    # regions), not a photo -- Floyd-Steinberg dithering (PIL's default)
    # speckles those flat regions with a visible noise pattern instead of
    # helping, since there's little gradient detail for it to preserve.
    return [Image.fromarray(f).convert("RGB").quantize(
        palette=palette_img, dither=Image.Dither.NONE) for f in frames]


def _label_frame(arr: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 7 * len(text) + 6, 16], fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return np.array(img)


def record_transformer_path_gif(
    task: RobosuiteTask, q_path: Sequence[np.ndarray], out_path: str,
    height: int = 256, width: int = 256, interp_steps: int = 6,
    frame_ms: int = 120, hold_ms: int = 1500, log=None,
):
    """Renders q_path (a sequence of joint configs, start -> target) from
    three side-by-side camera views into one animated GIF."""
    try:
        from PIL import Image
    except Exception as e:                                     # pragma: no cover
        if log:
            log(f"[video]   skipped: {e}")
        return None
    if len(q_path) < 1:
        if log:
            log("[video]   skipped: empty path")
        return None

    render_ik = RobosuiteIK(
        placement_initializer=_fixed_cube_initializer(task.cube_pos[0], task.cube_pos[1]),
        enable_render=True)
    render_ik.env.reset()

    world_cfg = task.world.cfg
    center = np.array([
        sum(world_cfg.x_range) / 2, sum(world_cfg.y_range) / 2, sum(world_cfg.z_range) / 2,
    ])
    cam_custom = mujoco.MjvCamera()
    cam_custom.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_custom.lookat = center
    cam_custom.distance = 1.2
    cam_custom.azimuth = 120           # diagonally opposite side (was -60, i.e. +180deg)
    cam_custom.elevation = -30

    frames = []
    q_smooth = _lerp_q_path(list(q_path), interp_steps)
    for q in q_smooth:
        render_ik.set_q(q)
        views = [
            _label_frame(render_ik.render("agentview", height, width), "agentview"),
            _label_frame(render_ik.render("frontview", height, width), "frontview"),
            _label_frame(render_ik.render(cam_custom, height, width), "isometric"),
        ]
        frames.append(np.concatenate(views, axis=1))            # side-by-side

    images = _quantize_frames_shared_palette(frames)
    durations = [frame_ms] * (len(images) - 1) + [hold_ms]
    images[0].save(out_path, save_all=True, append_images=images[1:],
                  duration=durations, loop=0, disposal=2)
    render_ik.close()
    if log:
        log(f"[video]   saved -> {out_path}")
    return out_path


def draw_goal_sphere(ax, center, radius: float) -> None:
    """Translucent gold sphere on a 3D axes marking the success region -- every
    point within task.cfg.eps_dist of the cube counts as reached. The 3D
    counterpart of the maze plots' gold goal-radius circle. Only looks
    spherical if the axes have equal scaling (set_box_aspect from the limits)."""
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    ax.plot_surface(center[0] + radius * np.cos(u) * np.sin(v),
                    center[1] + radius * np.sin(u) * np.sin(v),
                    center[2] + radius * np.cos(v),
                    color="gold", alpha=0.3, linewidth=0, shade=False,
                    antialiased=False)


def _point_along(P: np.ndarray, cum: np.ndarray, s: float):
    """Position after travelling arc length `s` along polyline P (`cum` is its
    cumulative arc length per vertex), and the trail drawn so far: every
    vertex already passed plus the current, partway-along-a-segment point."""
    if len(P) < 2:
        return P[0], P[:1]
    s = min(max(s, 0.0), cum[-1])
    i = min(max(int(np.searchsorted(cum, s, side="right")) - 1, 0), len(P) - 2)
    seg = cum[i + 1] - cum[i]
    frac = 0.0 if seg <= 1e-12 else min(max((s - cum[i]) / seg, 0.0), 1.0)
    pos = P[i] + frac * (P[i + 1] - P[i])
    return pos, np.vstack([P[: i + 1], pos])


def save_path_comparison_gif(
    graph: Graph, task: RobosuiteTask, t_nodes: Sequence[int],
    out_gif: str, out_png: str = None, n_frames: int = 60,
    frame_ms: int = 80, hold_ms: int = 2500, log=None,
):
    """Animated 3D comparison of the roadmap-Dijkstra path (crimson, solid) and
    the transformer's greedy path `t_nodes` (deepskyblue, dashed), both from
    the start to the node nearest the cube -- same colors/styles as the
    maze's final plot. Paths are EEF (x, y, z) positions from graph.pos.

    Each frame advances both paths by the same fraction of their own total
    arc length, so they leave together and arrive together regardless of
    how many nodes each has. The last frame is held for `hold_ms`, and (if
    out_png is given) also saved as a static image. Every explored graph node
    is drawn faintly behind the paths for context.

    The endpoint is the node NEAREST the cube, not necessarily the cube
    itself (collection may never have reached it) -- the title reports the
    remaining gap so the paths ending short of the star isn't confusing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
        from PIL import Image
    except Exception as e:                                     # pragma: no cover
        if log:
            log(f"[paths]   comparison skipped: {e}")
        return None

    g, prev = graph.dijkstra(0, return_prev=True)
    goal_node = graph.nearest(task.cube_pos)[0]
    d_nodes = dijkstra_path(graph, prev, goal_node)
    if not d_nodes or len(t_nodes) < 1:
        if log:
            log("[paths]   comparison skipped: empty path")
        return None

    Pd = np.array([graph.pos[i] for i in d_nodes])
    Pt = np.array([graph.pos[i] for i in t_nodes])
    d_cost, t_cost = float(g[goal_node]), path_length(graph, t_nodes)
    cum_d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pd, axis=0), axis=1))])
    cum_t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pt, axis=0), axis=1))])
    gap = float(np.linalg.norm(graph.pos[goal_node] - task.cube_pos))
    ratio = t_cost / d_cost if d_cost > 0 else float("nan")

    eps = float(task.cfg.eps_dist)
    allP = np.array(graph.pos)
    lo = np.minimum(allP.min(axis=0), task.cube_pos - eps)
    hi = np.maximum(allP.max(axis=0), task.cube_pos + eps)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(7.5, 6.5))
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.88)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(allP[:, 0], allP[:, 1], allP[:, 2], c="0.7", s=3, alpha=0.3,
               depthshade=False)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))
    ax.view_init(elev=30, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xlabel("x", labelpad=8); ax.set_ylabel("y", labelpad=8)
    ax.set_zlabel("z", labelpad=8)
    ax.scatter(*task.eef_start, c="white", edgecolors="k", s=70, label="start")
    ax.scatter(*task.cube_pos, marker="*", c="gold", edgecolors="k", s=220, label="cube")
    draw_goal_sphere(ax, task.cube_pos, eps)
    (trail_d,) = ax.plot([], [], [], color="crimson", lw=2.6,
                         label=f"roadmap Dijkstra (len={d_cost:.3f})")
    (trail_t,) = ax.plot([], [], [], color="deepskyblue", lw=2.0, ls="--",
                         label=f"transformer greedy (len={t_cost:.3f})")
    (dot_d,) = ax.plot([], [], [], "o", color="crimson", ms=7, mec="k")
    (dot_t,) = ax.plot([], [], [], "o", color="deepskyblue", ms=7, mec="k")
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="gold", alpha=0.4))
    labels.append(f"goal radius ({eps:.3f} m)")
    ax.legend(handles, labels, loc="upper left", fontsize=8)
    fig.suptitle(f"Dijkstra vs transformer path   (transformer/Dijkstra = {ratio:.3f})\n"
                 f"both end {gap:.3f} m from the cube (goal radius {eps:.3f} m)",
                 fontsize=10)

    frames = []
    for t in np.linspace(0.0, 1.0, n_frames):
        pd, trd = _point_along(Pd, cum_d, t * cum_d[-1])
        pt, trt = _point_along(Pt, cum_t, t * cum_t[-1])
        trail_d.set_data_3d(trd[:, 0], trd[:, 1], trd[:, 2])
        trail_t.set_data_3d(trt[:, 0], trt[:, 1], trt[:, 2])
        dot_d.set_data_3d([pd[0]], [pd[1]], [pd[2]])
        dot_t.set_data_3d([pt[0]], [pt[1]], [pt[2]])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    if out_png:
        fig.savefig(out_png, dpi=110)
    plt.close(fig)

    images = _quantize_frames_shared_palette(frames)
    durations = [frame_ms] * (len(images) - 1) + [hold_ms]
    images[0].save(out_gif, save_all=True, append_images=images[1:],
                  duration=durations, loop=0, disposal=2)
    if log:
        log(f"Roadmap Dijkstra path:    {len(d_nodes)} nodes, cost={d_cost:.4f}  |  "
            f"transformer/Dijkstra cost ratio = {ratio:.3f}")
        log(f"[paths]   saved -> {out_gif}" + (f" and {out_png}" if out_png else ""))
    return out_gif
