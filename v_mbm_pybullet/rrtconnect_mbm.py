"""Joint-space RRT-Connect baseline in the SAME geometry as the transformer pipeline.

Same `CollisionChecker` (valid / valid_motion at the task's motion_res), same extension range (task.cfg.step_size),
same joint limits; plus shortcut smoothing (random-pair shortcuts, fixed iteration count). This is the
like-for-like baseline until VAMP / OMPL comparisons are added.

Per seed it reports: success, wall time (planning + smoothing), joint-space and EEF path length (raw and
smoothed), and collision-check counts (deltas of the checker's counters).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from task_mbm import MBMTask


class _Tree:
    def __init__(self, root: np.ndarray, cap: int = 1024):
        self.q = np.empty((cap, root.size))
        self.parent = [-1]
        self.n = 1
        self.q[0] = root

    def add(self, q: np.ndarray, parent: int) -> int:
        if self.n == len(self.q):
            self.q = np.concatenate([self.q, np.empty_like(self.q)])
        self.q[self.n] = q
        self.parent.append(parent)
        self.n += 1
        return self.n - 1

    def nearest(self, q: np.ndarray) -> int:
        return int(np.argmin(np.sum((self.q[:self.n] - q) ** 2, axis=1)))

    def path_to_root(self, i: int) -> List[np.ndarray]:
        out = []
        while i != -1:
            out.append(self.q[i].copy())
            i = self.parent[i]
        return out                                   # node ... root


def _extend(tree: _Tree, q_target: np.ndarray, task: MBMTask):
    """One step from the nearest node toward q_target (at most step_size). Returns (status, node index);
    status is 'reached' (q_target itself added), 'advanced' or 'trapped'."""
    i = tree.nearest(q_target)
    q_near = tree.q[i]
    d = q_target - q_near
    dist = float(np.linalg.norm(d))
    if dist <= task.cfg.step_size:
        q_new, status = q_target.copy(), "reached"
    else:
        q_new, status = q_near + d * (task.cfg.step_size / dist), "advanced"
    if not task.checker.valid_motion(q_near, q_new):
        return "trapped", -1
    return status, tree.add(q_new, i)


def rrt_connect(task: MBMTask, q_start, q_goal, rng, max_iters: int = 20000,
                time_limit: float = 120.0) -> Optional[List[np.ndarray]]:
    """Returns the raw waypoint path start -> goal, or None if no path within max_iters / time_limit."""
    ta, tb = _Tree(np.asarray(q_start, float)), _Tree(np.asarray(q_goal, float))
    start_is_a = True
    t0 = time.perf_counter()
    for _ in range(max_iters):
        if time.perf_counter() - t0 > time_limit:
            return None
        q_rand = rng.uniform(task.q_low, task.q_high)
        status, i_new = _extend(ta, q_rand, task)
        if status != "trapped":
            q_new = ta.q[i_new]
            while True:                                        # connect the other tree to q_new
                status_b, j = _extend(tb, q_new, task)
                if status_b != "advanced":
                    break
            if status_b == "reached":
                path_a = ta.path_to_root(i_new)[::-1]          # root_a ... q_new
                path_b = tb.path_to_root(j)                    # q_new ... root_b
                path = path_a + path_b[1:]
                return path if start_is_a else path[::-1]
        ta, tb = tb, ta
        start_is_a = not start_is_a
    return None


def shortcut(path: Sequence[np.ndarray], task: MBMTask, rng, n_iters: int = 200) -> List[np.ndarray]:
    """Random-pair shortcut smoothing with a fixed iteration count."""
    path = [np.asarray(p, float) for p in path]
    for _ in range(n_iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j - i < 2:
            continue
        if task.checker.valid_motion(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def joint_path_length(path) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))


def run_rrtconnect_baseline(task: MBMTask, seeds: Sequence[int] = range(5), q_goal=None,
                            smooth_iters: int = 200, max_iters: int = 20000,
                            time_limit: float = 120.0, log=print) -> List[Dict]:
    """Runs RRT-Connect + shortcut smoothing once per seed on the first goal (MBM has exactly one).
    Returns one dict per seed (with the smoothed path under 'path')."""
    q_goal = task.goals[0] if q_goal is None else q_goal
    out = []
    for seed in seeds:
        rng = np.random.default_rng(10_000 + seed)
        s0, m0 = task.checker.n_state_checks, task.checker.n_motion_checks
        t0 = time.perf_counter()
        raw = rrt_connect(task, task.q_start, q_goal, rng, max_iters=max_iters, time_limit=time_limit)
        res = {"seed": seed, "success": raw is not None}
        if raw is not None:
            smooth = shortcut(raw, task, rng, smooth_iters)
            res.update(time_sec=time.perf_counter() - t0, n_waypoints_raw=len(raw), n_waypoints=len(smooth),
                       joint_len_raw=joint_path_length(raw), joint_len=joint_path_length(smooth),
                       path=smooth)
        else:
            res.update(time_sec=time.perf_counter() - t0)
        res["n_state_checks"] = task.checker.n_state_checks - s0
        res["n_motion_checks"] = task.checker.n_motion_checks - m0
        if raw is not None:
            res["eef_len_raw"] = task.eef_path_length(raw)
            res["eef_len"] = task.eef_path_length(res["path"])
        out.append(res)
        if log:
            if res["success"]:
                log(f"  [rrt-connect seed {seed}] ok  {res['time_sec']:.2f}s  joint {res['joint_len_raw']:.3f}"
                    f" -> {res['joint_len']:.3f} rad  eef {res['eef_len']:.3f} m  "
                    f"checks: {res['n_state_checks']} states / {res['n_motion_checks']} motions")
            else:
                log(f"  [rrt-connect seed {seed}] FAILED after {res['time_sec']:.1f}s "
                    f"({res['n_state_checks']} state checks)")
    return out


if __name__ == "__main__":
    import argparse
    from scene_mbm import DEFAULT_SCENES_DIR, scene_path
    from task_mbm import MBMTaskConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="table_pick")
    ap.add_argument("--problem", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    task = MBMTask(scene_path(DEFAULT_SCENES_DIR, a.scene, a.problem), MBMTaskConfig())
    run_rrtconnect_baseline(task, range(a.seeds))
    task.close()
