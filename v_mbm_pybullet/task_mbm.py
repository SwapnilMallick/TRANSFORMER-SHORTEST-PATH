"""MBM task: one Panda problem in joint space, with a pure-function environment.

States are 7-D joint configurations q. Sampling and collision checks are functions of q; there is no
"committed simulator state" to keep in sync (the PyBullet robots are moved explicitly right before a
collision query, a render, or FK). One `CollisionChecker` is the source of truth for validity.

Goal = one or several joint configurations `Q_goal` (MBM has exactly one). A goal is *reached* only through
the explicit exact-goal connection in collect_trajectory_mbm (a node placed exactly at q_goal), so
`reached(q)` is an exact match against the goal set, not a radius test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from collision_mbm import DEFAULT_STRATEGY, CollisionChecker, PyBulletChecker
from render_mbm import PyBulletRenderer
from robot_mbm import N_DOF
from scene_mbm import Scene, load_scene


@dataclass
class MBMTaskConfig:
    step_size: float = 0.15            # max joint-space action norm (rad)
    novelty_frac: float = 0.25         # merge radius = novelty_frac * step_size
    goal_connect: float = 0.5          # rad: try an exact goal connection within this joint distance
    motion_res: float = 0.02           # rad: interpolation resolution of valid_motion
    max_sample_tries: int = 25
    margin: float = 0.0                # collision margin (m)
    strategy: str = DEFAULT_STRATEGY

    @property
    def novelty_radius(self) -> float:
        return self.novelty_frac * self.step_size


class MBMTask:
    def __init__(self, scene_json_path, cfg: MBMTaskConfig = MBMTaskConfig(),
                 checker: Optional[CollisionChecker] = None, scene: Optional[Scene] = None):
        self.cfg = cfg
        self.scene = scene if scene is not None else load_scene(scene_json_path)
        self.q_start = self.scene.start.copy()
        self.goals: List[np.ndarray] = [g.copy() for g in self.scene.goals]
        self.checker = checker if checker is not None else PyBulletChecker(
            self.scene, motion_res=cfg.motion_res, margin=cfg.margin, strategy=cfg.strategy)
        self.renderer = PyBulletRenderer(self.scene)
        self.q_low, self.q_high = self.renderer.robot.q_low.copy(), self.renderer.robot.q_high.copy()
        self.diagonal = float(np.linalg.norm(self.q_high - self.q_low))
        self._validate()

    def _validate(self):
        """Refuse to run on an invalid problem (start/goal out of limits or in collision)."""
        for name, q in [("start", self.q_start)] + [(f"goal[{i}]", g) for i, g in enumerate(self.goals)]:
            if q.shape != (N_DOF,):
                raise ValueError(f"{self.scene.name}: {name} has shape {q.shape}, expected ({N_DOF},)")
            if np.any(q < self.q_low - 1e-5) or np.any(q > self.q_high + 1e-5):
                raise ValueError(f"{self.scene.name}: {name} is outside the joint limits")
            if not self.checker.valid(q):
                raise ValueError(f"{self.scene.name}: {name} is in collision under the PyBullet checker")

    # ------------------------------------------------------------------ sampling
    def sample_free(self, q, rng) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(dq, q_next): uniform-in-7-ball step (dir ~ N(0, I) normalised, radius step * u^(1/7)) whose
        straight segment is collision-free. Steps leaving the joint limits are rejected, not clipped.
        None if every attempt fails."""
        cfg = self.cfg
        for _ in range(cfg.max_sample_tries):
            d = rng.normal(size=N_DOF)
            d /= np.linalg.norm(d) + 1e-12
            dq = d * (cfg.step_size * rng.random() ** (1.0 / N_DOF))
            q_next = q + dq
            if np.any(q_next < self.q_low) or np.any(q_next > self.q_high):
                continue
            if self.checker.valid_motion(q, q_next):
                return dq, q_next
        return None

    # ------------------------------------------------------------------ goal
    def nearest_goal(self, q) -> Tuple[int, float]:
        d = [float(np.linalg.norm(q - g)) for g in self.goals]
        i = int(np.argmin(d))
        return i, d[i]

    def reached(self, q) -> bool:
        """q IS a goal configuration (goal nodes sit exactly at q_goal)."""
        return any(bool(np.array_equal(q, g)) for g in self.goals)

    # ------------------------------------------------------------------ geometry helpers
    def fk_eef(self, q) -> np.ndarray:
        """End-effector XYZ (plots only). Moves the renderer's robot; callers set q again before rendering."""
        return self.renderer.robot.fk_eef(q)

    def eef_polyline(self, q_path, max_seg: Optional[float] = None) -> np.ndarray:
        """(N, 3) end-effector points along a joint-space path, each segment densified to <= max_seg
        (default step_size) so paths with long edges (goal connection, RRT-Connect shortcuts) are measured alike."""
        max_seg = max_seg or self.cfg.step_size
        pts = []
        for a, b in zip(q_path[:-1], q_path[1:]):
            n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / max_seg)))
            for k in range(n):
                pts.append(self.fk_eef(a + (b - a) * (k / n)))
        if len(q_path):
            pts.append(self.fk_eef(q_path[-1]))
        return np.array(pts).reshape(-1, 3)

    def eef_path_length(self, q_path, max_seg: Optional[float] = None) -> float:
        P = self.eef_polyline(q_path, max_seg)
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0

    def close(self):
        for o in (self.checker, self.renderer):
            if hasattr(o, "close"):
                o.close()
