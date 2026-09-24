"""
Step 4: task definition for the robosuite Lift/Panda cube-reach experiment.

Two corrections made here based on direct measurement, not assumption (see
the session's step-4 discussion):
  - Ground-truth "reached" distance is computed from OUR OWN grip_site eef
    (ik.eef_pos()) to the cube, NOT robosuite's `gripper_to_cube_pos`
    observable -- that key references a different site than our IK's
    grip_site (measured ~0.018m apart, more than half of eps_dist=0.03),
    which would silently bias the success check if used.
  - "Frozen cube pose" is enforced with an explicit fixed placement
    initializer (ported from RRT-ROBOSUITE's `_fixed_initializer`), not just
    asserted. The cube's placement RNG turns out to be seeded once at ENV
    CONSTRUCTION, not per-reset: repeated .reset() calls on one already-built
    env looked deterministic (an earlier, flawed check), but 20 *fresh* env
    constructions showed real xy scatter (std ~0.017m, range ~0.055-0.058m --
    well past eps_dist=0.03). RobosuiteTask now does a two-phase build: probe
    the default sampler once to get a representative (x, y), then rebuild the
    real env with a zero-width UniformRandomSampler pinned to that point, so
    every future construction (not just every reset of one object) reproduces
    the same cube pose.

eps_dist=0.03 matches RRT-ROBOSUITE's convention. Workspace bounds come from
WorldConfig (robosuite_env.py); this module checks they actually contain the
frozen cube pose and start EEF pose rather than assuming so.

Usage:
    python task.py   # smoke test: prints frozen state, verifies determinism
                      # across FRESH env constructions, confirms reached() is
                      # False at start and True near the cube
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from robosuite.utils.placement_samplers import UniformRandomSampler

from ik_solver import RobosuiteIK
from robosuite_env import WorldConfig, RobosuiteWorld


def _fixed_cube_initializer(cx: float, cy: float) -> UniformRandomSampler:
    """Ported verbatim (config) from RRT-ROBOSUITE's _fixed_initializer:
    zero-width x/y range pins the cube's planar position; z is still placed
    by robosuite's normal table-drop logic relative to reference_pos/z_offset,
    which is what naturally reproduces the resting height (~0.83, matching
    what was measured for the default sampler)."""
    return UniformRandomSampler(
        name="ObjectSampler", x_range=[cx, cx], y_range=[cy, cy],
        rotation=0.0, rotation_axis="z", ensure_object_boundary_in_range=False,
        ensure_valid_placement=False, reference_pos=(0.0, 0.0, 0.8), z_offset=0.01)


@dataclass
class TaskConfig:
    eps_dist: float = 0.03                          # ground-truth success threshold (m)
    world: WorldConfig = field(default_factory=WorldConfig)


class RobosuiteTask:
    def __init__(self, cfg: TaskConfig = TaskConfig(), seed: int = None):
        """seed: if given, the probe env below is seeded with it, so the same
        seed always yields the same cube position (previously the probe was
        unseeded, so the cube differed on every construction even with the
        same run seed -- which confounds comparisons across runs, e.g. an
        alpha sweep). None keeps the old random-per-construction behaviour."""
        self.cfg = cfg

        # phase 1: probe the default (unfixed) sampler once, to get a
        # planar target. Seeded (when `seed` is given) so it is reproducible;
        # unseeded it differs on every construction.
        probe = RobosuiteIK(seed=seed)
        probe.env.reset()
        probe_xy = probe.cube_pos()[:2].copy()
        probe.close()

        # phase 2: rebuild with the cube's xy pinned to that point, so every
        # future construction (not just every reset of this one object)
        # reproduces the same pose
        self.ik = RobosuiteIK(
            placement_initializer=_fixed_cube_initializer(probe_xy[0], probe_xy[1]),
            seed=seed)
        self.world = RobosuiteWorld(self.ik, cfg.world)

        self.ik.env.reset()
        self.q_start = self.ik.get_q().copy()
        self.eef_start = self.ik.eef_pos().copy()
        self.cube_pos = self.ik.cube_pos().copy()

        if not self.world._in_bounds(self.cube_pos):
            raise ValueError(
                f"cube_pos {self.cube_pos} outside workspace bounds "
                f"x={cfg.world.x_range} y={cfg.world.y_range} z={cfg.world.z_range}")
        if not self.world._in_bounds(self.eef_start):
            raise ValueError(
                f"start EEF {self.eef_start} outside workspace bounds "
                f"x={cfg.world.x_range} y={cfg.world.y_range} z={cfg.world.z_range}")

    def verify_cube_frozen(self, n_checks: int = 5, atol: float = 3e-3) -> bool:
        """Checks the fixed placement initializer holds across repeated
        resets of this object AND across fresh env constructions (the
        harder, actually-meaningful test -- see module docstring for why
        reset-only checks were insufficient). atol=3mm: xy is pinned exactly
        by the fixed sampler (measured identical to the mm), but z has ~1-2mm
        of settling-simulation jitter across fresh constructions that the
        fixed sampler doesn't (and RRT-ROBOSUITE's original doesn't either)
        pin -- comfortably tighter than the ~55mm scatter this is guarding
        against, loose enough not to flag that harmless residual noise."""
        ok = True
        for _ in range(n_checks):
            self.ik.env.reset()
            if not np.allclose(self.ik.cube_pos(), self.cube_pos, atol=atol):
                ok = False
        self.reset_to_start()

        for _ in range(n_checks):
            fresh = RobosuiteIK(
                placement_initializer=_fixed_cube_initializer(
                    self.cube_pos[0], self.cube_pos[1]))
            fresh.env.reset()
            if not np.allclose(fresh.cube_pos(), self.cube_pos, atol=atol):
                ok = False
            fresh.close()
        return ok

    def reset_to_start(self) -> None:
        self.ik.env.reset()
        self.ik.set_q(self.q_start)

    def dist_to_cube(self, eef_pos=None) -> float:
        if eef_pos is None:
            eef_pos = self.ik.eef_pos()
        return float(np.linalg.norm(np.asarray(eef_pos, float) - self.cube_pos))

    def reached(self, eef_pos=None) -> bool:
        return self.dist_to_cube(eef_pos) < self.cfg.eps_dist

    def close(self) -> None:
        self.ik.close()


if __name__ == "__main__":
    task = RobosuiteTask()
    print("q_start:   ", task.q_start.round(4))
    print("eef_start: ", task.eef_start.round(4))
    print("cube_pos:  ", task.cube_pos.round(4))
    print("dist(start, cube):", round(task.dist_to_cube(), 4), " eps_dist:", task.cfg.eps_dist)
    print("reached() at start (expect False):", task.reached())

    print("\nverifying cube pose is stable across 5 resets...")
    print("verify_cube_frozen():", task.verify_cube_frozen())

    # drive the arm straight toward the cube via IK and confirm reached()
    # flips True once within eps_dist -- a real end-to-end check of the
    # success condition, not just a distance printout
    print("\ndriving toward the cube via IK...")
    q, ok = task.ik.solve_ik(task.cube_pos, q_init=task.q_start)
    task.ik.set_q(q)
    print("IK converged:", ok)
    print("dist(after IK, cube):", round(task.dist_to_cube(), 4))
    print("reached() after approach (expect True):", task.reached())

    task.close()
