"""
Step 3: MuJoCo-based collision checking, 3D sample_free, and a reset-and-replay
execution path, built on top of ik_solver.RobosuiteIK.

Collision rule (matches RRT-ROBOSUITE's rrt_enc convention): FORBIDDEN =
{table<->robot, robot<->robot (self-collision)}; gripper<->cube (and any
other robot<->cube) contact is ALLOWED, since reaching/touching the cube is
the point of the task. Geom->category classification is grounded in the
actual Lift/Panda model's body names (see the geom dump this was built
against), not guessed:
    table            -> body name == "table"
    cube             -> body name == "cube_main"
    robot (forbidden)-> body name startswith robot0_ / gripper0_ / fixed_mount0_
anything else (e.g. the "*_eef_target" marker bodies robosuite adds) is left
unclassified and never matches a forbidden pair.

Note on reset-and-replay: RRT-ROBOSUITE's Go-Explore variant resets the env
and replays a node's full joint-action history to reach it, because its
mujoco state can in principle be path-dependent. In this project we never
call mj_step (no physics integration) and the cube pose is frozen -- every
state transition is a direct qpos write + mj_forward, which is exactly what
RobosuiteIK.set_q already does. Under that kinematic-only model there is no
hidden path-dependent state (qvel/contact caches are recomputed fresh by
mj_forward every call), so replaying the whole history and directly calling
set_q(q) on the target node's stored q produce an *identical* resulting
simulator state. reset_and_replay() is still provided below (matching the
requested step), but collect_trajectory-style code can safely use the direct
set_q fast path instead -- see the smoke test at the bottom, which checks
this equivalence empirically rather than assuming it.

Usage:
    python robosuite_env.py   # smoke test: forbidden collision, free sample_free, replay equivalence
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ik_solver import IKConfig, RobosuiteIK


@dataclass
class WorldConfig:
    step_size: float = 0.04          # max |delta-translation action|, meters
    goal_bias: float = 0.0           # prob. of sampling straight toward the cube
    max_sample_tries: int = 25
    n_collision_substeps: int = 8    # matches RRT-ROBOSUITE default
    novelty_radius: float = 0.01     # roadmap-graph merge radius, meters (~step_size/4,
                                      # matching the maze's novelty_radius:step_size ratio)
    x_range: Tuple[float, float] = (-0.25, 0.25)
    y_range: Tuple[float, float] = (-0.25, 0.25)
    z_range: Tuple[float, float] = (0.80, 1.10)


def world_diagonal(cfg: WorldConfig) -> float:
    """3D workspace bounding-box diagonal -- the robosuite counterpart of the
    maze's sqrt(2)*L. Used to normalize Dijkstra-cost training labels and the
    veto's candidate cost into ~[0,1] units, matching
    new_maze_envs_v7/online_cost_transformer.py's Config.diagonal."""
    dx = cfg.x_range[1] - cfg.x_range[0]
    dy = cfg.y_range[1] - cfg.y_range[0]
    dz = cfg.z_range[1] - cfg.z_range[0]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def _classify_geom(model, geom_id: int) -> str:
    body = model.body_id2name(model.geom_bodyid[geom_id]) or ""
    if body == "table":
        return "table"
    if body == "cube_main":
        return "cube"
    if body.startswith("robot0_") or body.startswith("gripper0_") or body.startswith("fixed_mount0_"):
        return "robot"
    return "other"


class RobosuiteWorld:
    """IK + collision-aware 3D sampler over the Lift/Panda workspace."""

    def __init__(self, ik: RobosuiteIK, cfg: WorldConfig = WorldConfig()):
        self.ik = ik
        self.cfg = cfg
        m = ik.sim.model
        self._geom_class = [_classify_geom(m, i) for i in range(m.ngeom)]

    # -- collision --------------------------------------------------------
    def _forbidden_contact(self) -> bool:
        """True if any CURRENT contact (post mj_forward) is table<->robot or
        robot<->robot (self-collision). Cube contacts are always allowed."""
        d = self.ik.sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = self._geom_class[c.geom1], self._geom_class[c.geom2]
            pair = {g1, g2}
            if pair == {"table", "robot"}:
                return True
            if g1 == "robot" and g2 == "robot":
                return True
        return False

    def collides(self, q_from, q_to) -> bool:
        """Interpolates q_from -> q_to in joint space over
        cfg.n_collision_substeps steps (anti-tunnelling), checking forbidden
        contacts at each substep. Leaves the sim at q_to afterward."""
        q_from = np.asarray(q_from, float)
        q_to = np.asarray(q_to, float)
        n = self.cfg.n_collision_substeps
        for k in range(1, n + 1):
            q = q_from + (q_to - q_from) * (k / n)
            self.ik.set_q(q)
            if self._forbidden_contact():
                return True
        return False

    # -- sampling -----------------------------------------------------------
    def _in_bounds(self, p) -> bool:
        cfg = self.cfg
        return (cfg.x_range[0] <= p[0] <= cfg.x_range[1] and
                cfg.y_range[0] <= p[1] <= cfg.y_range[1] and
                cfg.z_range[0] <= p[2] <= cfg.z_range[1])

    def sample_free(self, rng) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Returns (action, s_next_eef, q_next) for a collision-free,
        IK-reachable delta-translation step from the CURRENT state, or None
        if every attempt this call failed (workspace/IK/collision)."""
        cfg = self.cfg
        q_from = self.ik.get_q()
        eef = self.ik.eef_pos()
        cube = self.ik.cube_pos()

        for _ in range(cfg.max_sample_tries):
            if cfg.goal_bias > 0 and rng.random() < cfg.goal_bias:
                v = cube - eef
                n = np.linalg.norm(v)
                dirv = v / n if n > 1e-9 else rng.normal(size=3)
            else:
                dirv = rng.normal(size=3)
                dirv /= np.linalg.norm(dirv) + 1e-12
            # uniform-in-ball radius (cube-root of a uniform sample), the 3D
            # analogue of the 2D sampler's sqrt() for uniform-in-disk
            mag = cfg.step_size * (rng.random() ** (1.0 / 3.0))
            a = dirv * mag
            target = eef + a
            if not self._in_bounds(target):
                continue

            q_to, ok = self.ik.solve_ik(target, q_init=q_from)
            if not ok:
                self.ik.set_q(q_from)      # restore before the next attempt
                continue
            if self.collides(q_from, q_to):
                self.ik.set_q(q_from)
                continue
            return a.astype(float), self.ik.eef_pos(), q_to

        self.ik.set_q(q_from)
        return None

    # -- reset-and-replay -----------------------------------------------
    def reset_and_replay(self, q_path) -> None:
        """Resets the env then replays a sequence of stored joint configs
        (root -> target node) via direct set_q calls. See module docstring:
        under this project's kinematic-only, frozen-cube execution model this
        is provably equivalent to a single set_q(q_path[-1]) call -- kept for
        parity with RRT-ROBOSUITE's design and as a defensive option if real
        physics stepping is ever added later."""
        self.ik.env.reset()
        for q in q_path:
            self.ik.set_q(q)


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    ik = RobosuiteIK()
    world = RobosuiteWorld(ik)

    print("geom class counts:",
          {c: world._geom_class.count(c) for c in set(world._geom_class)})

    # 1) forbidden-contact sanity check: drive the arm straight down through
    #    the table and confirm collides() actually catches it.
    q0 = ik.get_q()
    below_table = ik.eef_pos().copy()
    below_table[2] = 0.5       # well beneath the table surface
    q_bad, ok = ik.solve_ik(below_table, q_init=q0)
    print("\nIK toward below-table target converged:", ok)
    ik.set_q(q0)
    hit = world.collides(q0, q_bad)
    print("collides(q0 -> below-table q) =", hit, " (expect True)")
    ik.set_q(q0)

    # 2) sample_free: a handful of free-space steps should succeed
    n_ok = 0
    q_path = [ik.get_q()]
    t0 = time.time()
    for i in range(10):
        res = world.sample_free(rng)
        if res is not None:
            a, s_next, q_next = res
            n_ok += 1
            q_path.append(q_next)
            print(f"[{i}] action={a.round(4)} eef_next={s_next.round(4)}")
        else:
            print(f"[{i}] sample_free failed (all {world.cfg.max_sample_tries} tries)")
    dt = time.time() - t0
    print(f"\nsample_free: {n_ok}/10 succeeded, {dt:.2f}s total ({dt/10*1000:.1f} ms/call avg)")

    # 3) reset-and-replay vs. direct set_q equivalence check
    final_q_direct = q_path[-1].copy()
    ik.set_q(final_q_direct)
    eef_direct = ik.eef_pos().copy()

    world.reset_and_replay(q_path)
    eef_replay = ik.eef_pos().copy()

    print(f"\ndirect set_q EEF: {eef_direct.round(6)}")
    print(f"replay-to-same-q EEF: {eef_replay.round(6)}")
    print("equivalent:", bool(np.allclose(eef_direct, eef_replay, atol=1e-9)))

    ik.close()
