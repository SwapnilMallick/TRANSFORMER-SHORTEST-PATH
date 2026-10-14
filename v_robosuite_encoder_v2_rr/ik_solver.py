"""
Standalone damped-least-squares (DLS) Jacobian IK for the robosuite Lift/Panda
env, ported from RRT-ROBOSUITE's rrt_enc/rrt_ik_encoder.py (RobosuiteBackend),
decoupled from that repo's camera/controller/encoder machinery -- this project
only needs kinematic execution (set qpos directly + mj_forward, no physics
stepping, no rendering) and ground-truth gripper-cube distance, not vision.

Env creation matches Go-Explore RRT's style: has_renderer=False,
has_offscreen_renderer=False, use_camera_obs=False (no camera pipeline at
all), ignore_done=True (we drive termination ourselves via true distance).

Usage:
    python ik_solver.py    # smoke test: random reachable target, checks IK converges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np
import robosuite as suite

# robosuite logs an INFO line ("Loading controller configuration from ...")
# on every env build AND every env.reset() -- one per trajectory here, so
# hundreds per run. Raise its logger to WARNING so real warnings still show.
# (The few startup warnings printed during `import robosuite` itself come
# before this line runs and are unaffected.)
logging.getLogger("robosuite_logs").setLevel(logging.WARNING)


@dataclass
class IKConfig:
    env_name: str = "Lift"
    robot: str = "Panda"
    iters: int = 100
    tol: float = 1e-3
    damping: float = 1e-2
    max_step: float = 0.1
    seed: int = 0


class RobosuiteIK:
    """Wraps a robosuite env + DLS IK for the 'grip_site' (Panda gripper site).

    State is exposed/consumed as qpos over the 7 arm joints; EEF position is
    read from the grip site's world xyz. No physics stepping is used for
    IK/execution -- set_q() teleports the arm kinematically via qpos + forward,
    matching RRT-ROBOSUITE's Go-Explore variant ("execute = adopt the
    configuration")."""

    def __init__(self, cfg: IKConfig = IKConfig(), placement_initializer=None,
                enable_render: bool = False, seed: int = None):
        self.cfg = cfg
        make_kwargs = dict(
            env_name=cfg.env_name,
            robots=cfg.robot,
            has_renderer=False,
            has_offscreen_renderer=enable_render,
            use_camera_obs=False,
            ignore_done=True,
        )
        if placement_initializer is not None:
            make_kwargs["placement_initializer"] = placement_initializer
        if enable_render:
            # Soft resets for rendering-enabled envs. With robosuite's default
            # hard reset, every env.reset() (one per trajectory via
            # RobosuiteTask.reset_to_start) destroys and rebuilds env.sim and
            # its GL contexts. self.sim below is captured once, so it would
            # point at an orphaned sim, and after a few resets the
            # mujoco.Renderer built on it silently returned STALE frames
            # (measured: 25 different arm states rendering as 2-4 distinct
            # images), which would make every embedding meaningless. A soft
            # reset keeps env.sim (and hence self.sim and the renderer) alive.
            # Non-rendering envs keep the default, so nothing else changes.
            make_kwargs["hard_reset"] = False
        if seed is not None:
            # robosuite's env rng (which drives the cube placement sampler) is
            # otherwise seeded from OS entropy -- unseeded, so the cube's
            # position differed on every construction even with the same
            # numpy/torch seeds
            make_kwargs["seed"] = seed
        self.env = suite.make(**make_kwargs)
        self.env.reset()
        self.sim = self.env.sim
        self._renderer = None
        self._renderer_size = None

        m = self.sim.model
        grip = [n for n in m.site_names if n.endswith("grip_site")]
        if not grip:
            raise RuntimeError(f"no grip site found; sites={m.site_names}")
        self.grip_site_name = grip[0]
        self.grip_site_id = m.site_name2id(self.grip_site_name)

        # Panda's 7 arm joints (robot0_joint1..7) -- matches RRT-ROBOSUITE's
        # naming convention for the default single-robot robosuite setup.
        self.joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        self.qpos_idx = np.array(
            [m.get_joint_qpos_addr(n) for n in self.joint_names])
        jids = [m.joint_name2id(n) for n in self.joint_names]
        self._dof_idx = np.array([m.jnt_dofadr[j] for j in jids])
        jr = m.jnt_range
        self.jnt_low = jr[jids, 0].copy()
        self.jnt_high = jr[jids, 1].copy()

    # -- kinematic state -------------------------------------------------
    def set_q(self, q) -> None:
        self.sim.data.qpos[self.qpos_idx] = q
        self.sim.forward()

    def get_q(self) -> np.ndarray:
        return self.sim.data.qpos[self.qpos_idx].copy()

    def eef_pos(self) -> np.ndarray:
        return self.sim.data.site_xpos[self.grip_site_id].copy()

    def cube_pos(self) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[
            self.sim.model.body_name2id("cube_main")], float)

    def camera_names(self):
        """Names of the cameras defined in the sim (e.g. 'agentview',
        'sideview', 'frontview') -- used to validate the encoder's camera list."""
        try:
            return list(self.sim.model.camera_names)
        except AttributeError:
            m = self.sim.model._model if hasattr(self.sim.model, "_model") else self.sim.model
            return [m.camera(i).name for i in range(m.ncam)]

    # -- rendering (requires enable_render=True at construction) -----------
    def render(self, camera, height: int = 256, width: int = 256) -> np.ndarray:
        """Renders the CURRENT sim state from `camera` -- either a named
        camera string (e.g. 'agentview', 'frontview') or a mujoco.MjvCamera
        (for a custom free-camera view). Uses mujoco.Renderer directly
        rather than robosuite's use_camera_obs pipeline, since that pipeline
        only refreshes on reset()/step() -- this project drives the arm via
        set_q() + mj_forward (no stepping), so a direct render call is needed
        to get an up-to-date frame after every set_q().

        Explicitly hides geom group 0 (collision geometry -- e.g. the Panda's
        crude per-link collision meshes, hardcoded to flat colors like green)
        and shows only group 1 (the actual visual meshes). mujoco.Renderer's
        default MjvOption shows every geom group at once; robosuite's own
        on-screen viewer configures this correctly, but that configuration
        doesn't carry over when driving mujoco.Renderer directly like this.
        Without it, the collision and visual meshes z-fight (near-identical
        depth, near-identical geometry), which is what produced the
        speckled/discolored look on the arm -- not a texture or asset bug."""
        if self._renderer is None or self._renderer_size != (height, width):
            m = self.sim.model._model if hasattr(self.sim.model, "_model") else self.sim.model
            self._renderer = mujoco.Renderer(m, height=height, width=width)
            self._renderer_size = (height, width)
            self._scene_option = mujoco.MjvOption()
            self._scene_option.geomgroup[0] = 0
        d = self.sim.data._data if hasattr(self.sim.data, "_data") else self.sim.data
        self._renderer.update_scene(d, camera=camera, scene_option=self._scene_option)
        return self._renderer.render()

    # -- IK ----------------------------------------------------------------
    def solve_ik(self, target_xyz, q_init=None):
        """Damped-least-squares IK for the grip site. Returns (q, ok).
        Ported verbatim (algorithm) from RRT-ROBOSUITE's solve_ik."""
        cfg = self.cfg
        q = self.get_q().copy() if q_init is None else np.array(q_init, float).copy()
        m, d = self.sim.model, self.sim.data
        jacp = np.zeros((3, m.nv))
        for _ in range(cfg.iters):
            self.set_q(q)
            cur = d.site_xpos[self.grip_site_id]
            err = np.asarray(target_xyz, float) - cur
            if np.linalg.norm(err) < cfg.tol:
                return q, True
            mujoco.mj_jacSite(m._model if hasattr(m, "_model") else m,
                              d._data if hasattr(d, "_data") else d,
                              jacp, None, self.grip_site_id)
            J = jacp[:, self._dof_idx]                     # 3 x 7
            JJt = J @ J.T + (cfg.damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)
            n = np.linalg.norm(dq)
            if n > cfg.max_step:
                dq *= cfg.max_step / n
            q = np.clip(q + dq, self.jnt_low, self.jnt_high)
        self.set_q(q)
        final_err = np.linalg.norm(np.asarray(target_xyz, float) - self.eef_pos())
        return q, bool(final_err < cfg.tol)

    def close(self):
        self.env.close()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ik = RobosuiteIK()
    print("grip site:", ik.grip_site_name)
    print("joint limits low:", ik.jnt_low)
    print("joint limits high:", ik.jnt_high)

    start_eef = ik.eef_pos()
    print("start EEF pos:", start_eef)
    print("cube pos:", ik.cube_pos())

    n_ok = 0
    n_try = 15
    for i in range(n_try):
        delta = rng.uniform(-0.05, 0.05, size=3)
        target = start_eef + delta
        q, ok = ik.solve_ik(target)
        achieved = ik.eef_pos()
        err = np.linalg.norm(target - achieved)
        n_ok += int(ok)
        print(f"[{i}] target_delta={delta.round(3)} ok={ok} "
              f"final_err={err:.5f}")
        ik.set_q(q)  # leave arm at the achieved config for the next attempt

    print(f"\nconverged: {n_ok}/{n_try}")
    ik.close()
