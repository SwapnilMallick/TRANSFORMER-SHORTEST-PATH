"""Optional: compare the PyBullet checker with VAMP on random motions per scene. Skipped when VAMP is not
importable (run with ../mbm_vamp_render/.venv/bin/python -m pytest tests/test_vamp_compare.py -s to enable).

PyBullet checks true meshes against true primitives; VAMP checks robot SPHERES against capsules (cuboids in
`box`), so VAMP is expected to be the more conservative one: 'VAMP-invalid, PyBullet-valid' is common, the
reverse should be rare."""
import sys

import numpy as np
import pytest

vamp = pytest.importorskip("vamp")

import collision_mbm as C
import scene_mbm as S
from conftest import SCENES, ROOT, scene_json

sys.path.insert(0, str(ROOT.parent / "mbm_vamp_render"))
import load_mbm as L  # noqa: E402

N_MOTIONS = 5000


def _vamp_env(sc):
    p = L.Problem("panda", sc.scene, sc.index, sc.index + 1, sc.joint_names, sc.start, sc.goals,
                  sc.obstacles, sc.extra)
    return L.to_vamp_env(p)


@pytest.mark.parametrize("scene", SCENES)
def test_motion_verdicts_vs_vamp(scene):
    sc = S.load_scene(scene_json(scene, 0))
    env = _vamp_env(sc)
    ck = C.PyBulletChecker(sc, motion_res=0.02)
    rng = np.random.default_rng(0)
    lo, hi = ck.q_low, ck.q_high
    both_free = pb_only = vamp_only = both_blocked = 0
    n_state_dis = 0
    for _ in range(N_MOTIONS):
        q0 = rng.uniform(lo, hi)
        d = rng.normal(size=7)
        q1 = q0 + d / np.linalg.norm(d) * 0.15 * rng.random() ** (1 / 7)
        if not ck.in_limits(q1):
            continue
        n_state_dis += ck.valid(q0) != bool(vamp.panda.validate(q0.astype(np.float32), env))
        if not (ck.valid(q0) and vamp.panda.validate(q0.astype(np.float32), env)):
            continue                                              # compare motions from a common valid start
        a = ck.valid_motion(q0, q1)
        b = bool(vamp.panda.validate_motion(q0.astype(np.float32), q1.astype(np.float32), env))
        both_free += a and b
        both_blocked += (not a) and (not b)
        pb_only += a and not b
        vamp_only += b and not a
    n = both_free + both_blocked + pb_only + vamp_only
    print(f"{scene}: {n} motions from common-valid starts | agree {both_free + both_blocked} "
          f"(free {both_free}, blocked {both_blocked}) | PyBullet-free & VAMP-blocked {pb_only} "
          f"| VAMP-free & PyBullet-blocked {vamp_only} | disagreement {(pb_only + vamp_only) / n:.3%} "
          f"| random-q state disagreement {n_state_dis / N_MOTIONS:.3%}")
    assert n > 1000
    assert vamp_only / n < 0.10
