import numpy as np
import pytest

import scene_mbm as S
from conftest import empty_scene, scene_json
from task_mbm import MBMTask, MBMTaskConfig


@pytest.fixture(scope="module")
def task():
    t = MBMTask(scene_json("table_pick", 0), MBMTaskConfig())
    yield t
    t.close()


def test_sampler_respects_limits_step_and_validity(task):
    rng = np.random.default_rng(0)
    cfg = task.cfg
    starts = [task.q_start.copy()]
    hug = task.q_start.copy()
    hug[3] = task.q_low[3] + 0.02                                 # hugging the joint-4 lower limit
    hug[0] = task.q_high[0] - 0.02
    starts.append(hug)
    n_ok = 0
    for q in starts:
        for _ in range(300):
            out = task.sample_free(q, rng)
            if out is None:
                continue
            dq, q_next = out
            n_ok += 1
            assert np.linalg.norm(dq) <= cfg.step_size + 1e-12
            np.testing.assert_allclose(q_next, q + dq)
            assert np.all(q_next >= task.q_low) and np.all(q_next <= task.q_high)
            assert task.checker.valid(q_next) and task.checker.valid_motion(q, q_next)
    assert n_ok > 300


def test_diagonal_and_scene_validation(task):
    assert task.diagonal == pytest.approx(np.linalg.norm(task.q_high - task.q_low))
    assert task.reached(task.goals[0]) and not task.reached(task.q_start)
    # a start inside an obstacle is refused
    base = S.load_scene(scene_json("table_pick", 0))
    import robot_mbm as R
    p = R.load_robot(R.new_client()).fk_eef(base.start)
    blocker = {"name": "b", "type": "box", "position": p.tolist(), "quaternion_xyzw": [0, 0, 0, 1],
               "size": {"size": [0.3, 0.3, 0.3]}}
    with pytest.raises(ValueError, match="collision"):
        MBMTask(None, MBMTaskConfig(), scene=empty_scene(base, [blocker]))
    out_of_limits = base.start.copy()
    out_of_limits[3] = 1.0
    with pytest.raises(ValueError, match="limits"):
        MBMTask(None, MBMTaskConfig(), scene=empty_scene(base, [], start=out_of_limits))


def test_eef_path_length_is_densified(task):
    q = [task.q_start, task.goals[0]]
    assert task.eef_path_length(q) >= np.linalg.norm(task.fk_eef(q[1]) - task.fk_eef(q[0])) - 1e-9
