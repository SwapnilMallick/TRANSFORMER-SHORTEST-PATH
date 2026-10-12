import numpy as np
import pytest

import render_mbm as RM


@pytest.fixture(scope="module")
def rn(tp0):
    return RM.PyBulletRenderer(tp0)


def test_camera_names(rn):
    assert rn.camera_names() == ["front", "side", "top", "iso", "wrist"]
    with pytest.raises(ValueError):
        rn.render("nope", 32, 32)


def test_deterministic_pixels_for_equal_q(rn, tp0):
    for cam in rn.camera_names():
        rn.set_q(tp0.start)
        a = rn.render(cam, 112, 112)
        rn.set_q(tp0.goals[0])
        rn.render(cam, 112, 112)                                  # move away and come back
        rn.set_q(tp0.start)
        b = rn.render(cam, 112, 112)
        assert np.array_equal(a, b), cam


def test_different_q_gives_different_pixels(rn, tp0):
    for cam in rn.camera_names():
        rn.set_q(tp0.start)
        a = rn.render(cam, 112, 112)
        q = tp0.start.copy()
        q[0] += 0.4
        q[3] += 0.3
        rn.set_q(q)
        b = rn.render(cam, 112, 112)
        assert not np.array_equal(a, b), cam


def test_all_cameras_non_empty(rn, tp0):
    rn.set_q(tp0.start)
    for cam in rn.camera_names():
        im = rn.render(cam, 112, 112)
        assert im.shape == (112, 112, 3) and im.dtype == np.uint8
        assert (im < 250).any(axis=-1).mean() > 0.02, f"{cam} is (nearly) blank"
        assert im.std() > 5, cam


def test_wrist_camera_follows_the_hand(rn, tp0):
    rn.set_q(tp0.start)
    a = rn.render("wrist", 64, 64)
    q = tp0.start.copy()
    q[6] += 1.0                                                   # rotate the hand about its axis
    rn.set_q(q)
    assert not np.array_equal(a, rn.render("wrist", 64, 64))
