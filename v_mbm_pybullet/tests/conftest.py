import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scene_mbm as S  # noqa: E402

SCENES = ["table_pick", "table_under_pick", "box", "bookshelf_small", "bookshelf_tall", "bookshelf_thin", "cage"]


def scene_json(scene, idx):
    return S.scene_path(S.DEFAULT_SCENES_DIR, scene, idx)


@pytest.fixture(scope="session")
def tp0():
    return S.load_scene(scene_json("table_pick", 0))


def empty_scene(base: S.Scene, obstacles=(), start=None, goals=None, name="synthetic"):
    return S.Scene(name, 0, base.joint_names, base.start if start is None else start,
                   base.goals if goals is None else goals, list(obstacles), {})
