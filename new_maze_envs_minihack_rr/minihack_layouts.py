"""Extract raw grid layouts (MazeWalk / Corridor) from MiniHack.

This is the ONLY file that imports minihack / nle.  Geometry only: no monsters, door
mechanics, keys or items are transferred.  Output: layouts/raw/<name>.json plus native
renders in layouts/renders/minihack/.
"""
from __future__ import annotations

import argparse
import datetime
import json
import warnings
from collections import deque
from importlib import metadata
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym
import minihack  # noqa: F401  (registers envs)
from nle import nethack

HERE = Path(__file__).resolve().parent
EPISODE_STEP_LIMIT = 20000
SEARCH_TURNS = 25                                          # searches per dead end

# cmap indices (NetHack 3.6 defsym order; verified against `chars` at extraction time)
S_STONE, S_NDOOR, S_VODOOR, S_HODOOR, S_VCDOOR, S_HCDOOR = 0, 12, 13, 14, 15, 16
S_ROOM, S_DARKROOM, S_CORR, S_LITCORR, S_UPSTAIR, S_DNSTAIR, S_UPLADDER, S_DNLADDER = 19, 20, 21, 22, 23, 24, 25, 26
FREE_CMAP = {S_NDOOR, S_VODOOR, S_HODOOR, S_VCDOOR, S_HCDOOR, S_ROOM, S_DARKROOM, S_CORR, S_LITCORR,
             S_UPSTAIR, S_DNSTAIR, S_UPLADDER, S_DNLADDER}
DOOR_CMAP = {S_VODOOR, S_HODOOR, S_VCDOOR, S_HCDOOR}       # no diagonal moves through these
TILE = 16                                                  # pixel size of one map cell

SIZES = {"mazewalk": ["9x9", "15x15", "45x19"], "corridor": ["r2", "r3", "r5"]}


# --------------------------------------------------------------------------- #
# Environment discovery / creation
# --------------------------------------------------------------------------- #
def registered_ids():
    reg = gym.registry
    return sorted(k for k in reg if "MiniHack" in k and ("MazeWalk" in k or "Corridor" in k)
                  and "Battle" not in k)


def env_id_for(family, size, prefer_mapped=True):
    ids = set(registered_ids())
    if family == "mazewalk":
        mapped = f"MiniHack-MazeWalk-Mapped-{size}-v0"
        plain = f"MiniHack-MazeWalk-{size}-v0"
        if prefer_mapped and mapped in ids:
            return mapped, "mapped_variant"
        return plain, "frontier_explorer"
    eid = f"MiniHack-Corridor-{size.upper()}-v0"
    return eid, "frontier_explorer"


def corridor_des(n_rooms, room_w, room_h):
    """Custom Corridor level file: n rooms of fixed size joined by random corridors.

    Same as MiniHack's corridor{2,3,5}.des except that rooms have an explicit (w, h) instead of
    a random size.  NetHack places rooms on a 3x3 grid of the 79x21 map, so heights above ~6
    and widths above ~20 cannot be placed.
    """
    size = f"({room_w},{room_h})"
    rooms = [f'ROOM: "ordinary" , lit, random, random, {size} {{\n  STAIR: random, up\n}}',
             f'ROOM: "ordinary" , lit, random, random, {size} {{\n  STAIR: random, down\n}}']
    rooms += [f'ROOM: "ordinary" , lit, random, random, {size} {{\n}}'] * (n_rooms - 2)
    return 'LEVEL: "mylevel"\n\n' + "\n".join(rooms) + "\n\nRANDOM_CORRIDORS\n"


def make_env(env_id, des=None):
    """Env with glyph/blstats/chars/tty_chars/pixel observations and a large step limit.

    des: optional custom des-file string; then a MiniHackNavigation is built from it directly.
    """
    kw = dict(observation_keys=("glyphs", "chars", "blstats", "tty_chars", "pixel"))
    if des is not None:
        from minihack import MiniHackNavigation
        from minihack.envs.corridor import NAVIGATE_ACTIONS
        return MiniHackNavigation(des_file=des, actions=NAVIGATE_ACTIONS,
                                  max_episode_steps=EPISODE_STEP_LIMIT, **kw)
    try:
        env = gym.make(env_id, max_episode_steps=EPISODE_STEP_LIMIT, **kw)
    except Exception:
        env = gym.make(env_id, **kw)
    u = env.unwrapped
    if hasattr(u, "_max_episode_steps"):
        u._max_episode_steps = EPISODE_STEP_LIMIT
    return env


def seed_env(env, seed):
    """Same seed -> same level. reseed=False disables NLE's time-based reseeding."""
    env.unwrapped.seed(core=seed, disp=seed, reseed=False, lgen=seed)


def _reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step(env, a):
    out = env.step(a)
    if len(out) == 5:
        obs, r, term, trunc, info = out
        return obs, bool(term or trunc)
    obs, r, done, info = out
    return obs, bool(done)


# --------------------------------------------------------------------------- #
# Glyph classification
# --------------------------------------------------------------------------- #
def cmap_of(glyphs):
    """Per-cell cmap index, -1 where the glyph is not a map-symbol glyph."""
    out = -np.ones(glyphs.shape, dtype=np.int64)
    for g in np.unique(glyphs):
        if nethack.glyph_is_cmap(int(g)):
            out[glyphs == g] = int(g) - nethack.GLYPH_CMAP_OFF
    return out


def object_mask(glyphs):
    """Cells showing an object glyph (in these families: boulders lying on corridor floor)."""
    out = np.zeros(glyphs.shape, bool)
    for g in np.unique(glyphs):
        if nethack.glyph_is_object(int(g)):
            out[glyphs == g] = True
    return out


def free_mask(glyphs, agent_yx):
    """Free cells.  Deviation from "everything else is blocked": object glyphs (boulders) hide the
    floor under them and are pushed aside when walked into, so they count as free floor."""
    cm = cmap_of(glyphs)
    free = np.isin(cm, list(FREE_CMAP)) | object_mask(glyphs)
    free[agent_yx] = True
    return free, cm


def agent_pos(obs):
    bl = obs["blstats"]
    return int(bl[1]), int(bl[0])          # (row y, col x) in glyph coordinates


def sanity_check_glyph_mapping(obs):
    """Cross-check cmap indices against `chars` (open doors and walls share chars)."""
    cm = cmap_of(obs["glyphs"])
    ch = obs["chars"]
    for idx, want in ((S_DNSTAIR, ord(">")), (S_ROOM, ord(".")), (S_STONE, ord(" "))):
        sel = ch[cm == idx]
        assert len(sel) == 0 or (sel == want).all(), f"cmap {idx} does not map to {chr(want)!r}"


# --------------------------------------------------------------------------- #
# Frontier explorer
# --------------------------------------------------------------------------- #
DIRS = {(-1, 0): "N", (0, 1): "E", (1, 0): "S", (0, -1): "W",
        (-1, 1): "NE", (1, 1): "SE", (1, -1): "SW", (-1, -1): "NW"}


def explore(env, obs, log=print):
    """Walk the whole level in the discrete game.  Returns (obs, known_free, goal, steps, notes).

    A known free cell must be visited unless its eight neighbours are all displayed
    (non-stone): walking onto a cell reveals its neighbours, and true rock stays blank,
    so "unseen" can only be told apart from rock by having stood next to it.
    The goal (down stairs) is never entered because reaching it ends the episode.
    """
    actions = env.unwrapped.actions
    a_idx = {d: actions.index(getattr(nethack.CompassDirection, n)) for d, n in DIRS.items()}
    H, W = obs["glyphs"].shape
    known = np.zeros((H, W), bool)
    visited = np.zeros((H, W), bool)
    stuck = np.zeros((H, W), bool)
    ctype = -np.ones((H, W), dtype=np.int64)     # last seen terrain per cell (the agent hides its own)
    searched = np.zeros((H, W), bool)
    a_search = actions.index(nethack.Command.SEARCH)
    goal = None
    goals = set()
    steps = 0
    done = False

    def observe(o):
        nonlocal goal
        pos = agent_pos(o)
        fm, cm = free_mask(o["glyphs"], pos)
        known[:] |= fm
        ctype[cm >= 0] = cm[cm >= 0]
        visited[pos] = True
        ys, xs = np.nonzero(cm == S_DNSTAIR)
        if len(ys):
            goal = (int(ys[0]), int(xs[0]))
            goals.update(zip(ys.tolist(), xs.tolist()))
        return pos, cm

    pos, cm = observe(obs)
    while not done and steps < EPISODE_STEP_LIMIT:
        stone = np.pad(cm == S_STONE, 1, constant_values=False)
        near_stone = np.zeros((H, W), bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    near_stone |= stone[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        walk = known & ~stuck
        if goal:
            walk[goal] = False
        targets = walk & ~visited & near_stone
        searching = False
        if not targets.any():
            # Nothing left to walk to: search dead-end corridors for secret corridor squares/doors.
            fpad = np.pad(known, 1)
            nfree = sum(fpad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W].astype(int)
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
            targets = walk & visited & np.isin(ctype, (S_CORR, S_LITCORR)) & (nfree <= 1) & ~searched
            searching = True
            if not targets.any():
                break
        # BFS over walkable cells to the nearest target
        par = {pos: None}
        dq = deque([pos])
        hit = None
        while dq:
            cur = dq.popleft()
            if targets[cur] and (cur != pos or searching):
                hit = cur
                break
            for (dy, dx) in DIRS:
                nxt = (cur[0] + dy, cur[1] + dx)
                if not (0 <= nxt[0] < H and 0 <= nxt[1] < W) or nxt in par or not walk[nxt]:
                    continue
                if dy and dx and (ctype[cur] in DOOR_CMAP or ctype[nxt] in DOOR_CMAP):
                    continue
                par[nxt] = cur
                dq.append(nxt)
        if hit is None:
            break
        if searching:
            searched[hit] = True
        path = []
        c = hit
        while par[c] is not None:
            path.append(c)
            c = par[c]
        for nxt in reversed(path):
            d = (nxt[0] - pos[0], nxt[1] - pos[1])
            for attempt in range(8):        # closed doors take a turn or several to open
                obs, done = _step(env, a_idx[d])
                steps += 1
                new = agent_pos(obs)
                if new != pos or done:
                    break
            pos, cm = observe(obs)
            if new == nxt or done:
                continue
            stuck[nxt] = True               # blocked (e.g. locked door): plan around it
            break
        if searching and not done:
            before = int(known.sum())
            for _ in range(SEARCH_TURNS):   # each search finds an adjacent hidden square w.p. ~1/7
                obs, done = _step(env, a_search)
                steps += 1
                pos, cm = observe(obs)
                if done or int(known.sum()) > before:
                    break
    notes = {"stuck_cells": int(stuck.sum()),
             "stuck_at": [[int(y), int(x), int(cm[y, x])] for y, x in zip(*np.nonzero(stuck))],
             "object_cells_treated_free": int(object_mask(obs["glyphs"]).sum()), "terminated_early": bool(done),
             "unvisited_frontier": int((known & ~stuck & ~visited & near_stone).sum())}
    notes["goals_seen"] = len(goals)
    return obs, known, goal, steps, notes


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract(env_id, seed, method_hint, want_images=True, des=None):
    """One (env, seed) -> dict with the full-level grid, start, goal, images."""
    env = make_env(env_id, des)
    seed_env(env, seed)
    obs = _reset(env)
    sanity_check_glyph_mapping(obs)
    start = agent_pos(obs)
    steps, notes = 0, {}
    if method_hint == "mapped_variant":
        free, cm = free_mask(obs["glyphs"], start)
        ys, xs = np.nonzero(cm == S_DNSTAIR)
        goal = (int(ys[0]), int(xs[0])) if len(ys) else None
        n_goal = len(ys)
    else:
        obs, free, goal, steps, notes = explore(env, obs)
        n_goal = notes["goals_seen"]
    assert goal is not None and n_goal <= 1, f"expected exactly one down staircase, got {n_goal}"
    free[goal] = True
    assert free[start], "start must be free"
    ys, xs = np.nonzero(free)
    r0, r1 = max(ys.min() - 1, 0), min(ys.max() + 1, free.shape[0] - 1)
    c0, c1 = max(xs.min() - 1, 0), min(xs.max() + 1, free.shape[1] - 1)
    crop = free[r0:r1 + 1, c0:c1 + 1]
    crop = np.pad(crop, ((int(ys.min() - 1 < 0), int(ys.max() + 1 > free.shape[0] - 1)),
                         (int(xs.min() - 1 < 0), int(xs.max() + 1 > free.shape[1] - 1))))
    shift = (int(ys.min() - 1 < 0), int(xs.min() - 1 < 0))
    out = {
        "grid": ["".join("." if v else "X" for v in row) for row in crop],
        "start": [int(start[0] - r0 + shift[0]), int(start[1] - c0 + shift[1])],
        "goal": [int(goal[0] - r0 + shift[0]), int(goal[1] - c0 + shift[1])],
        "explorer_steps": steps, "notes": notes,
    }
    if want_images:
        out["pixel"] = obs["pixel"][r0 * TILE:(r1 + 1) * TILE, c0 * TILE:(c1 + 1) * TILE]
        out["text"] = "\n".join("".join(chr(c) for c in row[c0:c1 + 1]).rstrip()
                                for row in obs["tty_chars"][r0 + 1:r1 + 2])
    env.close()
    return out


def _versions():
    v = {}
    for pkg in ("nle", "minihack", "gymnasium", "gym"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return v


def fixed_corners(grid):
    """MazeWalk: start = top-left lattice node, goal = bottom-right lattice node.

    The cropped grid has a one-cell blocked border and lattice nodes at odd coordinates, so
    these are (1, 1) and (n_rows-2, n_cols-2); both must be free.
    """
    n, m = len(grid), len(grid[0])
    s, g = [1, 1], [n - 2, m - 2]
    assert grid[s[0]][s[1]] == "." and grid[g[0]][g[1]] == ".", "corner nodes must be free"
    return s, g


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", choices=["mazewalk", "corridor"], required=True)
    ap.add_argument("--size", required=True, help="9x9 | 15x15 | 45x19 | r2 | r3 | r5")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--out", default=str(HERE / "layouts"))
    ap.add_argument("--room-size", default=None, metavar="WxH",
                    help="corridor only: custom levels with fixed-size rooms, e.g. 12x5 (size becomes r5_12x5)")
    ap.add_argument("--random-endpoints", action="store_true",
                    help="keep MiniHack's random start/goal for MazeWalk (default: fixed opposite corners)")
    ap.add_argument("--no-mapped", action="store_true", help="force the frontier explorer for MazeWalk")
    a = ap.parse_args(argv)
    out = Path(a.out)
    print("registered MazeWalk/Corridor env ids:", registered_ids())
    env_id, method = env_id_for(a.family, a.size, not a.no_mapped)
    size = a.size.lower()
    des = None
    if a.room_size:
        assert a.family == "corridor", "--room-size only applies to the corridor family"
        rw, rh = (int(v) for v in a.room_size.lower().split("x"))
        n_rooms = int(size[1:])
        des = corridor_des(n_rooms, rw, rh)
        env_id = f"custom-corridor-{n_rooms}rooms-{rw}x{rh}"
        size = f"{size}_{rw}x{rh}"
    for seed in parse_seeds(a.seeds):
        r1 = extract(env_id, seed, method, des=des)
        r2 = extract(env_id, seed, method, want_images=False, des=des)
        assert (r1["grid"], r1["start"], r1["goal"]) == (r2["grid"], r2["start"], r2["goal"]), \
            f"non-deterministic level for {env_id} seed {seed}"
        name = f"{a.family}_{size}_s{seed}"
        native = {"native_start": r1["start"], "native_goal": r1["goal"]}
        if a.family == "mazewalk" and not a.random_endpoints:
            r1["start"], r1["goal"] = fixed_corners(r1["grid"])
            endpoints = "fixed_corners"
        else:
            endpoints = "minihack_random"
        rec = {"name": name, "family": a.family, "size": size, "seed": seed, "env_id": env_id,
               "grid": r1["grid"], "start": r1["start"], "goal": r1["goal"],
               "room_size": a.room_size, "endpoints": endpoints, **native, "reveal_method": method, "explorer_steps": r1["explorer_steps"], "explorer_notes": r1["notes"],
               "deterministic_check": True, "versions": _versions(),
               "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
        (out / "raw").mkdir(parents=True, exist_ok=True)
        (out / "raw" / f"{name}.json").write_text(json.dumps(rec))
        nat = out / "renders" / "minihack"
        nat.mkdir(parents=True, exist_ok=True)
        (nat / f"{name}.txt").write_text(r1["text"] + "\n")
        import matplotlib.image as mpimg
        mpimg.imsave(nat / f"{name}.png", r1["pixel"])
        g = r1["grid"]
        print(f"{name}: env={env_id} method={method} grid={len(g)}x{len(g[0])} "
              f"free={sum(r.count('.') for r in g)} steps={r1['explorer_steps']} notes={r1['notes']} deterministic=OK")


if __name__ == "__main__":
    main()
