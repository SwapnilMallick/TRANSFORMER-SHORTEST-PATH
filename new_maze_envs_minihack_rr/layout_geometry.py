"""Grid layouts -> continuous wall segments (no minihack/nle imports).

Coordinate convention (Task 2 relies on it): the domain is [0, W] x [0, H] with the
longer side equal to L (1.0 unless a per-size domain_L is configured).  Grid row 0 is the TOP row.  A cell (row, col) of an
n_rows x n_cols grid with cell size c = L / max(n_rows, n_cols) covers
    x in [col*c, (col+1)*c],   y in [(n_rows-1-row)*c, (n_rows-row)*c]
(y points up).  Cell centres are used for start and goal.

Two widening methods
    thinwall (A): MazeWalk lattice decoded to a logical maze; zero-thickness walls.
    dilate   (B): upsample by k, grow free space by m sub-cells, boundary segments.
"""
from __future__ import annotations

import argparse
import datetime
import heapq
import json
import platform
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # optional
    from scipy import ndimage as _ndi
except ImportError:  # pragma: no cover
    _ndi = None

HERE = Path(__file__).resolve().parent
DEFAULT_ENDPOINTS = HERE / "mazewalk_endpoints.json"
NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGH8 = NEIGH4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #
def parse_grid(rows) -> np.ndarray:
    return np.array([[ch == "." for ch in r] for r in rows], dtype=bool)


def grid_to_rows(free: np.ndarray) -> list:
    return ["".join("." if v else "X" for v in row) for row in free]


def grid_graph(free: np.ndarray):
    """4-connected graph of free cells -> (adj lists, idx array with -1 for blocked)."""
    idx = -np.ones(free.shape, dtype=np.int64)
    idx[free] = np.arange(int(free.sum()))
    adj = [[] for _ in range(int(free.sum()))]
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        ok = (a >= 0) & (b >= 0)
        for u, v in zip(a[ok].tolist(), b[ok].tolist()):
            adj[u].append(v)
            adj[v].append(u)
    return adj, idx


def bfs(adj, src: int) -> np.ndarray:
    dist = [-1] * len(adj)
    dist[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                dq.append(v)
    return np.array(dist)


def graph_components(adj):
    lab = -np.ones(len(adj), dtype=np.int64)
    n = 0
    for s in range(len(adj)):
        if lab[s] >= 0:
            continue
        lab[s] = n
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if lab[v] < 0:
                    lab[v] = n
                    dq.append(v)
        n += 1
    return lab, n


def label_grid(free: np.ndarray, connectivity: int = 4):
    """Component labels of free cells (-1 for blocked)."""
    nb = NEIGH4 if connectivity == 4 else NEIGH8
    lab = -np.ones(free.shape, dtype=np.int64)
    n = 0
    for r, c in zip(*np.nonzero(free)):
        if lab[r, c] >= 0:
            continue
        lab[r, c] = n
        dq = deque([(r, c)])
        while dq:
            a, b = dq.popleft()
            for dr, dc in nb:
                x, y = a + dr, b + dc
                if 0 <= x < free.shape[0] and 0 <= y < free.shape[1] and free[x, y] and lab[x, y] < 0:
                    lab[x, y] = n
                    dq.append((x, y))
        n += 1
    return lab, n


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Do label arrays a, b (equal length) induce the same partition?"""
    fw, bw = {}, {}
    for x, y in zip(a.tolist(), b.tolist()):
        if fw.setdefault(x, y) != y or bw.setdefault(y, x) != x:
            return False
    return True


# --------------------------------------------------------------------------- #
# 2a. Diagonal fix
# --------------------------------------------------------------------------- #
def diagonal_fix(free: np.ndarray):
    """Open one cell of every 2x2 block whose free cells are on one diagonal only.

    Candidate choice: the blocked cell with the fewest free 4-neighbours; ties go to
    the first in row-major order.  Repeats until no such block remains.  Asserts that
    4-connectivity of the result equals 8-connectivity of the input.
    Returns (fixed grid, [(row, col), ...]).
    """
    f = free.copy()
    n, m = f.shape
    fixes = []

    def nfree(r, c):
        return sum(0 <= r + dr < n and 0 <= c + dc < m and f[r + dr, c + dc] for dr, dc in NEIGH4)

    changed = True
    while changed:
        changed = False
        for r in range(n - 1):
            for c in range(m - 1):
                a, b, cc, d = f[r, c], f[r, c + 1], f[r + 1, c], f[r + 1, c + 1]
                if a and d and not b and not cc:
                    cands = [(r, c + 1), (r + 1, c)]
                elif b and cc and not a and not d:
                    cands = [(r, c), (r + 1, c + 1)]
                else:
                    continue
                pick = min(cands, key=lambda p: (nfree(*p), p))
                f[pick] = True
                fixes.append(pick)
                changed = True
    lab8, _ = label_grid(free, 8)
    lab4, _ = label_grid(f, 4)
    cells = np.nonzero(free)
    if not _same_partition(lab8[cells], lab4[cells]):
        raise AssertionError("diagonal fix: 4-connectivity after != 8-connectivity before")
    return f, [(int(r), int(c)) for r, c in fixes]


# --------------------------------------------------------------------------- #
# Segment utilities
# --------------------------------------------------------------------------- #
def boundary_unit_segments(free: np.ndarray) -> set:
    """Unit edges between free and blocked cells (outside counts as blocked).

    Vertex coordinates: x = column boundary, y = n_rows - row boundary (y up).
    """
    n, m = free.shape
    P = np.pad(free, 1)
    segs = set()
    hi, hj = np.nonzero(P[:-1, :] != P[1:, :])       # boundary between P[i] and P[i+1]
    for i, j in zip(hi.tolist(), hj.tolist()):
        segs.add(((j - 1, n - i), (j, n - i)))
    vi, vj = np.nonzero(P[:, :-1] != P[:, 1:])       # boundary between P[:,j] and P[:,j+1]
    for i, j in zip(vi.tolist(), vj.tolist()):
        segs.add(((j, n - i), (j, n - i + 1)))
    return segs


def merge_unit_segments(segs) -> list:
    """Merge touching collinear axis-aligned unit segments into maximal runs."""
    horiz, vert = {}, {}
    for (x0, y0), (x1, y1) in segs:
        if y0 == y1:
            horiz.setdefault(y0, []).append((min(x0, x1), max(x0, x1)))
        else:
            vert.setdefault(x0, []).append((min(y0, y1), max(y0, y1)))
    out = []
    for key, runs, is_h in [(k, v, True) for k, v in horiz.items()] + [(k, v, False) for k, v in vert.items()]:
        runs.sort()
        lo, hi = runs[0]
        for a, b in runs[1:]:
            if a <= hi:
                hi = max(hi, b)
            else:
                out.append((lo, key, hi, key) if is_h else (key, lo, key, hi))
                lo, hi = a, b
        out.append((lo, key, hi, key) if is_h else (key, lo, key, hi))
    return sorted(out)


def expand_to_unit_segments(merged) -> set:
    """Inverse of merge_unit_segments (for tests)."""
    s = set()
    for x0, y0, x1, y1 in merged:
        if y0 == y1:
            for x in range(int(min(x0, x1)), int(max(x0, x1))):
                s.add(((x, y0), (x + 1, y0)))
        else:
            for y in range(int(min(y0, y1)), int(max(y0, y1))):
                s.add(((x0, y), (x0, y + 1)))
    return s


# --------------------------------------------------------------------------- #
# 2d. Collision
# --------------------------------------------------------------------------- #
def _ccw(ax, ay, bx, by, cx, cy):
    # identical arithmetic to online_cost_transformer._ccw(a, b, c)
    return (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)


def _pt_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.maximum(L2, 1e-300), 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _collide_matrix(P, Q, walls, clearance):
    """(M,2),(M,2),(N,4) -> (M,N) bool collision matrix."""
    x1, y1 = P[:, 0, None], P[:, 1, None]
    x2, y2 = Q[:, 0, None], Q[:, 1, None]
    x3, y3, x4, y4 = (walls[None, :, i] for i in range(4))
    d1 = _ccw(x3, y3, x4, y4, x1, y1)
    d2 = _ccw(x3, y3, x4, y4, x2, y2)
    d3 = _ccw(x1, y1, x2, y2, x3, y3)
    d4 = _ccw(x1, y1, x2, y2, x4, y4)
    hit = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))       # _seg_intersect, verbatim
    if clearance <= 0:
        return hit
    cross = (d1 * d2 < 0) & (d3 * d4 < 0)
    dmin = np.minimum.reduce([
        _pt_seg_dist(x1, y1, x3, y3, x4, y4), _pt_seg_dist(x2, y2, x3, y3, x4, y4),
        _pt_seg_dist(x3, y3, x1, y1, x2, y2), _pt_seg_dist(x4, y4, x1, y1, x2, y2)])
    return hit | cross | (dmin < clearance)


def segment_collides(p, q, walls, clearance: float = 0.0) -> bool:
    """Does the motion segment p->q collide with any wall?

    walls: (N, 4) array of x1, y1, x2, y2.
    clearance == 0: exactly online_cost_transformer._seg_intersect against every wall,
        i.e. ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) with
        d1,d2 = ccw(wall, p), ccw(wall, q); d3,d4 = ccw(p q, wall ends), and
        ccw(a,b,c) = (c.y-a.y)*(b.x-a.x) - (b.y-a.y)*(c.x-a.x).  A zero orientation
        value counts as "not > 0" (same as the original), so grazing/collinear contacts
        follow the original's convention rather than being "strictly proper".
    clearance > 0: collision if the test above hits, the segments properly cross, or the
        segment-to-segment (Euclidean) distance to any wall is strictly below clearance.
        Passing exactly at distance == clearance is allowed.
    """
    walls = np.asarray(walls, dtype=float).reshape(-1, 4)
    if len(walls) == 0:
        return False
    P = np.asarray(p, dtype=float).reshape(1, 2)
    Q = np.asarray(q, dtype=float).reshape(1, 2)
    return bool(_collide_matrix(P, Q, walls, float(clearance)).any())


def _segments_collide_many(P, Q, walls, clearance):
    """Vectorised over pairs, chunked. Returns (M,) bool."""
    out = np.zeros(len(P), dtype=bool)
    step = max(1, int(2_000_000 // max(1, len(walls))))
    for i in range(0, len(P), step):
        out[i:i + step] = _collide_matrix(P[i:i + step], Q[i:i + step], walls, clearance).any(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Built layouts (either method)
# --------------------------------------------------------------------------- #
@dataclass
class Built:
    method: str
    params: dict
    W: float
    H: float
    mask: np.ndarray              # free mask (widened grid or logical cells)
    mask_cell: float              # domain size of one mask cell
    walls: np.ndarray             # (N, 4)
    start: tuple
    goal: tuple
    passage_width: float
    ref_len: float
    adj: list                     # widened graph adjacency
    cell_node: dict               # original free cell -> widened graph node
    graph_scale: float            # original grid units per graph step
    extra: dict = field(default_factory=dict)


def cell_center(r, c, n_rows, cell):
    return ((c + 0.5) * cell, (n_rows - 1 - r + 0.5) * cell)


def _scale_walls(merged, s):
    return np.array(merged, dtype=float).reshape(-1, 4) * s


# ---- 2b. Method A: thin-wall redraw --------------------------------------- #
def decode_lattice(free: np.ndarray):
    """Find parity offset (r0, c0) with every free cell a lattice node or connector.

    Returns (r0, c0, nodes, edges) or None.  nodes: set of (r, c); edges: set of
    ((r, c), (r, c)) node pairs joined by a free connector cell.
    """
    n, m = free.shape
    if not free.any():
        return None
    for r0 in (0, 1):
        for c0 in (0, 1):
            nodes, edges, ok = set(), {}, True
            for r, c in zip(*np.nonzero(free)):
                r, c = int(r), int(c)
                rn, cn = (r - r0) % 2 == 0, (c - c0) % 2 == 0
                if rn and cn:
                    nodes.add((r, c))
                elif rn or cn:
                    a, b = ((r, c - 1), (r, c + 1)) if rn else ((r - 1, c), (r + 1, c))
                    if not all(0 <= x < n and 0 <= y < m and free[x, y] for x, y in (a, b)):
                        ok = False
                        break
                    edges[(r, c)] = (a, b)
                else:
                    ok = False
                    break
            if ok:
                return r0, c0, nodes, edges
    return None


def logical_index(r0, c0, nodes):
    """Raw lattice node (r, c) -> logical (row, col), row 0 = top; plus logical (n_rows, n_cols)."""
    ij = {p: ((p[0] - r0) // 2, (p[1] - c0) // 2) for p in nodes}
    i0 = min(v[0] for v in ij.values())
    j0 = min(v[1] for v in ij.values())
    ij = {p: (a - i0, b - j0) for p, (a, b) in ij.items()}
    return ij, max(v[0] for v in ij.values()) + 1, max(v[1] for v in ij.values()) + 1


def apply_logical_endpoints(fixed, spec):
    """Config endpoints given as logical maze cells -> raw grid cells (row, col).

    spec = {"start": [row, col], "goal": [row, col]} in logical coordinates (row 0 = top,
    col 0 = left).  Raises ValueError with the maze size if a cell is not part of the maze.
    """
    dec = decode_lattice(fixed)
    if dec is None:
        raise ValueError("cannot apply logical endpoints: lattice decode failed")
    ij, nr, nc = logical_index(*dec[:3])
    inv = {v: k for k, v in ij.items()}
    out = []
    for key in ("start", "goal"):
        cell = tuple(spec[key])
        if cell not in inv:
            raise ValueError(f"{key} {list(cell)} is not a cell of the {nr}x{nc} logical maze "
                             f"(rows 0..{nr - 1}, cols 0..{nc - 1})")
        out.append(inv[cell])
    if out[0] == out[1]:
        raise ValueError("start and goal must differ")
    return out[0], out[1]


def build_thinwall(fixed: np.ndarray, start_cell, goal_cell, L: float = 1.0):
    """Method A. Returns Built, or None if the lattice decode fails."""
    dec = decode_lattice(fixed)
    if dec is None:
        return None
    r0, c0, nodes, connectors = dec
    ij, nr, nc = logical_index(r0, c0, nodes)
    cell = L / max(nr, nc)
    mask = np.zeros((nr, nc), dtype=bool)
    for a, b in ij.values():
        mask[a, b] = True
    edges = sorted({tuple(sorted((ij[a], ij[b]))) for a, b in connectors.values()})
    edge_set = set(edges)

    def has_edge(u, v):
        return tuple(sorted((u, v))) in edge_set

    # walls on every boundary without an edge (outer boundary included)
    segs = set()
    for i, j in zip(*np.nonzero(mask)):
        i, j = int(i), int(j)
        y_lo, y_hi = nr - 1 - i, nr - i
        for (di, dj), seg in (((-1, 0), ((j, y_hi), (j + 1, y_hi))), ((1, 0), ((j, y_lo), (j + 1, y_lo))),
                              ((0, -1), ((j, y_lo), (j, y_hi))), ((0, 1), ((j + 1, y_lo), (j + 1, y_hi)))):
            if not has_edge((i, j), (i + di, j + dj)):
                segs.add(seg)
    walls = _scale_walls(merge_unit_segments(segs), cell)

    # logical graph
    lnodes = sorted(ij.values())
    nid = {v: k for k, v in enumerate(lnodes)}
    adj = [[] for _ in lnodes]
    for u, v in edges:
        adj[nid[u]].append(nid[v])
        adj[nid[v]].append(nid[u])

    def to_node(cell_rc):
        if cell_rc in ij:
            return ij[cell_rc], False
        a, _ = connectors[cell_rc]
        return ij[a], True

    cell_node = {}
    for r, c in zip(*np.nonzero(fixed)):
        cell_node[(int(r), int(c))] = nid[to_node((int(r), int(c)))[0]]
    s_node, s_snap = to_node(tuple(start_cell))
    g_node, g_snap = to_node(tuple(goal_cell))
    steps = int(bfs(adj, nid[s_node])[nid[g_node]])
    return Built(
        method="thinwall", params={"k": None, "m": None},
        W=nc * cell, H=nr * cell, mask=mask, mask_cell=cell, walls=walls,
        start=cell_center(*s_node, nr, cell), goal=cell_center(*g_node, nr, cell),
        passage_width=cell, ref_len=steps * cell if steps >= 0 else float("nan"),
        adj=adj, cell_node=cell_node, graph_scale=2.0,
        extra={"logical_edges": [[a[0], a[1], b[0], b[1]] for a, b in edges],
               "lattice_offset": [r0, c0], "snapped": bool(s_snap or g_snap)})


# ---- 2c. Method B: upsample + grow ---------------------------------------- #
def dilate_mask(free: np.ndarray, m: int) -> np.ndarray:
    if m <= 0:
        return free.copy()
    if _ndi is not None:
        return _ndi.binary_dilation(free, structure=np.ones((2 * m + 1, 2 * m + 1), bool))
    n, w = free.shape
    P = np.pad(free, m)
    out = np.zeros_like(free)
    for dr in range(2 * m + 1):
        for dc in range(2 * m + 1):
            out |= P[dr:dr + n, dc:dc + w]
    return out


def build_dilate(fixed: np.ndarray, start_cell, goal_cell, k: int, m: int, L: float = 1.0) -> Built:
    n, w = fixed.shape
    c = L / max(n, w)
    sub = np.kron(fixed, np.ones((k, k), dtype=bool)).astype(bool)
    grown = dilate_mask(sub, m)
    s = c / k
    walls = _scale_walls(merge_unit_segments(boundary_unit_segments(grown)), s)
    adj, idx = grid_graph(grown)
    cell_node = {(int(r), int(cc)): int(idx[r * k + k // 2, cc * k + k // 2])
                 for r, cc in zip(*np.nonzero(fixed))}
    oadj, oidx = grid_graph(fixed)
    steps = int(bfs(oadj, int(oidx[tuple(start_cell)]))[int(oidx[tuple(goal_cell)])])
    return Built(
        method="dilate", params={"k": k, "m": m}, W=w * c, H=n * c,
        mask=grown, mask_cell=s, walls=walls,
        start=cell_center(*start_cell, n, c), goal=cell_center(*goal_cell, n, c),
        passage_width=(k + 2 * m) * s, ref_len=steps * c if steps >= 0 else float("nan"),
        adj=adj, cell_node=cell_node, graph_scale=1.0 / k, extra={})


# --------------------------------------------------------------------------- #
# 2e. Topology checks
# --------------------------------------------------------------------------- #
def check_components(fixed, b: Built, start_cell, goal_cell) -> bool:
    lab_o, n_o = label_grid(fixed, 4)
    lab_w, n_w = graph_components(b.adj)
    cells = sorted(b.cell_node)
    lo = np.array([lab_o[c] for c in cells])
    lw = np.array([lab_w[b.cell_node[c]] for c in cells])
    if n_o != n_w or not _same_partition(lo, lw):
        return False
    return bool(lab_w[b.cell_node[tuple(start_cell)]] == lab_w[b.cell_node[tuple(goal_cell)]])


def check_shortcuts(fixed, b: Built, seed: int = 0, max_all_pairs: int = 1500):
    """Returns (n_shortcuts, pairs_checked). Shortcut: widened < 0.5*original - 2."""
    oadj, oidx = grid_graph(fixed)
    cells = sorted(b.cell_node)
    onode = np.array([oidx[c] for c in cells])
    wnode = np.array([b.cell_node[c] for c in cells])
    rng = np.random.default_rng(seed)
    N = len(cells)
    if N <= max_all_pairs:
        jobs = [(i, np.arange(i + 1, N)) for i in range(N - 1)]
    else:  # 200 sources x 100 targets = 20,000 random pairs
        src = rng.choice(N, 200, replace=False)
        jobs = [(int(i), rng.choice(N, 100, replace=False)) for i in src]
    short = pairs = 0
    for i, tj in jobs:
        tj = tj[tj != i]
        if len(tj) == 0:
            continue
        od = bfs(oadj, int(onode[i]))[onode[tj]]
        wd = bfs(b.adj, int(wnode[i]))[wnode[tj]]
        ok = (od >= 0) & (wd >= 0)
        pairs += int(ok.sum())
        short += int((wd[ok] * b.graph_scale < 0.5 * od[ok] - 2).sum())
    return short, pairs


def sample_free(b: Built, n: int, rng) -> np.ndarray:
    rr, cc = np.nonzero(b.mask)
    pick = rng.integers(0, len(rr), n)
    u = rng.random((n, 2))
    nrows = b.mask.shape[0]
    x = (cc[pick] + u[:, 0]) * b.mask_cell
    y = (nrows - 1 - rr[pick] + u[:, 1]) * b.mask_cell
    return np.stack([x, y], axis=1)


def check_reachability(b: Built, n_points: int = 3000, seed: int = 0, max_tries: int = 3):
    """Random roadmap; returns (reachable, clearance, roadmap dict).

    Roadmap: points (start=0, goal=1), edges (E,2), path (indices start->goal or []).
    The point count doubles on failure, up to max_tries attempts.
    """
    rng = np.random.default_rng(seed)
    clearance = 0.05 * b.passage_width
    radius = 1.5 * b.passage_width
    road = None
    for attempt in range(max_tries):
        pts = np.vstack([np.array([b.start, b.goal]), sample_free(b, n_points * 2 ** attempt, rng)])
        iu_l, ju_l = [], []
        for a in range(0, len(pts), 400):
            d = np.linalg.norm(pts[a:a + 400, None, :] - pts[None, :, :], axis=2)
            ii, jj = np.nonzero(d <= radius)
            keep = jj > ii + a
            iu_l.append(ii[keep] + a)
            ju_l.append(jj[keep])
        iu, ju = np.concatenate(iu_l), np.concatenate(ju_l)
        bad = _segments_collide_many(pts[iu], pts[ju], b.walls, clearance)
        edges = np.stack([iu[~bad], ju[~bad]], axis=1)
        parent = list(range(len(pts)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in edges.tolist():
            parent[find(u)] = find(v)
        reachable = find(0) == find(1)
        path = _dijkstra(pts, edges, 0, 1) if reachable else []
        road = {"points": pts, "edges": edges, "path": np.array(path, dtype=np.int64)}
        if reachable:
            return True, clearance, road
    return False, clearance, road


def _dijkstra(pts, edges, s, t):
    adj = [[] for _ in range(len(pts))]
    for u, v in edges.tolist():
        w = float(np.linalg.norm(pts[u] - pts[v]))
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > dist.get(u, 1e300):
            continue
        for v, w in adj[u]:
            if d + w < dist.get(v, 1e300):
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (d + w, v))
    if t not in dist:
        return []
    path = [t]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return path[::-1]


def full_topology(fixed, b: Built, start_cell, goal_cell, seed=0, n_points=3000):
    comps = check_components(fixed, b, start_cell, goal_cell)
    n_short, pairs = check_shortcuts(fixed, b, seed)
    reach, clr, road = check_reachability(b, n_points, seed)
    topo = {"components_ok": comps, "n_shortcuts": n_short, "pairs_checked": pairs,
            "reachable": bool(reach), "clearance_used": clr, "m_final": b.params.get("m")}
    return topo, road


def topology_passes(t) -> bool:
    return bool(t["components_ok"] and t["n_shortcuts"] == 0 and t["reachable"])


def build_dilate_checked(fixed, start_cell, goal_cell, k=4, m=1, min_wall=1, seed=0, n_points=3000, L=1.0):
    """Method B with the m-retry rule. Returns (Built, topology, roadmap, retries)."""
    if k - 2 * m < min_wall:
        raise ValueError(f"k - 2m = {k - 2 * m} < {min_wall}")
    retries = 0
    while True:
        b = build_dilate(fixed, start_cell, goal_cell, k, m, L)
        topo, road = full_topology(fixed, b, start_cell, goal_cell, seed, n_points)
        if topology_passes(topo):
            return b, topo, road, retries
        if m == 0:
            raise ValueError(f"topology check fails even with m=0: {topo}")
        m -= 1
        retries += 1


# --------------------------------------------------------------------------- #
# Processing / JSON
# --------------------------------------------------------------------------- #
def _versions(raw):
    v = dict(raw.get("versions", {}))
    v.update({"python": platform.python_version(), "numpy": np.__version__})
    if _ndi is not None:
        import scipy
        v["scipy"] = scipy.__version__
    return v


def process_layout(raw: dict, method="auto", k=4, m=1, raw_file="", min_wall=1, seed=0, n_points=3000,
                   endpoints=None, domain_L=1.0):
    """Raw dict -> (processed dict, roadmap dict).

    endpoints: optional {layout name: {"start": [row, col], "goal": [row, col]}} in logical maze
    coordinates (MazeWalk); an entry for this layout overrides the raw start/goal.
    domain_L: length of the domain's longer side (1.0 by default); everything scales with it.
    """
    free0 = parse_grid(raw["grid"])
    start_cell, goal_cell = tuple(raw["start"]), tuple(raw["goal"])
    fixed, fixes = diagonal_fix(free0)
    ep_source = "raw"
    if endpoints and raw["name"] in endpoints:
        start_cell, goal_cell = apply_logical_endpoints(fixed, endpoints[raw["name"]])
        ep_source = "config"
    family = raw["family"]
    if method == "auto":
        method = "thinwall" if family == "mazewalk" else "dilate"
        fallback_ok = True
    else:
        fallback_ok = False
    decode = {"attempted": False, "success": None, "snapped": False}
    b = topo = road = None
    if method == "thinwall":
        decode["attempted"] = True
        b = build_thinwall(fixed, start_cell, goal_cell, domain_L)
        decode["success"] = b is not None
        if b is None:
            if not fallback_ok:
                raise ValueError("lattice decode failed for thinwall")
            method = "dilate"
        else:
            decode["snapped"] = b.extra["snapped"]
            topo, road = full_topology(fixed, b, start_cell, goal_cell, seed, n_points)
    if method == "dilate":
        b, topo, road, retries = build_dilate_checked(fixed, start_cell, goal_cell, k, m, min_wall, seed, n_points, domain_L)
        decode["retries"] = retries
    name = raw["name"]
    stem = f"{name}_{b.method}" + (f"_k{b.params['k']}m{b.params['m']}" if b.method == "dilate" else "")
    out = {
        "name": name, "family": family, "size": raw["size"], "seed": raw["seed"], "env_id": raw["env_id"],
        "method": b.method, "params": b.params,
        "domain": {"W": b.W, "H": b.H, "L": domain_L},
        "walls": b.walls.tolist(),
        "start": list(b.start), "goal": list(b.goal),
        "passage_width": b.passage_width,
        "reference_path_length": b.ref_len,
        "clearance_suggested": 0.05 * b.passage_width,
        "diagonal_fixes": [list(f) for f in fixes],
        "decode": decode,
        "topology": topo,
        "start_cell": list(start_cell), "goal_cell": list(goal_cell), "endpoints_source": ep_source,
        "raw_file": raw_file, "versions": _versions(raw),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        # extras (ignored by Task 2): shaded free space + logical maze for rendering
        "free_mask": {"cell": b.mask_cell, "grid": grid_to_rows(b.mask)},
        "logical_edges": b.extra.get("logical_edges"),
        "roadmap_file": f"{stem}_roadmap.npz",
        "stem": stem,
    }
    return out, road


def write_processed(proc: dict, road: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{proc['stem']}.json"
    p.write_text(json.dumps(proc))
    np.savez_compressed(out_dir / proc["roadmap_file"], **road)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", default=str(HERE / "layouts" / "raw"))
    ap.add_argument("--method", choices=["thinwall", "dilate", "auto"], default="auto")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "layouts" / "processed"))
    ap.add_argument("--endpoints", default=str(DEFAULT_ENDPOINTS),
                    help="JSON config of per-layout start/goal (logical cells); 'none' to disable")
    a = ap.parse_args(argv)
    endpoints, L_by_size = None, {}
    if a.endpoints.lower() != "none" and Path(a.endpoints).is_file():
        cfg = json.loads(Path(a.endpoints).read_text())
        endpoints = {k: v for k, v in cfg.items() if not k.startswith("_")}
        L_by_size = cfg.get("_domain_L", {})
    inp, out = Path(a.inp), Path(a.out)
    files = sorted(inp.glob("*.json")) if inp.is_dir() else [inp]
    rows, failed = [], []
    for f in files:
        raw = json.loads(f.read_text())
        try:
            rel = str(f.resolve().relative_to(out.resolve().parent))
        except ValueError:
            rel = str(f.resolve())
        try:
            proc, road = process_layout(raw, a.method, a.k, a.m, raw_file=rel, endpoints=endpoints,
                                       domain_L=float(L_by_size.get(f'{raw["family"]}_{raw["size"]}', 1.0)))
        except Exception as e:  # report and continue
            failed.append((f.name, repr(e)))
            print(f"FAIL {f.name}: {e!r}")
            continue
        write_processed(proc, road, out)
        t = proc["topology"]
        print(f"{proc['stem']}: method={proc['method']} decode={proc['decode']} fixes={len(proc['diagonal_fixes'])} "
              f"m_final={t['m_final']} pw={proc['passage_width']:.4f} ref={proc['reference_path_length']:.3f} "
              f"L={proc['domain']['L']} endpoints={proc['endpoints_source']} comps={t['components_ok']} shortcuts={t['n_shortcuts']}/{t['pairs_checked']} reach={t['reachable']}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
