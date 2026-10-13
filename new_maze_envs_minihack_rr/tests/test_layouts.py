import json

import numpy as np
import pytest

import layout_geometry as G
import render_layouts as R
import synthetic as S


def grid(*rows):
    return G.parse_grid(rows)


# ---- diagonal fix ---------------------------------------------------------- #
def test_diagonal_fix_opens_corner_only():
    g = grid("XXXX", "X.XX", "XX.X", "XXXX")
    f, fixes = G.diagonal_fix(g)
    assert len(fixes) == 1 and f.sum() == g.sum() + 1
    assert (f | g == f).all() and fixes[0] in [(1, 2), (2, 1)]
    assert G.label_grid(f, 4)[1] == G.label_grid(g, 8)[1] == 1


def test_diagonal_fix_leaves_clean_grid_untouched():
    g = grid("XXXXX", "X...X", "X.XXX", "XXXXX")
    f, fixes = G.diagonal_fix(g)
    assert fixes == [] and (f == g).all()


def test_diagonal_fix_connectivity_preserved_with_two_components():
    g = grid("XXXXXXX", "X.XXXXX", "XX.XXXX", "XXXXXXX", "XXXX.XX", "XXXXX.X", "XXXXXXX")
    f, fixes = G.diagonal_fix(g)
    assert len(fixes) == 2
    assert G.label_grid(g, 8)[1] == G.label_grid(f, 4)[1] == 2


# ---- Method A --------------------------------------------------------------- #
def test_thinwall_decodes_backtracker_exactly():
    g = S.backtracker_grid(5, 6, 3)
    r0, c0, nodes, conn = G.decode_lattice(g)
    assert (r0, c0) == (1, 1) and len(nodes) == 30 and len(conn) == 29    # spanning tree
    b = G.build_thinwall(g, (1, 1), (9, 11))
    assert b.mask.shape == (5, 6) and b.mask.all()
    assert len(b.extra["logical_edges"]) == 29
    assert b.passage_width == pytest.approx(1 / 6)


def test_lattice_parity_break_fails_cleanly():
    g = S.backtracker_grid(4, 4, 0)
    g[2, 2] = True                    # free cell off-lattice in both coordinates
    assert G.decode_lattice(g) is None and G.build_thinwall(g, (1, 1), (7, 7)) is None
    raw = S.raw_dict(g, "bad", "mazewalk", (1, 1), (7, 7))
    p, _ = G.process_layout(raw, "auto", n_points=300)
    assert p["decode"]["success"] is False and p["method"] == "dilate"


def test_thinwall_snaps_connector_start():
    g = S.backtracker_grid(3, 3, 0)
    conn = next(iter(G.decode_lattice(g)[3]))
    b = G.build_thinwall(g, conn, (1, 1))
    assert b.extra["snapped"]


# ---- Method B --------------------------------------------------------------- #
def two_corridors():
    return grid("XXXXXXXX", "X......X", "XXXXXXXX", "X......X", "XXXXXXXX")


def test_dilate_keeps_parallel_corridors_separate():
    g = two_corridors()
    b = G.build_dilate(g, (1, 1), (1, 6), 4, 1)
    assert G.graph_components(b.adj)[1] == 2
    assert b.passage_width == pytest.approx(6 * (1 / 8) / 4)


def test_dilate_merge_fails_topology_and_retry_lowers_m():
    g = two_corridors()
    # k=4, m=2 removes the whole 1-cell wall -> the two corridors merge
    b = G.build_dilate(g, (1, 1), (1, 6), 4, 2)
    assert G.graph_components(b.adj)[1] == 1
    assert not G.check_components(g, b, (1, 1), (1, 6))
    with pytest.raises(ValueError):
        G.build_dilate_checked(g, (1, 1), (1, 6), 4, 2)          # k - 2m < 1
    b, topo, road, retries = G.build_dilate_checked(g, (1, 1), (1, 6), 4, 2, min_wall=0, n_points=300)
    assert retries == 1 and topo["m_final"] == 1 and b.params["m"] == 1


# ---- segments --------------------------------------------------------------- #
def test_merge_covers_same_boundary():
    g = S.corridor_grid()
    unit = G.boundary_unit_segments(np.kron(g, np.ones((4, 4), bool)).astype(bool))
    merged = G.merge_unit_segments(unit)
    assert len(merged) < len(unit)
    assert G.expand_to_unit_segments(merged) == unit
    h = sorted((y0, x0, x1) for x0, y0, x1, y1 in merged if y0 == y1)
    for a, b in zip(h, h[1:]):                                   # maximal: nothing touches
        assert not (a[0] == b[0] and a[2] >= b[1])


# ---- topology --------------------------------------------------------------- #
def test_shortcut_detected():
    g = S.backtracker_grid(6, 6, 2)
    good = G.build_thinwall(g, (1, 1), (11, 11))
    assert G.check_shortcuts(g, good)[0] == 0
    nr, nc = good.mask.shape                     # remove every interior wall: open grid graph
    full = [[] for _ in range(nr * nc)]
    for i in range(nr):
        for j in range(nc):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < nr and j + dj < nc:
                    u, v = i * nc + j, (i + di) * nc + j + dj
                    full[u].append(v)
                    full[v].append(u)
    good.adj = full
    assert G.check_shortcuts(g, good)[0] > 0


# ---- collision -------------------------------------------------------------- #
def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def ref_intersect(p1, p2, p3, p4):
    d1, d2 = _ccw(p3, p4, p1), _ccw(p3, p4, p2)
    d3, d4 = _ccw(p1, p2, p3), _ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def test_collision_matches_seg_intersect():
    rng = np.random.default_rng(0)
    walls = rng.random((40, 4))
    for _ in range(300):
        p, q = rng.random(2), rng.random(2)
        assert G.segment_collides(p, q, walls) == any(ref_intersect(p, q, w[:2], w[2:]) for w in walls)
    w2 = rng.integers(0, 4, (30, 4)).astype(float)               # degenerate / collinear cases
    for _ in range(300):
        p, q = rng.integers(0, 4, 2).astype(float), rng.integers(0, 4, 2).astype(float)
        assert G.segment_collides(p, q, w2) == any(ref_intersect(p, q, w[:2], w[2:]) for w in w2)


def test_clearance_blocks_junction_and_narrow_gap():
    walls = np.array([[0, 0, 1, 0], [1, 0, 1, 1.0]])             # L-junction at (1, 0)
    assert G.segment_collides((0.9, -0.5), (1.1, 0.05), walls, 0.1)
    assert G.segment_collides((1.05, -0.5), (1.05, 0.5), walls, 0.1)   # passes 0.05 from the wall
    gap = np.array([[-1, 0, -0.05, 0], [0.05, 0, 1, 0.0]])       # gap of width 0.1
    assert not G.segment_collides((0, -1), (0, 1), gap, 0.0)
    assert not G.segment_collides((0, -1), (0, 1), gap, 0.04)
    assert G.segment_collides((0, -1), (0, 1), gap, 0.06)
    w = np.array([[0, 0, 1, 0.0]])
    assert not G.segment_collides((0, 0.1), (1, 0.1), w, 0.05)
    assert G.segment_collides((0, 0.1), (1, 0.1), w, 0.2)


# ---- coordinates ------------------------------------------------------------- #
def test_coordinate_convention():
    g = S.corridor_grid()          # 14 x 30 -> c = 1/30
    raw = S.raw_dict(g, "c", "corridor", (3, 3), (10, 24))
    p, _ = G.process_layout(raw, "dilate", n_points=300)
    c = 1 / 30
    assert p["domain"]["W"] == pytest.approx(1.0) and p["domain"]["H"] == pytest.approx(14 * c)
    # cell (row 3, col 3) covers x in [3c, 4c], y in [(14-1-3)c, (14-3)c]; start is its centre
    assert p["start"] == pytest.approx([3.5 * c, 10.5 * c])
    assert p["goal"] == pytest.approx([24.5 * c, 3.5 * c])
    mask = R.grid_array(p["free_mask"]["grid"])
    cell = p["free_mask"]["cell"]
    for key in ("start", "goal"):
        x, y = p[key]
        assert mask[int((p["domain"]["H"] - y) / cell), int(x / cell)]
    walls = np.array(p["walls"])
    assert walls[:, [0, 2]].min() >= 0 and walls[:, [0, 2]].max() <= 1 + 1e-9
    assert p["reference_path_length"] > 0
    assert p["clearance_suggested"] == pytest.approx(0.05 * p["passage_width"])


# ---- rendering ---------------------------------------------------------------- #
def test_render_produces_all_files(tmp_path):
    (tmp_path / "raw").mkdir()
    for name, fam, g, s, e in [("mazewalk_syn_s0", "mazewalk", S.backtracker_grid(4, 4, 0), (1, 1), (7, 7)),
                               ("corridor_syn_s0", "corridor", S.corridor_grid(), (3, 3), (10, 24))]:
        raw = S.raw_dict(g, name, fam, s, e)
        (tmp_path / "raw" / f"{name}.json").write_text(json.dumps(raw))
        p, road = G.process_layout(raw, "auto", raw_file=f"raw/{name}.json", n_points=400)
        G.write_processed(p, road, tmp_path / "processed")
    R.main(["--in", str(tmp_path / "processed"), "--out", str(tmp_path / "renders"), "--dpi", "50"])
    out = tmp_path / "renders"
    for stem in ("mazewalk_syn_s0_thinwall", "corridor_syn_s0_dilate_k4m1"):
        for rel in (f"grid/{stem}_raw.png", f"grid/{stem}_widened.png", f"continuous/{stem}.png",
                    f"continuous/{stem}.svg", f"continuous/{stem}_roadmap.png", f"previews/{stem}.png"):
            assert (out / rel).stat().st_size > 1000, rel
    for rel in ("gallery/mazewalk.png", "gallery/corridor.png", "gallery.md"):
        assert (out / rel).stat().st_size > 500, rel
    assert "corridor_syn_s0" in (out / "gallery.md").read_text()


# ---- endpoint config ---------------------------------------------------------- #
def test_endpoints_config_overrides_and_validates():
    g = S.backtracker_grid(4, 5, 1)
    raw = S.raw_dict(g, "mazewalk_x_s0", "mazewalk", (1, 1), (7, 9))
    cfg = {"mazewalk_x_s0": {"start": [0, 4], "goal": [3, 0]}}
    p, _ = G.process_layout(raw, "auto", n_points=300, endpoints=cfg)
    c = 1 / 5                                                       # 4x5 logical maze
    assert p["endpoints_source"] == "config"
    assert p["start"] == pytest.approx([4.5 * c, 3.5 * c]) and p["goal"] == pytest.approx([0.5 * c, 0.5 * c])
    assert p["start_cell"] == [1, 9] and p["goal_cell"] == [7, 1]      # raw grid cells
    assert p["topology"]["reachable"]
    other, _ = G.process_layout(raw, "auto", n_points=300, endpoints={"other": cfg["mazewalk_x_s0"]})
    assert other["endpoints_source"] == "raw"
    with pytest.raises(ValueError, match="not a cell"):
        G.process_layout(raw, "auto", n_points=300, endpoints={"mazewalk_x_s0": {"start": [0, 0], "goal": [4, 0]}})
    with pytest.raises(ValueError, match="differ"):
        G.process_layout(raw, "auto", n_points=300, endpoints={"mazewalk_x_s0": {"start": [0, 0], "goal": [0, 0]}})


def test_domain_L_scales_everything():
    g = S.backtracker_grid(4, 5, 1)
    raw = S.raw_dict(g, "mazewalk_x_s0", "mazewalk", (1, 1), (7, 9))
    p1, _ = G.process_layout(raw, "auto", n_points=300)
    p2, _ = G.process_layout(raw, "auto", n_points=300, domain_L=2.0)
    assert p2["domain"]["W"] == pytest.approx(2.0) and p2["domain"]["L"] == 2.0
    assert p2["passage_width"] == pytest.approx(2 * p1["passage_width"])
    assert np.array(p2["walls"]) == pytest.approx(2 * np.array(p1["walls"]))
    assert p2["start"] == pytest.approx(2 * np.array(p1["start"]))
    assert p2["reference_path_length"] == pytest.approx(2 * p1["reference_path_length"])
    assert p2["topology"]["reachable"] and p2["clearance_suggested"] == pytest.approx(2 * p1["clearance_suggested"])


# ---- derived two-room (mirror) layout ---------------------------------------------- #
def test_mirror_two_room_layout():
    import make_mirror_layout as M
    base = S.backtracker_grid(4, 4, 2)
    grid, r = M.mirror_two_room(base, bridge_row=3)
    n, w = base.shape
    assert grid.shape == (n, 2 * w + 1)
    assert (grid[:, :w][:, :w - 1] == base[:, :w - 1]).all()             # room A untouched (bar the connector col)
    assert (grid[:, w + 2:] == base[:, ::-1][:, 1:]).all()               # room B is the mirror image
    assert grid[r, w - 1] and grid[r, w] and grid[r, w + 1] and not grid[r - 2, w]   # one-cell bridge only
    start, goal = (1, 1), (2 * 2 + 1, 2 * w - 1)                          # goal: mirror of base cell (2, 0)
    assert M.logical_path_len(grid, start, goal) > 0
    raw = S.raw_dict(grid, "mazewalk_x_mirror2_s0", "mazewalk", start, goal)
    p, _ = G.process_layout(raw, "auto", n_points=300)
    assert p["decode"]["success"] and p["method"] == "thinwall" and p["topology"]["reachable"]
    with pytest.raises(ValueError):
        M.mirror_two_room(base, bridge_row=9)
