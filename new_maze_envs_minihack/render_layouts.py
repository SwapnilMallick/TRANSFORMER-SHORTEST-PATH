"""Render processed layouts (and their raw grids) to images + gallery.

Everything is drawn from the processed/raw JSON files on disk (never in-memory state),
so the images show exactly what the training code will load.  No minihack/nle imports;
native MiniHack renders are only read if extraction saved them in <out>/minihack/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
GOAL_RADIUS_FACTOR = 0.4
FREE_C, BLOCK_C, GROWN_C, FIX_C = "#eef3f8", "#2b2b2b", "#cfe0f0", "#ff7f0e"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_processed(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    d["_path"] = Path(path)
    return d


def load_raw(proc: dict):
    rf = proc.get("raw_file") or ""
    proot = proc["_path"].parent
    for cand in (Path(rf), proot.parent / rf, proot.parent / "raw" / f"{proc['name']}.json"):
        if cand.is_file():
            return json.loads(cand.read_text())
    return None


def load_roadmap(proc: dict):
    p = proc["_path"].parent / proc.get("roadmap_file", "")
    if p.is_file():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    return None


def grid_array(rows):
    return np.array([[ch == "." for ch in r] for r in rows], dtype=bool)


# --------------------------------------------------------------------------- #
# Panels (each draws onto a given axes)
# --------------------------------------------------------------------------- #
def _marker_kw(scale=1.0):
    return dict(start=dict(c="white", edgecolors="k", s=90 * scale, zorder=5),
                goal=dict(marker="*", c="gold", edgecolors="k", s=240 * scale, zorder=5))


def _cells_axes(ax, free, lines=False, title=None):
    n, m = free.shape
    ax.imshow(np.where(free, 1.0, 0.0), cmap=matplotlib.colors.ListedColormap([BLOCK_C, FREE_C]),
              vmin=0, vmax=1, extent=(0, m, n, 0), interpolation="nearest")
    if lines:
        ax.set_xticks(np.arange(0, m + 1), minor=True)
        ax.set_yticks(np.arange(0, n + 1), minor=True)
        ax.grid(which="minor", color="#888", lw=0.4)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=8)


def xy_to_cell(x, y, n_rows, cell):
    """domain (x, y) -> fractional (col, row) in a top-down grid of the given cell size."""
    return x / cell, n_rows - y / cell


def draw_raw(ax, proc, raw, scale=1.0):
    if raw is None:
        ax.text(0.5, 0.5, "raw file\nnot found", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    free = grid_array(raw["grid"])
    n, m = free.shape
    _cells_axes(ax, free, lines=max(n, m) <= 30, title=f"raw grid {m}x{n}")
    for r, c in proc["diagonal_fixes"]:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, fc=FIX_C, ec="none", zorder=3))
    if proc["diagonal_fixes"]:
        ax.text(0.01, 0.01, f"{len(proc['diagonal_fixes'])} diagonal fix(es) in orange", transform=ax.transAxes,
                fontsize=6, color=FIX_C, va="bottom")
    kw = _marker_kw(scale)
    (sr, sc), (gr, gc) = proc.get("start_cell", raw["start"]), proc.get("goal_cell", raw["goal"])
    ax.scatter(sc + 0.5, sr + 0.5, **kw["start"])
    ax.scatter(gc + 0.5, gr + 0.5, **kw["goal"])


def draw_widened(ax, proc, raw, scale=1.0):
    mask = grid_array(proc["free_mask"]["grid"])
    cell = proc["free_mask"]["cell"]
    n, m = mask.shape
    kw = _marker_kw(scale)
    if proc["method"] == "thinwall":
        _cells_axes(ax, mask, lines=True, title=f"logical maze {m}x{n} (cell = passage width)")
        for a, b, c, d in proc["logical_edges"] or []:
            ax.plot([b + .5, d + .5], [a + .5, c + .5], color="#1f77b4", lw=3, zorder=3, solid_capstyle="round")
        ii, jj = np.nonzero(mask)
        ax.scatter(jj + .5, ii + .5, s=14, c="#1f77b4", zorder=4)
    else:
        k = proc["params"]["k"]
        title = f"upsampled x{k}, grown by m={proc['params']['m']}"
        _cells_axes(ax, mask, lines=False, title=title)
        if raw is not None:
            fixed = grid_array(raw["grid"]).copy()
            for r, c in proc["diagonal_fixes"]:
                fixed[r, c] = True
            up = np.kron(fixed, np.ones((k, k), bool))
            grown = mask & ~up
            ax.imshow(np.ma.masked_where(~grown, grown), cmap=matplotlib.colors.ListedColormap([GROWN_C]),
                      extent=(0, m, n, 0), interpolation="nearest", zorder=2)
            ax.text(0.01, 0.01, "grown cells in light blue", transform=ax.transAxes, fontsize=6,
                    color="#4a7fb0", va="bottom")
    for key in ("start", "goal"):
        cx, ry = xy_to_cell(*proc[key], n, cell)
        ax.scatter(cx, ry, **kw[key])


def draw_continuous(ax, proc, scale=1.0, roadmap=None):
    W, H = proc["domain"]["W"], proc["domain"]["H"]
    pw = proc["passage_width"]
    mask = grid_array(proc["free_mask"]["grid"])
    ax.set_facecolor("#d9d9d9")
    ax.imshow(np.where(mask, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap([FREE_C]), vmin=0, vmax=1,
              extent=(0, mask.shape[1] * proc["free_mask"]["cell"], 0, mask.shape[0] * proc["free_mask"]["cell"]),
              interpolation="nearest", zorder=0)
    walls = np.array(proc["walls"]).reshape(-1, 2, 2)
    n_w = len(walls)
    lw = 1.6 if n_w < 300 else 1.0 if n_w < 1000 else 0.6
    if roadmap is not None:
        pts, edges = roadmap["points"], roadmap["edges"]
        ax.add_collection(LineCollection(pts[edges], colors="#1f77b4", lw=0.25, alpha=0.07, zorder=1))
        if len(roadmap["path"]):
            ax.plot(*pts[roadmap["path"]].T, color="crimson", lw=2 * scale, zorder=6)
    ax.add_collection(LineCollection(walls[:, :, :], colors="k", lw=lw, zorder=2, capstyle="projecting"))
    kw = _marker_kw(scale)
    ax.add_patch(plt.Circle(proc["goal"], GOAL_RADIUS_FACTOR * pw, color="gold", alpha=0.35, zorder=3))
    ax.scatter(*proc["start"], **kw["start"])
    ax.scatter(*proc["goal"], **kw["goal"])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")


def draw_scale_band(ax, pw, xspan):
    """Scale bar equal to one passage width."""
    ax.set_xlim(0, xspan)
    ax.set_ylim(0, 1.2 * pw)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([0.05 * pw, 1.05 * pw], [0.4 * pw, 0.4 * pw], color="k", lw=3, solid_capstyle="butt")
    ax.text(0.55 * pw, 0.6 * pw, f"passage width {pw:.4f}", ha="left", va="bottom", fontsize=6)


def _title(proc):
    p = proc["params"]
    par = "" if p["k"] is None else f", k={p['k']} m={p['m']}"
    return f"{proc['name']}  [{proc['method']}{par}]  passage width {proc['passage_width']:.4f}"


def fig_continuous(proc, roadmap=None, dpi=150):
    W, H, pw = proc["domain"]["W"], proc["domain"]["H"], proc["passage_width"]
    S = 8.0 / max(W, H)                       # inches per domain unit (same in both axes)
    xspan = max(W, 4.2 * pw)
    band = 1.2 * pw
    mx, mt, mb = 0.4, 0.55, 0.3
    Wf = max(xspan * S + 2 * mx, 4.5)
    Hf = H * S + band * S + mt + mb + 0.25
    fig = plt.figure(figsize=(Wf, Hf), dpi=dpi)
    x0 = (Wf - xspan * S) / 2
    main = fig.add_axes([x0 / Wf, (mb + band * S + 0.25) / Hf, W * S / Wf, H * S / Hf])
    draw_continuous(main, proc, roadmap=roadmap)
    bax = fig.add_axes([x0 / Wf, mb / Hf, xspan * S / Wf, band * S / Hf])
    draw_scale_band(bax, pw, xspan)
    fig.suptitle(_title(proc), fontsize=8, y=1 - 0.12 / Hf)
    return fig


# --------------------------------------------------------------------------- #
# Per-layout outputs
# --------------------------------------------------------------------------- #
def _save(fig, path: Path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight" if path.parent.name == "previews" else None)
    plt.close(fig)


def _single(draw, proc, raw, title, path, dpi):
    fig, ax = plt.subplots(figsize=(7, 7))
    draw(ax, proc, raw)
    fig.suptitle(f"{proc['name']}: {title}", fontsize=9)
    _save(fig, path, dpi)


def render_grid(proc, out, dpi):
    raw = load_raw(proc)
    name = proc["stem"]
    _single(draw_raw, proc, raw, "raw grid", out / "grid" / f"{name}_raw.png", dpi)
    _single(draw_widened, proc, raw, "widened", out / "grid" / f"{name}_widened.png", dpi)


def render_continuous(proc, out, dpi):
    name = proc["stem"]
    fig = fig_continuous(proc, None, dpi)
    _save(fig, out / "continuous" / f"{name}.svg", dpi)
    fig = fig_continuous(proc, None, dpi)
    _save(fig, out / "continuous" / f"{name}.png", dpi)
    road = load_roadmap(proc)
    _save(fig_continuous(proc, road, dpi), out / "continuous" / f"{name}_roadmap.png", dpi)


def render_preview(proc, out, dpi):
    raw = load_raw(proc)
    name = proc["stem"]
    fig, axs = plt.subplots(1, 4, figsize=(22, 6))
    nat = out / "minihack" / f"{proc['name']}.png"
    if nat.is_file():
        axs[0].imshow(plt.imread(nat))
        axs[0].set_title("native MiniHack", fontsize=8)
    else:
        axs[0].text(0.5, 0.5, "no native render", ha="center", va="center", transform=axs[0].transAxes)
    axs[0].axis("off")
    draw_raw(axs[1], proc, raw, 0.6)
    draw_widened(axs[2], proc, raw, 0.6)
    draw_continuous(axs[3], proc, scale=0.6)
    axs[3].set_title("continuous", fontsize=8)
    fig.suptitle(_title(proc), fontsize=10)
    fig.tight_layout()
    _save(fig, out / "previews" / f"{name}.png", dpi)


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #
def topo_ok(proc):
    t = proc["topology"]
    return bool(t["components_ok"] and t["n_shortcuts"] == 0 and t["reachable"])


def render_gallery(procs, out, dpi):
    (out / "gallery").mkdir(parents=True, exist_ok=True)
    fams = sorted({p["family"] for p in procs})
    for fam in fams:
        ps = [p for p in procs if p["family"] == fam and (out / "continuous" / f"{p['stem']}.png").is_file()]
        if not ps:
            print(f"gallery: no continuous renders for {fam}; run with --only continuous first")
            continue
        ncol = min(3, len(ps))
        nrow = -(-len(ps) // ncol)
        fig, axs = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        for ax, p in zip(axs.ravel(), ps):
            ax.imshow(plt.imread(out / "continuous" / f"{p['stem']}.png"))
            ax.set_title(f"{p['name']} [{p['method']}] pw={p['passage_width']:.3f} "
                         f"{'PASS' if topo_ok(p) else 'FAIL'}", fontsize=8)
        fig.tight_layout()
        _save(fig, out / "gallery" / f"{fam}.png", dpi)

    lines = ["# Layout gallery", "",
             "| thumb | name | family | size | seed | method | params | passage width | ref. path len | diag. fixes "
             "| decode | topology | images |", "|" + "---|" * 13]
    for p in sorted(procs, key=lambda p: (p["family"], p["size"], p["seed"], p["method"])):
        s = p["stem"]
        par = "" if p["params"]["k"] is None else f"k={p['params']['k']} m={p['params']['m']}"
        d = p["decode"]
        dec = "n/a" if not d["attempted"] else ("ok" + (" (snapped)" if d["snapped"] else "") if d["success"]
                                                 else "FAILED -> dilate fallback")
        t = p["topology"]
        top = ("pass" if topo_ok(p) else "FAIL") + f" (shortcuts {t['n_shortcuts']}/{t['pairs_checked']})"
        links = " · ".join(f"[{lab}]({path})" for lab, path in [
            ("raw", f"grid/{s}_raw.png"), ("widened", f"grid/{s}_widened.png"), ("cont", f"continuous/{s}.png"),
            ("svg", f"continuous/{s}.svg"), ("roadmap", f"continuous/{s}_roadmap.png"),
            ("preview", f"previews/{s}.png")])
        lines.append(f"| <img src=\"continuous/{s}.png\" width=\"120\"> | {p['name']} | {p['family']} | {p['size']} "
                     f"| {p['seed']} | {p['method']} | {par} | {p['passage_width']:.4f} "
                     f"| {p['reference_path_length']:.3f} | {len(p['diagonal_fixes'])} | {dec} | {top} | {links} |")
    lines += ["", "Contact sheets: " + ", ".join(f"[{f}](gallery/{f}.png)" for f in fams)]
    (out / "gallery.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", default=str(HERE / "layouts" / "processed"))
    ap.add_argument("--out", default=str(HERE / "layouts" / "renders"))
    ap.add_argument("--only", action="append", choices=["grid", "continuous", "previews", "gallery"])
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args(argv)
    inp, out = Path(a.inp), Path(a.out)
    files = sorted(inp.glob("*.json")) if inp.is_dir() else [inp]
    procs = [load_processed(f) for f in files]
    only = set(a.only or ["grid", "continuous", "previews", "gallery"])
    for p in procs:
        if "grid" in only:
            render_grid(p, out, a.dpi)
        if "continuous" in only:
            render_continuous(p, out, a.dpi)
        if "previews" in only:
            render_preview(p, out, a.dpi)
        print("rendered", p["stem"])
    if "gallery" in only:
        render_gallery(procs, out, a.dpi)


if __name__ == "__main__":
    main()
