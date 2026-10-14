"""
Frozen DINOv2 image encoder for the robosuite image-state experiment.

Every roadmap node gets ONE embedding: the frozen DINOv2 features of V rendered
camera views (default agentview / sideview / frontview) of the arm at that
node's stored joint configuration, shape (V, D) (D = 384 for ViT-S/14, 768 for
ViT-B/14; doubled with pool="cls_mean"). The transformer sees ONLY these
embeddings (plus the raw 3-D action) -- EEF xyz is used for geometry, labels
and the veto's comparison cost, never as a model input.

Standardization (why not LayerNorm)
-----------------------------------
The scene never changes except for the arm pose, so every embedding shares a
large common component (background, table, cube, robot base). Per-dimension
standardization over a calibration set of arm states -- (x - mean) / std with
the mean/std taken across STATES for each (view, dimension) -- removes that
shared component and rescales each dimension to comparable size. LayerNorm
normalizes WITHIN a single vector (across its own dimensions), so it cannot
remove a component that is common across vectors; it would leave the
transformer's input dominated by that shared offset. The stats are computed
once in calibrate() and then frozen, so every token in the replay buffer stays
comparable no matter when it was created.

Usage
-----
    python image_encoder.py            # embedding-sensitivity diagnostic:
                                       # does embedding distance track EEF
                                       # distance? -> embedding_sensitivity.png
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

_BASE_DIM = {"dinov2_vits14": 384, "dinov2_vitb14": 768}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_PATCH = 14


@dataclass
class EncoderConfig:
    model_name: str = "dinov2_vits14"            # or "dinov2_vitb14" (768-d)
    cameras: Tuple[str, ...] = ("agentview", "sideview", "frontview")
    image_size: int = 224                        # multiple of the ViT patch size (14)
    pool: str = "cls"                            # "cls" | "cls_mean" (CLS ++ mean patch tokens)
    device: str = "auto"                         # "auto" = cuda if available else cpu
    n_calib: int = 200                           # states used to fit the standardization stats
    view_dropout: float = 0.0                    # whole-view dropout inside the model (training only)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class ImageEncoder:
    """Frozen DINOv2 over V camera views of the CURRENT sim state."""

    def __init__(self, cfg: EncoderConfig):
        if cfg.model_name not in _BASE_DIM:
            raise ValueError(f"model_name must be one of {list(_BASE_DIM)}, got {cfg.model_name!r}")
        if cfg.pool not in ("cls", "cls_mean"):
            raise ValueError(f"pool must be 'cls' or 'cls_mean', got {cfg.pool!r}")
        assert cfg.image_size % _PATCH == 0, (
            f"image_size {cfg.image_size} must be a multiple of the patch size {_PATCH}")
        assert len(cfg.cameras) >= 1, "need at least one camera"
        self.cfg = cfg
        self.device = resolve_device(cfg.device)

        print(f"[encoder] loading {cfg.model_name} via torch.hub "
              f"(facebookresearch/dinov2); the FIRST run downloads the weights "
              f"to ~/.cache/torch/hub, later runs reuse them")
        self.model = torch.hub.load("facebookresearch/dinov2", cfg.model_name)
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)

        self._mean_t = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std_t = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.mean: Optional[np.ndarray] = None       # (V, D) frozen after calibrate()
        self.std: Optional[np.ndarray] = None
        self._cams_ok = False
        # cumulative cost of embed() (render + forward + standardize) so the
        # driver can report per-iteration embedding time / count
        self.embed_time = 0.0
        self.embed_count = 0

    # ------------------------------------------------------------------ dims
    @property
    def n_views(self) -> int:
        return len(self.cfg.cameras)

    @property
    def emb_dim(self) -> int:
        return _BASE_DIM[self.cfg.model_name] * (2 if self.cfg.pool == "cls_mean" else 1)

    # ------------------------------------------------------------- embedding
    def _check_cameras(self, ik) -> None:
        if self._cams_ok:
            return
        avail = ik.camera_names()
        missing = [c for c in self.cfg.cameras if c not in avail]
        if missing:
            raise ValueError(f"cameras {missing} not found in the sim; "
                             f"available cameras: {avail}")
        self._cams_ok = True

    @torch.no_grad()
    def embed_raw(self, ik) -> np.ndarray:
        """(V, D) raw (unstandardized) embeddings of the CURRENT sim state --
        every view is rendered from `ik` and all V go through ONE batched
        forward pass."""
        self._check_cameras(ik)
        S = self.cfg.image_size
        try:
            imgs = np.stack([ik.render(cam, S, S) for cam in self.cfg.cameras])   # (V,S,S,3) uint8
        except Exception as e:
            print("[encoder] rendering failed. On a headless machine try setting "
                  "MUJOCO_GL=egl (or osmesa) before running.")
            raise
        x = torch.from_numpy(imgs).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self._mean_t) / self._std_t
        if self.cfg.pool == "cls":
            z = self.model(x)                                    # normalized CLS token (V, D)
        else:
            f = self.model.forward_features(x)
            z = torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(dim=1)], dim=-1)
        return z.float().cpu().numpy()

    def embed(self, ik) -> np.ndarray:
        """(V, D) float32 standardized embeddings -- what gets stored per node
        and fed (flattened) into the transformer."""
        assert self.mean is not None, "call calibrate() first (stats are frozen after it)"
        t0 = time.perf_counter()
        z = (self.embed_raw(ik) - self.mean) / self.std
        self.embed_time += time.perf_counter() - t0
        self.embed_count += 1
        return z.astype(np.float32)

    # ----------------------------------------------------------- calibration
    def calibrate(self, task, rng, restart_every: int = 25):
        """Fits and FREEZES the per-(view, dim) standardization stats from
        cfg.n_calib arm states gathered by random walks with
        task.world.sample_free from the start config (restarting every
        `restart_every` steps or whenever sample_free fails). The start state
        is included. Always ends with task.reset_to_start().

        Returns (positions (N, 3), raw_embs (N, V, D)) for diagnostics."""
        ik, world = task.ik, task.world
        n_target = self.cfg.n_calib
        task.reset_to_start()
        pos = [ik.eef_pos()]
        raw = [self.embed_raw(ik)]
        since_restart, attempts = 0, 0
        max_attempts = 50 * n_target
        while len(raw) < n_target:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"calibration collected only {len(raw)}/{n_target} states after "
                    f"{attempts} sample_free attempts -- is the arm stuck?")
            if since_restart >= restart_every:
                task.reset_to_start()
                since_restart = 0
            cand = world.sample_free(rng)
            if cand is None:
                task.reset_to_start()
                since_restart = 0
                continue
            # sample_free leaves the sim at the new config on success
            pos.append(ik.eef_pos())
            raw.append(self.embed_raw(ik))
            since_restart += 1
        raw_arr = np.stack(raw)                                  # (N, V, D)
        self.mean = raw_arr.mean(axis=0)
        self.std = np.maximum(raw_arr.std(axis=0), 1e-6)
        task.reset_to_start()
        return np.stack(pos), raw_arr

    # ------------------------------------------------------------ persistence
    def save_stats(self, path: str) -> None:
        assert self.mean is not None, "nothing to save: calibrate() has not run"
        np.savez(path, mean=self.mean, std=self.std,
                 model_name=self.cfg.model_name, pool=self.cfg.pool,
                 cameras=np.array(self.cfg.cameras), image_size=self.cfg.image_size)

    def load_stats(self, path: str) -> None:
        d = np.load(path, allow_pickle=False)
        mean, std = d["mean"], d["std"]
        want = (self.n_views, self.emb_dim)
        if mean.shape != want or std.shape != want:
            raise ValueError(f"stats shape {mean.shape} does not match encoder {want}")
        self.mean, self.std = mean, std


# =========================================================================== #
# Diagnostic: does embedding distance track EEF distance?
# =========================================================================== #
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), pure numpy."""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0                               # 1-based average rank per unique value
    return avg[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    return np.sqrt(np.maximum(d2, 0.0))


def run_sensitivity_diagnostic(args) -> None:
    from task import RobosuiteTask

    task = RobosuiteTask(seed=args.seed, enable_render=True)
    enc_cfg = EncoderConfig(model_name=args.encoder, pool=args.pool,
                            device=args.device, n_calib=args.n_calib)
    enc = ImageEncoder(enc_cfg)
    rng = np.random.default_rng(args.seed + 2_000_003)
    print(f"[diag] calibrating on {enc_cfg.n_calib} random-walk states ...")
    t0 = time.perf_counter()
    pos, raw = enc.calibrate(task, rng)
    print(f"[diag] calibration took {time.perf_counter() - t0:.1f}s "
          f"({(time.perf_counter() - t0) / len(pos) * 1000:.0f} ms/state)")

    Z = (raw - enc.mean) / enc.std                               # (N, V, D) standardized
    N, V, _ = Z.shape
    iu = np.triu_indices(N, k=1)
    d_eef = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)[iu]
    near = d_eef < 0.05
    print(f"[diag] {N} states -> {len(d_eef)} pairs; {int(near.sum())} pairs with EEF dist < 5 cm")

    series = []
    for v, cam in enumerate(enc_cfg.cameras):
        series.append((cam, _pairwise_dist(Z[:, v, :])[iu]))
    series.append(("fused (all views)", _pairwise_dist(Z.reshape(N, -1))[iu]))

    print(f"\nSpearman rank correlation, EEF distance vs embedding distance "
          f"({enc_cfg.model_name}, pool={enc_cfg.pool}):")
    print(f"  {'view':<20}{'all pairs':>12}{'pairs < 5 cm':>16}")
    rho_all, rho_near = {}, {}
    for name, dist in series:
        rho_all[name] = spearman(d_eef, dist)
        rho_near[name] = spearman(d_eef[near], dist[near])
        print(f"  {name:<20}{rho_all[name]:>12.3f}{rho_near[name]:>16.3f}")
    print("\nNote: a weak correlation on the < 5 cm pairs means the embedding barely "
          "resolves small arm motions;\n      try pool='cls_mean' (--pool cls_mean) "
          "or dinov2_vitb14 (--encoder dinov2_vitb14).")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(series), figsize=(4.6 * len(series), 4.2))
    for ax, (name, dist) in zip(np.atleast_1d(axes), series):
        ax.scatter(d_eef, dist, s=2, alpha=0.15, c="tab:blue", rasterized=True)
        ax.axvline(0.05, color="tab:red", ls="--", lw=0.8)
        ax.set_xlabel("EEF distance (m)")
        ax.set_ylabel("embedding distance (standardized)")
        ax.set_title(f"{name}\nrho all={rho_all[name]:.2f} | <5cm={rho_near[name]:.2f}",
                     fontsize=9)
    fig.suptitle(f"Embedding sensitivity ({enc_cfg.model_name}, pool={enc_cfg.pool}, "
                 f"{N} states)", fontsize=10)
    fig.tight_layout()
    out = args.out
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[diag] saved -> {out}")
    task.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Embedding-sensitivity diagnostic.")
    p.add_argument("--encoder", default="dinov2_vits14", choices=list(_BASE_DIM))
    p.add_argument("--pool", default="cls", choices=["cls", "cls_mean"])
    p.add_argument("--n-calib", type=int, default=200, dest="n_calib")
    p.add_argument("--device", default="auto", help="auto (cuda else cpu) | cpu | mps | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="embedding_sensitivity.png")
    run_sensitivity_diagnostic(p.parse_args())
