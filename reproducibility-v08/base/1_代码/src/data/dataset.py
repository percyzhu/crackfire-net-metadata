"""
Crack Graph Dataset for charring area ratio time-series prediction.

Data flow:
    Crack JSON + FEM .npz -> Crack Graph + time-series label -> PyG Data

Graph structure (same as before):
    - Nodes: one per crack, features = encoded 6-tuple + derived features
    - Edges: fully connected between all cracks
    - Edge features: spatial relationship
    - Global features: crack count ratio, total area ratio, max depth ratio

Label: time series of 300C charring area fraction [T] at uniform time steps.
"""

import os
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


# ============================================================
# Beam constants (default, matching real experiment dimensions)
# ============================================================
DEFAULT_BEAM = {"L": 1.5, "W": 0.14, "H": 0.2}


# ============================================================
# Graph construction from crack parameters
# ============================================================

def encode_crack_node_features(cracks, beam=None):
    """Encode each crack's 6-tuple into node feature vector.

    Features per crack (11 dim):
        [0:4]  face one-hot (top, bottom, right, left)
        [4]    z_pos / L
        [5]    local_h / h_max
        [6]    c_width / W
        [7]    c_len / L
        [8]    c_depth / thickness
        [9]    c_len * c_depth / (W*H)  (crack area ratio)
        [10]   c_depth / thickness      (depth penetration ratio)
    """
    b = beam or DEFAULT_BEAM
    L, W, H = b["L"], b["W"], b["H"]

    features = []
    for c in cracks:
        face = int(c["face"])
        z, h, w, l, d = (float(c[k]) for k in ("z", "h", "w", "l", "d"))

        face_oh = [0.0] * 4
        face_oh[face] = 1.0

        thickness = H if face in (0, 1) else W
        h_max = (W / 2.0 - 0.05) if face in (0, 1) else (H / 2.0 - 0.05)
        h_max = max(h_max, 0.01)

        feat = face_oh + [
            z / L,
            h / h_max,
            w / W,
            l / L,
            d / thickness,
            (l * d) / (W * H),
            d / thickness,
        ]
        features.append(feat)

    return torch.tensor(features, dtype=torch.float32)


def encode_global_features(cracks, beam=None):
    """Encode beam-level global features [1, 3]."""
    b = beam or DEFAULT_BEAM
    L, W, H = b["L"], b["W"], b["H"]
    beam_surface = 2 * (W + H) * L

    total_area = sum(float(c["l"]) * float(c["w"]) for c in cracks)
    max_depth = max((float(c["d"]) for c in cracks), default=0.0)

    feat = [
        len(cracks) / 10.0,
        total_area / beam_surface,
        max_depth / max(W, H),
    ]
    return torch.tensor([feat], dtype=torch.float32)


def build_crack_edges(cracks, beam=None):
    """Build fully connected edges between all crack nodes.

    Edge features (5 dim): same_face, adj_face, delta_z, distance, combined_depth
    """
    b = beam or DEFAULT_BEAM
    L, W, H = b["L"], b["W"], b["H"]
    diag = (L**2 + W**2 + H**2) ** 0.5

    n = len(cracks)
    if n < 2:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 5), dtype=torch.float32))

    adjacent_pairs = {(0, 2), (0, 3), (1, 2), (1, 3),
                      (2, 0), (3, 0), (2, 1), (3, 1)}

    def center_3d(c):
        face, z, h, d = int(c["face"]), float(c["z"]), float(c["h"]), float(c["d"])
        if face == 0:   return np.array([h, H/2 - d/2, z])
        elif face == 1: return np.array([h, -H/2 + d/2, z])
        elif face == 2: return np.array([W/2 - d/2, h, z])
        else:           return np.array([-W/2 + d/2, h, z])

    centers = [center_3d(c) for c in cracks]

    src, dst, feats = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            fi, fj = int(cracks[i]["face"]), int(cracks[j]["face"])
            src.append(i)
            dst.append(j)
            feats.append([
                1.0 if fi == fj else 0.0,
                1.0 if (fi, fj) in adjacent_pairs else 0.0,
                abs(float(cracks[i]["z"]) - float(cracks[j]["z"])) / L,
                np.linalg.norm(centers[i] - centers[j]) / diag,
                (float(cracks[i]["d"]) + float(cracks[j]["d"])) / max(W, H),
            ])

    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(feats, dtype=torch.float32))


def cracks_to_pyg_data(cracks, charring_series=None, beam=None,
                       sample_id="", num_time_steps=61, duration=3600.0,
                       curve_type='iso834', curve_params=None):
    """Convert crack parameters to PyG Data with time-series label.

    Args:
        cracks: list of crack param dicts
        charring_series: [T] array of W_ratio_min time series, or None
        beam: beam dimensions dict
        sample_id: identifier
        num_time_steps: number of output time steps
        duration: simulation duration (seconds)
        curve_type: fire curve identifier (matches model.compute_fire_curve)
        curve_params: optional dict of curve-specific parameters

    Returns:
        PyG Data with x, edge_index, edge_attr, x_global, time_features, y
    """
    from src.gnn.model import make_time_features

    x = encode_crack_node_features(cracks, beam)
    x_global = encode_global_features(cracks, beam)
    edge_index, edge_attr = build_crack_edges(cracks, beam)

    tf = make_time_features(num_time_steps, duration, curve_type, curve_params)
    time_features = tf.squeeze(0)  # [T, 4]

    if charring_series is not None:
        y = torch.tensor(charring_series, dtype=torch.float32).unsqueeze(-1)
    else:
        y = torch.ones(num_time_steps, 1)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x_global=x_global,
        time_features=time_features,
        y=y,
        num_cracks=len(cracks),
        sample_id=sample_id,
    )
    return data


# ============================================================
# Charring area fraction extraction from FEM results
# ============================================================

def extract_charring_series(
    npz_path,
    target_times=None,
    num_time_steps=61,
    duration=3600.0,
    threshold_temp=300.0,
):
    """Extract effective section modulus ratio time series from FEM .npz.

    charring_ratios in .npz is now W_eff/W_0 (section modulus ratio):
    starts at 1.0 (fully effective) and decreases toward 0 as charring progresses.

    Args:
        npz_path: path to .npz
        target_times: [T_out] target time points
        num_time_steps: number of output steps
        duration: total duration
        threshold_temp: charring temperature (300C), used only for fallback

    Returns:
        section_modulus_ratios: [T_out] array in [0, 1], 1=fully intact, 0=fully charred
    """
    d = np.load(npz_path, allow_pickle=True)
    fem_times = d["time_steps"]

    if target_times is None:
        target_times = np.linspace(0, duration, num_time_steps)

    # Pre-computed section modulus ratio from extract_odb.py
    if "charring_ratios" in d:
        fem_ratios = d["charring_ratios"]
        # Charring is irreversible: enforce monotonic non-increasing
        fem_ratios = np.minimum.accumulate(fem_ratios)
        return np.interp(target_times, fem_times, fem_ratios)

    # Fallback: compute simple area-based ratio from temperature field
    if "temperatures" in d:
        temperatures = d["temperatures"]
        num_nodes = temperatures.shape[1]
        fractions = np.ones(len(target_times))
        for i, t_target in enumerate(target_times):
            idx = np.argmin(np.abs(fem_times - t_target))
            fractions[i] = 1.0 - np.sum(temperatures[idx] >= threshold_temp) / num_nodes
        return fractions

    return np.ones(len(target_times))


def extract_fire_resistance(modulus_series, target_times, limit=0.5):
    """Extract fire resistance from section modulus ratio time series.

    Fire resistance = first time when W_eff/W_0 drops to or below limit.

    Args:
        modulus_series: [T] section modulus ratios (1.0 -> 0.0)
        target_times: [T] time values
        limit: failure threshold (default 0.5 = 50% capacity lost)

    Returns:
        fire_resistance: time in seconds (or duration if never reached)
    """
    for i, ratio in enumerate(modulus_series):
        if ratio <= limit:
            return float(target_times[i])
    return float(target_times[-1])


# ============================================================
# PyG Dataset
# ============================================================

class CrackFireDataset(InMemoryDataset):
    """Crack graph dataset with charring time-series labels.

    Expected structure: root/raw/batch_run_log.json + sample_XXXX.npz files.

    Split modes:
        "random"                - random 70/15/15 split (default, backward compatible)
        "leave_one_curve_out"   - holdout all samples of one curve type as test
        "leave_crack_range_out" - holdout samples with crack count in [lo, hi] as test
    """

    def __init__(self, root, split="train", train_ratio=0.70, val_ratio=0.15,
                 num_time_steps=61, duration=3600.0, threshold_temp=300.0,
                 beam=None, transform=None, pre_transform=None,
                 split_mode="random", holdout_curve=None,
                 holdout_crack_range=None):
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_time_steps = num_time_steps
        self.duration = duration
        self.threshold_temp = threshold_temp
        self.beam = beam or DEFAULT_BEAM
        self.split_mode = split_mode
        self.holdout_curve = holdout_curve
        self.holdout_crack_range = holdout_crack_range
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["batch_run_log.json"]

    @property
    def processed_file_names(self):
        tag = self._split_tag()
        return [f"{self.split}_{tag}.pt"]

    def _split_tag(self):
        if self.split_mode == "leave_one_curve_out" and self.holdout_curve:
            return f"loco_{self.holdout_curve}"
        elif self.split_mode == "leave_crack_range_out" and self.holdout_crack_range:
            lo, hi = self.holdout_crack_range
            return f"lcro_{lo}_{hi}"
        return "random"

    def process(self):
        log_path = os.path.join(self.root, "raw", "batch_run_log.json")
        if not os.path.exists(log_path):
            self.save([], self.processed_paths[0])
            return

        with open(log_path) as f:
            log = json.load(f)

        crack_params_list = log.get("crack_params", [])
        results = log.get("results", [])
        fire_curves = log.get("fire_curves", [])

        valid = [i for i, r in enumerate(results)
                 if r.get("status") == "success" and i < len(crack_params_list)]

        if not valid:
            self.save([], self.processed_paths[0])
            return

        selected = self._compute_split(valid, crack_params_list, fire_curves)

        target_times = np.linspace(0, self.duration, self.num_time_steps)
        raw_dir = os.path.join(self.root, "raw")
        data_list = []

        for orig_idx in selected:
            cracks = crack_params_list[orig_idx]

            if orig_idx < len(results) and "sample_id" in results[orig_idx]:
                sid = "sample_%04d" % results[orig_idx]["sample_id"]
            else:
                sid = "sample_%04d" % orig_idx

            npz_path = os.path.join(raw_dir, sid + ".npz")
            if os.path.exists(npz_path):
                charring = extract_charring_series(
                    npz_path, target_times, self.num_time_steps,
                    self.duration, self.threshold_temp,
                )
            else:
                charring = None

            curve_info = fire_curves[orig_idx] if orig_idx < len(fire_curves) else {}
            curve_type = curve_info.get("type", "iso834")
            curve_params = curve_info.get("params", None)

            data = cracks_to_pyg_data(
                cracks, charring, self.beam, sid,
                self.num_time_steps, self.duration,
                curve_type=curve_type, curve_params=curve_params,
            )
            data.curve_type = curve_type
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

    def _compute_split(self, valid, crack_params_list, fire_curves):
        """Return list of original indices for the current split."""
        rng = np.random.RandomState(42)

        if self.split_mode == "leave_one_curve_out" and self.holdout_curve:
            holdout_idx = []
            rest_idx = []
            for orig_idx in valid:
                ci = fire_curves[orig_idx] if orig_idx < len(fire_curves) else {}
                ct = ci.get("type", "iso834")
                if ct == self.holdout_curve:
                    holdout_idx.append(orig_idx)
                else:
                    rest_idx.append(orig_idx)
            return self._split_rest_and_holdout(rest_idx, holdout_idx, rng)

        if self.split_mode == "leave_crack_range_out" and self.holdout_crack_range:
            lo, hi = self.holdout_crack_range
            holdout_idx = []
            rest_idx = []
            for orig_idx in valid:
                nc = len(crack_params_list[orig_idx])
                if lo <= nc <= hi:
                    holdout_idx.append(orig_idx)
                else:
                    rest_idx.append(orig_idx)
            return self._split_rest_and_holdout(rest_idx, holdout_idx, rng)

        # Default: random split
        n = len(valid)
        perm = rng.permutation(n)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        if self.split == "train":
            sel = perm[:n_train]
        elif self.split == "val":
            sel = perm[n_train:n_train + n_val]
        else:
            sel = perm[n_train + n_val:]
        return [valid[i] for i in sel]

    def _split_rest_and_holdout(self, rest_idx, holdout_idx, rng):
        """For leave-one-out modes: holdout is test, rest splits into train/val."""
        if self.split == "test":
            return holdout_idx
        n_rest = len(rest_idx)
        perm = rng.permutation(n_rest)
        val_frac = self.val_ratio / (self.train_ratio + self.val_ratio)
        n_val = int(n_rest * val_frac)
        if self.split == "val":
            return [rest_idx[i] for i in perm[:n_val]]
        return [rest_idx[i] for i in perm[n_val:]]
