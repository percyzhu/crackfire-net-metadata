"""Version 2 geometry features in the finite-element global coordinates.

The FE sketch transform reverses the sign of local h on top (0) and left (3).
Bottom (1) and right (2) preserve it. This correction is supported by the
independent tetrahedral-section audit, rather than inferred from the network.
Legacy functions remain selected unchanged by run_readiness. Do not feed these
new features to old weights and report the output as a reproduced benchmark.
"""
import numpy as np
import torch

FEATURE_VERSION = "fe-global-coordinate-features-v2"
DEFAULT_BEAM = {"L": 1.5, "W": 0.14, "H": 0.2}


def transverse_global_coordinate(crack):
    face = int(crack["face"])
    if face not in (0, 1, 2, 3):
        raise ValueError("Crack face must be 0, 1, 2 or 3")
    return (-1.0 if face in (0, 3) else 1.0) * float(crack["h"])


def crack_center_3d(crack, beam=None):
    b = beam or DEFAULT_BEAM
    face = int(crack["face"])
    z, d = float(crack["z"]), float(crack["d"])
    q = transverse_global_coordinate(crack)
    if face == 0:
        return np.array([q, b["H"] / 2 - d / 2, z])
    if face == 1:
        return np.array([q, -b["H"] / 2 + d / 2, z])
    if face == 2:
        return np.array([b["W"] / 2 - d / 2, q, z])
    return np.array([-b["W"] / 2 + d / 2, q, z])


def encode_crack_node_features(cracks, beam=None):
    if not cracks:
        raise ValueError("This dataset and protocol cover one or more cracks")
    b = beam or DEFAULT_BEAM
    L, W, H = b["L"], b["W"], b["H"]
    features = []
    for c in cracks:
        face = int(c["face"])
        z, w, length, depth = (float(c[k]) for k in ("z", "w", "l", "d"))
        thickness = H if face in (0, 1) else W
        h_scale = max((W / 2 - 0.05) if face in (0, 1) else (H / 2 - 0.05), 0.01)
        onehot = [float(face == f) for f in range(4)]
        # Keep the duplicated depth feature deliberately so that this version
        # isolates the coordinate correction; feature pruning is a later ablation.
        features.append(onehot + [z / L, transverse_global_coordinate(c) / h_scale,
            w / W, length / L, depth / thickness, length * depth / (W * H),
            depth / thickness])
    return torch.tensor(features, dtype=torch.float32)


def encode_global_features(cracks, beam=None):
    b = beam or DEFAULT_BEAM
    opening_area = sum(float(c["l"]) * float(c["w"]) for c in cracks)
    lateral_area = 2 * (b["W"] + b["H"]) * b["L"]
    depth = max((float(c["d"]) for c in cracks), default=0.)
    return torch.tensor([[len(cracks) / 10., opening_area / lateral_area,
                          depth / max(b["W"], b["H"])]], dtype=torch.float32)


def build_crack_edges(cracks, beam=None):
    b = beam or DEFAULT_BEAM
    if len(cracks) < 2:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 5), dtype=torch.float32)
    diagonal = np.sqrt(b["L"] ** 2 + b["W"] ** 2 + b["H"] ** 2)
    adjacent = {(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (3, 0), (2, 1), (3, 1)}
    centers = [crack_center_3d(c, b) for c in cracks]
    src, dst, features = [], [], []
    for i, ci in enumerate(cracks):
        for j, cj in enumerate(cracks):
            if i == j:
                continue
            fi, fj = int(ci["face"]), int(cj["face"])
            src.append(i); dst.append(j)
            features.append([float(fi == fj), float((fi, fj) in adjacent),
                abs(float(ci["z"]) - float(cj["z"])) / b["L"],
                np.linalg.norm(centers[i] - centers[j]) / diagonal,
                (float(ci["d"]) + float(cj["d"])) / max(b["W"], b["H"])])
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(features, dtype=torch.float32)
