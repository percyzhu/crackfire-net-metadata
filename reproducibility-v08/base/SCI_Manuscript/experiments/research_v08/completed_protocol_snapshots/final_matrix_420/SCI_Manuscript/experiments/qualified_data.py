"""Versioned loading of newly qualified fields-to-label datasets.

No archive fallback and no filtering based on scalar response variation.
Prescribed temperatures are evaluated from the actual solver amplitude table.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import physical_features as pf

FEATURE_VERSION = "fe-global-realized-amplitude-v3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_collection_digest(collection):
    """Bind identities to exact CPU input tensors, independently of dict order."""
    result = hashlib.sha256(b'qualified-tensor-collection-v1\n')
    for sid in sorted(collection):
        tensors = collection[sid]
        if isinstance(tensors, torch.Tensor):
            tensors = (tensors,)
        for index, tensor in enumerate(tensors):
            array = np.ascontiguousarray(tensor.detach().cpu().numpy())
            header = json.dumps([sid, index, array.dtype.str, list(array.shape)],
                                separators=(',', ':'), ensure_ascii=True).encode()
            result.update(len(header).to_bytes(8, 'little'))
            result.update(header)
            result.update(array.nbytes.to_bytes(8, 'little'))
            result.update(array.tobytes(order='C'))
    return result.hexdigest()


def geometry_id(case):
    payload = {"beam": case["beam"], "cracks": sorted(case["cracks"],
        key=lambda c: json.dumps(c, sort_keys=True))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def temporal_features(times, amplitude):
    times = np.asarray(times, dtype=float)
    amp = np.asarray(amplitude, dtype=float)
    if amp.ndim != 2 or amp.shape[1] != 2 or np.any(np.diff(amp[:, 0]) <= 0):
        raise ValueError("Invalid realized amplitude table")
    if amp[0, 0] > 0 or amp[-1, 0] < times[-1]:
        raise ValueError("Amplitude must explicitly include the complete solver period")
    if times[0] != 0 or times[-1] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid target evaluation times")
    # Integrate the actual piecewise-linear amplitude, not a coarse resampling.
    union = np.union1d(times, amp[:, 0])
    union = union[(union >= 0) & (union <= times[-1])]
    gas = np.interp(union, amp[:, 0], amp[:, 1])
    excess = gas - 20.
    integral = np.r_[0., np.cumsum(.5 * (excess[1:] + excess[:-1]) * np.diff(union))]
    temperature = np.interp(times, union, gas)
    # The prescribed scenario is known. A backward slope is used consistently.
    slope = np.r_[0., np.diff(temperature) / np.diff(times)]
    tf = np.column_stack((times / times[-1], temperature / 1200.,
        np.interp(times, union, integral) / (1000. * times[-1]), slope / 20.))
    if not np.isfinite(tf).all():
        raise ValueError("Nonfinite temporal features")
    return torch.tensor(tf, dtype=torch.float32)


def graph(case, times, amplitude):
    beam, cracks = case["beam"], case["cracks"]
    edge, attr = pf.build_crack_edges(cracks, beam)
    return (pf.encode_crack_node_features(cracks, beam), edge, attr,
        pf.encode_global_features(cracks, beam), temporal_features(times, amplitude))


class QualifiedDataset:
    def __init__(self, manifest, allow_software_fixture=False):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.sha256 = digest(self.path)
        self.fixture = self.manifest.get("purpose") == "SOFTWARE_VERIFICATION_ONLY"
        if self.fixture and not allow_software_fixture:
            raise ValueError("Manufactured software fixture is not a scientific dataset")
        if self.manifest.get("status") != "FROZEN_QUALIFIED":
            raise ValueError("Formal learning requires a frozen qualified dataset")
        self.records = self.manifest["cases"]
        ids = [c["sample_id"] for c in self.records]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Empty dataset or duplicate sample identity")
        self.by_id, self.graphs, self.targets = {}, {}, {}
        self.times = None
        for c in self.records:
            label = (self.path.parent / c["label_path"]).resolve()
            qualification = (self.path.parent / c["qualification_path"]).resolve()
            if digest(label) != c["label_sha256"] or digest(qualification) != c["qualification_sha256"]:
                raise ValueError("Dataset evidence hash mismatch: " + c["sample_id"])
            qa = json.loads(qualification.read_text(encoding="utf-8"))
            if qa.get("status") != "ADMITTED" or qa.get("sample_id") != c["sample_id"]:
                raise ValueError("Missing per-case admission: " + c["sample_id"])
            if qa.get("label_sha256") != c["label_sha256"]:
                raise ValueError("Qualification does not bind the exact label")
            if not self.fixture and (qa.get("minimum_temperature_C", -np.inf) < 19.9
                    or not qa.get("finite_complete_temperature_history", False)
                    or not qa.get("numerical_qualification_reference")):
                raise ValueError("Incomplete physical qualification: " + c["sample_id"])
            with np.load(label, allow_pickle=False) as data:
                times = np.asarray(data["time_s"], dtype=float)
                y = np.asarray(data["alpha"], dtype=float)
                amplitude = np.asarray(data["gas_amplitude"], dtype=float)
            if y.shape != times.shape or not np.isfinite(y).all() or np.any(y < 0) or np.any(y > 1+1e-7):
                raise ValueError("Invalid scalar labels; never clip them silently")
            if self.times is None:
                self.times = times
            if not np.array_equal(times, self.times):
                raise ValueError("Dataset evaluation times must be identical")
            if geometry_id(c) != c["geometry_id"]:
                raise ValueError("Geometry identity mismatch")
            sid = c["sample_id"]
            self.by_id[sid] = c
            self.graphs[sid] = graph(c, times, amplitude)
            self.targets[sid] = torch.tensor(y, dtype=torch.float32)

    def feature_tensor_sha256(self):
        return tensor_collection_digest(self.graphs)

    def target_tensor_sha256(self):
        return tensor_collection_digest(self.targets)

    def check_split(self, path):
        split = json.loads(Path(path).read_text(encoding="utf-8"))
        if split.get("dataset_sha256") != self.sha256:
            raise ValueError("Split refers to another dataset snapshot")
        if split.get("status") != "FROZEN_BEFORE_TRAINING":
            raise ValueError("Split is not frozen")
        memberships, geometry_sets = [], []
        for role in ("train", "validation", "test"):
            ids = split[role]
            if not ids or len(ids) != len(set(ids)) or any(i not in self.by_id for i in ids):
                raise ValueError("Invalid split membership: " + role)
            memberships.append(set(ids))
            geometry_sets.append({self.by_id[i]["geometry_id"] for i in ids})
        for i in range(3):
            for j in range(i):
                if memberships[i] & memberships[j] or geometry_sets[i] & geometry_sets[j]:
                    raise ValueError("Case or geometry leakage between splits")
        if set.union(*memberships) != set(self.by_id):
            raise ValueError("Every frozen case must be assigned or an explicit new subset frozen")
        return split
