"""Parameter-matched DeepSets control; no message passing or unused parameters.

Run this file for an architecture-only budget search and three-case smoke check.
No optimizer, training, checkpoint change, or scientific performance evaluation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

try:
    from .models import MLP, Surrogate
except ImportError:
    from models import MLP, Surrogate


class CapacityMatchedDeepSets(Surrogate):
    """Independent node MLP -> mean pool -> unchanged global/time/decoder.

    The ordinary DeepSets mode remains untouched. Only its crack encoder is
    replaced here. Edge tensors accepted by the shared interface are ignored.
    """

    def __init__(self, node_hidden_dim=232, node_mlp_layers=4,
                 hidden_dim=64, temporal_type="lstm", time_in_dim=4,
                 mlp_layers=2):
        super().__init__(mode="deepsets", hidden_dim=hidden_dim,
                         temporal_type=temporal_type, time_in_dim=time_in_dim,
                         mlp_layers=mlp_layers)
        if node_hidden_dim < 1 or node_mlp_layers < 2:
            raise ValueError("Node MLP needs a positive width and at least two affine layers")
        self.structure_head.crack_encoder = MLP(
            in_dim=11, out_dim=hidden_dim, hidden_dim=node_hidden_dim,
            num_layers=node_mlp_layers)
        self.capacity_config = {"node_hidden_dim": node_hidden_dim,
                                "node_mlp_layers": node_mlp_layers}


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def architecture_search():
    """Minimize count mismatch only; no sample, target, or score is inspected."""
    fixed = count(Surrogate(mode="deepsets")) - count(
        Surrogate(mode="deepsets").structure_head.crack_encoder)
    target = count(Surrogate(mode="gnn"))
    candidates = []
    # Width multiples of 8; moderate widths/depths avoid a one-layer huge bottleneck.
    for depth in range(2, 7):
        for width in range(32, 513, 8):
            node_count = (depth - 2) * width**2 + (3 * depth + 72) * width + 64
            total = fixed + node_count
            candidates.append({"node_mlp_layers": depth, "node_hidden_dim": width,
                "node_parameters": node_count, "total_parameters": total,
                "signed_difference": total - target,
                "absolute_difference": abs(total - target),
                "relative_difference": (total - target) / target,
                "within_one_percent": abs(total - target) <= target * .01})
    candidates.sort(key=lambda c: (c["absolute_difference"], c["node_mlp_layers"], c["node_hidden_dim"]))
    return target, fixed, candidates


def smoke(workspace, output):
    try:
        from .run_readiness import legacy_functions, graph, collate
        from . import physical_features
    except ImportError:
        from run_readiness import legacy_functions, graph, collate
        import physical_features

    torch.set_num_threads(4)
    torch.manual_seed(20260906)
    target_count, fixed_count, candidates = architecture_search()
    selected = candidates[0]
    kwargs = {k: selected[k] for k in ("node_hidden_dim", "node_mlp_layers")}
    model = CapacityMatchedDeepSets(**kwargs).eval()
    if count(model) != selected["total_parameters"] or not selected["within_one_percent"]:
        raise AssertionError("Parameter-budget mismatch")
    original_parameters = {mode: count(Surrogate(mode=mode)) for mode in
        ("fire_only", "global_stats", "deepsets", "gnn", "gnn_zero_edge_features")}
    forbidden = [name for name, _ in model.named_parameters()
                 if any(token in name for token in ("processors", "edge_encoder", "padding"))]
    if forbidden or any(not p.requires_grad for p in model.parameters()):
        raise AssertionError("Unexpected graph/unused/frozen parameters")

    metadata_path = workspace / "Abaqus/processed_pilot/raw/batch_run_log.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    indices = [next(i for i, cracks in enumerate(metadata["crack_params"])
                    if len(cracks) == n and metadata["results"][i]["status"] == "success")
               for n in (1, 2, 15)]
    ns = legacy_functions(workspace)
    for name in ("encode_crack_node_features", "encode_global_features", "build_crack_edges"):
        ns[name] = getattr(physical_features, name)
    graphs = [graph(ns, metadata["crack_params"][i], metadata["fire_curves"][i]) for i in indices]
    reversed_nodes = [graph(ns, list(reversed(metadata["crack_params"][i])), metadata["fire_curves"][i])
                      for i in indices]
    with torch.no_grad():
        joint = model(*collate(graphs)).squeeze(-1)
        alone = torch.cat([model(*collate([g])).squeeze(-1) for g in graphs])
        reversed_batch = model(*collate(list(reversed(graphs)))).squeeze(-1).flip(0)
        permuted = model(*collate(reversed_nodes)).squeeze(-1)
        # Remove all edges: a set model must not use relational input at all.
        edge_free = [(g[0], torch.empty((2, 0), dtype=torch.long),
                      torch.empty((0, 5), dtype=torch.float32), g[3], g[4]) for g in graphs]
        no_edges = model(*collate(edge_free)).squeeze(-1)
    errors = {"node_order_reversal": (joint-permuted).abs(),
              "batch_vs_single": (joint-alone).abs(),
              "batch_order_reversal": (joint-reversed_batch).abs(),
              "edge_removal": (joint-no_edges).abs()}
    tolerance = 1e-6
    if not all(bool(torch.isfinite(x).all()) and float(x.max()) <= tolerance for x in errors.values()):
        raise AssertionError("Set/batch invariance smoke failed")
    # Synthetic bounded targets exercise backward computation only; no FEM score.
    model.train()
    pred = model(*collate(graphs))
    synthetic_target = torch.linspace(1., 0., 61).reshape(1, 61, 1).expand(3, -1, -1)
    loss = torch.nn.functional.mse_loss(pred, synthetic_target)
    loss.backward()
    missing_gradients = [name for name, p in model.named_parameters() if p.grad is None]
    finite_gradients = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    if not bool(torch.isfinite(loss)) or not finite_gradients or missing_gradients:
        raise AssertionError("Forward/backward smoke failed")

    config = {"name": "capacity_matched_deepsets", "status": "implemented_and_smoke_checked_not_trained",
        "model_class": "capacity_matched_set.CapacityMatchedDeepSets", "model_kwargs": {
            **kwargs, "hidden_dim": 64, "temporal_type": "lstm", "time_in_dim": 4, "mlp_layers": 2},
        "feature_version": physical_features.FEATURE_VERSION,
        "fire_features": "legacy-v1 prescribed-fire four features; held identical across baselines",
        "label_version": "not selected; scientific target freeze pending; smoke used synthetic targets only",
        "pooling": "arithmetic mean over nodes within each graph",
        "node_feature_dim": 11, "global_feature_dim": 3,
        "architecture_selection": {"criterion": "minimum absolute parameter difference, then smaller depth, then width",
            "node_width_grid": "32..512 inclusive, step 8", "node_affine_depth_grid": [2, 3, 4, 5, 6],
            "target_parameters": target_count, "selected_parameters": count(model),
            "relative_difference": selected["relative_difference"],
            "uses_validation_or_test_scores": False},
        "shared_training_protocol": "Use the same frozen data, splits, seeds, optimizer, budget and selection rule as GNN; no training done here"}
    report = {"purpose": "capacity and computation checks only; no scientific prediction scores",
        "environment": {"python": sys.executable, "torch": torch.__version__, "device": "cpu", "threads": 4},
        "seed": 20260906, "config": config, "original_mode_parameter_counts": original_parameters,
        "budget": {"target_gnn": target_count, "matched_set": count(model), "fixed_shared": fixed_count,
                   "node_mlp": count(model.structure_head.crack_encoder),
                   "structural_branch": count(model.structure_head), "temporal_head": count(model.temporal_head),
                   "decoder": count(model.decoder), "signed_difference": count(model)-target_count},
        "no_graph_or_padding_parameter_names": not forbidden,
        "all_parameters_trainable_and_received_gradients": not missing_gradients,
        "forward_shape": list(pred.shape), "forward_finite": bool(torch.isfinite(pred).all()),
        "backward_finite": finite_gradients, "invariance_tolerance": tolerance,
        "source_hashes": {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(__file__).with_name("models.py"),
                                    Path(__file__).with_name("physical_features.py"), metadata_path)},
        "cases": []}
    for index, original in enumerate(indices):
        report["cases"].append({"sample_id": "sample_%04d" % metadata["results"][original]["sample_id"],
            "metadata_index": original, "num_cracks": len(metadata["crack_params"][original]),
            "curve_type": metadata["fire_curves"][original]["type"],
            "errors": {name: {"max_abs": float(value[index].max()), "mae": float(value[index].mean())}
                       for name, value in errors.items()}})
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "capacity_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0])); writer.writeheader(); writer.writerows(candidates)
    print(json.dumps({"budget": report["budget"], "cases": report["cases"],
                      "forward_finite": report["forward_finite"], "backward_finite": report["backward_finite"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("capacity_matched_set"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(workspace / "SCI_Manuscript/experiments"):
        raise ValueError("Output must remain under SCI_Manuscript/experiments")
    smoke(workspace, output)
