"""Read-only legacy checkpoint smoke audit; writes only an independent report.

Usage: python run_readiness.py --workspace D:/Work/Abstract
No dataset cache is created, no training is performed and no packages installed.
Feature functions are selected from the archived source AST, preserving legacy
normalization and prescribed-fire definitions. Their source hashes are recorded.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from models import Surrogate, from_checkpoint_config
import physical_features


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions_from_archive(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise RuntimeError("Archived feature-function inventory changed")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def legacy_functions(workspace):
    ns = {"torch": torch, "np": np, "math": math,
          "DEFAULT_BEAM": {"L": 1.5, "W": 0.14, "H": 0.2}}
    functions_from_archive(workspace / "1_代码/src/data/dataset.py",
        ["encode_crack_node_features", "encode_global_features", "build_crack_edges"], ns)
    functions_from_archive(workspace / "1_代码/src/gnn/model.py",
        ["compute_fire_curve", "make_time_features"], ns)
    return ns


def graph(ns, cracks, curve):
    edges, attrs = ns["build_crack_edges"](cracks)
    return (ns["encode_crack_node_features"](cracks), edges, attrs,
            ns["encode_global_features"](cracks),
            ns["make_time_features"](61, 3600., curve["type"], curve["params"]).squeeze(0))


def collate(graphs, device="cpu"):
    xs, edges, attrs, globals_, batches, temporal = [], [], [], [], [], []
    offset = 0
    for i, (x, edge, attr, global_, tf) in enumerate(graphs):
        xs.append(x); edges.append(edge + offset); attrs.append(attr)
        globals_.append(global_); temporal.append(tf)
        batches.append(torch.full((len(x),), i, dtype=torch.long))
        offset += len(x)
    return tuple(value.to(device) for value in
        (torch.cat(xs), torch.cat(edges, dim=1), torch.cat(attrs), torch.cat(globals_),
         torch.cat(batches), torch.stack(temporal)))


def run(workspace):
    torch.manual_seed(20260906)
    np.random.seed(20260906)
    torch.set_num_threads(4)
    ns = legacy_functions(workspace)
    raw = workspace / "Abaqus/processed_pilot/raw"
    metadata_path = raw / "batch_run_log.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # Select actual cases to cover the original empty-edge branch and batching.
    indices = [next(i for i, c in enumerate(metadata["crack_params"]) if len(c) == n)
               for n in (1, 2, 15)]
    graphs, targets, sample_info = [], [], []
    for index in indices:
        record = metadata["results"][index]
        if record["status"] != "success":
            raise ValueError("Smoke case is not successful")
        sample_id = "sample_%04d" % record["sample_id"]
        path = raw / (sample_id + ".npz")
        with np.load(path, allow_pickle=False) as data:
            if "charring_ratios" not in data or "time_steps" not in data:
                raise ValueError(f"Missing required legacy label in {path}")
            series = data["charring_ratios"]
            times = data["time_steps"]
            if not np.isfinite(series).all() or not np.isfinite(times).all():
                raise ValueError(f"Non-finite label in {path}")
            if np.any(np.diff(times) <= 0) or times[0] > 0 or times[-1] < 3600:
                raise ValueError(f"Invalid or incomplete history in {path}")
            y = np.interp(np.linspace(0, 3600, 61), times, np.minimum.accumulate(series))
        targets.append(y)
        graphs.append(graph(ns, metadata["crack_params"][index], metadata["fire_curves"][index]))
        sample_info.append({"sample_id": sample_id, "metadata_index": index,
                            "num_cracks": len(metadata["crack_params"][index]),
                            "curve_type": metadata["fire_curves"][index]["type"]})
    report = {
        "purpose": "readiness and tiny inference/backward checks; not a held-out benchmark",
        "environment": {"python": sys.executable, "version": platform.python_version(),
            "torch": torch.__version__, "torch_built_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pyg_installed": importlib.util.find_spec("torch_geometric") is not None},
        "source_hashes": {str(p.relative_to(workspace)): sha256(p) for p in
            [workspace / "1_代码/src/gnn/model.py", workspace / "1_代码/src/data/dataset.py",
             workspace / "1_代码/src/gnn/train.py", metadata_path,
             Path(__file__).with_name("models.py"), Path(__file__).with_name("physical_features.py")]},
        "samples": sample_info, "checkpoints": [], "baseline_smoke": [],
        "mathematical_equivalence_status": "Equations and state keys reviewed; direct PyG comparison pending",
    }
    for path in sorted((workspace / "3_最终成果/模型").glob("*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = from_checkpoint_config(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            together = model(*collate(graphs)).squeeze(-1)
            singles = torch.cat([model(*collate([g])).squeeze(-1) for g in graphs])
            reversed_batch = model(*collate(list(reversed(graphs)))).squeeze(-1).flip(0)
            reverse = graph(ns, list(reversed(metadata["crack_params"][indices[2]])),
                            metadata["fire_curves"][indices[2]])
            permuted = model(*collate([reverse])).squeeze(-1)
            # A sum loop is an independent check on destination-based aggregation.
            layer = model.structure_head.processors[0]
            x = torch.randn(3, 64); edge = torch.tensor([[0, 1, 2], [1, 2, 1]])
            e = torch.randn(3, 64)
            xn, en = layer(x, edge, e)
            manual = torch.zeros_like(x)
            for j in range(edge.shape[1]):
                manual[edge[1, j]] += en[j]
            expected_x = x + layer.node_mlp(torch.cat([x, manual], -1))
            layer_error = float((xn - expected_x).abs().max())
        gpu_error = None
        if torch.cuda.is_available():
            model.cuda()
            with torch.no_grad():
                gpu_pred = model(*collate(graphs, "cuda")).cpu().squeeze(-1)
            gpu_error = float((gpu_pred - together).abs().max())
        per_case_batch_errors = (together - singles).abs().max(dim=1).values.tolist()
        fixed = from_checkpoint_config(checkpoint["config"])
        fixed.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for layer in fixed.structure_head.processors:
            layer.isolated_node_policy = "always_update"
        fixed.eval()
        with torch.no_grad():
            fixed_together = fixed(*collate(graphs)).squeeze(-1)
            fixed_singles = torch.cat([fixed(*collate([g])).squeeze(-1) for g in graphs])
        report["checkpoints"].append({
            "path": str(path.relative_to(workspace)), "sha256": sha256(path),
            "epoch": checkpoint["epoch"], "config": checkpoint["config"],
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "strict_state_load": True, "reported_val_mse_sum_over_time": checkpoint["val_mse"],
            "corresponding_mean_per_sample_time_if_61_steps": checkpoint["val_mse"] / 61,
            "all_finite": bool(torch.isfinite(together).all()),
            "output_range": [float(together.min()), float(together.max())],
            "batch_vs_single_max_absolute_difference_per_case": per_case_batch_errors,
            "batch_vs_single_mean_absolute_difference_per_case": (together - singles).abs().mean(dim=1).tolist(),
            "batch_order_reversal_max_absolute_difference_per_case": (together - reversed_batch).abs().max(dim=1).values.tolist(),
            "batch_vs_single_time_of_max_difference_seconds": ((together - singles).abs().argmax(dim=1) * 60).tolist(),
            "permutation_max_absolute_difference_15_cracks": float((permuted - singles[2:3]).abs().max()),
            "message_sum_loop_max_absolute_difference": layer_error,
            "cpu_gpu_max_absolute_difference": gpu_error,
            "always_update_batch_vs_single_max_absolute_difference_per_case":
                (fixed_together - fixed_singles).abs().max(dim=1).values.tolist(),
            "always_update_status": "invariance-only smoke with legacy weights; retraining required for new results",
        })
    target = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(-1)
    physical_ns = dict(ns)
    for key in ("encode_crack_node_features", "encode_global_features", "build_crack_edges"):
        physical_ns[key] = getattr(physical_features, key)
    physical_graphs = [graph(physical_ns, metadata["crack_params"][i], metadata["fire_curves"][i])
                       for i in indices]
    for feature_version, current_graphs in (("legacy-v1", graphs),
                                           (physical_features.FEATURE_VERSION, physical_graphs)):
        for mode in ("fire_only", "global_stats", "deepsets", "gnn", "gnn_zero_edge_features"):
            torch.manual_seed(20260906)
            model = Surrogate(mode=mode)
            pred = model(*collate(current_graphs))
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            report["baseline_smoke"].append({"mode": mode, "feature_version": feature_version,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "forward_shape": list(pred.shape), "finite_loss": bool(torch.isfinite(loss)),
                "finite_gradients": all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                        for p in model.parameters())})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("readiness_report.json"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(workspace / "SCI_Manuscript/experiments"):
        raise ValueError("Report destination must remain within SCI_Manuscript/experiments")
    report = run(workspace)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "environment": report["environment"],
                      "checkpoints": [{k: v for k, v in c.items() if k != "config"}
                                      for c in report["checkpoints"]]},
                     ensure_ascii=False, indent=2))
