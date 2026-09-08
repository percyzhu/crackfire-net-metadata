"""Bind an already-complete independent audit to its immutable nine-file snapshot.

Read only the requested protocol's integrity entries. Never creates aggregate or
engineering results, retrains, relaxes replay tolerance, or overwrites a report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    review = Path(__file__).resolve().parent
    root = review.parents[3]
    experiment = root / "SCI_Manuscript/experiments/research_v08"
    protocol = args.protocol
    assert protocol.startswith("loco_") and protocol.replace("_", "").isalnum()
    snapshot = experiment / "completed_protocol_snapshots" / f"{protocol}_35"
    manifest_path = snapshot / "snapshot_manifest.json"
    audit_path = review / f"{protocol}_independent_results_audit_v2.json"
    topology_path = review / f"{protocol}_same_weight_topology_audit.json"
    required = [manifest_path, audit_path, topology_path]
    if not all(path.exists() for path in required):
        print(json.dumps({"status": "WAITING_FOR_COMPLETE_INDEPENDENT_AUDITS_AND_SNAPSHOT", "performance_read": False}))
        return
    audit, topology = read(audit_path), read(topology_path)
    assert audit["runs"] == 35 and audit["protocol"] == protocol
    assert audit["status"] == "COMPLETE_PROTOCOL_ARITHMETIC_DEVELOPMENT_AUDIT_PASSED_NOT_FULL_MATRIX_OR_SUBMISSION_REVIEW"
    assert topology["status"] == "COMPLETE_PROTOCOL_TOPOLOGY_BINDINGS_AND_ARITHMETIC_CHECKED"
    assert topology["protocol"] == protocol and topology["paired_sparse_views"] == 30
    manifest = read(manifest_path)
    assert manifest["protocol"] == protocol and manifest["protocol_runs"] == 35
    entries = {item["name"]: item for item in manifest["files"]}
    statistics = ["per_case_metrics.csv", "strata_by_seed.csv", "summary.json", "time_metrics.csv", "topology_by_seed.csv"]
    engineering = [f"{protocol}_engineering_proxy_ci.json", f"{protocol}_engineering_proxy_ci.csv"]
    assert set(entries) == set(statistics + engineering + ["report.json", "run_integrity.json"])
    checked = []
    for name, item in entries.items():
        path = snapshot / name
        assert digest(path) == item["sha256"] and path.stat().st_size == item["bytes"], name
        checked.append({"name": name, "sha256": item["sha256"], "bytes": item["bytes"]})
    for name in statistics:
        assert digest(experiment / "evaluation" / protocol / name) == entries[name]["sha256"], name
    for name in engineering:
        assert digest(review / name) == entries[name]["sha256"], name
    summary_path = experiment / "evaluation" / protocol / "summary.json"
    assert audit["source_hashes"][str(summary_path)] == entries["summary.json"]["sha256"]
    assert audit["source_hashes"][str(review / engineering[0])] == entries[engineering[0]]["sha256"]

    # Verify frozen provenance independently of the running queue's later state.
    plan_path = experiment / "comparison_plan_v08_420.json"
    plan = read(plan_path)
    plan_hash = digest(plan_path)
    assert plan_hash == manifest["plan_sha256"]
    assert len(plan["source_sha256"]) == 12
    for relative, expected in plan["source_sha256"].items():
        assert digest(root / relative) == expected, relative
    wrapper_path = experiment / "io_recovery_patch1.py"
    amendment_path = experiment / "recovery_io_patch1/amendment.json"
    amendment = read(amendment_path)
    assert amendment["original_plan_sha256"] == plan_hash
    assert amendment["wrapper_sha256"] == digest(wrapper_path)
    runtime = manifest.get("runtime_verification", {})
    if "source_sha256" in runtime:
        assert runtime["source_sha256"] == plan["source_sha256"]
    for key, path in [("wrapper_sha256", wrapper_path), ("amendment_sha256", amendment_path)]:
        if key in runtime:
            assert runtime[key] == digest(path)

    # Ignore non-target entries from the full current integrity snapshot.
    integrity = read(snapshot / "run_integrity.json")
    runs = [item for item in integrity["audits"] if item["protocol"] == protocol]
    assert len(runs) == 35
    cells = {(item["model"], item["seed"]) for item in runs}
    assert cells == {(model, seed) for model in plan["models"] for seed in plan["seeds"]}
    replays = []
    for item in runs:
        assert item["status"] == "PASSED_BINDINGS"
        replays.append(item["computational_replay"])
        replays.extend(item["topology_computational_replay"].values())
    assert len(replays) == 65
    for replay in replays:
        assert replay["cpu_diagnostic_tolerance"] == 3e-6
        if replay["cpu_within_original_tolerance"]:
            assert replay["cpu_max_abs_difference"] < 3e-6
            assert replay["status"] == "CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE"
        else:
            assert replay["cpu_max_abs_difference"] >= 3e-6
            assert replay["same_backend_gpu_bitwise_equal"] is True
            assert replay["gpu_max_abs_difference"] == 0.0
            assert replay["status"] == "GPU_IDENTITY_EXACT_CPU_BACKEND_VARIATION_RECORDED"
    dataset = read(Path(plan["manifest_path"]))
    assert digest(Path(plan["manifest_path"])) == plan["manifest_sha256"]
    cases = [item for item in dataset["cases"] if item["fire_family"] == protocol.removeprefix("loco_")]
    assert len(cases) == audit["independent_geometries"]
    result = {
        "status": "SNAPSHOT_BOUND_TO_INDEPENDENT_COMPLETE_PROTOCOL_AUDIT",
        "protocol": protocol,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256": digest(manifest_path),
        "snapshot_files_checked": checked,
        "all7_protocol_outputs_identical_to_audited_live_files": True,
        "current12_frozen_sources_plan_wrapper_amendment_verified": True,
        "complete_runs": 35,
        "primary_and_sparse_replay_calls": len(replays),
        "replay_status_counts": dict(Counter(item["status"] for item in replays)),
        "max_CPU_replay_difference": max(item["cpu_max_abs_difference"] for item in replays),
        "unique_prescribed_fire_parameter_objects": len({json.dumps(item["fire"], sort_keys=True) for item in cases}),
        "other_family_prediction_scores_inspected": False,
        "manuscript_or70run_package_modified": False,
        "review_files_sha256": {path.name: digest(path) for path in [audit_path, topology_path]},
        "auditor_script_sha256": digest(Path(__file__)),
        "submission_pass": False,
    }
    output = args.output or review / f"{protocol}_snapshot_binding_review.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "protocol": protocol, "output": str(output), "replay_status_counts": result["replay_status_counts"], "max_CPU_replay_difference": result["max_CPU_replay_difference"]}))


if __name__ == "__main__":
    main()
