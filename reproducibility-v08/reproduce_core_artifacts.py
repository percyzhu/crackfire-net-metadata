"""Verify released saved results and recreate the manuscript's core table and figure.

This entry point performs no training and never alters bundled data or model files.
It writes all generated outputs to a fresh directory below ``reproduced_outputs``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "base"
ADDENDUM = ROOT / "attention-addendum"

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def summary_row(path: Path, model: str):
    data = read_json(path)
    rows = [r for r in data["model_summaries"] if r["model"] == model]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {model} row in {path}")
    metric = rows[0]["metrics"]["equal_case_MAE"]
    return metric["mean"], metric["sample_sd"]

def run(command: list[str]):
    subprocess.run(command, cwd=ROOT, check=True)

def write_table(output: Path):
    base_eval = BASE / "SCI_Manuscript/experiments/research_v08/evaluation"
    attention_eval = ADDENDUM / "SCI_Manuscript/experiments/research_v09_set_attention/evaluation_complete60_v1"
    sources = {"Fire only": (base_eval, "fire_only"), "Matched set": (base_eval, "capacity_matched_deepsets"), "Sum graph": (base_eval, "gnn"), "Set attention (exploratory)": (attention_eval, "set_attention")}
    rows = []
    for label, (folder, model) in sources.items():
        ordinary = summary_row(folder / "iid997/summary.json", model)
        count = summary_row(folder / "lcro_9_15/summary.json", model)
        rows.append({"representation": label, "ordinary_test_mean_MAE": ordinary[0], "ordinary_test_sample_SD": ordinary[1], "larger_count_test_mean_MAE": count[0], "larger_count_test_sample_SD": count[1], "ordinary_test_percentage_points": f"{100*ordinary[0]:.3f} ± {100*ordinary[1]:.3f}", "larger_count_test_percentage_points": f"{100*count[0]:.3f} ± {100*count[1]:.3f}"})
    with (output / "Table3_core_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    table = ["| Representation | Ordinary test | Larger-count test |", "|---|---:|---:|"]
    table.extend(f"| {r['representation']} | {r['ordinary_test_percentage_points']} | {r['larger_count_test_percentage_points']} |" for r in rows)
    (output / "Table3_core_comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return rows

def read_count_values(path: Path, allowed_models: set[str]):
    result: dict[str, dict[int, list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model in allowed_models:
                result.setdefault(model, {}).setdefault(int(row["crack_count"]), []).append(float(row["MAE"]))
    return {m: {n: sum(v) / len(v) for n, v in counts.items()} for m, counts in result.items()}

def draw_count_figure(output: Path):
    base = read_count_values(BASE / "SCI_Manuscript/experiments/research_v08/evaluation/lcro_9_15/per_case_metrics.csv", {"fire_only", "capacity_matched_deepsets", "gnn"})
    attention = read_count_values(ADDENDUM / "SCI_Manuscript/experiments/research_v09_set_attention/evaluation_complete60_v1/lcro_9_15/per_case_metrics.csv", {"set_attention"})
    curves = [("Fire only", "fire_only", "#707070", base), ("Matched set", "capacity_matched_deepsets", "#357D9B", base), ("Sum graph", "gnn", "#0072B2", base), ("Set attention (exploratory)", "set_attention", "#A1518B", attention)]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 73 / 25.4))
    for label, model, color, values in curves:
        x = sorted(values[model]); ax.plot(x, [100 * values[model][n] for n in x], marker="o", ms=3.5, lw=1.4, color=color, label=label)
    ax.set(xlabel="Held-out crack count", ylabel="Trajectory MAE (percentage points)", xticks=list(range(9, 16)))
    ax.grid(axis="y", color="#E6ECEF", lw=0.7); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left"); ax.set_title("Fixed-weight prediction on the larger-count holdout", loc="left", weight="bold")
    fig.text(.125, .015, "All count-holdout cases are from source batch 005; crack count and source composition are confounded.", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for ext in ("pdf", "png"): fig.savefig(output / f"Figure_core_count_transfer.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--bootstrap", action="store_true"); parser.add_argument("--output", type=Path); args = parser.parse_args()
    output = args.output or ROOT / "reproduced_outputs" / dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(BASE / "verify_results.py"), "--output", str(output / "base_saved_results_verification.json")]
    if args.bootstrap: command.append("--bootstrap")
    run(command)
    run([sys.executable, str(ADDENDUM / "verify_addendum.py"), "--base", str(BASE), "--output", str(output / "attention_saved_results_verification.json")])
    rows = write_table(output); draw_count_figure(output)
    (output / "reproduction_summary.json").write_text(json.dumps({"status": "PASSED_RELEASED_SAVED_RESULTS_VERIFICATION_AND_ARTIFACT_REPRODUCTION", "base_checkpoints_verified": 420, "attention_checkpoints_verified": 60, "bootstrap_recomputed": args.bootstrap, "table_rows": rows, "generated": ["Table3_core_comparison.csv", "Table3_core_comparison.md", "Figure_core_count_transfer.pdf", "Figure_core_count_transfer.png"], "training_performed": False}, indent=2) + "\n", encoding="utf-8")
    print(output)

if __name__ == "__main__": main()
