"""Generate FEM validation figures for thesis (PDF, Chinese labels).

Figures:
  F1: mesh_sensitivity.pdf  — 5 mesh sizes vs experimental T-t curves
  F2: cp_comparison.pdf     — EC5 Cp(T) original vs smoothed + T-t validation

Usage:
    conda run -n abaqus-gnn python scripts/generate_fig_fem.py
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from scripts.figures.plot_config import setup_chinese_style, savefig, THESIS_FIG_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

setup_chinese_style()

# Paths. Override FEM_DIR when validation data is stored outside the archive.
FEM_DIR = os.environ.get("FEM_DIR", "data/fem/mesh_sensitivity")
EXP_PATH = "src/data/T-t_experimental_curve.xlsx"

# ---- EC5 specific heat data (from fire_model.py) ----
EC5_CP_ORIGINAL = [
    (1530, 20), (1770, 99), (13600, 100), (13600, 110), (13600, 120),
    (2120, 121), (2000, 200), (1620, 250), (710, 300), (750, 350),
    (1000, 400), (1400, 600), (1650, 800), (1650, 1200),
]
EC5_CP_SMOOTH = [
    (1530, 20), (1770, 80), (2516, 90), (7648, 100), (11240, 105),
    (12267, 110), (11240, 115), (7648, 120), (2516, 130), (2000, 140),
    (2000, 200), (1620, 250), (710, 300), (750, 350),
    (1000, 400), (1400, 600), (1650, 800), (1650, 1200),
]


def load_fem_curves(json_path):
    with open(json_path) as f:
        data = json.load(f)
    times = np.array(data["times"]) / 60.0
    curves = {}
    for key, info in data["depth_curves"].items():
        depth_mm = info.get("depth_mm", info.get("depth_cm", 0) * 10)
        temps = [t if t is not None else float("nan") for t in info["temperatures"]]
        curves[depth_mm] = (times, np.array(temps))
    return curves


def load_experimental():
    df = pd.read_excel(EXP_PATH, sheet_name="Sheet1")
    exp_t = df["t"].values / 60.0
    exp_curves = {
        15: (df["T15a"].values + df["T15b"].values) / 2,
        30: (df["T30a"].values + df["T30b"].values) / 2,
        40: (df["T40a"].values + df["T40b"].values) / 2,
    }
    return exp_t, exp_curves, df


# =================================================================
# F1: Mesh sensitivity (5 seeds vs experiment)
# =================================================================
def plot_mesh_sensitivity():
    print("[F1] mesh_sensitivity.pdf")
    seeds = ["50mm", "30mm", "20mm", "15mm", "10mm"]
    exp_t, exp_curves, df = load_experimental()
    exp_depth_mm = {15: "15mm", 30: "30mm", 40: "40mm"}

    fem = {}
    for seed in seeds:
        p = os.path.join(FEM_DIR, f"temp_{seed}.json")
        if os.path.exists(p):
            fem[seed] = load_fem_curves(p)
        else:
            print(f"  WARNING: {p} not found, skipping")

    if not fem:
        print("  No FEM data — skipping F1")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(seeds)))
    ls_map = {"50mm": "-", "30mm": "--", "20mm": "-.", "15mm": ":", "10mm": "-"}
    seed_cn = {"50mm": "50 mm", "30mm": "30 mm", "20mm": "20 mm",
               "15mm": "15 mm", "10mm": "10 mm"}

    for idx, depth in enumerate([15, 30, 40]):
        ax = axes[idx]
        col_a = f"T{depth}a"; col_b = f"T{depth}b"
        if col_a in df.columns:
            ax.plot(exp_t, df[col_a].values, color="gray", alpha=0.35, lw=0.8,
                    label="实验组 a")
            ax.plot(exp_t, df[col_b].values, color="gray", alpha=0.35, lw=0.8,
                    label="实验组 b")
        ax.plot(exp_t, exp_curves[depth], color="black", lw=2.2,
                label="实验均值")

        for si, seed in enumerate(seeds):
            if seed not in fem:
                continue
            best_d, best_diff = None, 999
            for d_mm in fem[seed]:
                diff = abs(d_mm - depth)
                if diff < best_diff:
                    best_diff = diff; best_d = d_mm
            if best_d is not None and best_diff <= 5.0:
                t, temps = fem[seed][best_d]
                ax.plot(t, temps, lw=1.8, ls=ls_map.get(seed, "-"),
                        color=colors[si],
                        label=f"FEM {seed_cn[seed]}（d={best_d:.0f} mm）")

        ax.set_title(f"深度 = {depth} mm", fontsize=16)
        ax.set_xlabel("时间（min）")
        if idx == 0:
            ax.set_ylabel("温度（°C）")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)

    plt.tight_layout()
    savefig(fig, "mesh_sensitivity")


# =================================================================
# F2: Cp comparison — two panels
#   Left:  Cp(T) curves (original EC5 stepwise vs smoothed bell)
#   Right: Temperature validation at 15mm depth (smooth vs original vs exp)
# =================================================================
def plot_cp_comparison():
    print("[F2] cp_comparison.pdf")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={'width_ratios': [1, 1.1]})

    # ---- Left panel: Cp(T) curves ----
    T_orig = [t for _, t in EC5_CP_ORIGINAL]
    Cp_orig = [c for c, _ in EC5_CP_ORIGINAL]
    T_smooth = [t for _, t in EC5_CP_SMOOTH]
    Cp_smooth = [c for c, _ in EC5_CP_SMOOTH]

    ax1.plot(T_orig, Cp_orig, color='#1f77b4', lw=2.0, marker='.',
             markersize=4,
             label='EC5 原始（阶跃）')
    ax1.plot(T_smooth, Cp_smooth, color='#d62728', lw=2.0, marker='.',
             markersize=4, label='面积守恒平滑')

    ax1.axvspan(80, 140, alpha=0.08, color='orange')
    ax1.annotate('水分蒸发\n潜热峰', xy=(110, 13000), fontsize=9,
                 ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                           ec='orange', alpha=0.8))

    ax1.set_xlabel("温度（°C）")
    ax1.set_ylabel("比热容（J/(kg·K)）")
    ax1.set_title("(a) EC5 比热容曲线对比")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 400)
    ax1.set_ylim(0, 15000)
    ax1.grid(True, alpha=0.3)

    # ---- Right panel: Temperature-time validation ----
    smooth_path = os.path.join(FEM_DIR, "temp_20mm.json")
    orig_path = os.path.join(FEM_DIR, "temp_20mm_origcp.json")

    if os.path.exists(smooth_path) and os.path.exists(orig_path):
        smooth = load_fem_curves(smooth_path)
        orig = load_fem_curves(orig_path)
        exp_t, exp_curves, _ = load_experimental()

        depth = 15
        ax2.plot(exp_t, exp_curves[depth], 'k-', lw=2.2, label='实验均值')
        for label, data, color, ls in [
            ("平滑 Cp（410 步）", smooth, '#d62728', "--"),
            ("原始 Cp（4828 步）", orig, '#1f77b4', ":"),
        ]:
            if depth in data:
                t, temps = data[depth]
                ax2.plot(t, temps, color=color, lw=2, ls=ls, label=label)

                fem_interp = np.interp(exp_t, t, temps, left=temps[0],
                                       right=temps[-1])
                n = min(len(exp_curves[depth]), len(fem_interp))
                rmse = np.sqrt(np.mean((fem_interp[:n] - exp_curves[depth][:n])**2))
                print(f"    {label}: RMSE@{depth}mm = {rmse:.1f}°C")

        ax2.set_xlabel("时间（min）")
        ax2.set_ylabel("温度（°C）")
        ax2.set_title(f"(b) 平滑化验证（20 mm 种子，深度 {depth} mm）")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 60)
    else:
        ax2.text(0.5, 0.5, "FEM 数据文件不可用\n(E 盘)", transform=ax2.transAxes,
                 ha='center', va='center', fontsize=14, color='gray')
        ax2.set_title("(b) 平滑化验证")

    plt.tight_layout()
    savefig(fig, "cp_comparison")


# =================================================================
if __name__ == "__main__":
    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {THESIS_FIG_DIR}\n")
    plot_mesh_sensitivity()
    plot_cp_comparison()
    print("\nDone — F1, F2 generated.")
