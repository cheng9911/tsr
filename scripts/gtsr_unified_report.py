"""Render the complete comparison and preserve diagnostics without selecting cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gtsr_unified_eval import ARMS, METHODS, ROOT, SE3_SEEDS, sha  # noqa: E402

LABELS = {
    "fixed": "Fixed",
    "frame": "Frame",
    "scalar_finite_gp": "Scalar GP finite",
    "frame_affine": "Frame affine",
    "scalar_affine_gp": "Scalar GP affine",
    "full_affine": "Full affine",
    "diagonal_affine": "Diagonal affine",
    "context_rbf": "Context RBF",
    "tpgmm_additive": "TP-GMM additive",
    "tpgmm_se3": "TP-GMM SE(3)",
    "pdiag": "Pdiag finite",
}
DISPLAY_METRICS = {
    "local_action_mm": "局部动作",
    "anchored_pose_mm": "共同名义位姿",
    "native_pose_mm": "原生位姿",
    "anchored_task_proxy_mm": "共同名义任务代理",
    "native_task_proxy_mm": "原生任务代理",
}


def save_figure(fig, folder, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Root directory containing results/")
    args = parser.parse_args()
    results = args.out / "results"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["complete"] and summary["status_counts"] == {"ok": 396, "failed": 0}, summary["status_counts"]
    rows = summary["summary"]
    cells = [json.loads(p.read_text()) for p in sorted((results / "cells").glob("*.json"))]
    assert len(cells) == 396 and len({(r["arm"], r["seed"], r["subset_id"], r["method"]) for r in cells}) == 396
    assert all(sha(p) == digest for p, digest in summary["manifest"]["source_sha256"].items())
    assert len(list((results / "cells").glob("*.npz"))) == 396
    for row in rows:
        assert row["successful_cells"] == 18 and row["failed_cells"] == 0
    lookup = {(r["arm"], r["method"]): r for r in rows}
    table = [
        "# 完整比较表",
        "",
        "单位：mm-equivalent；均值 ± 跨 seed 样本 SD（n=3）。越低越好。",
        "先在每个 seed 内平均六个训练子集；不能将 18 个单元视为独立重复。",
        "",
    ]
    for group, description in (
        ("mixed", "留出 mixed（每单元 30 条）"),
        ("isolated_equal_generators", "isolated 六 generator 等权"),
        ("isolated", "isolated 14 条条件等权（yaw 有四条，其余各两条）"),
    ):
        for arm in ARMS:
            table += [
                f"## {arm}：{description}",
                "",
                "| 方法 | " + " | ".join(DISPLAY_METRICS.values()) + " |",
                "|---|" + "---:|" * len(DISPLAY_METRICS),
            ]
            for method in METHODS:
                item = lookup[(arm, method)]["groups"][group]
                table.append(
                    "| "
                    + LABELS[method]
                    + " | "
                    + " | ".join(f"{item['mean'][m]:.3f} ± {item['sd'][m]:.3f}" for m in DISPLAY_METRICS)
                    + " |"
                )
            table.append("")
    table += [
        "## yaw 单轴干预的阶段分解",
        "",
        "共同名义轨道；pre 为 phase 0–1，post 为 phase 2–3。",
        "圆孔全程都采用轴向对称任务代理。阶段标签是 oracle 标签，不能当作预测器已识别释放事件。",
        "",
        "| arm | 方法 | pre 位姿 | pre 任务代理 | post 位姿 | post 任务代理 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    yaw_rows = []
    for arm in ARMS:
        for method in METHODS:
            selected = [r for r in cells if r["arm"] == arm and r["method"] == method]
            values = {}
            for interval, phases in (("pre", (0, 1)), ("post", (2, 3))):
                for metric in ("anchored_pose_mm", "anchored_task_proxy_mm"):
                    values[f"{interval}_{metric}"] = float(
                        np.mean([r["summary"]["yaw"]["phase_means"][str(q)][metric] for r in selected for q in phases])
                    )
            yaw_rows.append({"arm": arm, "method": method, **values})
            table.append(f"| {arm} | {LABELS[method]} | " + " | ".join(f"{v:.3f}" for v in values.values()) + " |")
    (args.out / "tables.md").write_text("\n".join(table) + "\n")

    paired = []
    for arm in ARMS:
        for group in ("mixed", "isolated_equal_generators", "isolated", "yaw"):
            for metric in DISPLAY_METRICS:
                pdiag = lookup[(arm, "pdiag")]["groups"][group]
                for method in METHODS:
                    if method == "pdiag":
                        continue
                    other = lookup[(arm, method)]["groups"][group]
                    differences = {
                        str(s): pdiag["per_seed"][str(s)][metric] - other["per_seed"][str(s)][metric] for s in SE3_SEEDS
                    }
                    paired.append(
                        dict(
                            arm=arm,
                            group=group,
                            metric=metric,
                            baseline=method,
                            pdiag_minus_baseline=differences,
                            mean_difference=float(np.mean(list(differences.values()))),
                            seeds_pdiag_lower=sum(d < 0 for d in differences.values()),
                        )
                    )

    historical_path = ROOT / "docs/superpowers/specs/2026-09-08-gtsr-sprint-f1-adequacy-source-data.json"
    historical = json.loads(historical_path.read_text())
    drift = []
    for old in historical["per_seed"]:
        selected = [c for c in cells if c["method"] == "pdiag" and c["arm"] == old["arm"] and c["seed"] == old["seed"]]
        assert len(selected) == 6
        new = float(np.mean([c["summary"][old["generator"]]["mean"]["world_action_mm"] for c in selected]))
        drift.append(
            dict(
                arm=old["arm"],
                seed=old["seed"],
                generator=old["generator"],
                historical_mm=old["response_error"] * 1000,
                rebuilt_mm=new,
                difference_mm=new - old["response_error"] * 1000,
            )
        )
    diagnostic = {
        "comparison_run_hash": summary["manifest"]["run_hash"],
        "all_source_hashes_unchanged": True,
        "maximum_passive_coordinate_difference_mm": max(c["invariance_max_abs"] for c in cells),
        "historical_source_sha256": sha(historical_path),
        "historical_response_reproduction": drift,
        "maximum_historical_error_difference_mm": max(abs(r["difference_mm"]) for r in drift),
        "yaw_phase_breakdown": yaw_rows,
        "paired_seed_differences": paired,
        "warning_cells": [
            {k: c[k] for k in ("arm", "seed", "subset_id", "method", "warnings")} for c in cells if c["warnings"]
        ],
    }
    (results / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for i, arm in enumerate(ARMS):
        for j, (metric, title) in enumerate(
            (
                ("local_action_mm", "Local response"),
                ("anchored_pose_mm", "Pose: shared nominal"),
                ("native_pose_mm", "Pose: native prediction"),
            )
        ):
            ax = axes[i, j]
            for k, method in enumerate(METHODS):
                item = lookup[(arm, method)]["groups"]["mixed"]
                color = "#D55E00" if method == "pdiag" else "#0072B2"
                ax.errorbar(item["mean"][metric], k, xerr=item["sd"][metric], fmt="o", color=color, capsize=3)
                ax.plot([v[metric] for v in item["per_seed"].values()], [k] * 3, "|", color=color, alpha=0.6)
            ax.set(
                yticks=range(len(METHODS)),
                yticklabels=[LABELS[m] for m in METHODS],
                xlabel="mm-equivalent (lower is better)",
                title=f"{arm}: {title}",
                xlim=(0, None),
            )
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Held-out mixed conditions | 30 train / 30 test per cell | mean ± seed SD (n=3)")
    save_figure(fig, figures, "unified_comparison")

    preference = json.loads((results / "task_preference.json").read_text())
    probes = preference["physical_probes"]["attempts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    pre = probes[:4]
    axes[0].bar(range(4), [r["final_clearance_mm"] for r in pre], color=["#009E73", "#D55E00", "#009E73", "#D55E00"])
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(
        xticks=range(4),
        xticklabels=["−30°\nmatched", "−30°\nmismatch", "+30°\nmatched", "+30°\nmismatch"],
        ylabel="Key clearance at planned stop (mm)",
        title="Before release: matching affects passage",
    )
    post = [r for r in probes if r["probe_type"] == "arbitrary_postclear_yaw" and r["final_success"]]
    axes[1].plot([-35, 35], [-35, 35], "--", color="gray", alpha=0.6)
    axes[1].scatter([r["post_clear_yaw_deg"] for r in post], [r["final_yaw_deg"] for r in post], s=80, color="#009E73")
    axes[1].set(
        xlabel="Requested final axial yaw (deg)",
        ylabel="Measured final axial yaw (deg)",
        title="After release: both yaw choices succeed",
        xlim=(-40, 40),
        ylim=(-40, 40),
    )
    axes[1].text(
        0.04,
        0.95,
        "2 completions + 1 planning failure\nAdaptive retries; all attempts retained\nNot a success-rate estimate",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.suptitle("Existing simulator interventions | 6 conditions, 7 attempts | one geometry")
    save_figure(fig, figures, "task_vs_preference")
    print(
        json.dumps(
            {
                "complete_cells": len(cells),
                "warnings": len(diagnostic["warning_cells"]),
                "max_invariance_mm": diagnostic["maximum_passive_coordinate_difference_mm"],
                "max_historical_difference_mm": diagnostic["maximum_historical_error_difference_mm"],
            }
        )
    )


if __name__ == "__main__":
    main()
