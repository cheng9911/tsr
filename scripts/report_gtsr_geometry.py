"""Verify, aggregate and plot all frozen synthetic geometry benchmark cells."""

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
from scripts.benchmark_gtsr_geometry import (  # noqa: E402
    Case,
    collision,
    edge_valid,
    expected_acceptance,
    sample_states,
    sha,
    task_valid,
)

LABELS = {
    "tsr_box": "Single TSR",
    "tsr_union_4": "4 TSR union",
    "tsr_union_16": "16 TSR union",
    "tsr_union_64": "64 TSR union",
    "tsr_chain_independent": "Independent TSRChain",
    "gtsr_poe": "G-TSR PoE",
    "tsr_box_repaired": "TSR + coupling correction",
}
COLORS = ["#222222", "#56B4E9", "#0072B2", "#CC79A7", "#999999", "#D55E00", "#009E73"]


def save(fig, folder, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root, results = args.directory, args.directory / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    config = manifest["config"]
    assert all(sha(p) == digest for p, digest in manifest["sources"].items())
    files = sorted((results / "cells").glob("*.json"))
    rows = [json.loads(p.read_text()) for p in files]
    cases = [
        Case(h, eps, *config["q_range_rad"], *config["x_range_m"])
        for h in config["pitches_m_per_rad"]
        for eps in config["tolerances_m"]
    ]
    keys = {(c.key, m, s) for c in cases for m in config["methods"] for s in config["seeds"]}
    assert len(rows) == len(keys) == 1120
    assert {(r["case"], r["method"], r["seed"]) for r in rows} == keys
    assert all(r["run_hash"] == manifest["run_hash"] and r["status"] == "ok" for r in rows)
    case_map = {c.key: c for c in cases}
    max_path_excess = 0.0
    for row in rows:
        assert row["geometry"]["coverage"] == 1.0
        assert [p["budget"] for p in row["planning"]] == config["planning_proposal_budgets"]
        assert all(p["proposals_used"] <= p["budget"] for p in row["planning"])
        assert all(p["success"] or p["proposals_used"] == p["budget"] for p in row["planning"])
        assert row["planning"][-1]["success"] == (row["solution"] is not None)
        if row["solution"] is not None:
            case, path = case_map[row["case"]], np.array(row["solution"])
            np.testing.assert_allclose(path[0], [0, case.qlo, case.pitch * case.qlo], atol=1e-12)
            np.testing.assert_allclose(path[-1], [0, case.qhi, case.pitch * case.qhi], atol=1e-12)
            assert all(
                edge_valid(case, a, b, config["configuration_obstacles_x_q"]) for a, b in zip(path[:-1], path[1:])
            )
            # Different verification grid from the 101-point run-time check.
            dense = np.concatenate([a + np.linspace(0, 1, 137)[:, None] * (b - a) for a, b in zip(path[:-1], path[1:])])
            assert task_valid(case, dense).all() and not collision(dense, config["configuration_obstacles_x_q"]).any()
            max_path_excess = max(
                max_path_excess,
                float(np.maximum(abs(dense[:, 2] - case.pitch * dense[:, 1]) - case.tolerance, 0).max()),
            )
    lookup = {(r["case"], r["method"], r["seed"]): r for r in rows}
    canonical_pairs = 0
    for case in cases:
        if case.pitch != 0:
            continue
        for seed in config["seeds"]:
            base = lookup[(case.key, "tsr_box", seed)]
            for method in config["methods"]:
                other = lookup[(case.key, method, seed)]
                for a, b in zip(base["planning"], other["planning"]):
                    for field in ("success", "proposals_used", "vertices", "edges_checked", "accepted_proposals"):
                        assert a[field] == b[field]
                if base["solution"] is not None:
                    np.testing.assert_allclose(base["solution"], other["solution"], atol=1e-14)
                canonical_pairs += 1
    aggregate = []
    max_probability_error = 0.0
    table = [
        "# TSR / G-TSR 全部几何与规划结果",
        "",
        "每种场景20个配对算法随机 seed。每 seed 几何样本10,000；规划列为找到可行路径的次数/20。",
        "均值±SD指跨算法 seed 的波动。规划失败保留；这些不是物理执行成功率。",
        "",
    ]
    for case in cases:
        table += [
            f"## h={case.pitch:g} m/rad，容差 ±{case.tolerance * 1000:g} mm",
            "",
            "| 方法 | 几何有效率 %（均值±SD） | 理论 % | 覆盖率 % | "
            "128 提议 | 512 提议 | 2048 提议 | 平均实际提议数（上限2048） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in config["methods"]:
            selected = [lookup[(case.key, method, s)] for s in config["seeds"]]
            valid = np.array([r["geometry"]["valid_fraction"] for r in selected])
            expected = expected_acceptance(case, method)
            error = abs(float(valid.mean()) - expected)
            max_probability_error = max(max_probability_error, error)
            geometric = {}
            for field in selected[0]["geometry"]:
                values = [r["geometry"][field] for r in selected if r["geometry"][field] is not None]
                geometric[field] = (
                    None if not values else dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))
                )
            planning = []
            for bi, budget in enumerate(config["planning_proposal_budgets"]):
                items = [r["planning"][bi] for r in selected]
                lengths = [p["path_length_mm_equivalent"] for p in items if p["success"]]
                planning.append(
                    dict(
                        budget=budget,
                        n=len(items),
                        successes=sum(p["success"] for p in items),
                        success_fraction=float(np.mean([p["success"] for p in items])),
                        mean_proposals_used=float(np.mean([p["proposals_used"] for p in items])),
                        mean_planning_seconds=float(np.mean([p["planning_seconds"] for p in items])),
                        path_length_mean_among_successes=None if not lengths else float(np.mean(lengths)),
                    )
                )
            aggregate.append(
                dict(
                    case=case.key,
                    pitch=case.pitch,
                    tolerance=case.tolerance,
                    method=method,
                    geometry=geometric,
                    planning=planning,
                )
            )
            table.append(
                f"| {LABELS[method]} | {valid.mean() * 100:.3f} ± {valid.std(ddof=1) * 100:.3f} | "
                f"{expected * 100:.3f} | 100 | "
                + " | ".join(f"{p['successes']}/20" for p in planning)
                + f" | {planning[-1]['mean_proposals_used']:.1f} |"
            )
        table.append("")
    table += [
        "## 计量说明",
        "",
        "- 覆盖率的解析值为1，并用独立可行样本检查；没有通过收缩区域提高有效率。",
        "- 错误接受率采用共同单 TSR 紧致盒中的无效查询为条件分布；"
        "非零 h 时等分 K 并集理论值为1/K，h=0时无此类查询，记null。",
        "- sampler_seconds计时完整2048提议池；planning_seconds只计搜索到首次成功或预算耗尽，"
        "均不含最终密集核验。二者不能相加称作各预算端到端时间。",
        "- 路径长度只在成功运行中定义，失败保持null；不能据此比较全部运行的质量。",
        "- 同一运行的三个预算检查点相互依赖，不是三份独立试验。",
    ]
    (root / "tables.md").write_text("\n".join(table) + "\n")
    summary = dict(
        run_hash=manifest["run_hash"],
        cells=len(rows),
        planning_outcomes=3360,
        summaries=aggregate,
        max_mean_valid_probability_error=max_probability_error,
        max_path_excess_m=max_path_excess,
        canonical_method_seed_checks=canonical_pairs,
    )
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    verification = dict(
        all_source_hashes_unchanged=True,
        unique_complete_cells=len(rows),
        canonical_equivalence_checks=canonical_pairs,
        all_saved_paths_verified=True,
        output_sha256={str(p): sha(p) for p in files},
        report_script_sha256=sha(__file__),
    )
    (results / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    # All eight scenarios shown; the illustrative cloud uses the middle nonzero pitch.
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True, layout="constrained")
    by_method = {(r["case"], r["method"]): r for r in aggregate}
    for case in cases:
        ax = axes[config["tolerances_m"].index(case.tolerance), config["pitches_m_per_rad"].index(case.pitch)]
        for method, color in zip(config["methods"], COLORS):
            item = by_method[(case.key, method)]
            ax.plot(
                config["planning_proposal_budgets"],
                [p["success_fraction"] * 100 for p in item["planning"]],
                color=color,
                marker="o",
                linestyle="--" if method in ("tsr_chain_independent", "tsr_box_repaired") else "-",
                label=LABELS[method],
                alpha=0.85,
            )
        ax.set(
            xscale="log",
            xticks=config["planning_proposal_budgets"],
            xticklabels=config["planning_proposal_budgets"],
            ylim=(-3, 103),
            title=f"h={case.pitch:g} m/rad | tolerance ±{case.tolerance * 1000:g} mm",
        )
        ax.grid(alpha=0.2)
        ax.minorticks_off()
    for ax in axes[1]:
        ax.set_xlabel("Proposal budget (including rejected samples)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Solved paths (%) | 20 algorithm seeds")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("Common reduced-space RRT | fixed configuration obstacles | synthetic geometry, no robot execution")
    save(fig, figures, "planning_all_cases")

    case = Case(0.05, 0.0005)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True, layout="constrained")
    u = np.random.default_rng(3911).random((1800, 3))
    q = np.linspace(case.qlo, case.qhi, 300)
    for ax, method in zip(axes, ("tsr_box", "tsr_union_4", "tsr_union_64", "gtsr_poe")):
        states = sample_states(case, method, u)
        valid = task_valid(case, states)
        ax.fill_between(
            q,
            (case.pitch * q - case.tolerance) * 1000,
            (case.pitch * q + case.tolerance) * 1000,
            color="#009E73",
            alpha=0.3,
        )
        for mask, color in ((~valid, "#D55E00"), (valid, "#0072B2")):
            ax.scatter(states[mask, 1], states[mask, 2] * 1000, s=3, alpha=0.4, color=color, rasterized=True)
        ax.set(
            title=f"{LABELS[method]}\nAnalytic valid fraction: {expected_acceptance(case, method) * 100:.2f}%",
            xlabel="Yaw q (rad)",
        )
    axes[0].set_ylabel("Axial position z (mm)")
    fig.suptitle(
        "Tight bounds, full coverage | h=0.05 m/rad, tolerance ±0.5 mm | blue: valid; orange: outside task band"
    )
    save(fig, figures, "tight_region_samples")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, eps in zip(axes, config["tolerances_m"]):
        for h, color in zip(config["pitches_m_per_rad"][1:], ("#0072B2", "#D55E00", "#009E73")):
            case = Case(h, eps)
            ks = np.logspace(0, 3, 200)
            probabilities = 2 * eps / (abs(h) * (case.qhi - case.qlo) / ks + 2 * eps)
            ax.plot(ks, probabilities * 100, color=color, label=f"h={h:g} m/rad")
            for k, method in ((1, "tsr_box"), (4, "tsr_union_4"), (16, "tsr_union_16"), (64, "tsr_union_64")):
                item = by_method[(case.key, method)]["geometry"]["valid_fraction"]
                ax.errorbar(k, item["mean"] * 100, yerr=item["sd"] * 100, fmt="o", color=color, capsize=3)
        ax.axhline(100, color="black", linestyle="--", lw=1, label="Exact G-TSR / corrected TSR")
        ax.set(
            xscale="log",
            xlabel="Number of tight TSR boxes K",
            ylabel="Constraint-valid samples (%)",
            title=f"Tolerance ±{eps * 1000:g} mm",
            ylim=(0, 105),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Prescribed equal-q partition: analytic curves and mean ± seed SD; "
        "no claim of globally optimal TSR approximation"
    )
    save(fig, figures, "complexity_acceptance")
    print(json.dumps({k: v for k, v in summary.items() if k != "summaries"}))


if __name__ == "__main__":
    main()
