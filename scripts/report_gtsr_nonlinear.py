"""Validate raw decision-gate records and report evidence without novelty inflation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from gtsr_nonlinear import METHODS, Task, connection_maximum
from gtsr_nonlinear_reference import reference_projection

ROOT = Path(__file__).resolve().parents[1]


def read_rows(folder, name):
    return [json.loads(line) for line in (folder / f"{name}.jsonl").read_text().splitlines()]


def keyed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate record {key}")
        result[key] = row
    return result


def projection_correct(row):
    if row.get("exception") or not row["reference"]["verified"]:
        return False
    return bool(
        abs(row["gap_to_reference"]) <= 1e-7
        and row["checks"]["band_violation"] <= 1e-10
        and row["checks"]["domain_violation"] <= 1e-10
        and row["checks"]["off_axis_error"] <= 1e-10
        and abs(row["distance"] - row["independent_distance"]) <= 1e-12
    )


def reference_fraction(rows):
    eligible = sum(row["reference"]["verified"] for row in rows)
    return sum(map(projection_correct, rows)) / eligible if eligible else None


def audit(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    completion = json.loads((folder / "completion.json").read_text())
    tables = {
        name: read_rows(folder, name)
        for name in ("samples", "jacobians", "projections", "connections", "approximations")
    }
    base = list(itertools.product(manifest["a"], manifest["rho"], manifest["seeds"], METHODS))
    fields = ("a", "rho", "seed", "method")
    for name, n, suffix in (("samples", 256, "sample"), ("jacobians", 16, "query")):
        records = keyed(tables[name], (*fields, suffix))
        expected = {(*key, i) for key in base for i in range(n)}
        if set(records) != expected:
            raise ValueError(f"missing/extra {name} records")
    projections = keyed(tables["projections"], (*fields, "weight", "query"))
    nqueries = 4 if manifest["smoke"] else 8
    expected = {(*key, w, i) for key in base for w in manifest["weights"] for i in range(nqueries)}
    if set(projections) != expected or completion["projection_records"] != len(expected):
        raise ValueError("incomplete projection factorial")
    connections = keyed(tables["connections"], (*fields, "edge", "mode"))
    expected = {(*key, i, mode) for key in base for i in range(32) for mode in ("coordinate", "pose_linear")}
    if set(connections) != expected:
        raise ValueError("incomplete connection factorial")
    approximations = keyed(tables["approximations"], ("a", "rho", "segments"))
    expected = set(itertools.product(manifest["a"], manifest["rho"], (1, 4, 16, 64)))
    if set(approximations) != expected:
        raise ValueError("incomplete approximation factorial")
    reference_cache = {}
    for row in tables["projections"]:
        key = (row["a"], row["rho"], row["seed"], row["weight"], row["query"])
        if key not in reference_cache:
            if row["reference"]["verified"]:
                reference_cache[key] = reference_projection(
                    Task(row["a"], row["rho"]), row["query_theta"], row["query_height"], row["weight"]
                )
            else:
                reference_cache[key] = row["reference"]
        recomputed = reference_cache[key]
        if (
            row["reference"]["verified"] and abs(recomputed["distance"] - row["reference"]["distance"]) > 1e-14
        ) or recomputed["verified"] != row["reference"]["verified"]:
            raise ValueError("reference mismatch")
        if not row.get("exception") and row["reference"]["distance"] is not None:
            if abs(row["distance"] - row["reference"]["distance"] - row["gap_to_reference"]) > 1e-14:
                raise ValueError("incorrect reference gap")
        if row["counts"]["ik_calls"] != 0 or row["counts"]["collision_checks"] != 0:
            raise ValueError("unexpected robot checks")
    for row in tables["connections"]:
        maximum = connection_maximum(Task(row["a"], row["rho"]), row["first"], row["last"], row["mode"])
        if abs(maximum - row["maximum_residual"]) > 1e-14:
            raise ValueError("connection bound mismatch")
        if bool(maximum <= row["rho"] + 1e-10) != row["valid"]:
            raise ValueError("incorrect connection classification")
        if maximum + 1e-12 < max(row["dense_maximum"], row["mapped_check_maximum"]):
            raise ValueError("continuous bound below independent observations")
    for row in tables["approximations"]:
        predicted = abs(row["a"]) / (4 * row["segments"] ** 2)
        if max(abs(predicted - row["predicted_maximum"]), abs(predicted - row["observed_maximum"])) > 1e-12:
            raise ValueError("approximation formula mismatch")
    sources = json.loads((folder / "source_manifest.json").read_text())
    for path, digest in sources.items():
        snapshot = folder / "source_snapshot" / path
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError(f"snapshot changed: {path}")
    analytic_pair_maximum = 0.0
    analytic_count_mismatches = 0
    matched_inputs = 0
    for a, rho, seed, weight, query in itertools.product(
        manifest["a"], manifest["rho"], manifest["seeds"], manifest["weights"], range(nqueries)
    ):
        paired = [projections[(a, rho, seed, method, weight, query)] for method in METHODS]
        if len({(row["query_theta"], row["query_height"]) for row in paired}) != 1:
            raise ValueError("unmatched query inputs")
        matched_inputs += 1
        first, second = paired[:2]
        if not first.get("exception") and not second.get("exception"):
            analytic_pair_maximum = max(analytic_pair_maximum, abs(first["distance"] - second["distance"]))
            analytic_count_mismatches += first["counts"] != second["counts"]
    return (
        manifest,
        tables,
        dict(
            records={name: len(rows) for name, rows in tables.items()},
            paired_query_cases=matched_inputs,
            distinct_pose_query_settings=len(
                {(r["a"], r["rho"], r["weight"], r["query_theta"], r["query_height"]) for r in tables["projections"]}
            ),
            unverified_references=sum(not r["verified"] for r in reference_cache.values()),
            analytic_pair_max_distance_difference=analytic_pair_maximum,
            analytic_pair_count_mismatches=analytic_count_mismatches,
            source_snapshot_files=len(sources),
        ),
    )


def render(folder, out):
    manifest, tables, checks = audit(folder)
    out.mkdir(parents=True, exist_ok=False)
    summary = {"audit": checks, "methods": {}}
    for method in METHODS:
        rows = [r for r in tables["projections"] if r["method"] == method]
        samples = [r for r in tables["samples"] if r["method"] == method]
        jacobians = [r for r in tables["jacobians"] if r["method"] == method]
        available = [r for r in rows if not r.get("exception")]
        summary["methods"][method] = dict(
            projection_queries=len(rows),
            exceptions=len(rows) - len(available),
            optimizer_success=sum(r.get("solver_reported_success", r["success"]) for r in rows),
            reference_eligible=sum(r["reference"]["verified"] for r in rows),
            reference_correct=sum(map(projection_correct, rows)),
            max_absolute_gap=max(
                (
                    abs(r["gap_to_reference"])
                    for r in available
                    if r["reference"]["verified"] and r["gap_to_reference"] is not None
                ),
                default=None,
            ),
            median_ms=float(np.median([r["elapsed_seconds"] * 1000 for r in rows])),
            median_map_calls=float(np.median([r["counts"]["map_calls"] for r in rows])),
            median_jacobian_calls=float(np.median([r["counts"]["jacobian_calls"] for r in rows])),
            samples=len(samples),
            valid_samples=sum(
                r["band_violation"] <= 1e-10 and r["domain_violation"] <= 1e-10 and r["off_axis_error"] <= 1e-10
                for r in samples
            ),
            max_sample_matrix_difference=max(r["matrix_max_difference"] for r in samples),
            max_jacobian_error=max(r["max_absolute_error"] for r in jacobians),
            connections={
                mode: dict(total=len(selected), valid=sum(r["valid"] for r in selected))
                for mode in ("coordinate", "pose_linear")
                for selected in [[r for r in tables["connections"] if r["method"] == method and r["mode"] == mode]]
            },
        )
    failures = [r for r in tables["projections"] if not projection_correct(r) or not r["success"]]
    summary["decision"] = (
        "No new capability follows from shared operators; assess any numerical differences separately."
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "projection_flags.json").write_text(json.dumps(failures, indent=2, allow_nan=False) + "\n")
    lines = [
        "# 最小显式基线＋变螺距：开发结果",
        "",
        f"输入批次：`{folder.name}`。开发验证；不作跨任务泛化或统计显著性结论。",
        "",
        "## 投影与调用成本",
        "",
        "正确投影以独立参考误差与区域约束判定，分母为参考已验证的查询；未验证参考另报。",
        "优化器成功的分母包含全部查询。时间包含共同投影流程，排除导入/预热。",
        "",
        "| 实现 | 参考正确 | 求解器成功 | 最大距离误差 m | 中位时间 ms | 中位映射调用 | 中位导数调用 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, s in summary["methods"].items():
        gap_text = "—" if s["max_absolute_gap"] is None else f"{s['max_absolute_gap']:.3e}"
        lines.append(
            f"| {method} | {s['reference_correct']}/{s['reference_eligible']} | "
            f"{s['optimizer_success']}/{s['projection_queries']} | {gap_text} | "
            f"{s['median_ms']:.3f} | {s['median_map_calls']:.1f} | {s['median_jacobian_calls']:.1f} |"
        )
    lines += [
        "",
        "## 采样、导数与连续连接",
        "",
        "| 实现 | 带内样本 | Jacobian 最大误差 | 坐标映射连接合法 | 位姿线性连接合法 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, s in summary["methods"].items():
        coordinate, pose = s["connections"]["coordinate"], s["connections"]["pose_linear"]
        lines.append(
            f"| {method} | {s['valid_samples']}/{s['samples']} | {s['max_jacobian_error']:.3e} | "
            f"{coordinate['valid']}/{coordinate['total']} | {pose['valid']}/{pose['total']} |"
        )
    lines += [
        "",
        "## 解读与下一步",
        "",
        "- 三种实现调用相同生产投影器，显式基线拥有同一映射与导数信息。这一比较检验一致性和实现成本。",
        "- 坐标连接的带内性质来自映射与凸参数域，直接参数化同样获得；不能算 G-TSR 独有保证。",
        "- 自动微分耗时属于本次 PyTorch CPU eager 后端开销，不能推出 G-TSR 固有更快。",
        "- 参考仅适用于本批同轴查询；60 位根细化没有区间认证。求解器标志和参考误差不同的查询保留。",
        "- IK 与碰撞检查次数都是 0。本批不能论证机器人成功率、关节连续性或障碍避碰。",
        "- 若没有独立的算法增量，按决策节点一暂缓扩大曲面/机器人实验。",
        "",
        f"参考未验证查询：{checks['unverified_references']}；解析对照最大距离差："
        f"{checks['analytic_pair_max_distance_difference']:.3e}；解析对照调用次数不一致查询："
        f"{checks['analytic_pair_count_mismatches']}。",
        "",
        "原始标志见 `projection_flags.json`，完整记录与源码快照在输入批次目录。",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    plot(out, tables, manifest)
    return summary


def plot(out, tables, manifest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["G-TSR\nanalytic", "Direct\nanalytic", "Direct\nautodiff"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    runtime = [[r["elapsed_seconds"] * 1000 for r in tables["projections"] if r["method"] == m] for m in METHODS]
    axes[0].boxplot(runtime, tick_labels=labels, showfliers=False)
    axes[0].set(ylabel="Projection time (ms)", title="Same solver; implementation cost")
    fractions = [reference_fraction([r for r in tables["projections"] if r["method"] == m]) for m in METHODS]
    axes[1].bar(range(3), [value if value is not None else 0 for value in fractions], color="#4477AA")
    for index, value in enumerate(fractions):
        if value is None:
            axes[1].text(index, 0.05, "unknown", ha="center")
    axes[1].set(
        xticks=range(3),
        xticklabels=labels,
        ylim=(0, 1.08),
        ylabel="Reference-correct fraction",
        title="Independent scalar reference",
    )
    for mode, label, color in (
        ("coordinate", "Coordinate connection (all methods)", "#228833"),
        ("pose_linear", "Pose-linear connection (all methods)", "#CC6677"),
    ):
        rates = []
        for a in manifest["a"]:
            rows = [r for r in tables["connections"] if r["a"] == a and r["mode"] == mode]
            rates.append(sum(r["valid"] for r in rows) / len(rows))
        axes[2].plot(manifest["a"], rates, "o-", color=color, label=label)
    axes[2].set(
        xlabel="Quadratic coefficient (m/rad²)",
        ylabel="Continuous band-valid fraction",
        ylim=(-0.05, 1.08),
        title="Shared mapping property",
    )
    axes[2].legend(fontsize=7, loc="lower center")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(out / f"decision_gate.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.input.resolve(), args.out.resolve()), indent=2))
