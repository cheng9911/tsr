"""Audit every coupled-gate cell and preserve old evidence before reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from benchmark_gtsr_coupled import METHODS, SEEDS, fingerprints
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = ["Single TSR", "4 TSR", "16 TSR", "64 TSR", "TSR + correction", "G-TSR"]
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(folder):
    hashes = json.loads((folder / "geometry/source_hashes.json").read_text())
    assert hashes == fingerprints(), "experiment source mismatch"
    prior = json.loads((folder / "preservation_before.json").read_text())
    changed = [p for p, digest in prior.items() if not (ROOT / p).exists() or sha(ROOT / p) != digest]
    assert not changed, changed
    preservation = {"checked": len(prior), "changed": changed}
    (folder / "preservation_check.json").write_text(json.dumps(preservation, indent=2) + "\n")
    geometry = [json.loads(p.read_text()) for p in sorted((folder / "geometry").glob("a*.json"))]
    expected = {
        (a, k, b, s, m)
        for a in (0.0, 0.8)
        for k, b in (("proposals", 128), ("proposals", 512), ("milliseconds", 20), ("milliseconds", 100))
        for s in SEEDS
        for m in METHODS
    }
    assert len(geometry) == len(expected) == 960
    assert {(r["angle"], r["budget_kind"], r["budget"], r["seed"], r["method"]) for r in geometry} == expected
    for row in geometry:
        assert 0 <= row["collision_free_accepted"] <= row["band_accepted"] <= row["proposals"]
        assert row["geometry_seconds"] >= row["sampling_seconds"]
        assert row["within_budget"] == (
            row["budget_kind"] == "proposals" or row["geometry_seconds"] <= row["budget"] / 1000
        )
        assert row["path_found"] == (row["route"] is not None)
        if row["path_found"]:
            route = np.array([[0, 0], *row["route"]])
            assert np.all(np.diff(route[:, 0]) > 0)
            assert max(np.diff(route[:, 0])) <= 0.025 + 1e-12
            assert max(np.abs(route[:, 1] - row["angle"] * route[:, 0])) <= 0.03 + 1e-12
    by_key = {(r["angle"], r["budget_kind"], r["budget"], r["seed"], r["method"]): r for r in geometry}
    robot_files = sorted(p for p in (folder / "robot").glob("*.json") if not p.name.endswith(".started.json"))
    robots = [json.loads(p.read_text()) for p in robot_files]
    expected_robot = {(a, b, s, m) for a in (0.0, 0.8) for b in (128, 512) for s in SEEDS[:3] for m in METHODS}
    assert len(robots) == len(expected_robot) == 72
    assert {(r["angle"], r["budget"], r["seed"], r["method"]) for r in robots} == expected_robot
    trace_count = 0
    trace_audit = []
    for path, row in zip(robot_files, robots):
        source = by_key[(row["angle"], "proposals", row["budget"], row["seed"], row["method"])]
        assert row["physical_attempt"] == source["path_found"]
        original = folder / "geometry" / f"a{row['angle']}_proposals{row['budget']}_{row['seed']}_{row['method']}.json"
        marker = json.loads(Path(str(path.with_suffix("")) + ".started.json").read_text())
        assert marker["geometry_sha256"] == sha(original) and marker["source_hashes"] == hashes
        if not row["physical_attempt"]:
            assert not row["success"] and not path.with_suffix(".h5").exists()
            continue
        assert row["route"] == source["route"]
        trace_count += 1
        with h5py.File(path.with_suffix(".h5")) as handle:
            episode = handle["episode_0"]
            assert len(episode["actions"]) == row["steps"]
            assert len(episode["peg_pose"]) == row["steps"] + 1
            force = np.linalg.norm(episode["contact_force"][:], axis=1).max()
            assert abs(force - row["max_force_N"]) < 1e-9
            peg, socket = episode["peg_pose"][:], episode["socket_pose"][:]
            rp = Rotation.from_quat(peg[:, [4, 5, 6, 3]])
            rs = Rotation.from_quat(socket[:, [4, 5, 6, 3]])
            rotations = (rs.inv() * rp).as_matrix()
            local = rs.inv().apply(peg[:, :3] - socket[:, :3])
            selected = np.isin(episode["solver_phase"][:], [4, 6])
            progress = (0.122 - local[selected, 2]) / 0.04
            lateral = np.linalg.norm(local[selected, :2], axis=1)
            axis = np.arccos(np.clip(rotations[selected, 2, 2], -1, 1))
            yaw = np.arctan2(rotations[selected, 1, 0], rotations[selected, 0, 0])
            yaw_error = np.abs(yaw - row["angle"] * np.clip(progress, 0, 1))
            exact_clearance = 0.045 - (
                local[-1, 2] - 0.037 * rotations[-1, 2, 2] + np.abs(rotations[-1, 2]) @ np.array([0.021, 0.014, 0.001])
            )
            trace_audit.append(
                dict(
                    file=path.name,
                    h5_sha256=sha(path.with_suffix(".h5")),
                    success=row["success"],
                    exact_pin_clearance_mm=float(exact_clearance * 1000),
                    state_sha256=hashlib.sha256(
                        b"".join(episode[key][:].tobytes() for key in ("peg_pose", "qpos", "contact_force", "actions"))
                    ).hexdigest(),
                )
            )
            if row["success"]:
                assert bool(episode["success"][-1]) and row["crossing_witness"]
                assert episode["obj_to_goal_dist"][-1] < 0.003
                assert episode["axis_angle_err"][-1] < 0.05
                assert episode["key_clearance_margin"][-1] > 0.001
                assert all(p["verified"] for p in row["plans"])
                assert row["max_force_N"] <= 80
                # Independent quaternion/trace reconstruction, including tilted pin top.
                assert exact_clearance > 0.001
                assert progress[0] <= 0.05 and progress[-1] > 1
                assert np.all(progress >= np.maximum.accumulate(progress) - 0.05)
                assert max(lateral) <= 0.003 and max(axis) <= 0.05 and max(yaw_error) <= 0.06
                passage = row["passage"]
                assert passage[0]["progress"] <= 0.05 and passage[-1]["progress"] > 1
                progress = np.array([p["progress"] for p in passage])
                assert np.all(progress >= np.maximum.accumulate(progress) - 0.05)
                assert all(
                    p["lateral_m"] <= 0.003 and p["axis_rad"] <= 0.05 and p["yaw_error_rad"] <= 0.06 for p in passage
                )
    geo_summary, robot_summary = [], []
    for angle in (0.0, 0.8):
        for kind, budget in (("proposals", 128), ("proposals", 512), ("milliseconds", 20), ("milliseconds", 100)):
            for method in METHODS:
                rows = [
                    r
                    for r in geometry
                    if (r["angle"], r["budget_kind"], r["budget"], r["method"]) == (angle, kind, budget, method)
                ]
                proposals = sum(r["proposals"] for r in rows)
                geo_summary.append(
                    dict(
                        angle=angle,
                        kind=kind,
                        budget=budget,
                        method=method,
                        n=len(rows),
                        path_found=sum(r["path_found"] for r in rows),
                        band_acceptance=sum(r["band_accepted"] for r in rows) / max(proposals, 1),
                        median_proposals=float(np.median([r["proposals"] for r in rows])),
                        median_ms=1000 * float(np.median([r["geometry_seconds"] for r in rows])),
                        deadline_misses=sum(not r["within_budget"] for r in rows),
                    )
                )
        for budget in (128, 512):
            for method in METHODS:
                rows = [r for r in robots if (r["angle"], r["budget"], r["method"]) == (angle, budget, method)]
                attempts = [r for r in rows if r["physical_attempt"]]
                robot_summary.append(
                    dict(
                        angle=angle,
                        budget=budget,
                        method=method,
                        n=len(rows),
                        attempted=len(attempts),
                        success=sum(r["success"] for r in rows),
                        reasons=dict(Counter(r["stop_reason"] for r in rows)),
                        max_net_force_N=max((r["max_force_N"] for r in attempts), default=None),
                        median_plan_audit_s=float(np.median([r["plan_and_audit_seconds"] for r in attempts]))
                        if attempts
                        else None,
                    )
                )
    # Straight controls should be checked empirically, including physics outcomes.
    control_equal = []
    for budget in (128, 512):
        for seed in SEEDS[:3]:
            rows = [r for r in robots if r["angle"] == 0 and r["budget"] == budget and r["seed"] == seed]
            control_equal.append(
                dict(
                    budget=budget,
                    seed=seed,
                    equal_success=len({r["success"] for r in rows}) == 1,
                    error_range_mm=float(np.ptp([r["final_error_mm"] for r in rows if r["physical_attempt"]])),
                    equal_state_trajectories=len(
                        {
                            r["state_sha256"]
                            for r in trace_audit
                            if r["file"].startswith(f"a0p0_proposals{budget}_{seed}_")
                        }
                    )
                    == 1,
                )
            )
    summary = dict(
        preservation=preservation,
        geometry=geo_summary,
        robot=robot_summary,
        audited_traces=trace_count,
        straight_control=control_equal,
    )
    (folder / "trace_audit.json").write_text(json.dumps(trace_audit, indent=2) + "\n")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# 耦合几何与物理执行：本轮结果",
        "",
        f"旧数据校验：{len(prior)} 个文件哈希不变。"
        f"新结果：960 格几何比较，72 格端到端比较，{trace_count} 次实际物理尝试。",
        "",
        "所有数字均为开发实验；机器人每格只有 3 个采样种子，不报告统计显著性或泛化成功率。",
        "",
        "## 几何路径与采样",
        "",
        "时间包含参数化构建、采样、过滤与局部连接。时间预算列按总几何时间判定超时；不含机器人求解。",
        "",
        "| 通道 rad | 预算 | 方法 | 带内接受率 | 预算内路径 | 几何时间中位数 ms | 超时格数 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in geo_summary:
        lines.append(
            f"| {r['angle']} | {r['budget']} {r['kind']} | {r['method']} | {100 * r['band_acceptance']:.1f}% | "
            f"{r['path_found']}/{r['n']} | {r['median_ms']:.2f} | {r['deadline_misses']} |"
        )
    lines += [
        "",
        "## 固定控制器的物理执行",
        "",
        "分母包括几何路径失败；物理尝试单列。合力为 peg/socket 净合力，不能替代逐接触最大载荷。",
        "",
        "| 通道 rad | 提议次数 | 方法 | 物理尝试 | 端到端成功 | 最大净合力 N |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in robot_summary:
        force = "—" if r["max_net_force_N"] is None else f"{r['max_net_force_N']:.2f}"
        lines.append(
            f"| {r['angle']} | {r['budget']} | {r['method']} | {r['attempted']}/{r['n']} | "
            f"{r['success']}/{r['n']} | {force} |"
        )
    lines += [
        "",
        "## 失败记录",
        "",
        "```json",
        json.dumps(dict(Counter(r["stop_reason"] for r in robots if not r["success"])), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 证据边界",
        "",
        "- 该任务的螺旋关系是预先给定的；显式修正基线拥有同样的信息。已知解析中心线也可直接生成路径。",
        "- 结论针对本采样接口和固定局部连接器。不能据此声称 G-TSR 胜过所有 TSRChain 或全局规划方法。",
        "- 最大 4 mm 连接半径是共同算法设置；目前没有半径消融。",
        "- 实际轨迹检查允许命令 yaw 带以外 0.03 rad 跟踪裕量，不是严格带内跟踪的证明。",
        "- 规划采用离散关节插值审核；抓持圆柱使用网格近似；不是连续碰撞的数学认证。",
        "- 只验证了固定场景、已知几何和初态；尚缺几何误差、姿态扰动、多个任务与硬件实验。",
        "",
        "复现协议见 [protocol.md](protocol.md)，汇总数据见 [summary.json](summary.json)，"
        "源码快照见 `source_batch_v1/`。",
        "",
    ]
    (folder / "results.md").write_text("\n".join(lines))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(METHODS))
    for budget, color, offset in ((128, "#0072B2", -0.18), (512, "#D55E00", 0.18)):
        geo = [r for r in geo_summary if r["angle"] == 0.8 and r["kind"] == "proposals" and r["budget"] == budget]
        rob = [r for r in robot_summary if r["angle"] == 0.8 and r["budget"] == budget]
        for ax, values in (
            (axes[0, 0], [r["band_acceptance"] * 100 for r in geo]),
            (axes[0, 1], [r["path_found"] for r in geo]),
            (axes[1, 0], [r["median_ms"] for r in geo]),
            (axes[1, 1], [r["success"] for r in rob]),
        ):
            ax.bar(x + offset, values, width=0.35, color=color, label=f"{budget} proposals")
    for ax, title, ylabel in zip(
        axes.flat,
        (
            "A  Helical gate: sampling",
            "B  Geometric paths (20 seeds)",
            "C  Complete geometric stage",
            "D  Physical task success (3 seeds)",
        ),
        ("Band acceptance (%)", "Paths / 20", "Median time (ms)", "Successes / 3"),
    ):
        ax.set(title=title, ylabel=ylabel, xticks=x, xticklabels=LABELS)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_ylim(0, 21)
    axes[1, 1].set_ylim(0, 3.2)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(folder / f"coupled_results.{ext}", dpi=200)
    plt.close(fig)
    print(
        json.dumps(
            {
                "geometry_cells": len(geometry),
                "robot_cells": len(robots),
                "traces": trace_count,
                "successes": sum(r["success"] for r in robots),
                "preservation": preservation,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    report(parser.parse_args().directory)
