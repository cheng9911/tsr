"""Independent uncertainty audit: command certificate, observed tracking, physical completion."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from benchmark_gtsr_uncertainty import ERRORS, LEVELS, SEEDS, hashes
from gtsr_uncertainty import EPSILON, METHODS, RHO, TRUE_B, TRUE_H, Q
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(folder):
    assert json.loads((folder / "sources.json").read_text()) == hashes()
    prior = json.loads((folder / "preservation_before.json").read_text())
    changed = [name for name, sha in prior.items() if digest(Path(name)) != sha]
    assert not changed
    cells = json.loads((folder / "samples.json").read_text())
    assert len(cells) == 735
    assert {(r["db"], r["dh"], r["method"], r["seed"]) for r in cells} == {
        (db, dh, method, seed) for db in ERRORS for dh in ERRORS for method in METHODS for seed in SEEDS
    }
    methods = []
    for method in METHODS:
        rows = [r for r in cells if r["method"] == method]
        safe = [r for r in rows if r["certified"]]
        methods.append(
            dict(
                method=method,
                certified_cells=len(safe),
                certified_proposals=sum(r["samples"] for r in safe),
                certified_accepted=sum(r["accepted"] for r in safe),
                certified_violations=sum(r["accepted_violations"] for r in safe),
                certified_raw_violations=sum(r["raw_violations"] for r in safe),
                outside_accepted=sum(r["accepted"] for r in rows if not r["certified"]),
                outside_violations=sum(r["accepted_violations"] for r in rows if not r["certified"]),
            )
        )
    for row in cells:
        data = np.asarray(row["values"])
        theta, z = data[:, 0], data[:, 1]
        estimated = z - (TRUE_B + row["db"]) - (TRUE_H + row["dh"]) * theta
        actual = z - TRUE_B - TRUE_H * theta
        assert row["certified"] == (abs(row["db"]) + Q * abs(row["dh"]) + RHO <= EPSILON + 1e-12)
        assert row["raw_violations"] == int(sum(abs(actual) > EPSILON + 1e-12))
        np.testing.assert_allclose(data[:, 2], estimated, atol=1e-14, rtol=0)
        np.testing.assert_allclose(data[:, 3], actual, atol=1e-14, rtol=0)
        keep = abs(estimated) <= RHO + 1e-12
        assert int(sum(keep)) == row["accepted"]
        assert int(sum((abs(actual) > EPSILON + 1e-12) & keep)) == row["accepted_violations"]
        if row["certified"]:
            assert row["accepted_violations"] == 0
    routes = json.loads((folder / "routes.json").read_text())
    assert len(routes) == len(LEVELS) * len(METHODS) * len(SEEDS[:2]) == 30
    assert {(r["level"], r["method"], r["seed"]) for r in routes} == {
        (level, method, seed) for level in LEVELS for method in METHODS for seed in SEEDS[:2]
    }
    audits = []
    for cell in routes:
        name = f"l{cell['level']}_{cell['seed']}_{cell['method']}".replace(".", "p")
        stem = folder / "robot" / name
        record = json.loads(stem.with_suffix(".json").read_text())
        assert json.loads(Path(str(stem) + ".started.json").read_text()) == cell
        audit = dict(
            level=cell["level"],
            seed=cell["seed"],
            method=cell["method"],
            certified=cell["certified"],
            attempted=cell["path_found"],
            raw_success=record["success"],
            success=False,
            reason=record["stop_reason"],
            target_in_band=False,
            actual_in_band=None,
        )
        if not cell["path_found"]:
            assert not record["success"] and not stem.with_suffix(".h5").exists()
            audits.append(audit)
            continue
        assert record["route"] == cell["route"]
        route = np.asarray(cell["route"])
        assert np.all(np.diff(route[:, 0]) < 0)
        assert np.max(abs(route[:, 0] - cell["b"] - cell["h"] * route[:, 1])) <= RHO + 1e-12
        assert np.min(route[:, 1]) >= 0 and np.max(route[:, 1]) <= Q
        target_error = float(np.max(abs(route[:, 0] - TRUE_B - TRUE_H * route[:, 1])))
        audit.update(target_in_band=target_error <= EPSILON + 1e-12, max_target_residual_mm=target_error * 1000)
        if cell["certified"]:
            assert audit["target_in_band"]
        with h5py.File(stem.with_suffix(".h5")) as handle:
            ep = handle["episode_0"]
            assert len(ep["actions"]) == record["steps"] and len(ep["peg_pose"]) == record["steps"] + 1
            peg, socket = ep["peg_pose"][:], ep["socket_pose"][:]
            rp, rs = Rotation.from_quat(peg[:, [4, 5, 6, 3]]), Rotation.from_quat(socket[:, [4, 5, 6, 3]])
            rotations = (rs.inv() * rp).as_matrix()
            local = rs.inv().apply(peg[:, :3] - socket[:, :3])
            yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
            residual = local[:, 2] - TRUE_B - TRUE_H * yaw
            phases = ep["solver_phase"][:]
            active = (local[:, 2] >= 0.082) & (local[:, 2] <= 0.122) & np.isin(phases, [3, 4, 6])
            actual_max = float(np.max(abs(residual[active]))) if any(active) else None
            clearance = 0.045 - (
                local[-1, 2] - 0.037 * rotations[-1, 2, 2] + np.abs(rotations[-1, 2]) @ np.array([0.021, 0.014, 0.001])
            )
            force = float(np.max(np.linalg.norm(ep["contact_force"][:], axis=1)))
            assert abs(force - record["max_force_N"]) < 1e-9
            completion = (
                record["success"]
                and bool(ep["success"][-1])
                and clearance > 0.001
                and ep["axis_angle_err"][-1] < 0.05
                and ep["obj_to_goal_dist"][-1] < 0.003
            )
            tracking_z, tracking_yaw, tracking_checks = [], [], 0
            targets = record["targets"]
            for i, target in enumerate(targets):
                if target["phase"] != "enter_key":
                    continue
                start = target["state_index"] + 1
                end = targets[i + 1]["state_index"] + 1 if i + 1 < len(targets) else len(local)
                indices = np.arange(start, end)
                indices = indices[active[indices]]
                if not len(indices):
                    continue
                dz, dt = abs(local[indices, 2] - target["depth"]), abs(yaw[indices] - target["yaw"])
                bound = abs(cell["db"]) + Q * abs(cell["dh"]) + RHO + dz + abs(TRUE_H) * dt
                assert np.all(abs(residual[indices]) <= bound + 1e-12)
                tracking_z.extend(dz.tolist())
                tracking_yaw.extend(dt.tolist())
                tracking_checks += len(indices)
            observed_bound = (
                (abs(cell["db"]) + Q * abs(cell["dh"]) + RHO + max(tracking_z) + abs(TRUE_H) * max(tracking_yaw))
                if tracking_checks
                else None
            )
            audit.update(
                success=bool(completion),
                actual_in_band=(actual_max <= EPSILON + 1e-12) if actual_max is not None else None,
                max_actual_residual_mm=actual_max * 1000 if actual_max is not None else None,
                exact_clearance_mm=float(clearance * 1000),
                max_net_force_N=force,
                observed_tracking_bound_mm=observed_bound * 1000 if observed_bound is not None else None,
                tracking_inequality_checks=tracking_checks,
                h5_sha256=digest(stem.with_suffix(".h5")),
            )
        audits.append(audit)
    physical = []
    for level in LEVELS:
        for method in METHODS:
            rows = [r for r in audits if r["level"] == level and r["method"] == method]
            physical.append(
                dict(
                    level=level,
                    method=method,
                    n=len(rows),
                    certified=rows[0]["certified"],
                    attempted=sum(r["attempted"] for r in rows),
                    target_in_band=sum(r["target_in_band"] for r in rows),
                    actual_in_band=sum(r["actual_in_band"] is True for r in rows),
                    success=sum(r["success"] for r in rows),
                    reasons=dict(Counter(r["reason"] for r in rows)),
                )
            )
    result = dict(
        methods=methods,
        physical=physical,
        audited=audits,
        matched_coordinates=json.loads((folder / "matched_coordinates.json").read_text()),
        preserved_files=len(prior),
        changes=changed,
        raw_success_reclassified=sum(r["raw_success"] and not r["success"] for r in audits),
    )
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 几何估计误差界：完整结果",
        "",
        "## 保证范围内的生产采样输出",
        "",
        "| 方法 | 保证内格数 | 过滤后样本 | 过滤后违例 | 原始提议违例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in methods:
        lines.append(
            f"| {r['method']} | {r['certified_cells']} | {r['certified_accepted']} | "
            f"{r['certified_violations']} | {r['certified_raw_violations']} |"
        )
    lines += [
        "",
        "## 相同 100 ms 几何预算的物理批次",
        "",
        "每行预定两次；没有执行的格子仍在总分母内。实际带内只表示已记录的孔内轨迹片段，失败可能提前终止。",
        "目标带内和任务成功是不同指标；超出充分条件不等于必然失败。",
        "",
        "| λ | 方法 | 有命令保证 | 物理尝试 | 全路径目标带内 | 已记录实际片段带内 | 物理任务成功 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in physical:
        lines.append(
            f"| {r['level']} | {r['method']} | {r['certified']} | {r['attempted']}/2 | "
            f"{r['target_in_band']}/2 | {r['actual_in_band']}/2 | {r['success']}/2 |"
        )
    lines += [
        "",
        "采样分布不同，因此超出保证范围的违例比例不能归结为表示能力差异。",
        "实际跟踪误差界由本次轨迹事后测量，不是事先给出的控制器鲁棒保证。",
        f"旧数据 {len(prior)} 个文件哈希不变；原始成功经精确针片间隙复核后重新分类 "
        f"{result['raw_success_reclassified']} 次。",
        "",
    ]
    (folder / "results.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        matrix = np.zeros((len(ERRORS), len(ERRORS)))
        for i, db in enumerate(ERRORS):
            for j, dh in enumerate(ERRORS):
                rows = [r for r in cells if (r["db"], r["dh"], r["method"]) == (db, dh, method)]
                matrix[i, j] = sum(r["accepted_violations"] for r in rows) / sum(r["accepted"] for r in rows)
        im = ax.imshow(matrix, vmin=0, vmax=1, origin="lower", cmap="magma")
        for i, db in enumerate(ERRORS):
            for j, dh in enumerate(ERRORS):
                if abs(db) + Q * abs(dh) + RHO <= EPSILON + 1e-12:
                    ax.plot(j, i, marker="s", markersize=13, markerfacecolor="none", markeredgecolor="cyan")
        ax.set(
            title=method,
            xticks=range(7),
            yticks=range(7),
            xticklabels=[v * 1000 for v in ERRORS],
            yticklabels=[v * 1000 for v in ERRORS],
            xlabel="Pitch error (mm/rad)",
            ylabel="Offset error (mm)",
        )
    fig.colorbar(im, ax=axes, label="True-band violation fraction after estimated-band filtering", shrink=0.75)
    fig.suptitle("Cyan squares: certified by the common bound; sampling laws differ outside it")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(folder / f"uncertainty_boundary.{ext}", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                methods=methods,
                robot_attempts=sum(r["attempted"] for r in audits),
                robot_successes=sum(r["success"] for r in audits),
                preserved=len(prior),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    run(parser.parse_args().directory)
