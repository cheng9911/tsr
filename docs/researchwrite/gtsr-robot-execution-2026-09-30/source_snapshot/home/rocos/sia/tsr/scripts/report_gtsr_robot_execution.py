"""Verify complete physical execution records and summarize all planned attempts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gtsr_robot_execution import cell_key, sha, validate_preparation  # noqa: E402


def verify(directory, folder="execution"):
    validate_preparation(directory)
    out = directory / folder
    manifest = json.loads((out / "manifest.json").read_text())
    assert sha(directory / "preparation.json") == manifest["preparation_sha256"], "preparation metadata drift"
    changed = [path for path, expected in manifest["sources"].items() if sha(path) != expected]
    if changed:
        raise ValueError(f"execution input drift: {changed}")
    config = json.loads((directory / "protocol.json").read_text())
    cells = json.loads((out / "cells.json").read_text())
    keys = [cell_key(c) for c in cells]
    assert len(keys) == len(set(keys))
    if folder == "execution":
        expected = {
            cell_key(dict(arm=a, context=c["name"], seed=s, method=m))
            for a in config["arms"]
            for c in config["contexts"]
            for s in config["execution_seeds"]
            for m in config["methods"]
        }
        assert set(keys) == expected, "missing planned factorial cells"
    records, hashes, traced = [], {}, 0
    initial_states, initial_differences, failure_diagnostics = {}, [], []
    for cell, key in zip(cells, keys):
        path = out / f"{key}.json"
        record = json.loads(path.read_text())
        assert all(record[k] == v for k, v in cell.items())
        hashes[path.name] = sha(path)
        path = out / f"{key}.h5"
        if path.exists() and "steps" in record:
            with h5py.File(path, "r") as handle:
                g = handle["episode_0"]
                n = len(g["actions"])
                assert n == record["steps"]
                assert len(g["peg_pose"]) == len(g["success"]) == n + 1
                assert len(g["terminated"]) == len(g["truncated"]) == n
                for ds in g.values():
                    assert np.all(np.isfinite(ds[:])), f"nonfinite {path.name}/{ds.name}"
                force = float(np.max(np.linalg.norm(g["contact_force"][:], axis=1)))
                np.testing.assert_allclose(force, record["max_net_contact_force_N"], atol=1e-10)
                env_success = bool(g["success"][-1])
                assert env_success == record.get("env_success", env_success)
                assert record["success"] == (env_success and record["status"] == "executed")
                np.testing.assert_allclose(g["obj_to_goal_dist"][-1] * 1000, record["final_position_error_mm"])
                np.testing.assert_allclose(g["axis_angle_err"][-1], record["final_axis_error_rad"])
                geometric_success = g["obj_to_goal_dist"][-1] < 0.012 and g["axis_angle_err"][-1] < 0.15
                assert env_success == bool(geometric_success)
                if record["success"]:
                    assert force <= config["force_abort_N"], "successful run exceeded the common force guard"
                group = (cell["arm"], cell["context"], cell["seed"])
                initial = np.concatenate([g[name][0].reshape(-1) for name in ("qpos", "peg_pose", "socket_pose")])
                if group in initial_states:
                    difference = float(np.max(np.abs(initial - initial_states[group])))
                    assert difference < 1e-7, "unpaired initial state"
                    initial_differences.append(difference)
                else:
                    initial_states[group] = initial
                if not record["success"]:
                    force_index = int(np.argmax(np.linalg.norm(g["contact_force"][:], axis=1)))
                    failure_diagnostics.append(
                        {
                            **cell,
                            "stop_reason": record["stop_reason"],
                            "last_phase": int(g["solver_phase"][-1]),
                            "peak_force_phase": int(g["solver_phase"][force_index]),
                            "max_net_contact_force_N": force,
                            "final_key_clearance_mm": float(g["key_clearance_margin"][-1]) * 1000,
                            "final_position_error_mm": record["final_position_error_mm"],
                            "final_axis_error_rad": record["final_axis_error_rad"],
                        }
                    )
            hashes[path.name] = sha(path)
            traced += 1
        else:
            assert not record["success"], "success without an execution trace"
            if path.exists():
                hashes[path.name] = sha(path)  # retain incomplete traces without treating them as validated
        records.append(record)
    # Exact same intended central targets under TSR/PoE is the compatibility contract.
    lookup = {cell_key(r): r for r in records}
    differences = []
    for row in records:
        if not row["method"].startswith("gtsr_"):
            continue
        other = lookup.get(cell_key({**row, "method": "tsr_" + row["method"][5:]}))
        if other is None:
            continue
        a = {t["waypoint"]: t for t in row.get("targets", []) if t["candidate"] == 0}
        b = {t["waypoint"]: t for t in other.get("targets", []) if t["candidate"] == 0}
        for waypoint in a.keys() & b.keys():
            difference = float(np.max(np.abs(np.array(a[waypoint]["peg_target"]) - b[waypoint]["peg_target"])))
            assert difference < 1e-12
            differences.append(difference)
    checks = {
        "planned": len(cells),
        "recorded": len(records),
        "physical_traces": traced,
        "source_hashes_verified": len(manifest["sources"]),
        "output_sha256": hashes,
        "paired_central_targets_checked": len(differences),
        "max_paired_target_difference": max(differences, default=None),
        "paired_initial_states_checked": len(initial_differences),
        "max_paired_initial_state_difference": max(initial_differences, default=None),
        "failure_diagnostics": failure_diagnostics,
    }
    return records, checks


def verify_pdiag_targets(directory):
    from tsr.gtsr.profile6 import ResponseProfile6D, se3_from_pose6
    from tsr.gtsr.response import ResponseOperator

    config = json.loads((directory / "protocol.json").read_text())
    metadata = json.loads((directory / "preparation.json").read_text())["source_metadata"]
    differences = []
    root = Path(__file__).resolve().parents[1]
    with np.load(directory / "targets.npz", allow_pickle=False) as targets:
        for arm_index, arm in enumerate(config["arms"]):
            path = (
                root
                / "docs/researchwrite/gtsr-unified-eval-2026-09-30/artifacts/pdiag"
                / (f"{arm}_{config['source_seed']}_{config['source_subset']}.npz")
            )
            assert sha(path) == metadata[f"{arm}/pdiag"]["production_file_sha256"]
            with np.load(path, allow_pickle=False) as artifact:
                profile = ResponseProfile6D(
                    artifact["theta"],
                    alpha_max=float(artifact["alpha_max"]),
                    basis_width=float(artifact["basis_width"]),
                )
            reference = targets["reference"][arm_index]
            operator = ResponseOperator(reference, np.eye(6))
            for ci, context in enumerate(config["contexts"]):
                relation = reference @ se3_from_pose6(context["pose6"])
                for phase in range(4):
                    for i, rho in enumerate(np.linspace(0, 1, 25)):
                        predicted = operator.action(relation, profile.alpha(phase, rho))
                        frozen = targets["actions"][arm_index, 2, ci, phase * 25 + i]
                        differences.append(float(np.max(np.abs(predicted - frozen))))
    assert max(differences) < 1e-12
    return {"production_actions_checked": len(differences), "max_difference": max(differences)}


def plot_results(directory, summary, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "TSR + Frame",
        "G-TSR + Frame",
        "TSR + Pdiag",
        "G-TSR + Pdiag",
        "TSR + Fixed",
        "TSR + Full affine",
        "TSR + TP-GMM",
    ]
    contexts = [c["name"] for c in config["contexts"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.6), constrained_layout=True)
    for axis, arm in zip(axes, config["arms"]):
        rows = [next(r for r in summary if r["arm"] == arm and r["method"] == m) for m in config["methods"]]
        values = np.array(
            [[r["contexts"][c]["successes"] / r["contexts"][c]["attempts"] for c in contexts] for r in rows]
        )
        axis.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for j, row in enumerate(rows):
            for i, c in enumerate(contexts):
                n, total = row["contexts"][c]["successes"], row["contexts"][c]["attempts"]
                axis.text(
                    i, j, f"{n}/{total}", ha="center", va="center", color="white" if n / total > 0.65 else "black"
                )
        axis.set_xticks(range(len(contexts)), contexts, rotation=20, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(arm.replace("_", " "))
    fig.suptitle(
        "Panda / PhysX execution: successes / all attempts\nThree development seeds per fixed context", fontsize=12
    )
    folder = directory / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / f"execution_success.{extension}", dpi=220)
    plt.close(fig)


def summarize(records, arms, methods):
    rows = []
    for arm in arms:
        for method in methods:
            subset = [r for r in records if r["arm"] == arm and r["method"] == method]
            traced = [r for r in subset if "steps" in r]
            rows.append(
                {
                    "arm": arm,
                    "method": method,
                    "attempts": len(subset),
                    "successes": sum(r["success"] for r in subset),
                    "traces": len(traced),
                    "max_force_N": max((r["max_net_contact_force_N"] for r in traced), default=None),
                    "mean_planning_seconds_traced_attempts_only": float(
                        np.mean([r["planning_seconds"] for r in traced])
                    )
                    if traced
                    else None,
                    "planning_time_measurement_count": len(traced),
                    "stops": dict(Counter(r["stop_reason"] for r in subset)),
                    "contexts": {
                        c: {
                            "attempts": sum(r["context"] == c for r in subset),
                            "successes": sum(r["success"] for r in subset if r["context"] == c),
                        }
                        for c in sorted({r["context"] for r in subset})
                    },
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    config = json.loads((directory / "protocol.json").read_text())
    records, checks = verify(directory)
    checks["pdiag_production_operator"] = verify_pdiag_targets(directory)
    summary = summarize(records, config["arms"], config["methods"])
    (directory / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# 全部物理执行尝试",
        "",
        "分母包含预定条件中的全部失败。每个任务/方法12次尝试，3个执行seed×4个固定context。",
        "不是12个独立任务实例，不能据此进行总体显著性声明。",
        "",
        "| 任务 | 方法 | 总成功 | nominal | translation | yaw25 | mixed | 最大净接触力 N |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        counts = [
            f"{row['contexts'][c]['successes']}/{row['contexts'][c]['attempts']}"
            for c in ("nominal", "translation", "yaw25", "mixed")
        ]
        force = "NA" if row["max_force_N"] is None else f"{row['max_force_N']:.3f}"
        lines.append(
            f"| {row['arm']} | {row['method']} | {row['successes']}/{row['attempts']} | "
            + " | ".join(counts)
            + f" | {force} |"
        )
    lines += ["", "## 失败与停止原因", "", "| 任务 | 方法 | 原因及次数 |", "|---|---|---|"]
    for row in summary:
        lines.append(f"| {row['arm']} | {row['method']} | {json.dumps(row['stops'], ensure_ascii=False)} |")
    (directory / "tables.md").write_text("\n".join(lines) + "\n")
    plot_results(directory, summary, config)
    print(json.dumps({k: v for k, v in checks.items() if k not in ("output_sha256", "failure_diagnostics")}, indent=2))
    print("successes", sum(r["success"] for r in records), "/", len(records))


if __name__ == "__main__":
    main()
