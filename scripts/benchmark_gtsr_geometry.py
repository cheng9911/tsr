"""Paired TSR/G-TSR geometry and reduced-space planning benchmark.

This evaluates analytic configuration constraints, not robot execution or CBiRRT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsr.gtsr.charts import PoEChart  # noqa: E402
from tsr.gtsr.domains import CoordinateDomain  # noqa: E402
from tsr.gtsr.region import GTSR  # noqa: E402
from tsr.tsr import TSR  # noqa: E402
from tsr.tsr_chain import TSRChain  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Case:
    pitch: float
    tolerance: float
    qlo: float = -0.5
    qhi: float = 0.5
    xlo: float = -0.012
    xhi: float = 0.012

    @property
    def key(self):
        return f"h{self.pitch:g}_eps{self.tolerance:g}"

    def bounds(self, lo=None, hi=None):
        lo, hi = self.qlo if lo is None else lo, self.qhi if hi is None else hi
        return min(self.pitch * lo, self.pitch * hi) - self.tolerance, max(
            self.pitch * lo, self.pitch * hi
        ) + self.tolerance


def box_count(method):
    return int(method.rsplit("_", 1)[1]) if method.startswith("tsr_union_") else 1


def expected_acceptance(case, method):
    if method in ("gtsr_poe", "tsr_box_repaired"):
        return 1.0
    return 2 * case.tolerance / (abs(case.pitch) * (case.qhi - case.qlo) / box_count(method) + 2 * case.tolerance)


def sample_states(case, method, uniform):
    """Equivalent vectorized coordinate maps, checked against production forward maps."""
    u = np.asarray(uniform)
    x = case.xlo + (case.xhi - case.xlo) * u[:, 0]
    q = case.qlo + (case.qhi - case.qlo) * u[:, 1]
    if method == "gtsr_poe":
        z = case.pitch * q + case.tolerance * (2 * u[:, 2] - 1)
    else:
        k = box_count(method)
        index = np.minimum((u[:, 1] * k).astype(int), k - 1)
        lo = case.qlo + (case.qhi - case.qlo) * index / k
        hi = case.qlo + (case.qhi - case.qlo) * (index + 1) / k
        zlo = np.minimum(case.pitch * lo, case.pitch * hi) - case.tolerance
        zhi = np.maximum(case.pitch * lo, case.pitch * hi) + case.tolerance
        z = zlo + (zhi - zlo) * u[:, 2]
        if method == "tsr_box_repaired":
            z = case.pitch * q + np.clip(z - case.pitch * q, -case.tolerance, case.tolerance)
    return np.column_stack([x, q, z])


def task_valid(case, states, slack=1e-12):
    s = np.asarray(states)
    return (
        (s[..., 0] >= case.xlo - slack)
        & (s[..., 0] <= case.xhi + slack)
        & (s[..., 1] >= case.qlo - slack)
        & (s[..., 1] <= case.qhi + slack)
        & (np.abs(s[..., 2] - case.pitch * s[..., 1]) <= case.tolerance + slack)
    )


def representation_contains(case, method, states):
    if method in ("gtsr_poe", "tsr_box_repaired"):
        return task_valid(case, states)
    s = np.asarray(states)
    inside = np.zeros(s.shape[:-1], dtype=bool)
    edges = np.linspace(case.qlo, case.qhi, box_count(method) + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        zlo, zhi = case.bounds(lo, hi)
        inside |= (
            (s[..., 1] >= lo - 1e-12)
            & (s[..., 1] <= hi + 1e-12)
            & (s[..., 2] >= zlo - 1e-12)
            & (s[..., 2] <= zhi + 1e-12)
        )
    return inside & (s[..., 0] >= case.xlo - 1e-12) & (s[..., 0] <= case.xhi + 1e-12)


def pose(state):
    x, q, z = state
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0, x], [s, c, 0, 0], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)


def production_check(case, uniforms):
    """Check the actual package APIs rather than relabeling a separate sampler."""
    chart = PoEChart([[1, 0, 0, 0, 0, 0], [0, 0, case.pitch, 0, 0, 1], [0, 0, 1, 0, 0, 0]], representation="space")
    region = GTSR(
        chart=chart,
        domain=CoordinateDomain.from_bounds(
            np.array([[case.xlo, case.xhi], [case.qlo, case.qhi], [-case.tolerance, case.tolerance]]),
            chart.periodic,
            chart.periods,
        ),
    )
    maximum = 0.0
    for u, state in zip(uniforms, sample_states(case, "gtsr_poe", uniforms)):
        result = region.to_transform([state[0], state[1], case.tolerance * (2 * u[2] - 1)])
        maximum = max(maximum, float(np.max(np.abs(result - pose(state)))))
    for method in ("tsr_box", "tsr_union_4", "tsr_union_16", "tsr_union_64", "tsr_chain_independent"):
        k = box_count(method)
        states = sample_states(case, method, uniforms)
        for u, state in zip(uniforms, states):
            index = min(int(u[1] * k), k - 1)
            lo = case.qlo + (case.qhi - case.qlo) * index / k
            hi = case.qlo + (case.qhi - case.qlo) * (index + 1) / k
            bounds = np.zeros((6, 2))
            bounds[0] = [case.xlo, case.xhi]
            bounds[2] = case.bounds(lo, hi)
            bounds[5] = [lo, hi]
            coordinates = [state[0], 0, state[2], 0, 0, state[1]]
            if method == "tsr_chain_independent":
                translation, rotation = bounds.copy(), np.zeros((6, 2))
                translation[5] = 0
                rotation[5] = bounds[5]
                chain = TSRChain(TSRs=[TSR(Bw=translation), TSR(Bw=rotation)])
                result = chain.to_transform([[state[0], 0, state[2], 0, 0, 0], [0, 0, 0, 0, 0, state[1]]])
            else:
                result = TSR(Bw=bounds).to_transform(coordinates)
            maximum = max(maximum, float(np.max(np.abs(result - pose(state)))))
    assert maximum < 1e-12, maximum
    return maximum


def collision(states, obstacles):
    s = np.asarray(states)
    found = np.zeros(s.shape[:-1], dtype=bool)
    for rectangle in obstacles:
        found |= (
            (s[..., 0] >= rectangle[0][0])
            & (s[..., 0] <= rectangle[0][1])
            & (s[..., 1] >= rectangle[1][0])
            & (s[..., 1] <= rectangle[1][1])
        )
    return found


def segment_hits_rectangle(a, b, rectangle):
    """Closed segment / closed rectangle intersection by the slab method."""
    enter, leave = 0.0, 1.0
    for j, (lo, hi) in enumerate(rectangle):
        delta = b[j] - a[j]
        if abs(delta) < 1e-15:
            if a[j] < lo or a[j] > hi:
                return False
        else:
            t0, t1 = (lo - a[j]) / delta, (hi - a[j]) / delta
            enter, leave = max(enter, min(t0, t1)), min(leave, max(t0, t1))
            if enter > leave:
                return False
    return True


def edge_valid(case, a, b, obstacles):
    # The task band and coordinate bounds are convex in this configuration slice.
    return bool(
        task_valid(case, a)
        and task_valid(case, b)
        and not any(segment_hits_rectangle(a, b, rect) for rect in obstacles)
    )


def known_path(case):
    points = np.array([[0, -0.5], [0.008, -0.3], [0.008, -0.05], [-0.008, 0.05], [-0.008, 0.3], [0, 0.5]])
    return np.column_stack([points, case.pitch * points[:, 1]])


def run_rrt(case, proposals, goal_choices, config):
    budgets = config["planning_proposal_budgets"]
    obstacles, scale = config["configuration_obstacles_x_q"], np.asarray(config["metric_scale_x_q_z"])
    start, goal = np.array([0, case.qlo, case.pitch * case.qlo]), np.array([0, case.qhi, case.pitch * case.qhi])
    nodes = np.empty((max(budgets) + 2, 3))
    nodes[0] = start
    parents = [-1]
    n = 1
    accepted = edges = 0
    snapshots, solution = [], None
    started = time.perf_counter()
    for used, (proposal, choose_goal) in enumerate(zip(proposals, goal_choices), 1):
        target = goal if choose_goal else proposal
        if task_valid(case, target) and not collision(target, obstacles):
            accepted += 1
            differences = (nodes[:n] - target) * scale
            nearest = int(np.argmin(np.einsum("ij,ij->i", differences, differences)))
            delta = target - nodes[nearest]
            length = float(np.linalg.norm(delta * scale))
            if length > 1e-14:
                candidate = nodes[nearest] + delta * min(1.0, config["steering_step_m_equivalent"] / length)
                edges += 1
                if edge_valid(case, nodes[nearest], candidate, obstacles):
                    nodes[n] = candidate
                    parents.append(nearest)
                    n += 1
                    if np.linalg.norm((candidate - goal) * scale) <= config["goal_connection_radius_m_equivalent"]:
                        edges += 1
                        if edge_valid(case, candidate, goal, obstacles):
                            indices, index = [], n - 1
                            while index >= 0:
                                indices.append(index)
                                index = parents[index]
                            solution = np.vstack([nodes[indices[::-1]], goal])
        if used in budgets or solution is not None:
            snapshots.append(
                dict(
                    budget=used,
                    success=solution is not None,
                    proposals_used=used,
                    accepted_proposals=accepted,
                    edges_checked=edges,
                    vertices=n,
                    planning_seconds=time.perf_counter() - started,
                )
            )
        if solution is not None:
            break
    if solution is not None:
        first = snapshots[-1]
        snapshots = [s for s in snapshots if s["budget"] in budgets and s["budget"] < first["proposals_used"]]
        snapshots += [{**first, "budget": budget} for budget in budgets if budget >= first["proposals_used"]]
    assert [s["budget"] for s in snapshots] == budgets
    for s in snapshots:
        s["path_length_mm_equivalent"] = None
        s["path_max_violation_m"] = None
        s["dense_collision_count"] = None
        if s["success"]:
            dense = np.concatenate(
                [a[None] + np.linspace(0, 1, 101)[:, None] * (b - a)[None] for a, b in zip(solution[:-1], solution[1:])]
            )
            assert task_valid(case, dense).all() and not collision(dense, obstacles).any()
            assert all(edge_valid(case, a, b, obstacles) for a, b in zip(solution[:-1], solution[1:]))
            s["path_length_mm_equivalent"] = float(
                np.linalg.norm(np.diff(solution, axis=0) * scale, axis=1).sum() * 1000
            )
            s["path_max_violation_m"] = float(
                np.maximum(abs(dense[:, 2] - case.pitch * dense[:, 1]) - case.tolerance, 0).max()
            )
            s["dense_collision_count"] = int(collision(dense, obstacles).sum())
    return snapshots, solution


def geometry_record(case, method, uniform, query, feasible):
    states = sample_states(case, method, uniform)
    valid = task_valid(case, states)
    query_valid = task_valid(case, query)
    query_accepted = representation_contains(case, method, query)
    repaired = (
        sample_states(case, "tsr_box", uniform)[:, 2] - states[:, 2]
        if method == "tsr_box_repaired"
        else np.zeros(len(states))
    )
    return dict(
        valid_fraction=float(valid.mean()),
        expected_valid_fraction=expected_acceptance(case, method),
        coverage=float(representation_contains(case, method, feasible).mean()),
        false_acceptance_on_invalid_queries=float(query_accepted[~query_valid].mean())
        if (~query_valid).any()
        else None,
        mean_excess_mm=float(
            np.maximum(abs(states[:, 2] - case.pitch * states[:, 1]) - case.tolerance, 0).mean() * 1000
        ),
        maximum_excess_mm=float(
            np.maximum(abs(states[:, 2] - case.pitch * states[:, 1]) - case.tolerance, 0).max() * 1000
        ),
        correction_fraction=float((abs(repaired) > 1e-12).mean()),
        mean_correction_mm=float(abs(repaired).mean() * 1000),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [
        Path(__file__),
        args.protocol,
        args.protocol.with_suffix(".md"),
        ROOT / "src/tsr/tsr.py",
        ROOT / "src/tsr/tsr_chain.py",
        ROOT / "src/tsr/utils.py",
        *[ROOT / "src/tsr/gtsr" / f for f in ("charts.py", "domains.py", "region.py", "metrics.py", "_validation.py")],
    ]
    manifest = dict(
        sources={str(p): sha(p) for p in paths},
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        config=config,
    )
    run_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["run_hash"] = run_hash
    mpath = args.out / "manifest.json"
    if mpath.exists():
        assert json.loads(mpath.read_text())["run_hash"] == run_hash, "Inputs changed; use a new output directory"
    else:
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    cells = args.out / "cells"
    cells.mkdir(exist_ok=True)
    cases = [
        Case(h, eps, *config["q_range_rad"], *config["x_range_m"])
        for h in config["pitches_m_per_rad"]
        for eps in config["tolerances_m"]
    ]
    api_checks = {}
    for ci, case in enumerate(cases):
        api_checks[case.key] = production_check(case, np.random.default_rng(101 + ci).random((32, 3)))
        certificate = known_path(case)
        assert all(
            edge_valid(case, a, b, config["configuration_obstacles_x_q"])
            for a, b in zip(certificate[:-1], certificate[1:])
        )
        for seed in config["seeds"]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, ci, 10]))
            uniform = rng.random((config["geometry_samples_per_seed"], 3))
            query = sample_states(case, "tsr_box", rng.random(uniform.shape))
            feasible = sample_states(case, "gtsr_poe", rng.random(uniform.shape))
            plan_random = np.random.default_rng(np.random.SeedSequence([seed, ci, 20])).random(
                (max(config["planning_proposal_budgets"]), 4)
            )
            for method in config["methods"]:
                name = f"{case.key}_{seed}_{method}"
                path = cells / f"{name}.json"
                if path.exists():
                    saved = json.loads(path.read_text())
                    assert saved["run_hash"] == run_hash
                    continue
                started = time.perf_counter()
                proposals = sample_states(case, method, plan_random[:, :3])
                sampler_seconds = time.perf_counter() - started
                planning, solution = run_rrt(case, proposals, plan_random[:, 3] < config["goal_bias"], config)
                record = dict(
                    case=case.key,
                    pitch=case.pitch,
                    tolerance=case.tolerance,
                    seed=seed,
                    method=method,
                    run_hash=run_hash,
                    status="ok",
                    geometry=geometry_record(case, method, uniform, query, feasible),
                    planning=planning,
                    sampler_seconds=sampler_seconds,
                    solution=None if solution is None else solution.tolist(),
                )
                path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"{case.key} seed={seed} complete", flush=True)
    records = [json.loads(p.read_text()) for p in sorted(cells.glob("*.json"))]
    expected = len(cases) * len(config["seeds"]) * len(config["methods"])
    assert len(records) == expected and all(r["run_hash"] == run_hash for r in records)
    result = dict(
        run_hash=run_hash,
        count=len(records),
        planning_outcomes=len(records) * len(config["planning_proposal_budgets"]),
        production_forward_max_errors=api_checks,
    )
    (args.out / "completion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
