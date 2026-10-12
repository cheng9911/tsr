"""Certify affine-band error bounds and evaluate estimated geometry without truth leakage."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from gtsr_uncertainty import (
    EPSILON,
    METHODS,
    RHO,
    TRUE_B,
    TRUE_H,
    EstimatedSampler,
    Q,
    estimated_route,
    exact_fixed_error_bound,
    uncertainty_bound,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(930401, 930406))
ERRORS = (-0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002)
LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0)


def hashes():
    paths = [
        Path(__file__),
        ROOT / "scripts/gtsr_uncertainty.py",
        ROOT / "scripts/gtsr_uncertain_robot.py",
        ROOT / "scripts/gtsr_coupled_geometry.py",
        ROOT / "scripts/gtsr_coupled_env.py",
        *sorted((ROOT / "src/tsr").rglob("*.py")),
    ]
    paths += [
        ROOT.parent / "maniskill_spectrum/phase_switch_symmetry" / name
        for name in ("phase_switch_symmetry_env.py", "collect_phase_switch_rollouts.py")
    ]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def corner_checks():
    rows = []
    for db, dh in itertools.product((0, 0.0005, 0.001, 0.002), repeat=2):
        for rho in (0, 0.0005, 0.0015):
            values = [
                eb + eh * theta + eta
                for eb, eh, theta, eta in itertools.product((-db, db), (-dh, dh), (0, Q), (-rho, rho))
            ]
            computed = max(abs(v) for v in values)
            bound = uncertainty_bound(db, dh, rho)
            assert abs(computed - bound) < 1e-14
            rows.append(
                dict(
                    delta_b=db,
                    delta_h=dh,
                    rho=rho,
                    vertex_max=computed,
                    bound=bound,
                    certified=bound <= EPSILON + 1e-12,
                )
            )
    return rows


def sample_cell(db, dh, method, seed):
    sampler = EstimatedSampler(method, TRUE_B + db, TRUE_H + dh)
    rng = np.random.default_rng(seed)
    records = []
    for uniform in rng.random((256, 2)):
        pose = sampler.sample(uniform)
        theta, z = float(np.arctan2(pose[1, 0], pose[0, 0])), float(pose[2, 3])
        estimated_error = z - sampler.b - sampler.h * theta
        actual_error = z - TRUE_B - TRUE_H * theta
        records.append([theta, z, estimated_error, actual_error])
    values = np.asarray(records)
    keep = np.abs(values[:, 2]) <= RHO + 1e-12
    bound = uncertainty_bound(abs(db), abs(dh), RHO)
    violation = np.abs(values[:, 3]) > EPSILON + 1e-12
    if bound <= EPSILON + 1e-12:
        assert not np.any(violation & keep)
    return dict(
        db=db,
        dh=dh,
        method=method,
        seed=seed,
        samples=256,
        accepted=int(sum(keep)),
        raw_violations=int(sum(violation)),
        accepted_violations=int(sum(violation & keep)),
        bound=bound,
        exact_bound=exact_fixed_error_bound(db, dh, RHO),
        certified=bound <= EPSILON + 1e-12,
        values=records,
    )


def matched_coordinate_control():
    """Equal theta/eta through all production APIs, removing output-law differences."""
    rng, maximum, count = np.random.default_rng(930400), 0.0, 0
    for db, dh in itertools.product(ERRORS, ERRORS):
        samplers = {m: EstimatedSampler(m, TRUE_B + db, TRUE_H + dh) for m in METHODS}
        for u, v in rng.random((100, 2)):
            theta, eta = Q * u, RHO * (2 * v - 1)
            reference = samplers["gtsr"].sample([u, v])
            for method in ("tsr_union_64", "tsr_projected"):
                sampler = samplers[method]
                n = len(sampler.regions)
                k = min(int(u * n), n - 1)
                lo, hi = Q * k / n, Q * (k + 1) / n
                low = min(sampler.h * lo, sampler.h * hi) - RHO
                high = max(sampler.h * lo, sampler.h * hi) + RHO
                adjusted_v = (sampler.h * theta + eta - low) / (high - low)
                pose = sampler.sample([u, adjusted_v])
                maximum = max(maximum, float(np.max(np.abs(pose - reference))))
            count += 1
    assert maximum < 1e-10
    return dict(paired_coordinates=count, maximum_matrix_entry_difference=maximum)


def plan_cell(b, h, method, seed, budget_ms=100):
    # Only estimated geometry and independently known entry/exit heights here.
    tick = time.perf_counter()
    sampler, rng = EstimatedSampler(method, b, h), np.random.default_rng(seed)
    accepted, proposals = [], 0
    while time.perf_counter() - tick < budget_ms / 2000:
        pose = sampler.sample(rng.random(2))
        proposals += 1
        z, theta = float(pose[2, 3]), float(np.arctan2(pose[1, 0], pose[0, 0]))
        if abs(sampler.residual(pose)) <= RHO + 1e-12 and 0.082 <= z <= 0.122:
            accepted.append([z, theta])
    route = estimated_route(accepted, b, h)
    seconds = time.perf_counter() - tick
    within_budget = seconds <= budget_ms / 1000
    return dict(
        b=b,
        h=h,
        method=method,
        seed=seed,
        budget_ms=budget_ms,
        proposals=proposals,
        accepted=len(accepted),
        geometry_seconds=seconds,
        within_budget=within_budget,
        route=route if within_budget else None,
        path_found=route is not None and within_budget,
    )


def run_geometry(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "sources.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    source = hashes()
    save(manifest, source)
    save(out / "corners.json", corner_checks())
    save(out / "matched_coordinates.json", matched_coordinate_control())
    cells = [
        sample_cell(db, dh, method, seed) for db, dh, method, seed in itertools.product(ERRORS, ERRORS, METHODS, SEEDS)
    ]
    save(out / "samples.json", cells)
    routes = []
    for level, seed in itertools.product(LEVELS, SEEDS[:2]):
        for method in np.random.default_rng(seed).permutation(METHODS):
            db, dh = level * 0.0004, level * 0.0006
            cell = plan_cell(TRUE_B + db, TRUE_H + dh, str(method), seed)
            cell.update(level=level, db=db, dh=dh, certified=uncertainty_bound(db, dh, RHO) <= EPSILON + 1e-12)
            # Truth enters only after planning, for evaluation.
            cell["max_target_true_residual"] = (
                max(abs(z - TRUE_B - TRUE_H * t) for z, t in cell["route"]) if cell["route"] else None
            )
            routes.append(cell)
    save(out / "routes.json", routes)
    assert hashes() == source
    save(out / "environment.json", dict(python=sys.version, numpy=np.__version__))
    print(
        json.dumps(
            dict(
                cells=len(cells),
                samples=sum(c["samples"] for c in cells),
                routes=sum(c["path_found"] for c in routes),
                route_cells=len(routes),
            )
        )
    )


def run_robot(out):
    import sqlite3  # noqa: F401

    from gtsr_uncertain_robot import run_trial

    source = json.loads((out / "sources.json").read_text())
    assert hashes() == source
    folder = out / "robot"
    folder.mkdir(exist_ok=True)
    for cell in json.loads((out / "routes.json").read_text()):
        stem = f"l{cell['level']}_{cell['seed']}_{cell['method']}".replace(".", "p")
        output = folder / stem
        if output.with_suffix(".json").exists():
            raise FileExistsError(output)
        save(folder / (stem + ".started.json"), cell)
        if cell["path_found"]:
            result = run_trial(output, cell["b"], cell["h"], cell["route"], cell["seed"])
        else:
            result = dict(success=False, status="geometry_failed", stop_reason="geometric_path_not_found")
            save(output.with_suffix(".json"), result)
        assert hashes() == source
        print(f"CELL {stem}: {result['success']} {result['stop_reason']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("geometry", "robot"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (run_geometry if args.stage == "geometry" else run_robot)(args.output)
