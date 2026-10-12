"""Matched geometric sampling and optional physical execution; never overwrite cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from gtsr_coupled_geometry import Gate, RegionSampler, coordinates, in_band

METHODS = ("tsr", "tsr_union_4", "tsr_union_16", "tsr_union_64", "tsr_projected", "gtsr")
SEEDS = tuple(range(930201, 930221))
ROOT = Path(__file__).resolve().parents[1]


def make_route(gate, accepted, max_gap=0.1, execution_step=0.025):
    """Common monotone local connector in (progress, yaw), not a global planner.

    All methods receive identical endpoints, band and connector. A 4 mm axial
    connection radius limits each edge; execution interpolates to <=1 mm.
    The yaw band is convex in these coordinates. Physical slice clearances are
    additionally checked at every execution waypoint and midpoint.
    """
    points = sorted([[0.0, 0.0], *accepted, [1.0, gate.angle]])
    if np.max(np.diff(np.asarray(points)[:, 0])) > max_gap + 1e-12:
        return None
    selected, index = [points[0]], 0
    while index < len(points) - 1:
        nxt = index + 1
        while nxt + 1 < len(points) and points[nxt + 1][0] - points[index][0] <= max_gap + 1e-12:
            nxt += 1
        selected.append(points[nxt])
        index = nxt
    route = []
    for first, last in zip(selected[:-1], selected[1:]):
        count = max(1, int(np.ceil((last[0] - first[0]) / execution_step)))
        for fraction in np.linspace(0, 1, 2 * count + 1)[1:]:
            s, yaw = np.asarray(first) + fraction * (np.asarray(last) - first)
            if abs(yaw - gate.angle * s) > gate.yaw_tolerance + 1e-12 or gate.pin_clearance(s, yaw) <= 0:
                return None
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            route.append((np.asarray(first) + fraction * (np.asarray(last) - first)).tolist())
    return route


def geometric_trial(angle, method, seed, budget_kind, budget):
    # RNG creation is included in wall time. Same stream prefix for each method.
    tick = time.perf_counter()
    rng = np.random.default_rng(seed)
    gate = Gate(angle=angle)
    sampler = RegionSampler(gate, method)
    accepted, proposals, band_count = [], 0, 0
    # Reserve half the deadline for connection/validation; count both stages.
    while proposals < budget if budget_kind == "proposals" else time.perf_counter() - tick < budget / 2000:
        pose = sampler.sample(rng.random(2))
        proposals += 1
        if in_band(gate, pose):
            band_count += 1
            s, yaw = coordinates(gate, pose)
            if gate.pin_clearance(s, yaw) > 0:
                accepted.append([s, yaw])
    sampling_seconds = time.perf_counter() - tick
    route = make_route(gate, accepted)
    total_seconds = time.perf_counter() - tick
    # Time budgets apply to the complete geometric stage, including connector.
    within_budget = budget_kind == "proposals" or total_seconds <= budget / 1000
    return {
        "angle": angle,
        "method": method,
        "seed": seed,
        "budget_kind": budget_kind,
        "budget": budget,
        "proposals": proposals,
        "band_accepted": band_count,
        "collision_free_accepted": len(accepted),
        "sampling_seconds": sampling_seconds,
        "geometry_seconds": total_seconds,
        "within_budget": within_budget,
        "path_found": route is not None and within_budget,
        "route": route if within_budget else None,
        "path_before_deadline_check": route is not None,
        "expected_band_acceptance": sampler.expected_band_acceptance(),
    }


def fingerprints():
    paths = [
        *sorted((ROOT / "scripts").glob("gtsr_coupled_*.py")),
        Path(__file__),
        *sorted((ROOT / "src/tsr").rglob("*.py")),
    ]
    paths += [
        ROOT.parent / "maniskill_spectrum/phase_switch_symmetry" / name
        for name in ("phase_switch_symmetry_env.py", "collect_phase_switch_rollouts.py")
    ]
    return {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def run_geometry(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "source_hashes.json"
    hashes = fingerprints()
    if manifest.exists():
        if json.loads(manifest.read_text()) != hashes:
            raise RuntimeError("source changed: use a new output directory")
    else:
        manifest.write_text(json.dumps(hashes, indent=2) + "\n")
    for angle in (0.0, 0.8):
        for kind, budget in (("proposals", 128), ("proposals", 512), ("milliseconds", 20), ("milliseconds", 100)):
            for seed in SEEDS:
                # Deterministic balanced order reduces timing-order bias.
                order = np.random.default_rng(seed).permutation(METHODS)
                for method in order:
                    path = output / f"a{angle}_{kind}{budget}_{seed}_{method}.json"
                    if path.exists():
                        continue
                    result = geometric_trial(angle, str(method), seed, kind, budget)
                    path.write_text(json.dumps(result, indent=2) + "\n")
    if fingerprints() != hashes:
        raise RuntimeError("source changed during run")


def run_robot(geometry, output):
    from gtsr_coupled_robot import run_pilot

    output.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((geometry / "source_hashes.json").read_text())
    if fingerprints() != hashes:
        raise RuntimeError("geometry/robot source mismatch")
    for angle in (0.0, 0.8):
        for budget in (128, 512):
            for seed in SEEDS[:3]:
                for method in np.random.default_rng(seed).permutation(METHODS):
                    name = f"a{angle}_proposals{budget}_{seed}_{method}"
                    # with_suffix would truncate names containing decimal angle;
                    # attach an explicit extension-free stem for robot output.
                    target = output / name.replace(".", "p")
                    if target.with_suffix(".json").exists():
                        if "physical_attempt" in json.loads(target.with_suffix(".json").read_text()):
                            continue
                        raise RuntimeError(f"incomplete cell metadata: {target}")
                    source = geometry / (name + ".json")
                    cell = json.loads(source.read_text())
                    marker = Path(str(target) + ".started.json")
                    if marker.exists():
                        raise RuntimeError(f"incomplete cell requires explicit failure accounting: {marker}")
                    marker.write_text(
                        json.dumps(
                            {
                                "geometry_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                "source_hashes": hashes,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    if cell["path_found"]:
                        record = run_pilot(target, angle, "sampled", seed, scene_collision=True, route=cell["route"])
                    else:
                        record = {
                            "success": False,
                            "status": "geometry_failed",
                            "stop_reason": "geometric_path_not_found",
                        }
                    record.update({k: cell[k] for k in ("angle", "method", "seed", "budget", "geometry_seconds")})
                    record["physical_attempt"] = cell["path_found"]
                    target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
                    if fingerprints() != hashes:
                        raise RuntimeError("source changed during run")
                    print(f"CELL {name}: {record['status']} success={record['success']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("geometry", "robot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", type=Path)
    args = parser.parse_args()
    if args.stage == "geometry":
        run_geometry(args.output)
    else:
        run_robot(args.geometry, args.output)


if __name__ == "__main__":
    main()
