"""Run the preregistered first decision gate; outputs are never overwritten."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
from gtsr_nonlinear import (
    METHODS,
    Task,
    anchor_transform,
    check_pose,
    coaxial_pose,
    connection_maximum,
    finite_difference_body_jacobian,
    implementation,
    piecewise_height,
)
from gtsr_nonlinear_reference import reference_projection

from tsr.gtsr.metrics import SE3Metric
from tsr.gtsr.projection import project

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (930501, 930502)


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def emit(handle, row):
    handle.write(json.dumps(row, allow_nan=False) + "\n")
    handle.flush()


def queries(task, rng):
    return [
        ("center", 0.35, task.height(0.35)),
        ("upper_band", 0.7, task.height(0.7) + task.rho),
        ("lower_band", 0.2, task.height(0.2) - task.rho),
        ("before_domain", -0.1, task.height(-0.1)),
        ("after_domain", 1.1, task.height(1.1)),
        ("axial_outside", 0.55, task.height(0.55) + 0.025),
        ("random", float(rng.uniform(-0.1, 1.1)), float(task.height(0.5) + rng.uniform(-0.08, 0.08))),
        ("random", float(rng.uniform(-0.1, 1.1)), float(task.height(0.5) + rng.uniform(-0.08, 0.08))),
    ]


def snapshot(out):
    paths = [
        *sorted((ROOT / "src/tsr").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*nonlinear*.py")),
        ROOT / "tests/gtsr/test_nonlinear_operators.py",
        ROOT / "pyproject.toml",
        out.parent / "protocol.md",
    ]
    records = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(out / "source_manifest.json", records)


def run(out, smoke):
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out)
    start = time.perf_counter()
    import torch

    torch.set_num_threads(1)
    import_seconds = time.perf_counter() - start
    coefficients = (0.0, 0.08) if smoke else (-0.08, -0.02, 0.0, 0.02, 0.08)
    widths = (0.0005,) if smoke else (0.0, 0.0005)
    seeds = SEEDS[:1] if smoke else SEEDS
    manifest = dict(
        smoke=smoke,
        a=coefficients,
        rho=widths,
        seeds=seeds,
        weights=[0.02, 0.1],
        methods=METHODS,
        python=sys.version,
        numpy=np.__version__,
        scipy=scipy.__version__,
        torch=torch.__version__,
        platform=platform.platform(),
        argv=sys.argv,
        torch_import_seconds=import_seconds,
        input_distribution="uniform_parameter_box",
        shared_solver="unmodified tsr.gtsr.projection.project",
        projection_starts="lower upper midpoint",
        numpy_seed_policy="matched within task/seed; method order randomized per query",
        ik_calls=0,
        collision_checks=0,
    )
    save(out / "manifest.json", manifest)
    warmup = {}
    for method in METHODS:
        begin = time.perf_counter()
        region = implementation(Task(0.02, 0.0005), method)
        transform = region.to_transform([0.5, 0])
        project(region, transform, SE3Metric(0.1), starts=[[0, -0.0005], [1, 0.0005], [0.5, 0]])
        warmup[method] = time.perf_counter() - begin
    save(out / "warmup.json", warmup)
    handles = {
        name: (out / f"{name}.jsonl").open("x")
        for name in ("samples", "jacobians", "projections", "connections", "approximations")
    }
    total_projections = 0
    try:
        for a in coefficients:
            for rho in widths:
                task = Task(a, rho)
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    coordinates = np.array([task.domain().map_from_unit_cube(u) for u in rng.random((256, 2))])
                    query_list = queries(task, rng)
                    if smoke:
                        query_list = query_list[:4]
                    regions = {m: implementation(task, m) for m in METHODS}
                    meta = dict(a=a, rho=rho, seed=seed)
                    for i, z in enumerate(coordinates):
                        transforms = {m: region.to_transform(z) for m, region in regions.items()}
                        for method, transform in transforms.items():
                            emit(
                                handles["samples"],
                                dict(
                                    **meta,
                                    sample=i,
                                    method=method,
                                    z=z.tolist(),
                                    matrix_max_difference=float(np.max(np.abs(transform - transforms[METHODS[0]]))),
                                    **check_pose(task, transform),
                                ),
                            )
                    for j, z in enumerate(coordinates[:16]):
                        expected = finite_difference_body_jacobian(
                            lambda x: coaxial_pose(x[0], task.b + task.h * x[0] + task.a * x[0] ** 2 + x[1]), z
                        )
                        for method, region in regions.items():
                            derivative = region.chart.jacobian(z)
                            emit(
                                handles["jacobians"],
                                dict(
                                    **meta,
                                    query=j,
                                    method=method,
                                    z=z.tolist(),
                                    max_absolute_error=float(np.max(np.abs(derivative - expected))),
                                    min_singular_value=float(np.linalg.svd(derivative, compute_uv=False)[-1]),
                                ),
                            )
                    for index in range(32):
                        first, last = coordinates[2 * index : 2 * index + 2]
                        for mode in ("coordinate", "pose_linear"):
                            maximum = connection_maximum(task, first, last, mode)
                            dense_t = np.linspace(0, 1, 2001)
                            theta = first[0] + dense_t * (last[0] - first[0])
                            eta = first[1] + dense_t * (last[1] - first[1])
                            if mode == "coordinate":
                                height = task.height(theta) + eta
                            else:
                                height = (1 - dense_t) * (task.height(first[0]) + first[1]) + dense_t * (
                                    task.height(last[0]) + last[1]
                                )
                            dense_max = float(np.max(np.abs(height - task.b - task.h * theta - task.a * theta**2)))
                            for method, region in regions.items():
                                matrices = []
                                for t in np.linspace(0, 1, 17):
                                    if mode == "coordinate":
                                        matrices.append(region.to_transform((1 - t) * first + t * last))
                                    else:
                                        t0, t1 = region.to_transform(first), region.to_transform(last)
                                        local0 = np.linalg.solve(anchor_transform(), t0)
                                        local1 = np.linalg.solve(anchor_transform(), t1)
                                        th = (1 - t) * first[0] + t * last[0]
                                        matrices.append(
                                            anchor_transform()
                                            @ coaxial_pose(th, (1 - t) * local0[2, 3] + t * local1[2, 3])
                                        )
                                observed = max(abs(check_pose(task, p)["eta"]) for p in matrices)
                                emit(
                                    handles["connections"],
                                    dict(
                                        **meta,
                                        method=method,
                                        edge=index,
                                        mode=mode,
                                        first=first.tolist(),
                                        last=last.tolist(),
                                        maximum_residual=maximum,
                                        valid=bool(maximum <= rho + 1e-10),
                                        dense_maximum=dense_max,
                                        mapped_check_maximum=observed,
                                        dense_checks=2001,
                                        mapped_checks=17,
                                        ik_calls=0,
                                        collision_checks=0,
                                    ),
                                )
                    starts = [
                        task.domain().cont_lower,
                        task.domain().cont_upper,
                        (task.domain().cont_lower + task.domain().cont_upper) / 2,
                    ]
                    for weight in (0.02, 0.1):
                        metric = SE3Metric(weight)
                        for i, (kind, theta, height) in enumerate(query_list):
                            query_pose = anchor_transform() @ coaxial_pose(theta, height)
                            reference = reference_projection(task, theta, height, weight)
                            for method in rng.permutation(METHODS):
                                region = regions[method]
                                region.chart.counts.reset()
                                row = dict(
                                    **meta,
                                    method=str(method),
                                    weight=weight,
                                    query=i,
                                    kind=kind,
                                    query_theta=theta,
                                    query_height=height,
                                    reference=reference,
                                )
                                begin = time.perf_counter()
                                try:
                                    result = project(region, query_pose, metric, starts=starts)
                                    elapsed = time.perf_counter() - begin
                                    counts = asdict(region.chart.counts)
                                    independent_distance = metric.distance(result.transform, query_pose)
                                    row.update(
                                        elapsed_seconds=elapsed,
                                        counts=counts,
                                        z=result.z.tolist(),
                                        distance=result.distance,
                                        independent_distance=independent_distance,
                                        success=result.success,
                                        optimality=result.optimality,
                                        message=result.message,
                                        gap_to_reference=result.distance - reference["distance"],
                                        checks=check_pose(task, result.transform),
                                        exception=None,
                                    )
                                except Exception as exc:
                                    row.update(
                                        elapsed_seconds=time.perf_counter() - begin,
                                        counts=asdict(region.chart.counts),
                                        exception=repr(exc),
                                        success=False,
                                    )
                                emit(handles["projections"], row)
                                total_projections += 1
                    print(json.dumps(dict(**meta, projections_written=total_projections)), flush=True)
                for k in (1, 4, 16, 64):
                    theta = np.linspace(0, task.q, 8193)
                    residual = piecewise_height(task, theta, k) - task.height(theta)
                    emit(
                        handles["approximations"],
                        dict(
                            a=a,
                            rho=rho,
                            segments=k,
                            predicted_maximum=abs(a) * task.q**2 / (4 * k * k),
                            observed_maximum=float(np.max(np.abs(residual))),
                            meaning="centerline approximation only; not DSS or a planning failure",
                        ),
                    )
    finally:
        for handle in handles.values():
            handle.close()
    save(out / "completion.json", dict(projection_records=total_projections, finished=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run(arguments.out, arguments.smoke)
