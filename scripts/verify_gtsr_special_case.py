"""Measure exact TSR embedding and the region-count/acceptance trade-off."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from gtsr_coupled_geometry import Gate, RegionSampler, in_band

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.gtsr.helpers import make_pair, random_bounds, random_se3, transform_error  # noqa: E402


def run(output):
    if output.exists():
        raise FileExistsError(output)
    sources = [
        Path(__file__),
        ROOT / "scripts/gtsr_coupled_geometry.py",
        ROOT / "tests/gtsr/helpers.py",
        *sorted((ROOT / "src/tsr").rglob("*.py")),
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    embedding = []
    for seed in (0, 1, 7, 42, 20260902):
        rng = np.random.default_rng(seed)
        errors = []
        for _ in range(1000):
            first, last, bounds = random_se3(rng), random_se3(rng), random_bounds(rng)
            tsr, gtsr = make_pair(T0_w=first, Tw_e=last, Bw=bounds)
            coordinates = gtsr.domain.sample(rng)
            errors.append(transform_error(tsr.to_transform(coordinates), gtsr.to_transform(coordinates)))
        maximum = np.max(errors, axis=0)
        assert np.all(maximum <= 1e-10)
        embedding.append(
            dict(seed=seed, samples=1000, max_translation_m=float(maximum[0]), max_rotation_rad=float(maximum[1]))
        )
    capacity = []
    methods = (
        "tsr",
        "tsr_union_4",
        "tsr_union_16",
        "tsr_union_64",
        "tsr_union_128",
        "tsr_union_254",
        "tsr_union_512",
        "tsr_projected",
        "gtsr",
    )
    for angle in (0.0, 0.8):
        for method in methods:
            sampler = RegionSampler(Gate(angle=angle), method)
            for seed in range(930301, 930311):
                uniforms = np.random.default_rng(seed).random((1000, 2))
                tick = time.perf_counter()
                accepted = sum(in_band(sampler.gate, sampler.sample(u)) for u in uniforms)
                capacity.append(
                    dict(
                        angle=angle,
                        method=method,
                        seed=seed,
                        samples=1000,
                        regions=sampler.count,
                        accepted=int(accepted),
                        expected_acceptance=sampler.expected_band_acceptance(),
                        sampling_and_check_seconds=time.perf_counter() - tick,
                    )
                )
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    result = dict(
        source_hashes=hashes,
        numpy_version=np.__version__,
        python=sys.version,
        embedding=embedding,
        capacity=capacity,
        construction="equal progress slabs; tight independent yaw/depth boxes; full band coverage",
        required_boxes_for_expected_95_percent=int(np.ceil(0.8 * 0.95 / (0.06 * 0.05))),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                embedding_samples=5000,
                capacity_samples=180000,
                max_translation_m=max(r["max_translation_m"] for r in embedding),
                max_rotation_rad=max(r["max_rotation_rad"] for r in embedding),
                required_boxes=result["required_boxes_for_expected_95_percent"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
