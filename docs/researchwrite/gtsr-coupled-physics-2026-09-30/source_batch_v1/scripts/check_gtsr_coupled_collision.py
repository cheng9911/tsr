"""Cross-check reduced centered pin clearance against independent FCL boxes."""

import argparse
import json
import sqlite3  # noqa: F401
from pathlib import Path

import mplib
import numpy as np
from gtsr_coupled_geometry import Gate
from transforms3d.euler import euler2quat


def check(output):
    if output.exists():
        raise FileExistsError(output)
    fcl = mplib.collision_detection.fcl
    rng, rows = np.random.default_rng(930199), []
    for angle in (0.0, 0.8):
        gate = Gate(angle=angle)
        walls = [
            fcl.CollisionObject(fcl.Box(2 * half), position, euler2quat(0, 0, yaw))
            for half, position, yaw in gate.walls()
        ]
        for _ in range(200):
            progress, yaw = rng.uniform(0.001, 0.999), rng.uniform(-0.15, angle + 0.15)
            pin = fcl.CollisionObject(
                fcl.Box(2 * gate.pin_halfx, 2 * gate.pin_halfy, 2 * gate.pin_halfheight),
                np.array([0, 0, gate.upper - gate.travel * progress]),
                euler2quat(0, 0, yaw),
            )
            collision = any(fcl.collide(pin, wall, fcl.CollisionRequest()).is_collision() for wall in walls)
            clearance = gate.pin_clearance(progress, yaw)
            rows.append(
                {
                    "angle": angle,
                    "progress": progress,
                    "yaw": yaw,
                    "clearance_m": clearance,
                    "fcl_collision": collision,
                    "agree": collision == (clearance < 0),
                }
            )
    result = {"cases": len(rows), "mismatches": sum(not r["agree"] for r in rows), "rows": rows}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    assert result["mismatches"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    check(parser.parse_args().output)
