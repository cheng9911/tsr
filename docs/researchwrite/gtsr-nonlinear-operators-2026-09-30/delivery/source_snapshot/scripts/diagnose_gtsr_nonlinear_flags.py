"""Read-only scalar diagnosis of stored boundary KKT flags; does not change results."""

import argparse
import json
from pathlib import Path

import numpy as np


def diagnose(folder):
    flags = []
    for line in (folder / "projections.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("exception") or row["success"]:
            continue
        a, rho, w = row["a"], row["rho"], row["weight"]
        theta, eta = row["z"]
        axial = 0.12 + 0.04 * theta + a * theta**2 + eta - row["query_height"]
        angular = theta - row["query_theta"]
        exact_gradient = np.array([2 * axial * (0.04 + 2 * a * theta) + 2 * w * w * angular, 2 * axial])

        def objective(z):
            t, e = z
            return (0.12 + 0.04 * t + a * t * t + e - row["query_height"]) ** 2 + w * w * (t - row["query_theta"]) ** 2

        z, lower, upper = np.array(row["z"]), np.array([0, -rho]), np.array([1, rho])
        finite_gradient = np.zeros(2)
        for j in range(2):
            plus, minus = z.copy(), z.copy()
            plus[j] = min(z[j] + 1e-6, upper[j])
            minus[j] = max(z[j] - 1e-6, lower[j])
            if plus[j] > minus[j]:
                finite_gradient[j] = (objective(plus) - objective(minus)) / (plus[j] - minus[j])

        def projected_norm(gradient):
            result = gradient.copy()
            for j in range(2):
                if upper[j] - lower[j] <= 1e-12:
                    result[j] = 0
                elif z[j] <= lower[j] + 1e-12:
                    result[j] = min(result[j], 0)
                elif z[j] >= upper[j] - 1e-12:
                    result[j] = max(result[j], 0)
            return float(max(abs(result)))

        flags.append(
            dict(
                method=row["method"],
                a=a,
                rho=rho,
                weight=w,
                seed=row["seed"],
                query=row["query"],
                original_message=row["message"],
                reference_gap=row["gap_to_reference"],
                exact_gradient=exact_gradient.tolist(),
                clipped_difference_gradient=finite_gradient.tolist(),
                analytic_kkt_norm=projected_norm(exact_gradient),
                clipped_difference_kkt_norm=projected_norm(finite_gradient),
                eta_distance_to_boundary=rho - abs(eta),
            )
        )
    return dict(
        flags=flags,
        count=len(flags),
        explanation="Near a box boundary the clipped 1e-6 finite difference is asymmetric. "
        "Its O(step) bias is comparable to the 1e-6 KKT tolerance; scalar analytic gradients "
        "isolate this diagnostic effect. Production code and raw success flags are unchanged.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.input)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("flags", result["count"])
    if result["flags"]:
        print("maximum analytic KKT norm", max(x["analytic_kkt_norm"] for x in result["flags"]))
        print(
            "clipped difference KKT range",
            min(x["clipped_difference_kkt_norm"] for x in result["flags"]),
            max(x["clipped_difference_kkt_norm"] for x in result["flags"]),
        )
