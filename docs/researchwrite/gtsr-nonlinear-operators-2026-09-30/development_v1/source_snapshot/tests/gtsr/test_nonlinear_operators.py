"""Independent correctness checks for the nonlinear decision-gate experiment."""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from benchmark_gtsr_nonlinear import emit, safe_reference  # noqa: E402
from gtsr_nonlinear import (  # noqa: E402
    METHODS,
    Task,
    check_pose,
    coaxial_pose,
    connection_maximum,
    finite_difference_body_jacobian,
    implementation,
    piecewise_height,
)
from gtsr_nonlinear_reference import reference_projection  # noqa: E402

from tsr.gtsr.charts import PoEChart  # noqa: E402
from tsr.gtsr.metrics import SE3Metric  # noqa: E402
from tsr.gtsr.projection import project  # noqa: E402


class NonlinearOperatorsTest(unittest.TestCase):
    def test_nonfinite_failure_preserves_record_and_next_row(self):
        handle = io.StringIO()
        emit(handle, dict(distance=float("nan"), optimality=float("inf"), success=True))
        emit(handle, dict(distance=0.0, optimality=0.0, success=True))
        rows = [json.loads(line) for line in handle.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["success"])
        self.assertIsNone(rows[0]["distance"])
        self.assertEqual(len(rows[0]["nonfinite_fields"]), 2)
        self.assertTrue(rows[1]["success"])

    def test_reference_exception_remains_unknown(self):
        with patch("benchmark_gtsr_nonlinear.reference_projection", side_effect=RuntimeError("injected")):
            result = safe_reference(Task(0.08, 0), 0.3, 0.1, 0.1)
        self.assertFalse(result["verified"])
        self.assertIsNone(result["distance"])
        self.assertIn("injected", result["refinement_failures"][0])

    def test_invalid_parameters(self):
        for a, rho in [(np.nan, 0), (0, -0.1)]:
            with self.assertRaises(ValueError):
                Task(a, rho)

    def test_constant_pitch_matches_production_poe(self):
        task = Task(0, 0.0005)
        region = implementation(task, METHODS[0])
        chart = PoEChart([[0, 0, task.h, 0, 0, 1], [0, 0, 1, 0, 0, 0]], "space")
        offset = np.eye(4)
        offset[2, 3] = task.b
        for theta in np.linspace(0, 1, 11):
            z = [theta, 0.0002]
            np.testing.assert_allclose(region.chart.forward(z), offset @ chart.forward(z), atol=1e-14)
            np.testing.assert_allclose(region.chart.jacobian(z), chart.jacobian(z), atol=1e-14)

    def test_mapped_samples_and_autodiff(self):
        task = Task(-0.08, 0.0005)
        regions = [implementation(task, m) for m in METHODS]
        for theta in np.linspace(0, 1, 7):
            z = [theta, -0.0004]
            expected = finite_difference_body_jacobian(regions[0].chart.forward, z)
            for region in regions:
                np.testing.assert_allclose(region.chart.jacobian(z), expected, atol=2e-10)
                np.testing.assert_allclose(region.to_transform(z), regions[0].to_transform(z), atol=1e-14)
                self.assertLess(check_pose(task, region.to_transform(z))["band_violation"], 1e-12)

    def test_all_methods_enforce_domain(self):
        for method in METHODS:
            region = implementation(Task(0.08, 0.0005), method)
            for z in ([-0.01, 0], [0.5, 0.001], [np.nan, 0]):
                with self.assertRaises(ValueError):
                    region.to_transform(z)

    def test_reference_exact_on_region(self):
        for rho in (0, 0.0005):
            task = Task(0.08, rho)
            result = reference_projection(task, 0.3, task.height(0.3), 0.1)
            self.assertTrue(result["verified"])
            self.assertLess(result["distance"], 1e-14)

    def test_reference_matches_independent_global_search(self):
        # Independent stochastic solver used only as a cross-check of algebraic enumeration.
        for a, rho, theta, height in [
            (-0.08, 0, 0.7, 0.17),
            (0.08, 0.0005, -0.1, 0.2),
            (-0.08, 0.0005, 1.1, 0.1),
            (0, 0, 0.4, 0.2),
        ]:
            task, weight = Task(a, rho), 0.02
            ref = reference_projection(task, theta, height, weight)
            result = differential_evolution(
                lambda z: max(abs(task.height(z[0]) - height) - rho, 0) ** 2 + weight**2 * (z[0] - theta) ** 2,
                [(0, 1)],
                seed=431,
                tol=1e-12,
                polish=True,
            )
            self.assertTrue(ref["verified"])
            self.assertAlmostEqual(ref["distance"], np.sqrt(result.fun), delta=1e-9)

    def test_reference_rejects_yaw_branch(self):
        with self.assertRaises(ValueError):
            reference_projection(Task(0.08, 0.0005), 4, 0.1, 0.1)

    def test_projection_objective_and_reference(self):
        for rho in (0, 0.0005):
            task = Task(0.08, rho)
            query = implementation(task, METHODS[0]).T0_w @ coaxial_pose(0.4, 0.18)
            reference = reference_projection(task, 0.4, 0.18, 0.1)
            for method in METHODS:
                region = implementation(task, method)
                result = project(region, query, SE3Metric(0.1), starts=[[0, -rho], [1, rho], [0.5, 0]])
                self.assertAlmostEqual(result.distance, reference["distance"], delta=1e-7)
                self.assertLessEqual(check_pose(task, result.transform)["band_violation"], 1e-10)

    def test_endpoints_valid_but_pose_connection_leaves_band(self):
        task = Task(0.08, 0.0005)
        self.assertEqual(connection_maximum(task, [0, 0], [1, 0], "coordinate"), 0)
        self.assertAlmostEqual(connection_maximum(task, [0, 0], [1, 0], "pose_linear"), 0.02)

    def test_continuous_connection_extrema(self):
        rng = np.random.default_rng(507)
        for a in (-0.08, 0, 0.08):
            task = Task(a, 0.0005)
            for _ in range(20):
                start, end = [task.domain().map_from_unit_cube(u) for u in rng.random((2, 2))]
                t = np.linspace(0, 1, 10001)
                theta = (1 - t) * start[0] + t * end[0]
                height = (1 - t) * (task.height(start[0]) + start[1]) + t * (task.height(end[0]) + end[1])
                dense = max(abs(height - task.height(theta)))
                exact = connection_maximum(task, start, end, "pose_linear")
                self.assertGreaterEqual(exact + 1e-14, dense)
                self.assertAlmostEqual(exact, dense, delta=3e-10)

    def test_piecewise_error_formula(self):
        for a in (-0.08, 0, 0.08):
            task = Task(a, 0)
            for k in (1, 4, 16, 64):
                points = np.linspace(0, 1, 8193)
                observed = max(abs(piecewise_height(task, points, k) - task.height(points)))
                self.assertAlmostEqual(observed, abs(a) / (4 * k * k), delta=1e-14)

    def test_counters_include_autodiff_forward_work(self):
        region = implementation(Task(0.08, 0.0005), "explicit_autodiff")
        region.chart.jacobian([0.5, 0])
        self.assertEqual(region.chart.counts.jacobian_calls, 1)
        self.assertEqual(region.chart.counts.map_calls, 2)
        self.assertEqual(region.chart.counts.autodiff_map_calls, 1)


if __name__ == "__main__":
    unittest.main()
