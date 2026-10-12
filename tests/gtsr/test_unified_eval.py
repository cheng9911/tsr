"""Independent numerical checks for the new exploratory evaluation protocol."""

import numpy as np
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation

from scripts.gtsr_unified_eval import (
    evaluate_predictions,
    finite_actions,
    inverse,
    log_coordinates,
    residual,
    task_proxy,
)
from tsr.gtsr.metrics import se3_hat, se3_vee


def test_log_matches_independent_matrix_log_for_coupled_motion():
    rng = np.random.default_rng(417)
    twists = rng.normal(size=(12, 6)) * [0.02, 0.02, 0.02, 0.4, 0.4, 0.4]
    transforms = np.array([expm(se3_hat(x)) for x in twists])
    reference = np.array([se3_vee(np.real(logm(t))) for t in transforms])
    np.testing.assert_allclose(log_coordinates(transforms), reference, atol=1e-11, rtol=0)
    np.testing.assert_allclose(log_coordinates(np.eye(4)), np.zeros(6), atol=1e-15)


def test_log_near_pi_and_small_angle():
    for angle in (1e-8, np.pi - 1e-5):
        twist = np.array([0.01, -0.02, 0.03, angle, 0, 0])
        np.testing.assert_allclose(log_coordinates(expm(se3_hat(twist))), twist, atol=1e-10)


def test_passive_frame_change_preserves_primary_metrics():
    reference = expm(se3_hat([0.1, -0.2, 0.3, 0.2, -0.1, 0.3]))
    baseline = np.tile(reference, (8, 1, 1))
    phases = np.repeat(np.arange(4), 2)
    contexts = np.array([[0.01, 0, 0, 0, 0, 0.2], [-0.01, 0.02, 0, 0.1, 0, 0]])
    actions = finite_actions(reference, contexts, np.full((8, 6), 0.5))
    observed = finite_actions(reference, contexts, np.full((8, 6), 0.9)) @ baseline
    native = actions @ baseline
    original = evaluate_predictions(native, actions, baseline, observed, reference, "keyed", phases)
    g = expm(se3_hat([1, -0.3, 0.5, 0.4, 0.2, -0.3]))
    shifted = evaluate_predictions(
        g @ native, g @ actions @ inverse(g), g @ baseline, g @ observed, g @ reference, "keyed", phases
    )
    keep = [0, *range(2, 10)]
    np.testing.assert_allclose(original[..., keep], shifted[..., keep], atol=1e-9)
    assert np.max(abs(original[..., 1] - shifted[..., 1])) > 1


def test_axial_preference_is_distinct_from_axis_alignment():
    ref = np.eye(4)
    ref[:3, :3] = Rotation.from_rotvec([0.4, -0.3, 0.2]).as_matrix()
    observed = np.tile(ref, (1, 4, 1, 1))
    yaw = np.eye(4)
    yaw[:3, :3] = Rotation.from_euler("z", 0.5).as_matrix()
    prediction = observed @ yaw  # rotate about the actual body insertion axis
    phases = np.arange(4)
    np.testing.assert_allclose(task_proxy(prediction, observed, "circular_honest", phases), 0, atol=1e-12)
    keyed = task_proxy(prediction, observed, "keyed", phases)
    np.testing.assert_allclose(keyed[0, :2], 15, atol=1e-12)
    np.testing.assert_allclose(keyed[0, 2:], 0, atol=1e-12)
    assert np.min(residual(prediction, observed)) > 14.9
    tilt = np.eye(4)
    tilt[:3, :3] = Rotation.from_euler("x", 0.2).as_matrix()
    assert np.min(task_proxy(observed @ tilt, observed, "circular_honest", phases)) > 5.9


def test_zero_nominal_anchor_removes_intercept_without_erasing_context():
    zero = expm(se3_hat([0.1, -0.02, 0.03, 0.2, 0.3, -0.2]))
    baseline = expm(se3_hat([-0.03, 0.04, 0.02, -0.1, 0.2, 0.1]))
    action = expm(se3_hat([0.02, 0, 0, 0, 0, 0.3]))
    native = action @ zero
    np.testing.assert_allclose((native @ inverse(zero)) @ baseline, action @ baseline, atol=1e-12)
    assert residual(native, action @ baseline) > 1
