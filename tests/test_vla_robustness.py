import numpy as np

from vla_robustness.env import PushEnv
from vla_robustness.evaluate import evaluate, run_episode
from vla_robustness.metrics import expected_calibration_error, summarize, wilson_interval
from vla_robustness.perturbations import apply_perturbation
from vla_robustness.policy import ColorBlobPolicy


def test_env_is_deterministic_and_images_are_valid():
    a = PushEnv(seed=7).reset(seed=7)
    b = PushEnv(seed=7).reset(seed=7)
    assert np.allclose(a.state, b.state)
    assert a.image.shape == (64, 64, 3)
    assert 0.0 <= a.image.min() <= a.image.max() <= 1.0


def test_color_policy_can_push_without_perturbation():
    result = run_episode(PushEnv(seed=1), ColorBlobPolicy(), "none", 0.0, seed=1)
    assert result.success


def test_perturbations_keep_shape_and_range():
    image = PushEnv(seed=2).reset(seed=2).image
    for name in ("brightness", "contrast", "noise", "blur", "occlusion", "shift"):
        out = apply_perturbation(image, name, 1.0, np.random.default_rng(0))
        assert out.shape == image.shape
        assert 0.0 <= out.min() <= out.max() <= 1.0


def test_metrics_are_sane():
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high
    assert summarize(8, 10)["success_rate"] == 0.8
    assert 0.0 <= expected_calibration_error(np.array([0.9, 0.1]), np.array([1, 0])) <= 1.0


def test_evaluate_returns_report():
    report = evaluate(episodes=5, perturbation="brightness", severity=0.2)
    assert report["episodes"] == 5
    assert 0.0 <= report["success_rate"] <= 1.0
