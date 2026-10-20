"""Policies used by the harness. The default policy is image-only."""

from __future__ import annotations

import numpy as np


def _centroid(image: np.ndarray, predicate) -> np.ndarray | None:
    mask = predicate(image)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


class ColorBlobPolicy:
    """A deliberately simple image-only policy.

    It finds the blue agent and red block by color, then pushes the block
    toward the green goal. Perturbations that change color or position can
    break it, which is the point of the benchmark.
    """

    def act(self, image: np.ndarray) -> np.ndarray:
        red = _centroid(image, lambda x: (x[..., 0] > 0.55) & (x[..., 1] < 0.35) & (x[..., 2] < 0.35))
        blue = _centroid(image, lambda x: (x[..., 2] > 0.55) & (x[..., 0] < 0.35) & (x[..., 1] < 0.45))
        green = _centroid(image, lambda x: (x[..., 1] > 0.45) & (x[..., 0] < 0.35) & (x[..., 2] < 0.35))
        if red is None or blue is None:
            return np.zeros(2, dtype=np.float32)
        target = red
        if green is not None and np.linalg.norm(red - green) < 4.0:
            target = green
        direction = target - blue
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return np.array([direction[0] / norm, -direction[1] / norm], dtype=np.float32) * 0.1


class ZeroPolicy:
    def act(self, image: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)
