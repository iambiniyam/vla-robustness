"""Image perturbations used to stress a visual policy."""

from __future__ import annotations

import numpy as np


def _clip(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def brightness(image: np.ndarray, severity: float) -> np.ndarray:
    return _clip(image * (1.0 + severity))


def contrast(image: np.ndarray, severity: float) -> np.ndarray:
    return _clip((image - 0.5) * (1.0 + severity) + 0.5)


def gaussian_noise(image: np.ndarray, severity: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return _clip(image + rng.normal(0.0, 0.08 * severity, image.shape))


def blur(image: np.ndarray, severity: float) -> np.ndarray:
    if severity <= 0:
        return _clip(image)
    radius = max(1, round(severity))
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    total = np.zeros_like(image)
    count = 0
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            total += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
            count += 1
    return _clip(total / count)


def occlude(image: np.ndarray, severity: float) -> np.ndarray:
    out = image.copy()
    size = int(8 + 10 * severity)
    height, width = out.shape[:2]
    x = int(width * 0.58)
    y = int(height * 0.18)
    out[y : y + size, x : x + size] = 0.0
    return _clip(out)


def shift(image: np.ndarray, severity: float) -> np.ndarray:
    pixels = round(4 * severity)
    if pixels == 0:
        return _clip(image)
    return _clip(np.roll(image, shift=(pixels, -pixels), axis=(0, 1)))


PERTURBATIONS = {
    "none": lambda image, severity, rng: _clip(image),
    "brightness": lambda image, severity, rng: brightness(image, severity),
    "contrast": lambda image, severity, rng: contrast(image, severity),
    "noise": gaussian_noise,
    "blur": lambda image, severity, rng: blur(image, severity),
    "occlusion": lambda image, severity, rng: occlude(image, severity),
    "shift": lambda image, severity, rng: shift(image, severity),
}


def apply_perturbation(
    image: np.ndarray,
    name: str = "none",
    severity: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation: {name}")
    return PERTURBATIONS[name](np.asarray(image, dtype=np.float32), severity, rng or np.random.default_rng(0))
