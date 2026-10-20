"""A tiny image-based pushing environment used to exercise the harness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class StepResult:
    image: np.ndarray
    state: np.ndarray
    reward: float
    done: bool
    success: bool


class PushEnv:
    """Push a block to a goal with a point agent.

    This is intentionally small. It is a testbed for the evaluation harness,
    not a replacement for PushT, LIBERO, or a real robot.
    """

    size = 64
    max_steps = 70

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> StepResult:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent = np.array([0.18, 0.18], dtype=np.float32)
        self.block = np.array([0.45, 0.45], dtype=np.float32) + self.rng.uniform(-0.05, 0.05, 2)
        self.goal = np.array([0.84, 0.84], dtype=np.float32)
        self.steps = 0
        return StepResult(self.render(), self.state(), 0.0, False, False)

    def state(self) -> np.ndarray:
        return np.concatenate([self.agent, self.block]).astype(np.float32)

    def render(self) -> np.ndarray:
        image = Image.new("RGB", (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        for i in range(1, 4):
            xy = int(self.size * i / 4)
            draw.line((xy, 0, xy, self.size), fill=(230, 230, 230))
            draw.line((0, xy, self.size, xy), fill=(230, 230, 230))
        goal = self._pixel(self.goal)
        block = self._pixel(self.block)
        agent = self._pixel(self.agent)
        draw.ellipse((goal[0] - 5, goal[1] - 5, goal[0] + 5, goal[1] + 5), outline=(0, 150, 0), width=2)
        draw.rectangle((block[0] - 5, block[1] - 5, block[0] + 5, block[1] + 5), fill=(210, 40, 40))
        draw.ellipse((agent[0] - 4, agent[1] - 4, agent[0] + 4, agent[1] + 4), fill=(40, 80, 210))
        return np.asarray(image, dtype=np.float32) / 255.0

    def _pixel(self, point: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(point[0], 0, 1) * (self.size - 1))
        y = int((1 - np.clip(point[1], 0, 1)) * (self.size - 1))
        return x, y

    def step(self, action: np.ndarray) -> StepResult:
        action = np.clip(np.asarray(action, dtype=np.float32), -0.12, 0.12)
        self.agent = np.clip(self.agent + action, 0.0, 1.0)
        if np.linalg.norm(self.agent - self.block) < 0.13:
            self.block = np.clip(self.block + action * 0.85, 0.0, 1.0)
        self.steps += 1
        success = bool(np.linalg.norm(self.block - self.goal) < 0.07)
        done = success or self.steps >= self.max_steps
        reward = 1.0 if success else -0.01
        return StepResult(self.render(), self.state(), reward, done, success)
