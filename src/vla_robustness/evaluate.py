"""Run episodes under perturbations and write a compact JSON report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .env import PushEnv
from .metrics import summarize
from .perturbations import apply_perturbation
from .policy import ColorBlobPolicy


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    perturbation: str
    severity: float


def run_episode(env: PushEnv, policy, perturbation: str, severity: float, seed: int) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    result = env.reset(seed=seed)
    done = False
    while not done:
        image = apply_perturbation(result.image, perturbation, severity, rng)
        action = policy.act(image)
        result = env.step(action)
        done = result.done
    return EpisodeResult(result.success, env.steps, perturbation, severity)


def evaluate(episodes: int = 40, perturbation: str = "none", severity: float = 1.0) -> dict[str, object]:
    env = PushEnv(seed=0)
    policy = ColorBlobPolicy()
    results = [run_episode(env, policy, perturbation, severity, seed) for seed in range(episodes)]
    successes = sum(item.success for item in results)
    report = summarize(successes, episodes)
    report.update(
        {
            "perturbation": perturbation,
            "severity": severity,
            "mean_steps": float(np.mean([item.steps for item in results])),
            "failures": sum(not item.success for item in results),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--perturbation", default="none")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.episodes, args.perturbation, args.severity)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
