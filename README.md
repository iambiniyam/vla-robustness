# vla-robustness

A small, reproducible harness for measuring how a visual policy behaves under image perturbations.

This is a research scaffold, not a claim that a VLA has been benchmarked. The default policy is a color-blob heuristic on a tiny pushing environment. It exists to make the evaluation path executable before plugging in LeRobot, OpenPI, or a real robot.

## What it measures

- success rate with Wilson confidence intervals
- mean steps to completion
- robustness under brightness, contrast, noise, blur, occlusion, and pixel shift
- optional calibration and risk-coverage utilities for selective policies

## Run

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
vla-robustness --episodes 40 --perturbation none
vla-robustness --episodes 40 --perturbation occlusion --severity 1.0
pytest -q
```

## Why this exists

A policy that succeeds in clean images but fails under a lighting change is not deployable. The harness makes the perturbation and the failure visible before adding a larger model.

## Next integration

- LeRobot `PushT` adapter
- LIBERO or SimplerEnv adapter
- VLA policy adapters for OpenPI, OpenVLA, and SmolVLA
- failure taxonomy and episode videos
