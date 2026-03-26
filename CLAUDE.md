# CLAUDE.md — quant-sim (quantsim-bench on PyPI)

> Operating instructions for Claude Code on this repo.

## What Is This?

Simulates the effect of different quantization levels (Q2, Q3, Q4, Q8, FP16) on model quality and VRAM usage. Helps users decide which quantization to use before downloading a model.

**Status:** Published 0.1.0 on PyPI as `quantsim-bench`. Finished product. No further development planned.

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.1.0 (PyPI as quantsim-bench) |
| Tests | Basic |
| Dependencies | torch, transformers |

## What This Repo IS

- A finished simulation tool for model quantization comparison
- Helps answer "should I use Q4 or Q8 for my use case?"

## What This Repo IS NOT

- Not actively developed
- Not related to TurboQuant (different kind of quantization)

## Related Projects

| Project | What |
|---------|------|
| **FlockRun** | Parent project |

## Commands

```bash
pip install quantsim-bench
quantsim-bench simulate --model "meta-llama/Llama-3.1-8B"
```

## PyPI

- Account: back2matching
- Package: quantsim-bench 0.1.0 (note: PyPI name differs from repo name)
