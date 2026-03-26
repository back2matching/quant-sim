# quant-sim Documentation

> Internal docs for the quant-sim codebase. User-facing docs live in the root [README.md](../README.md).

## Documents

| Doc | What |
|-----|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the benchmark works end-to-end: model discovery, speed tests, quality tests, recommendation algorithm, leaderboard |
| [reference/CODEBASE-MAP.md](reference/CODEBASE-MAP.md) | Every file, every function, line counts |
| [reference/QUALITY-TESTS.md](reference/QUALITY-TESTS.md) | All 20 quality test questions, categories, grading rules |

## Quick Reference

- **PyPI package:** `quantsim-bench` (v0.1.0)
- **CLI command:** `quant-sim`
- **Entry point:** `quant_sim.cli:main`
- **Dependencies:** `requests` (only runtime dep). Optional: `matplotlib` for charts.
- **Requires:** Ollama running locally, NVIDIA GPU with nvidia-smi
- **Tests:** 13 (pytest), all in `tests/test_quality.py`
- **Total source:** ~864 lines across 7 Python files

## Project Status

Finished product. Published on PyPI. No further development planned.
