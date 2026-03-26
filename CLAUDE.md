# CLAUDE.md -- quant-sim

> Operating instructions for Claude Code on this repo.

## What Is This?

Quantization benchmark tool for local LLMs via Ollama. Benchmarks every quant level of a model on your GPU for speed and quality, then recommends the best tradeoff.

**PyPI:** `quantsim-bench` v0.1.0. **CLI command:** `quant-sim`. **Status:** Finished product.

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.1.0 |
| PyPI name | quantsim-bench |
| CLI command | quant-sim |
| Source files | 7 (864 lines) |
| Test file | 1 (105 lines, 13 tests) |
| Dependencies | requests (only runtime dep) |
| Python | >= 3.10 |
| License | Apache 2.0 |

## Architecture

Pipeline: **Discover models -> Detect GPU -> Benchmark each (speed + quality) -> Recommend -> Display**.

6 modules, each with a single responsibility:

| Module | Lines | Purpose |
|--------|-------|---------|
| `bench.py` | 199 | Benchmark engine: speed tests, quality tests, VRAM measurement, recommendation algorithm, table formatting |
| `cli.py` | 161 | CLI entry point (argparse), dispatches to benchmark/GPU/list/leaderboard flows |
| `ollama.py` | 193 | Ollama HTTP API client: model listing, metadata, pulling, inference, quant tag discovery |
| `quality.py` | 104 | 20-question quality test (facts, math, coding, reasoning), grading rules |
| `gpu.py` | 36 | NVIDIA GPU detection via nvidia-smi |
| `leaderboard.py` | 158 | Community leaderboard via GitHub Issues (submit + view) |

## Recommendation Algorithm

Two-tier: quality >= 80% -> pick fastest in that group. All below 80% -> pick highest quality, break ties by speed.

## Quality Test

20 questions, 4 categories (5 each): facts, math, coding, reasoning. Three grading types:
- `contains:X` -- case-insensitive substring match
- `exact:X` -- case-insensitive exact match
- `code:X` -- case-sensitive substring match (for Python syntax)

Thinking tags (`<think>...</think>`) are stripped before grading.

Full question list: [docs/reference/QUALITY-TESTS.md](docs/reference/QUALITY-TESTS.md).

## Commands

```bash
# Install
pip install quantsim-bench
pip install -e .  # dev install from source

# Run
quant-sim qwen2.5:7b              # benchmark one model
quant-sim --local                  # benchmark all local models
quant-sim --local --quick          # fast mode
quant-sim --local --speed-only     # skip quality test
quant-sim --gpu                    # show GPU info
quant-sim --list                   # list local models
quant-sim --leaderboard            # view community results

# Test
pytest tests/
pytest tests/test_quality.py -v

# Build + publish
python -m build
twine upload dist/*
```

## Key Files

```
quant_sim/
  __init__.py        -- package metadata, __version__
  bench.py           -- benchmark engine (QuantResult, recommend, format_table)
  cli.py             -- CLI entry point (main)
  gpu.py             -- GPU detection (GpuInfo, detect_gpu)
  ollama.py          -- Ollama client (generate, discover_quant_tags, pull_model)
  quality.py         -- quality test (QUALITY_TESTS, grade_response, run_quality_benchmark)
  leaderboard.py     -- GitHub Issues leaderboard (submit_results, view_leaderboard)
tests/
  test_quality.py    -- 13 tests (grading, GPU import, Ollama import, recommend, quant extraction)
docs/
  README.md          -- doc index
  ARCHITECTURE.md    -- how the benchmark works end-to-end
  reference/
    CODEBASE-MAP.md  -- every file, every function, line counts
    QUALITY-TESTS.md -- all 20 questions with grading criteria
```

## Ollama API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Health check |
| `GET /api/tags` | List local models, file sizes |
| `POST /api/show` | Model metadata (quant level) |
| `POST /api/pull` | Download model (streaming) |
| `POST /api/chat` | Inference (non-streaming, returns timing) |

## Gotchas

- PyPI name (`quantsim-bench`) differs from repo name (`quant-sim`) and package name (`quant_sim`)
- `_extract_quant_from_name` is the fallback when Ollama metadata doesn't report quant level
- VRAM measurement requires nvidia-smi on PATH
- Speed prompts use `max_tokens=100`, quality prompts use `max_tokens=200`
- Quick mode: 1 speed prompt x 1 run + 5 quality questions. Full mode: 3 prompts x 3 runs + 20 questions.
- `discover_quant_tags` generates candidates that may not exist on Ollama's registry. Non-existent tags fail at pull time and are skipped.
- Leaderboard uses GitHub Issues API. Submitting requires `GITHUB_TOKEN` env var. Viewing is unauthenticated.

## Further Reading

- [docs/README.md](docs/README.md) -- doc index
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) -- pipeline details
- [docs/reference/CODEBASE-MAP.md](docs/reference/CODEBASE-MAP.md) -- file inventory
- [docs/reference/QUALITY-TESTS.md](docs/reference/QUALITY-TESTS.md) -- quality test details
