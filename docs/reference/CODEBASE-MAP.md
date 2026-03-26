# Codebase Map

Every file in quant-sim with functions and line counts.

**Total:** 864 lines of source code, 105 lines of tests, 7 source files, 1 test file.

## Package: `quant_sim/` (864 lines)

### `__init__.py` (13 lines)

Package metadata and docstring.

- `__version__` -- `"0.1.0"`

### `bench.py` (199 lines)

Core benchmark engine. Coordinates speed and quality testing across quant levels.

**Data classes:**
- `QuantResult` -- benchmark result for one model/quant: tag, quant label, size, VRAM, speed, quality, fit status, recommendation flag, error

**Constants:**
- `SPEED_PROMPTS` -- 3 prompts used for speed measurement (trivial, medium, complex)

**Functions:**
- `measure_vram() -> int` -- current VRAM usage in MB via nvidia-smi
- `benchmark_one_quant(model_tag, gpu, quick, quality) -> QuantResult` -- full benchmark of one model: VRAM fit check, warm-up, speed test (3 prompts x 3 runs), quality test (20 questions)
- `recommend(results) -> QuantResult | None` -- pick best quant: quality >= 80% -> fastest, else highest quality
- `format_table(results, gpu) -> str` -- format results as text table with recommendation explanation

### `cli.py` (161 lines)

CLI entry point. Parses args, orchestrates the benchmark run.

**Functions:**
- `main()` -- argparse setup, dispatch to GPU info / list / leaderboard / benchmark flow

**CLI arguments:**
| Arg | Short | Description |
|-----|-------|-------------|
| `model` | | Model to benchmark (positional, optional) |
| `--quick` | `-q` | Quick mode (1 prompt, 1 run, 5 quality questions) |
| `--speed-only` | | Skip quality benchmark |
| `--json` | `-j` | Save results to JSON file |
| `--gpu` | | Show GPU info and exit |
| `--list` | | List local models |
| `--tags` | | Comma-separated quant tags to test |
| `--no-pull` | | Only test locally available models |
| `--local` | | Benchmark all locally installed models |
| `--submit` | | Submit results to community leaderboard |
| `--leaderboard` | | View community leaderboard |

### `gpu.py` (36 lines)

GPU detection via nvidia-smi.

**Data classes:**
- `GpuInfo` -- name, vram_total_mb, vram_used_mb, vram_free_mb

**Functions:**
- `detect_gpu() -> GpuInfo | None` -- query nvidia-smi for GPU name and VRAM stats

### `ollama.py` (193 lines)

Ollama API client. Handles model listing, metadata, pulling, inference, and quant tag discovery.

**Constants:**
- `OLLAMA_BASE` -- `"http://localhost:11434"`

**Data classes:**
- `InferenceResult` -- model, prompt_tokens, generated_tokens, prompt_eval_rate, eval_rate, total_duration_ms, response

**Functions:**
- `check_ollama() -> bool` -- ping Ollama health endpoint
- `list_local_models() -> list[dict]` -- get all locally installed models from `/api/tags`
- `get_model_size_gb(model_name) -> float` -- look up model file size in GB
- `get_model_quant_level(model_name) -> str` -- detect quant level from model metadata (details -> quantization_level -> model_info file_type -> name extraction)
- `_extract_quant_from_name(name) -> str` -- fallback: parse quant level from model tag string
- `pull_model(model_name) -> bool` -- download a model via streaming pull API with progress
- `generate(model, prompt, max_tokens, temperature) -> InferenceResult | None` -- run inference via chat API (non-streaming)
- `discover_quant_tags(model_base) -> list[str]` -- find all quant variants: local matches + generated common tag patterns

### `quality.py` (104 lines)

20-question quality test covering facts, math, coding, and reasoning.

**Data classes:**
- `QualityQuestion` -- category, prompt, check (grading rule), weight

**Constants:**
- `QUALITY_TESTS` -- list of 20 `QualityQuestion` instances (5 per category)

**Functions:**
- `grade_response(question, response) -> bool` -- grade one response: strip thinking tags, apply check rule (contains/exact/code)
- `run_quality_benchmark(generate_fn, n_questions) -> dict` -- run N questions, return score (0-100), per-category breakdown, details

### `leaderboard.py` (158 lines)

Community leaderboard via GitHub Issues. No backend server.

**Constants:**
- `REPO` -- `"back2matching/quant-sim"`
- `RESULTS_LABEL` -- `"benchmark-result"`

**Functions:**
- `generate_hardware_id(gpu_name) -> str` -- anonymous hardware fingerprint: sha256(gpu + os + arch)[:12]
- `format_submission(results, gpu_info) -> dict` -- format results for submission (version, timestamp, GPU, OS, results array)
- `submit_results(results, gpu_info) -> str | None` -- create GitHub Issue with benchmark results, returns issue URL
- `view_leaderboard() -> list[dict]` -- fetch benchmark-result issues, parse JSON from body

## Tests: `tests/` (105 lines)

### `test_quality.py` (105 lines)

13 tests across 5 test classes:

| Class | Tests | What |
|-------|-------|------|
| `TestGrading` | 4 | contains match, code match, thinking tag stripping, case insensitivity |
| `TestGpuDetection` | 1 | import smoke test (works without GPU) |
| `TestOllamaClient` | 1 | import smoke test (works without Ollama) |
| `TestBenchModule` | 4 | quant name extraction, recommend picks fastest above 80%, recommend picks highest when all below, recommend skips errors, recommend returns None on empty |
| `TestOllamaModule` | 2 | quant name extraction, discover includes base model |

All tests run offline (no Ollama or GPU required).

## Config Files

| File | Lines | Purpose |
|------|-------|---------|
| `pyproject.toml` | 32 | Package config, dependencies, CLI entry point |
| `.gitignore` | 6 | Exclude pycache, eggs, dist, build, pytest, JSON |
| `LICENSE` | -- | Apache 2.0 |
| `README.md` | 147 | User-facing docs, examples, usage |
| `CLAUDE.md` | -- | Claude Code instructions |
