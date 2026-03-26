# Architecture

How quant-sim works, from CLI invocation to recommendation.

## Overview

quant-sim answers one question: "Which quantization level should I run on my GPU?" It benchmarks every available quant of a model (or all local models) for speed and quality, then picks the best tradeoff.

The pipeline: **Discover models -> Detect GPU -> Benchmark each (speed + quality) -> Recommend best -> Display results**.

## Pipeline Stages

### 1. Model Discovery (`ollama.py: discover_quant_tags`)

Given a base model like `qwen2.5:7b`, the tool finds all quantization variants to test.

**Strategy:**
1. Query Ollama's `/api/tags` for locally installed models matching the base name
2. Generate candidate tags from common Ollama naming patterns: `base:Xb-instruct-{quant}` and `base:{quant}`
3. Standard quant levels tried: Q3_K_S, Q4_K_M, Q5_K_M, Q6_K, Q8_0

The base model tag always goes first. Local variants come next, then generated candidates.

With `--local` mode, discovery is skipped entirely. All locally installed models are tested.

With `--tags`, the user provides an explicit comma-separated list.

### 2. GPU Detection (`gpu.py: detect_gpu`)

Runs `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free` to get GPU name, total VRAM, current usage, and free VRAM. Returns `None` if no NVIDIA GPU is present.

Used for two things:
- VRAM fit check before loading a model
- Display in results table and leaderboard submissions

### 3. VRAM Fit Check (`bench.py: benchmark_one_quant`)

Before benchmarking, checks if the model will fit: `(model_size_gb * 1024 + 2048) < gpu.vram_total_mb`. The 2GB overhead accounts for KV cache, CUDA context, and working memory.

Models that don't fit are skipped with a "Won't fit in VRAM" note.

### 4. Speed Benchmark (`bench.py: benchmark_one_quant`)

**Warm-up:** One short inference (`"Hi"`, 5 tokens max) to force Ollama to load the model into VRAM. VRAM is measured before and after to capture peak usage.

**Speed test:** Runs 3 prompts (1 in quick mode) across 3 iterations each (1 in quick mode):
- `"What is 2+2?"` (trivial)
- `"Explain the theory of relativity in 3 sentences."` (medium)
- `"Write a Python function to find the longest common subsequence of two strings."` (complex)

Each prompt generates up to 100 tokens via Ollama's chat API (`/api/chat`). Two metrics are captured per inference:
- **prompt_eval_rate:** tokens/sec for processing the input prompt
- **eval_rate:** tokens/sec for generating output tokens

Final speed = average eval_rate across all runs.

### 5. Quality Benchmark (`quality.py: run_quality_benchmark`)

20 questions across 4 categories (5 each): facts, math, coding, reasoning. See [QUALITY-TESTS.md](reference/QUALITY-TESTS.md) for the full question set.

Each question has a grading rule (check string):
- **`contains:X`** -- response must contain X (case-insensitive)
- **`exact:X`** -- response must exactly match X (case-insensitive)
- **`code:X`** -- response must contain X (case-sensitive, for code syntax)

Before grading, `<think>...</think>` tags are stripped (Qwen-style thinking models wrap reasoning in these tags).

In quick mode, only 5 questions are used instead of 20.

Score = (correct / total) * 100, reported as a percentage.

### 6. Recommendation Algorithm (`bench.py: recommend`)

Two-tier selection:

1. **Filter:** Only models that fit in VRAM, had no errors, and have eval_rate > 0.
2. **If any model scores >= 80% quality:** Pick the fastest one from that group. Speed wins when quality is good enough.
3. **If ALL models score < 80%:** Pick highest quality. Break ties by speed.

This means:
- A Q4 at 85% quality and 130 tok/s beats a Q8 at 95% quality and 70 tok/s
- But a Q3 at 60% quality loses to a Q4 at 70% quality, even if the Q3 is faster

The 80% threshold is the "good enough" line. Below it, you're losing too much to quantization.

### 7. Output Formatting (`bench.py: format_table`)

Results are displayed as a table with columns: Quant, Size, VRAM, Speed (tok/s), Quality (%), Note.

The recommended model gets a `* BEST *` tag. Below the table, a recommendation block explains:
- What was picked and its stats
- Why it was picked (fastest above 80%, or highest quality available)
- Runner-up comparison (faster but lower quality, or better quality but slower)

Failed inferences ("Inference failed", "Model not available") are silently omitted from the table.

### 8. Community Leaderboard (`leaderboard.py`)

Results are stored as GitHub Issues on the `back2matching/quant-sim` repo. No backend server needed.

**Submit** (`--submit`): Creates a GitHub Issue with the `benchmark-result` label. Issue body contains a markdown table and a raw JSON block in a `<details>` tag. Requires `GITHUB_TOKEN` env var.

**View** (`--leaderboard`): Fetches open issues with the `benchmark-result` label, parses the JSON from each issue body, displays a summary grouped by GPU.

Hardware fingerprinting: `sha256(gpu_name + os + arch)[:12]` generates an anonymous hardware ID. No personal info is collected.

## Data Flow

```
CLI args
  |
  v
discover_quant_tags(model)  -- or --  list_local_models()
  |
  v
detect_gpu()
  |
  v
for each tag:
  benchmark_one_quant(tag, gpu)
    |-> measure_vram() (before/after)
    |-> generate() x N  (speed)
    |-> run_quality_benchmark()  (quality)
    |-> QuantResult
  |
  v
recommend(results)  -- pick best
  |
  v
format_table(results, gpu)  -- display
  |
  v
optional: --json, --submit
```

## Ollama API Usage

All inference goes through Ollama's HTTP API at `localhost:11434`:

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Health check (`check_ollama`) |
| `GET /api/tags` | List local models, get file sizes |
| `POST /api/show` | Get model metadata (quant level, file type) |
| `POST /api/pull` | Download a model (streaming progress) |
| `POST /api/chat` | Run inference (non-streaming, returns timing stats) |

All requests use `stream: False` for chat. Pull uses streaming to show download progress.
