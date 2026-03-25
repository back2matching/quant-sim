# quant-sim

Which quantization should I use? One command tells you.

```bash
pip install quant-sim
quant-sim qwen2.5:7b
```

Benchmarks every quantization level of a model on YOUR GPU. Measures speed, quality, and VRAM. Tells you the best tradeoff.

## Why

Every model on Ollama has 5+ quantization levels. You ask Reddit "should I use Q4_K_M or Q5_K_M?" and get 10 different answers. The right answer depends on YOUR GPU, YOUR tasks, YOUR quality threshold.

No existing tool benchmarks speed AND quality across quant levels automatically:
- ollamabench, llm-benchmark, LocalScore: speed only
- lm-evaluation-harness: quality only, manual setup
- ollama-grid-search: prompt tuning, not quant comparison

quant-sim does both in one command.

## Example: Compare All Quant Levels

```
$ quant-sim qwen2.5:7b --quick --speed-only

Quant          Size    VRAM      Speed  Quality Note
------------ ------ ------- ---------- -------- ---------------
Q3_K_S         3.3G  15004M    128.8/s       --
Q4_K_M         4.4G   7885M    134.2/s       -- * BEST *
Q5_K_M         5.1G   8532M    105.8/s       --
Q6_K           5.8G   9160M     89.0/s       --
Q8_0           7.5G  10813M     69.7/s       --

Recommendation: Use Q4_K_M (qwen2.5:7b-instruct-q4_k_m).
  134 tok/s, 4.4 GB.
```

## Example: Benchmark All Local Models

```
$ quant-sim --local --quick --speed-only

Quant          Size    VRAM      Speed  Note
------------ ------ ------- ---------- ---------------
Q4_K_M         7.5G   7857M    117.9/s * BEST *
Q4_K_M         4.4G   7888M    112.0/s
Q5_K_M         5.1G   8532M    101.2/s
Q4_K_M         4.9G   8619M     98.6/s
Q6_K           5.8G   9220M     89.1/s
Q4_K_M         6.1G  10717M     80.4/s
Q4_K_M         6.1G  10723M     75.9/s
Q8_0           7.5G  11096M     72.7/s
Q4_K_M         8.6G  12375M     50.9/s
Q3_K_M        13.4G  15843M      2.1/s  (CPU offload)
```

*Real output from RTX 4080 16GB with 11 models installed.*

## Install

```bash
pip install quant-sim  # coming soon — for now: pip install -e . from source
```

Requires: Ollama running locally, NVIDIA GPU.

## Usage

```bash
# Benchmark a model (auto-discovers quant variants, pulls if needed)
quant-sim qwen2.5:7b

# Benchmark ALL locally installed models (no downloads)
quant-sim --local

# Quick mode (~2 min instead of ~10 min)
quant-sim qwen2.5:7b --quick

# Speed only (skip quality test)
quant-sim --local --quick --speed-only

# Don't download anything (only test what's already installed)
quant-sim qwen2.5:7b --no-pull

# Compare specific tags
quant-sim test --tags "qwen3:8b,qwen3:14b,qwen3.5:9b"

# Save results as JSON
quant-sim qwen2.5:7b --json results.json

# Show GPU info
quant-sim --gpu

# List local models
quant-sim --list
```

## What It Measures

| Metric | How |
|--------|-----|
| **Speed** | Tokens/sec via Ollama chat API (prompt + generation) |
| **Quality** | 20 built-in questions: facts, math, coding, reasoning |
| **VRAM** | Peak GPU memory via nvidia-smi during inference |
| **Size** | Model file size from Ollama |

## Quality Test

Built-in 20-question test covering:
- **Facts** (5): capitals, science, literature
- **Math** (5): arithmetic, word problems
- **Coding** (5): Python functions, one-liners
- **Reasoning** (5): logic puzzles, trick questions

## How It Works

1. Discovers available quantization variants of your model
2. For each variant: loads model, measures VRAM, runs speed prompts, runs quality questions
3. Grades quality answers automatically (keyword matching, code syntax checking)
4. Recommends the best tradeoff: highest quality above 80%, then fastest

## Community Leaderboard

Share your benchmarks. Compare your GPU against others.

```bash
# Submit results after benchmarking
quant-sim --local --quick --submit

# View community results
quant-sim --leaderboard
```

Results stored as GitHub issues — no backend server needed. Set `GITHUB_TOKEN` env var to submit.

## Requirements

- Python 3.10+
- Ollama installed and running
- NVIDIA GPU with nvidia-smi

## License

Apache 2.0
