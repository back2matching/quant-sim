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

## Example Output

```
GPU: NVIDIA GeForce RTX 4080 (16376 MB VRAM, 3448 MB free)

Quant          Size    VRAM      Speed  Quality Note
------------ ------ ------- ---------- -------- ---------------
Q4_K_M         6.1G  10801M     65.4/s      60% * BEST *
Q4_K_M         4.9G   8714M     76.1/s      40%
Q4_K_M         8.6G  12602M     45.7/s      60%

Recommendation: Use Q4_K_M (Qwen/Qwen3-8B).
  60% quality at 65 tok/s, 6.1 GB.
```

## Install

```bash
pip install quant-sim
```

Requires: Ollama running locally, NVIDIA GPU.

## Usage

```bash
# Benchmark a model (auto-discovers quant variants)
quant-sim qwen2.5:7b

# Quick mode (~2 min instead of ~10 min)
quant-sim llama3.1:8b --quick

# Speed only (skip quality test)
quant-sim mistral:7b --speed-only

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

## Requirements

- Python 3.10+
- Ollama installed and running
- NVIDIA GPU with nvidia-smi

## License

Apache 2.0
