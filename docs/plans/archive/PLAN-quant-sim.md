# ExecPlan: quant-sim — Which Quantization Should I Use?

> One command benchmarks every quant level of a model on YOUR GPU. Tells you the best speed/quality tradeoff.

**Created:** 2026-03-25
**Repo:** [github.com/back2matching/quant-sim](https://github.com/back2matching/quant-sim) (live, NOT on PyPI yet)
**Validated by:** Agent team research + Eng Ops review. No competing tool exists.

---

## The Problem

Every model on Ollama has 5+ quantization levels (Q3_K_S, Q4_K_M, Q5_K_M, Q6_K, Q8_0...). Users ask "which should I use?" on r/LocalLLaMA daily. The answer depends on YOUR GPU, YOUR use case, YOUR quality threshold. Nobody has a tool that answers this automatically.

**Existing tools and why they don't solve this:**
- ollamabench (speed only, single model, no quality)
- ollama-grid-search (manual model selection, prompt tuning not quant comparison)
- llm-benchmark (speed only, no quality)
- LocalScore (speed only, no quality)
- lm-evaluation-harness (quality only, no automation for quant comparison)

**The gap:** No tool auto-pulls multiple quants, benchmarks speed AND quality, and recommends the best one.

---

## The Product

```bash
pip install quant-sim

# One command — benchmarks all quant levels of a model on your GPU
quant-sim qwen2.5:14b

# Output:
# Model: qwen2.5:14b on NVIDIA RTX 4080 (16GB)
#
# | Quant    | Size  | VRAM  | Speed    | Quality | Recommendation |
# |----------|-------|-------|----------|---------|----------------|
# | Q3_K_S   | 6.0G  | 8.1G  | 45 tok/s | 82%     |                |
# | Q4_K_M   | 8.5G  | 10.8G | 38 tok/s | 94%     | ★ Best value   |
# | Q5_K_M   | 10.3G | 12.9G | 32 tok/s | 97%     |                |
# | Q6_K     | 12.0G | 14.5G | 28 tok/s | 99%     |                |
# | Q8_0     | 15.0G | OOM   | —        | —       | Won't fit      |
#
# Recommendation: Use Q4_K_M. 94% quality at 38 tok/s. Best speed/quality tradeoff.
```

---

## Phase 1: Core CLI + Ollama Backend (3-4 days)

### 1.1 Project Setup
- ✅ Repo created: github.com/back2matching/quant-sim
- ✅ GPU detection (nvidia-smi: name, VRAM total/free/used)
- ✅ Ollama API client (list, show, pull, generate via chat API, discover tags)

### 1.2 Quant Discovery
- ✅ Discovers local variants of base model
- ✅ Model size from Ollama tags API
- ⬜ Auto-discover remote quant tags from Ollama library (currently generates common patterns)
- ⬜ Auto-pull quants that aren't downloaded (pull works, discovery needs improvement)

### 1.3 Speed Benchmark
- ✅ Standardized prompts (short, medium, long)
- ✅ Prompt eval tok/s + generation tok/s
- ✅ Peak VRAM measurement via nvidia-smi during inference
- ✅ 3 runs per quant in full mode, 1 in quick mode

### 1.4 Quality Benchmark
- ✅ 20 built-in questions (facts, math, coding, reasoning)
- ✅ Auto-grading (keyword match, code syntax check)
- ✅ Quality score 0-100% per quant
- ✅ Thinking-model support (Qwen3.5 <think> tag handling)

### 1.5 Report + Recommendation
- ✅ Terminal table with size, VRAM, speed, quality
- ✅ JSON output (--json flag)
- ✅ Smart recommendation (quality >= 80% then fastest)
- ✅ "Won't fit" detection

### Success Criteria
- `quant-sim qwen2.5:7b` produces a comparison table in < 10 minutes
- Recommendation matches what an expert would suggest
- Works on any Ollama model

---

## Phase 2: Polish + Publish (2-3 days)

### 2.1 UX
- ✅ `--quick` mode (1 run, fewer prompts)
- ✅ `--speed-only` mode
- ✅ `--json` output
- ✅ `--local` mode (benchmark all installed models)
- ✅ `--no-pull` flag
- ⬜ Progress bar during benchmarks
- ⬜ Color-coded terminal output

### 2.2 Testing
- ✅ 15 unit tests (grading, recommendation engine, quant extraction, discovery)
- ✅ Tested on 11 models, recommendations verified sensible
- ✅ Agent team code review (Eng Ops gpt-5.3-codex) — 4 findings fixed

### 2.3 Publish
- ✅ Agent team review complete (Eng Ops reviewed code, 4 findings addressed)
- ✅ GitHub repo live with README showing real RTX 4080 benchmark output
- ⬜ User approval before PyPI publish
- ⬜ PyPI: `pip install quant-sim`
- ⬜ Reddit r/LocalLLaMA post with benchmark tables

---

## Phase 3: Community Leaderboard (THE differentiator)

Research shows tools with leaderboards get 500+ stars vs 50 for CLI-only.
The flywheel: run benchmark -> see score -> compare to others -> share -> more users.

- ⬜ `--submit` flag: upload anonymized results to public leaderboard
- ⬜ Leaderboard backend: simple JSON API (GitHub Pages or Supabase)
- ⬜ Leaderboard web UI: "What's the fastest model on RTX 4080?" with community data
- ⬜ Chart generation in CLI (Pareto curve: speed vs quality)
- ⬜ `--compare model1 model2` (cross-model comparison)

---

## What We Reuse

| From | What | How |
|------|------|-----|
| kvcache-bench | GPU detection, Ollama API, chart generation | Copy + adapt |
| turboquant | Quality benchmark methodology | Adapt prompts |
| FlockRun harness | Quality grading approach | Adapt scoring |

---

## Competitive Advantage

1. **One command** — no config files, no manual model selection
2. **Speed AND quality** — nobody else measures both
3. **Recommendation engine** — tells you THE answer, not just data
4. **Your GPU, your data** — not theoretical calculators
5. **Built by someone who understands quantization deeply** (we know the internals)
