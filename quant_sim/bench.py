"""Core benchmark engine: runs speed + quality comparison across quant levels."""

import time
from dataclasses import dataclass, asdict
from typing import Optional
from quant_sim.gpu import detect_gpu, GpuInfo
from quant_sim.ollama import generate, get_model_info, InferenceResult
from quant_sim.quality import run_quality_benchmark, QUALITY_TESTS


@dataclass
class QuantResult:
    model_tag: str
    quant_label: str
    size_gb: float
    vram_peak_mb: int
    prompt_eval_rate: float  # tok/s
    eval_rate: float  # tok/s
    quality_score: float  # 0-100
    quality_by_category: dict
    fits_in_vram: bool
    is_recommended: bool = False
    error: Optional[str] = None


# Speed benchmark prompts (short, medium, long)
SPEED_PROMPTS = [
    "What is 2+2?",
    "Explain the theory of relativity in 3 sentences.",
    "Write a Python function to find the longest common subsequence of two strings. Include docstring and type hints.",
]


def benchmark_one_quant(
    model_tag: str,
    gpu: Optional[GpuInfo],
    quick: bool = False,
    quality: bool = True,
) -> QuantResult:
    """Benchmark a single quantization level."""

    # Get model size
    info = get_model_info(model_tag)
    if not info:
        return QuantResult(
            model_tag=model_tag, quant_label=_extract_quant(model_tag),
            size_gb=0, vram_peak_mb=0, prompt_eval_rate=0, eval_rate=0,
            quality_score=0, quality_by_category={}, fits_in_vram=False,
            error="Model not available",
        )

    # Extract size from model info
    size_bytes = info.get("size", 0)
    if isinstance(info.get("model_info"), dict):
        # Try to get from parameters
        pass
    size_gb = round(size_bytes / (1024**3), 1) if size_bytes else 0

    # Check if it fits
    fits = True
    if gpu and size_gb > 0:
        fits = (size_gb * 1024) < (gpu.vram_free_mb + 500)  # 500MB margin

    if not fits:
        return QuantResult(
            model_tag=model_tag, quant_label=_extract_quant(model_tag),
            size_gb=size_gb, vram_peak_mb=0, prompt_eval_rate=0, eval_rate=0,
            quality_score=0, quality_by_category={}, fits_in_vram=False,
            error="Won't fit in VRAM",
        )

    # Speed benchmark
    n_runs = 1 if quick else 3
    prompt_rates = []
    eval_rates = []

    for prompt in (SPEED_PROMPTS[:1] if quick else SPEED_PROMPTS):
        for _ in range(n_runs):
            result = generate(model_tag, prompt, max_tokens=100)
            if result:
                prompt_rates.append(result.prompt_eval_rate)
                eval_rates.append(result.eval_rate)

    avg_prompt_rate = round(sum(prompt_rates) / len(prompt_rates), 1) if prompt_rates else 0
    avg_eval_rate = round(sum(eval_rates) / len(eval_rates), 1) if eval_rates else 0

    # Quality benchmark
    quality_result = {"score": 0, "by_category": {}}
    if quality:
        n_questions = 5 if quick else 20

        def gen_fn(prompt: str) -> str:
            r = generate(model_tag, prompt, max_tokens=200)
            return r.response if r else ""

        quality_result = run_quality_benchmark(gen_fn, n_questions)

    return QuantResult(
        model_tag=model_tag,
        quant_label=_extract_quant(model_tag),
        size_gb=size_gb,
        vram_peak_mb=0,  # TODO: measure via nvidia-smi polling
        prompt_eval_rate=avg_prompt_rate,
        eval_rate=avg_eval_rate,
        quality_score=quality_result["score"],
        quality_by_category=quality_result.get("by_category", {}),
        fits_in_vram=True,
    )


def recommend(results: list[QuantResult]) -> Optional[QuantResult]:
    """Pick the best quant: highest quality above 80%, then fastest."""
    valid = [r for r in results if r.fits_in_vram and not r.error and r.eval_rate > 0]
    if not valid:
        return None

    # Quality threshold: 80%
    good_quality = [r for r in valid if r.quality_score >= 80]
    if good_quality:
        # Among good quality, pick fastest
        best = max(good_quality, key=lambda r: r.eval_rate)
    else:
        # No good quality options — pick highest quality
        best = max(valid, key=lambda r: r.quality_score)

    best.is_recommended = True
    return best


def _extract_quant(model_tag: str) -> str:
    """Extract quantization label from model tag."""
    tag = model_tag.lower()
    for q in ["q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_k_s", "q4_k_m",
              "q5_k_s", "q5_k_m", "q6_k", "q8_0", "fp16", "f16"]:
        if q in tag:
            return q.upper()
    return "default"


def format_table(results: list[QuantResult], gpu: Optional[GpuInfo]) -> str:
    """Format results as a terminal-friendly table."""
    lines = []

    if gpu:
        lines.append(f"GPU: {gpu.name} ({gpu.vram_total_mb} MB VRAM, {gpu.vram_free_mb} MB free)")
    lines.append("")
    lines.append(f"{'Quant':<12} {'Size':>6} {'VRAM':>7} {'Speed':>10} {'Quality':>8} {'Note'}")
    lines.append(f"{'-'*12} {'-'*6} {'-'*7} {'-'*10} {'-'*8} {'-'*20}")

    for r in results:
        if r.error:
            lines.append(f"{r.quant_label:<12} {r.size_gb:>5.1f}G {'--':>7} {'--':>10} {'--':>8} {r.error}")
        else:
            note = "* BEST *" if r.is_recommended else ""
            lines.append(
                f"{r.quant_label:<12} {r.size_gb:>5.1f}G "
                f"{r.vram_peak_mb:>6}M " if r.vram_peak_mb else f"{r.quant_label:<12} {r.size_gb:>5.1f}G {'--':>7} "
                + f"{r.eval_rate:>8.1f}/s {r.quality_score:>7.0f}% {note}"
            )

    # Recommendation
    rec = recommend(results)
    if rec:
        lines.append("")
        lines.append(f"Recommendation: Use {rec.quant_label}. {rec.quality_score:.0f}% quality at {rec.eval_rate:.0f} tok/s.")

    return "\n".join(lines)
