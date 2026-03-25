"""Core benchmark engine: speed + quality comparison across quant levels."""

import subprocess
from dataclasses import dataclass, asdict
from typing import Optional
from quant_sim.gpu import detect_gpu, GpuInfo
from quant_sim.ollama import generate, get_model_size_gb, get_model_quant_level
from quant_sim.quality import run_quality_benchmark


@dataclass
class QuantResult:
    model_tag: str
    quant_label: str
    size_gb: float
    vram_peak_mb: int
    prompt_eval_rate: float
    eval_rate: float
    quality_score: float
    quality_by_category: dict
    fits_in_vram: bool
    is_recommended: bool = False
    error: Optional[str] = None


SPEED_PROMPTS = [
    "What is 2+2?",
    "Explain the theory of relativity in 3 sentences.",
    "Write a Python function to find the longest common subsequence of two strings.",
]


def measure_vram() -> int:
    """Current VRAM usage in MB."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(float(r.stdout.strip()))
    except Exception:
        return 0


def benchmark_one_quant(
    model_tag: str,
    gpu: Optional[GpuInfo],
    quick: bool = False,
    quality: bool = True,
) -> QuantResult:
    """Benchmark a single quantization level."""

    quant_label = get_model_quant_level(model_tag)
    size_gb = get_model_size_gb(model_tag)

    # Check if it fits (rough estimate: model needs ~size + 2GB overhead)
    fits = True
    if gpu and size_gb > 0:
        fits = (size_gb * 1024 + 2048) < gpu.vram_total_mb

    if not fits:
        return QuantResult(
            model_tag=model_tag, quant_label=quant_label,
            size_gb=size_gb, vram_peak_mb=0, prompt_eval_rate=0, eval_rate=0,
            quality_score=0, quality_by_category={}, fits_in_vram=False,
            error="Won't fit in VRAM",
        )

    # Warm up (first inference loads model)
    vram_before = measure_vram()
    warmup = generate(model_tag, "Hi", max_tokens=5)
    if not warmup:
        return QuantResult(
            model_tag=model_tag, quant_label=quant_label,
            size_gb=size_gb, vram_peak_mb=0, prompt_eval_rate=0, eval_rate=0,
            quality_score=0, quality_by_category={}, fits_in_vram=True,
            error="Inference failed",
        )
    vram_after = measure_vram()
    vram_peak = vram_after  # Model now loaded

    # Speed benchmark
    n_runs = 1 if quick else 3
    prompts = SPEED_PROMPTS[:1] if quick else SPEED_PROMPTS
    prompt_rates = []
    eval_rates = []

    for prompt in prompts:
        for _ in range(n_runs):
            result = generate(model_tag, prompt, max_tokens=100)
            if result and result.eval_rate > 0:
                prompt_rates.append(result.prompt_eval_rate)
                eval_rates.append(result.eval_rate)

    avg_prompt_rate = round(sum(prompt_rates) / len(prompt_rates), 1) if prompt_rates else 0
    avg_eval_rate = round(sum(eval_rates) / len(eval_rates), 1) if eval_rates else 0

    # Quality benchmark
    quality_result = {"score": 0, "by_category": {}}
    if quality:
        n_q = 5 if quick else 20

        def gen_fn(prompt: str) -> str:
            r = generate(model_tag, prompt, max_tokens=200)
            return r.response if r else ""

        quality_result = run_quality_benchmark(gen_fn, n_q)

    return QuantResult(
        model_tag=model_tag,
        quant_label=quant_label,
        size_gb=size_gb,
        vram_peak_mb=vram_peak,
        prompt_eval_rate=avg_prompt_rate,
        eval_rate=avg_eval_rate,
        quality_score=quality_result["score"],
        quality_by_category=quality_result.get("by_category", {}),
        fits_in_vram=True,
    )


def recommend(results: list[QuantResult]) -> Optional[QuantResult]:
    """Pick best quant: among those with quality >= 80%, pick fastest. If none >= 80%, pick highest quality."""
    valid = [r for r in results if r.fits_in_vram and not r.error and r.eval_rate > 0]
    if not valid:
        return None

    good = [r for r in valid if r.quality_score >= 80]
    if good:
        best = max(good, key=lambda r: r.eval_rate)
    else:
        # All below 80% -- pick highest quality, break ties by speed
        best = max(valid, key=lambda r: (r.quality_score, r.eval_rate))

    return best


def format_table(results: list[QuantResult], gpu: Optional[GpuInfo]) -> str:
    # Find recommendation first (don't mutate results)
    rec = recommend(results)
    if rec:
        rec.is_recommended = True
    lines = []
    if gpu:
        lines.append(f"GPU: {gpu.name} ({gpu.vram_total_mb} MB VRAM, {gpu.vram_free_mb} MB free)")
    lines.append("")

    header = f"{'Quant':<12} {'Size':>6} {'VRAM':>7} {'Speed':>10} {'Quality':>8} {'Note'}"
    lines.append(header)
    lines.append(f"{'-'*12} {'-'*6} {'-'*7} {'-'*10} {'-'*8} {'-'*15}")

    for r in results:
        # Skip failed inference (bad tag guesses that don't exist)
        if r.error and r.error in ("Inference failed", "Model not available"):
            continue
        note = ""
        if r.error:
            note = r.error
            lines.append(f"{r.quant_label:<12} {r.size_gb:>5.1f}G {'--':>7} {'--':>10} {'--':>8} {note}")
        else:
            if r.is_recommended:
                note = "* BEST *"
            vram_str = f"{r.vram_peak_mb}M" if r.vram_peak_mb else "--"
            lines.append(
                f"{r.quant_label:<12} {r.size_gb:>5.1f}G {vram_str:>7} "
                f"{r.eval_rate:>8.1f}/s {r.quality_score:>7.0f}% {note}"
            )

    if rec:
        lines.append("")
        lines.append(f"Recommendation: {rec.quant_label} ({rec.model_tag})")
        lines.append(f"  Speed:   {rec.eval_rate:.0f} tok/s")
        if rec.quality_score > 0:
            lines.append(f"  Quality: {rec.quality_score:.0f}%")
        lines.append(f"  Size:    {rec.size_gb:.1f} GB")
        lines.append(f"  VRAM:    {rec.vram_peak_mb} MB")

        # Explain WHY this was picked
        valid = [r for r in results if r.fits_in_vram and not r.error and r.eval_rate > 0]
        if len(valid) > 1:
            lines.append("")
            lines.append("  Why this one?")
            if rec.quality_score >= 80:
                lines.append(f"    Quality >= 80% ({rec.quality_score:.0f}%), fastest in that group.")
            else:
                lines.append(f"    Highest quality available ({rec.quality_score:.0f}%).")

            # Show runner-up
            others = [r for r in valid if r.model_tag != rec.model_tag]
            if others:
                runner = max(others, key=lambda r: (r.quality_score, r.eval_rate))
                lines.append(f"  Runner-up: {runner.quant_label} ({runner.model_tag})")
                lines.append(f"    {runner.eval_rate:.0f} tok/s, {runner.quality_score:.0f}% quality, {runner.size_gb:.1f} GB")
                if runner.eval_rate > rec.eval_rate:
                    lines.append(f"    (faster but lower quality)")
                elif runner.quality_score > rec.quality_score:
                    lines.append(f"    (better quality but slower)")

    return "\n".join(lines)
