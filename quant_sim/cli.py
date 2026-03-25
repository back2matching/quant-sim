"""CLI entry point: quant-sim command."""

import argparse
import json
import sys
from quant_sim.gpu import detect_gpu
from quant_sim.ollama import check_ollama, list_local_models, pull_model, discover_quant_tags
from quant_sim.bench import benchmark_one_quant, recommend, format_table


def main():
    parser = argparse.ArgumentParser(
        prog="quant-sim",
        description="Which quantization should I use? Benchmarks every quant level on YOUR GPU.",
    )
    parser.add_argument("model", nargs="?", help="Model to benchmark (e.g., qwen2.5:7b)")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (~2 min instead of ~10 min)")
    parser.add_argument("--speed-only", action="store_true", help="Skip quality benchmark")
    parser.add_argument("--json", "-j", help="Save results to JSON file")
    parser.add_argument("--gpu", action="store_true", help="Show GPU info and exit")
    parser.add_argument("--list", action="store_true", help="List local models")
    parser.add_argument("--tags", help="Comma-separated quant tags to test (instead of auto-discovery)")
    parser.add_argument("--no-pull", action="store_true", help="Only test locally available models (don't download)")
    parser.add_argument("--local", action="store_true", help="Benchmark all locally installed models")
    parser.add_argument("--submit", action="store_true", help="Submit results to community leaderboard (requires GITHUB_TOKEN)")
    parser.add_argument("--leaderboard", action="store_true", help="View community leaderboard")
    args = parser.parse_args()

    # Leaderboard view
    if args.leaderboard:
        from quant_sim.leaderboard import view_leaderboard
        results = view_leaderboard()
        if not results:
            print("No community results yet. Be the first: quant-sim --local --submit")
            return
        print(f"\nCommunity Leaderboard ({len(results)} submissions)\n")
        for sub in results:
            gpu = sub.get("gpu", {})
            print(f"  {gpu.get('name', '?')} ({gpu.get('vram_mb', 0)} MB)")
            for r in sub.get("results", [])[:3]:
                rec = " *" if r.get("recommended") else ""
                print(f"    {r['quant']:<10} {r['model']:<30} {r['speed_tok_s']:>6.1f} tok/s{rec}")
            print()
        return

    # GPU info
    if args.gpu:
        gpu = detect_gpu()
        if gpu:
            print(f"GPU: {gpu.name}")
            print(f"VRAM: {gpu.vram_used_mb}/{gpu.vram_total_mb} MB ({gpu.vram_free_mb} MB free)")
        else:
            print("No NVIDIA GPU detected.")
        return

    # Check Ollama
    if not check_ollama():
        print("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    # List models
    if args.list:
        models = list_local_models()
        if models:
            print("Local models:")
            for m in models:
                size_gb = m.get("size", 0) / (1024**3)
                print(f"  {m['name']:<40s} {size_gb:.1f} GB")
        else:
            print("No models found.")
        return

    gpu = detect_gpu()

    # --local mode: benchmark all locally installed models
    if args.local:
        models = list_local_models()
        tags = [m["name"] for m in models]
        args.model = "all local models"
        args.no_pull = True
    elif not args.model:
        print("Usage: quant-sim <model>")
        print("       quant-sim qwen2.5:7b")
        print("       quant-sim --local          (benchmark all installed models)")
        print("       quant-sim --local --quick   (fast comparison of all models)")
        sys.exit(1)
    elif args.tags:
        tags = args.tags.split(",")
    else:
        tags = discover_quant_tags(args.model)

    print(f"\n{'='*60}")
    print(f"quant-sim v0.1.0")
    print(f"{'='*60}")
    if gpu:
        print(f"GPU: {gpu.name} ({gpu.vram_total_mb} MB VRAM)")
    print(f"Model: {args.model}")
    label = "models" if getattr(args, 'local', False) else "quantization levels"
    print(f"Testing {len(tags)} {label}...")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"{'='*60}\n")

    # Benchmark each quant level
    from dataclasses import asdict
    results = []

    for i, tag in enumerate(tags):
        print(f"[{i+1}/{len(tags)}] {tag}...", end=" ", flush=True)

        # Check if model is available locally
        local_models = [m["name"] for m in list_local_models()]
        tag_found = any(tag == m or tag + ":latest" == m or m.startswith(tag + ":") for m in local_models)
        if not tag_found:
            if getattr(args, 'no_pull', False):
                print("SKIP (not local, --no-pull)")
                continue
            print("pulling...", end=" ", flush=True)
            success = pull_model(tag)
            if not success:
                print("SKIP (not available)")
                continue

        result = benchmark_one_quant(
            tag, gpu,
            quick=args.quick,
            quality=not args.speed_only,
        )
        results.append(result)

        if result.error:
            print(f"{result.error}")
        else:
            q_str = f", quality={result.quality_score:.0f}%" if result.quality_score > 0 else ""
            print(f"{result.eval_rate:.1f} tok/s{q_str}")

    # Recommend
    rec = recommend(results)

    # Format output
    print(f"\n{format_table(results, gpu)}")

    # Save JSON
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.json}")

    # Submit to leaderboard
    if args.submit:
        from quant_sim.leaderboard import submit_results
        print("\nSubmitting to community leaderboard...", end=" ")
        gpu_dict = {"name": gpu.name, "vram_total_mb": gpu.vram_total_mb} if gpu else {}
        url = submit_results([asdict(r) for r in results], gpu_dict)
        if url:
            print(f"Done! {url}")
        else:
            print("Failed (set GITHUB_TOKEN env var)")


if __name__ == "__main__":
    main()
