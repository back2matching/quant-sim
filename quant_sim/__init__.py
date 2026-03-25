"""
quant-sim: Which quantization should I use?

One command benchmarks every quant level of a model on YOUR GPU.
Measures speed, quality, and VRAM. Recommends the best tradeoff.

Usage:
    quant-sim qwen2.5:7b
    quant-sim llama3.1:8b --quick
    quant-sim mistral:7b --json results.json
"""

__version__ = "0.1.0"
