"""Ollama API client for quant-sim."""

import json
import requests
import re
from dataclasses import dataclass
from typing import Optional

OLLAMA_BASE = "http://localhost:11434"


def check_ollama() -> bool:
    try:
        return requests.get(f"{OLLAMA_BASE}/", timeout=3).status_code == 200
    except Exception:
        return False


def list_local_models() -> list[dict]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        return r.json().get("models", [])
    except Exception:
        return []


def get_model_size_gb(model_name: str) -> float:
    """Get model file size in GB from Ollama tags API."""
    try:
        models = list_local_models()
        for m in models:
            # Match exact name or name:latest
            if m["name"] == model_name or m["name"] == model_name + ":latest":
                return round(m.get("size", 0) / (1024**3), 1)
            # Partial match
            if model_name in m["name"] or m["name"].startswith(model_name):
                return round(m.get("size", 0) / (1024**3), 1)
        return 0
    except Exception:
        return 0


def get_model_quant_level(model_name: str) -> str:
    """Detect quantization level from model metadata."""
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/show", json={"model": model_name}, timeout=10)
        if r.status_code != 200:
            return "unknown"
        data = r.json()
        # Check details
        details = data.get("details", {})
        ql = details.get("quantization_level", "")
        if ql:
            return ql.upper()
        # Check model_info for file_type
        info = data.get("model_info", {})
        for k, v in info.items():
            if "file_type" in k.lower() and isinstance(v, (int, str)):
                ft = str(v)
                # Map common file type IDs
                ft_map = {"2": "Q4_0", "7": "Q8_0", "15": "Q4_K_M", "17": "Q5_K_M", "18": "Q6_K"}
                return ft_map.get(ft, f"FT{ft}")
        # Try from model name
        return _extract_quant_from_name(model_name)
    except Exception:
        return _extract_quant_from_name(model_name)


def _extract_quant_from_name(name: str) -> str:
    """Extract quant level from model name/tag."""
    name_lower = name.lower()
    for q in ["q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_k_s", "q4_k_m",
              "q5_0", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "fp16", "f16"]:
        if q in name_lower:
            return q.upper()
    return "default"


def pull_model(model_name: str) -> bool:
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/pull", json={"model": model_name}, stream=True, timeout=600)
        last_pct = 0
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if "total" in data and "completed" in data and data["total"] > 0:
                    pct = int(data["completed"] / data["total"] * 100)
                    if pct >= last_pct + 20:
                        print(f"{pct}%...", end=" ", flush=True)
                        last_pct = pct
                if data.get("status") == "success":
                    return True
        return True
    except Exception as e:
        print(f"pull failed: {e}")
        return False


@dataclass
class InferenceResult:
    model: str
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_rate: float
    eval_rate: float
    total_duration_ms: float
    response: str


def generate(model: str, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> Optional[InferenceResult]:
    """Run inference via chat API (handles thinking models)."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            },
            timeout=300,
        )
        data = r.json()
        if "error" in data:
            return None

        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)
        prompt_eval_dur = data.get("prompt_eval_duration", 1) / 1e9
        eval_dur = data.get("eval_duration", 1) / 1e9

        msg = data.get("message", {})
        response = msg.get("content", "")

        return InferenceResult(
            model=model,
            prompt_tokens=prompt_eval_count,
            generated_tokens=eval_count,
            prompt_eval_rate=round(prompt_eval_count / prompt_eval_dur, 1) if prompt_eval_dur > 0 else 0,
            eval_rate=round(eval_count / eval_dur, 1) if eval_dur > 0 else 0,
            total_duration_ms=data.get("total_duration", 0) / 1e6,
            response=response,
        )
    except Exception:
        return None


def discover_quant_tags(model_base: str) -> list[str]:
    """
    Discover available quantization tags for a model.

    Strategy:
    1. Check local models for variants of the same base
    2. Generate common Ollama tag patterns to try pulling
    """
    base = model_base.split(":")[0] if ":" in model_base else model_base

    # Check what's already local
    local = list_local_models()
    local_names = [m["name"] for m in local]

    # Find local variants of this base model
    local_variants = [n for n in local_names if n.startswith(base + ":") or n.startswith(base + "/")]

    # Common Ollama tag patterns for quantized variants
    # Format: model:size-quant or model:quant
    size_match = re.search(r':(\d+b)', model_base.lower())
    size = size_match.group(1) if size_match else ""

    candidate_tags = []

    # Add the base model first
    candidate_tags.append(model_base)

    # Add local variants we already have
    for v in local_variants:
        if v != model_base and v not in candidate_tags:
            candidate_tags.append(v)

    # Generate common quant tag patterns to try
    quant_levels = ["q3_K_S", "q4_K_M", "q5_K_M", "q6_K", "q8_0"]
    for q in quant_levels:
        # Try: base:Xb-instruct-quant (common for instruct models)
        if size:
            tag = f"{base}:{size}-instruct-{q.lower()}"
            if tag not in candidate_tags:
                candidate_tags.append(tag)
        # Try: base:quant
        tag = f"{base}:{q.lower()}"
        if tag not in candidate_tags:
            candidate_tags.append(tag)

    return candidate_tags
