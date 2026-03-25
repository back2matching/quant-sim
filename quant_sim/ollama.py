"""Ollama API client for quant-sim."""

import json
import requests
import time
from dataclasses import dataclass
from typing import Optional

OLLAMA_BASE = "http://localhost:11434"


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_local_models() -> list[dict]:
    """List models already downloaded in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        return r.json().get("models", [])
    except Exception:
        return []


def get_model_info(model_name: str) -> Optional[dict]:
    """Get info about a specific model."""
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/show", json={"model": model_name}, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def pull_model(model_name: str, progress_callback=None) -> bool:
    """Pull a model from Ollama registry. Returns True on success."""
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/pull", json={"model": model_name}, stream=True, timeout=600)
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if progress_callback and "total" in data and "completed" in data:
                    progress_callback(data["completed"], data["total"], data.get("status", ""))
                if data.get("status") == "success":
                    return True
        return True
    except Exception as e:
        print(f"  Pull failed: {e}")
        return False


def delete_model(model_name: str) -> bool:
    """Delete a model from Ollama."""
    try:
        r = requests.delete(f"{OLLAMA_BASE}/api/delete", json={"model": model_name}, timeout=30)
        return r.status_code == 200
    except Exception:
        return False


@dataclass
class InferenceResult:
    model: str
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_rate: float  # tok/s
    eval_rate: float  # tok/s
    total_duration_ms: float
    response: str


def generate(model: str, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> Optional[InferenceResult]:
    """Run a single inference call via chat API (handles thinking models)."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
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

        # Get content (chat API separates thinking from content)
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
    except Exception as e:
        return None


def discover_quant_tags(model_base: str) -> list[str]:
    """
    Given a model base name (e.g., "qwen2.5:14b"), discover available quantization tags.

    Ollama models typically have tags like:
    - qwen2.5:14b (default, usually Q4_K_M)
    - qwen2.5:14b-instruct-q3_K_S
    - qwen2.5:14b-instruct-q4_K_M
    - qwen2.5:14b-instruct-q5_K_M
    etc.

    We check both the Ollama library page and local models.
    """
    # For now, return common quant suffixes to try
    # TODO: scrape Ollama library page for actual available tags
    base = model_base.split(":")[0] if ":" in model_base else model_base
    size = model_base.split(":")[1] if ":" in model_base else ""

    # Common quantization tags in Ollama
    quant_levels = ["q3_K_S", "q4_K_M", "q5_K_M", "q6_K", "q8_0"]

    tags = []
    for q in quant_levels:
        if size:
            tags.append(f"{base}:{size}-instruct-{q}")
        else:
            tags.append(f"{base}:{q}")

    # Also include the default tag
    tags.insert(0, model_base)

    return tags
