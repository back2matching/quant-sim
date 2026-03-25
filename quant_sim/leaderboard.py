"""Community leaderboard: submit and view benchmark results."""

import json
import hashlib
import platform
import requests
from datetime import datetime
from dataclasses import asdict
from typing import Optional

# Leaderboard API (GitHub Gist as simple JSON store)
# For v1: results are submitted as GitHub issues on the quant-sim repo
# This avoids needing a backend server entirely
REPO = "back2matching/quant-sim"
RESULTS_LABEL = "benchmark-result"


def generate_hardware_id(gpu_name: str) -> str:
    """Generate anonymous hardware fingerprint (GPU + OS, no personal info)."""
    raw = f"{gpu_name}:{platform.system()}:{platform.machine()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def format_submission(results: list, gpu_info: dict) -> dict:
    """Format benchmark results for submission."""
    return {
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "gpu": {
            "name": gpu_info.get("name", "unknown"),
            "vram_mb": gpu_info.get("vram_total_mb", 0),
            "hardware_id": generate_hardware_id(gpu_info.get("name", "")),
        },
        "os": platform.system(),
        "results": [
            {
                "model": r.get("model_tag", ""),
                "quant": r.get("quant_label", ""),
                "size_gb": r.get("size_gb", 0),
                "vram_mb": r.get("vram_peak_mb", 0),
                "speed_tok_s": r.get("eval_rate", 0),
                "quality_pct": r.get("quality_score", 0),
                "recommended": r.get("is_recommended", False),
            }
            for r in results
            if not r.get("error")
        ],
    }


def submit_results(results: list, gpu_info: dict) -> Optional[str]:
    """
    Submit benchmark results to the community leaderboard.

    Uses GitHub Issues API to store results (no backend needed).
    Returns the issue URL on success, None on failure.
    """
    submission = format_submission(results, gpu_info)

    # Format as a readable GitHub issue
    gpu = submission["gpu"]
    title = f"Benchmark: {gpu['name']} ({len(submission['results'])} models)"

    body_lines = [
        f"## Benchmark Results",
        f"",
        f"**GPU:** {gpu['name']} ({gpu['vram_mb']} MB VRAM)",
        f"**OS:** {submission['os']}",
        f"**Date:** {submission['timestamp'][:10]}",
        f"**quant-sim:** v{submission['version']}",
        f"",
        f"| Model | Quant | Size | VRAM | Speed | Quality |",
        f"|-------|-------|------|------|-------|---------|",
    ]

    for r in submission["results"]:
        rec = " *" if r["recommended"] else ""
        body_lines.append(
            f"| {r['model']} | {r['quant']} | {r['size_gb']:.1f}G | "
            f"{r['vram_mb']}M | {r['speed_tok_s']:.1f}/s | "
            f"{r['quality_pct']:.0f}%{rec} |"
        )

    body_lines.extend([
        f"",
        f"<details>",
        f"<summary>Raw JSON</summary>",
        f"",
        f"```json",
        json.dumps(submission, indent=2),
        f"```",
        f"</details>",
    ])

    body = "\n".join(body_lines)

    # Submit as GitHub issue (requires GITHUB_TOKEN env var)
    import os
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("  Set GITHUB_TOKEN to submit results.")
        print("  Get a token at: https://github.com/settings/tokens")
        return None

    try:
        r = requests.post(
            f"https://api.github.com/repos/{REPO}/issues",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={
                "title": title,
                "body": body,
                "labels": [RESULTS_LABEL],
            },
            timeout=15,
        )
        if r.status_code == 201:
            return r.json().get("html_url")
        else:
            print(f"  Submit failed: {r.status_code} {r.text[:200]}")
            return None
    except Exception as e:
        print(f"  Submit failed: {e}")
        return None


def view_leaderboard() -> list[dict]:
    """
    Fetch community results from the leaderboard.

    Reads benchmark-result labeled issues from the GitHub repo.
    """
    try:
        r = requests.get(
            f"https://api.github.com/repos/{REPO}/issues",
            params={"labels": RESULTS_LABEL, "state": "open", "per_page": 50},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=15,
        )
        if r.status_code != 200:
            return []

        results = []
        for issue in r.json():
            # Parse JSON from the issue body
            body = issue.get("body", "")
            if "```json" in body:
                json_str = body.split("```json")[1].split("```")[0].strip()
                try:
                    data = json.loads(json_str)
                    results.append(data)
                except json.JSONDecodeError:
                    pass
        return results
    except Exception:
        return []
