"""GPU detection via nvidia-smi."""

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int


def detect_gpu() -> Optional[GpuInfo]:
    """Detect NVIDIA GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(",")
        if len(parts) < 4:
            return None
        return GpuInfo(
            name=parts[0].strip(),
            vram_total_mb=int(float(parts[1].strip())),
            vram_used_mb=int(float(parts[2].strip())),
            vram_free_mb=int(float(parts[3].strip())),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
