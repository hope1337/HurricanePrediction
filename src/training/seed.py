"""Determinism, device selection, and environment recording."""

from __future__ import annotations

import platform
import random
import subprocess

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(preference: str = "auto") -> torch.device:
    """'auto' -> CUDA when available else CPU; 'cpu'/'cuda' force the choice."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")
    return torch.device(preference)


def env_info() -> dict:
    """Library/framework/device versions for experiment logs."""
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        git = "unknown"
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "git_sha": git, "numpy": np.__version__}
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info
