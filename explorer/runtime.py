"""Paths shared by the explorer API and the training jobs it starts.

The page itself runs in ``explorer/.venv``. Prediction, training, and W&B use
the repo-root ``.venv`` built by ``uv sync`` from ``pyproject.toml``.
"""

import threading
from pathlib import Path

EXPLORER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPLORER_ROOT.parent
TRAIN_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
LOG_DIR = REPO_ROOT / "lightning_logs"
JOBS_DIR = EXPLORER_ROOT / "cache" / "jobs"

# Held while a prediction is in flight or a training job is being started,
# so the two cannot take the GPU at the same moment.
gpu_lock = threading.Lock()


class TrainEnvError(RuntimeError):
    """The repo-root training environment is not ready."""


def train_python():
    """Interpreter that has torch, Hydra, and W&B."""
    if not TRAIN_PYTHON.is_file():
        raise TrainEnvError(
            "Missing repo .venv. From the repo root run: uv sync"
        )
    return TRAIN_PYTHON
