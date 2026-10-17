"""Talk to the long-lived predict worker, and list checkpoints on disk."""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from runtime import EXPLORER_ROOT, LOG_DIR, REPO_ROOT, TrainEnvError, train_python

_lock = threading.Lock()
_proc = None
_log = None


class PredictError(RuntimeError):
    """The worker refused the request or exited."""


def list_checkpoints():
    """``.ckpt`` files under ``lightning_logs``, newest first. No torch."""
    rows = []
    if not LOG_DIR.is_dir():
        return {"checkpoints": rows}
    for ckpt in LOG_DIR.glob("*/checkpoints/*.ckpt"):
        if not ckpt.is_file():
            continue
        exp = ckpt.parent.parent
        name = ckpt.name
        if name.startswith("best"):
            kind = "best"
        elif "last" in name:
            kind = "last"
        else:
            kind = "other"
        rows.append({
            "exp_name": exp.name,
            "file": name,
            "path": str(ckpt.resolve()),
            "mtime": ckpt.stat().st_mtime,
            "ds2": "_ds2" in exp.name,
            "kind": kind,
        })
    rows.sort(key=lambda row: row["mtime"], reverse=True)
    return {"checkpoints": rows}


def resolve_checkpoint(path):
    """Absolute checkpoint path, restricted to ``lightning_logs``."""
    ckpt = Path(path).resolve()
    root = LOG_DIR.resolve()
    if ckpt.suffix != ".ckpt" or not ckpt.is_file() or root not in ckpt.parents:
        raise PredictError("checkpoint must be a .ckpt file under lightning_logs")
    return ckpt


def _log_handle():
    global _log
    path = EXPLORER_ROOT / "cache" / "predict_worker.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    if _log is None or _log.closed:
        _log = open(path, "ab", buffering=0)
    return _log


def _ensure():
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    python = train_python()
    _proc = subprocess.Popen(
        [str(python), "-u", str(EXPLORER_ROOT / "predict_worker.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=_log_handle(),
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return _proc


def _stop_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def stop():
    """End the worker so a training job can take the GPU."""
    global _proc
    with _lock:
        proc = _proc
        _proc = None
        _stop_proc(proc)


def _readline(proc, timeout):
    box = []

    def read():
        box.append(proc.stdout.readline())

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    return box[0] if box else b""


def _read_json(proc, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(0.1, deadline - time.time())
        line = _readline(proc, remaining)
        if line is None:
            raise PredictError("prediction timed out")
        if line == b"":
            raise PredictError("predict worker exited")
        text = line.decode("utf-8", "replace").strip()
        if text.startswith("{"):
            return json.loads(text)
    raise PredictError("prediction timed out")


def request(payload, timeout=180):
    """Send one command and return the worker's JSON object."""
    global _proc
    with _lock:
        try:
            proc = _ensure()
        except TrainEnvError:
            raise
        try:
            proc.stdin.write((json.dumps(payload) + "\n").encode())
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            _proc = None
            _stop_proc(proc)
            raise PredictError("predict worker exited") from exc
        try:
            return _read_json(proc, timeout)
        except PredictError:
            _proc = None
            _stop_proc(proc)
            raise
