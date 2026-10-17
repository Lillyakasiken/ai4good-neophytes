"""GPU snapshot from nvidia-smi.

The training queue uses the same snapshot to decide the GPU is free.
No Python package is required.
"""

import csv
import io
import shutil
import subprocess


def _blank(value):
    text = (value or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]", "[NOT SUPPORTED]"}:
        return None
    return text


def _float(value):
    text = _blank(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int(value):
    number = _float(value)
    if number is None:
        return None
    return int(number)


def _rows(args):
    if shutil.which("nvidia-smi") is None:
        raise FileNotFoundError("nvidia-smi is not installed")
    proc = subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "nvidia-smi failed").strip()
        raise RuntimeError(detail)
    return list(csv.reader(io.StringIO(proc.stdout)))


def snapshot():
    """One row per GPU, plus the compute processes holding memory."""
    try:
        gpu_rows = _rows([
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,power.limit",
        ])
        try:
            app_rows = _rows([
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            ])
        except RuntimeError:
            app_rows = []
    except FileNotFoundError as exc:
        return {"available": False, "detail": str(exc), "gpus": []}
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "detail": str(exc), "gpus": []}

    by_uuid = {}
    gpus = []
    for row in gpu_rows:
        if len(row) < 9:
            continue
        item = {
            "index": _int(row[0]),
            "name": _blank(row[2]) or "GPU",
            "util": _float(row[3]),
            "memory_used": _float(row[4]),
            "memory_total": _float(row[5]),
            "temperature": _float(row[6]),
            "power": _float(row[7]),
            "power_limit": _float(row[8]),
            "processes": [],
        }
        gpus.append(item)
        uuid = _blank(row[1])
        if uuid:
            by_uuid[uuid] = item

    for row in app_rows:
        if len(row) < 4:
            continue
        proc = {
            "pid": _int(row[1]),
            "name": _blank(row[2]) or "process",
            "memory": _float(row[3]),
        }
        target = by_uuid.get(_blank(row[0]))
        if target is None and gpus:
            target = gpus[0]
        if target is not None:
            target["processes"].append(proc)

    return {"available": True, "detail": "", "gpus": gpus}


def compute_busy():
    """True when some process holds GPU memory.

    ``nvidia-smi`` missing counts as free: the queue then only avoids
    overlapping its own job.
    """
    info = snapshot()
    if not info["available"]:
        return False
    return any(gpu["processes"] for gpu in info["gpus"])
