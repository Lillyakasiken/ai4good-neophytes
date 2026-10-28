"""Recent W&B runs for the explorer.

Imported by the API, this starts the repo-root interpreter, which has wandb.
Run as a script, it prints one JSON object and does not import the API.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = "ai4good-neophytes"
# The shared project is on this team, not on a personal entity.
TEAM = "max-zehnder-university-of-z-rich"
METRICS = ("val_f1_species", "test_f1_species", "val_f1", "train_loss")


def _stamp(value):
    if not value:
        return ""
    return value if isinstance(value, str) else str(value)


def _finished_at(run):
    """When a finished run stopped.

    wandb-metadata.json only records startedAt. The last heartbeat is the
    completion time W&B keeps once the run state is finished.
    """
    if run.state != "finished":
        return ""
    try:
        heartbeat = run.heartbeat_at
    except AttributeError:
        return ""
    return _stamp(heartbeat)


def _entity(api):
    """Team that owns the project, unless WANDB_ENTITY names another one."""
    preferred = os.environ.get("WANDB_ENTITY") or TEAM
    candidates = [preferred]
    if api.default_entity and api.default_entity not in candidates:
        candidates.append(api.default_entity)
    errors = []
    for entity in candidates:
        try:
            api.project(PROJECT, entity)
            return entity
        except Exception as exc:
            errors.append(f"{entity}/{PROJECT}: {exc}")
    raise RuntimeError(
        f"Could not find project {PROJECT} under {preferred}. " + " ".join(errors)
    )


def _local_exp_names():
    """Experiment folders that actually have a checkpoint on disk."""
    root = Path(__file__).resolve().parent.parent / "lightning_logs"
    if not root.is_dir():
        return []
    names = []
    for path in sorted(root.iterdir()):
        ckpt_dir = path / "checkpoints"
        if path.is_dir() and ckpt_dir.is_dir() and any(ckpt_dir.glob("*.ckpt")):
            names.append(path.name)
    return names


def _metrics(run):
    summary = dict(run.summary or {})
    metrics = {}
    for key in METRICS:
        value = summary.get(key)
        if isinstance(value, (int, float)) and value == value:
            metrics[key] = float(value)
    return metrics


def _row(run):
    config = dict(run.config or {})
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": _stamp(run.created_at),
        "finished_at": _finished_at(run),
        "encoder": config.get("encoder_name"),
        "learning_rate": config.get("learning_rate"),
        "metrics": _metrics(run),
    }


def _absorb(rows, index, run):
    """Keep one row per run name. A later match only fills metrics that are missing."""
    row = _row(run)
    name = row["name"]
    slot = index.get(name)
    if slot is None:
        index[name] = len(rows)
        rows.append(row)
        return
    kept = rows[slot]["metrics"]
    for key, value in row["metrics"].items():
        kept.setdefault(key, value)


def _script_payload(limit):
    import wandb

    api = wandb.Api()
    entity = _entity(api)
    path = f"{entity}/{PROJECT}"
    rows = []
    index = {}
    recent = api.runs(path, order="-created_at", per_page=limit)
    seen = 0
    for run in recent:
        _absorb(rows, index, run)
        seen += 1
        if seen >= limit:
            break
    # The recent page is newer few-shot runs. Checkpoints on disk are older,
    # and the prediction list joins scores by this exact folder name.
    wanted = [name for name in _local_exp_names() if name not in index]
    for start in range(0, len(wanted), 40):
        chunk = wanted[start:start + 40]
        matched = api.runs(
            path,
            filters={"display_name": {"$in": chunk}},
            per_page=100,
        )
        for run in matched:
            _absorb(rows, index, run)
    return {"available": True, "detail": "", "project": path, "runs": rows}


def main():
    limit = 30
    if len(sys.argv) == 3 and sys.argv[1] == "--limit":
        limit = max(1, min(100, int(sys.argv[2])))
    try:
        payload = _script_payload(limit)
    except Exception as exc:
        detail = str(exc)
        if "api_key" in detail or "Not logged in" in detail or "login" in detail.lower():
            detail = "W&B is not logged in for the repo .venv. From the repo root run: uv run wandb login"
        payload = {"available": False, "detail": detail, "project": PROJECT, "runs": []}
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")


def fetch(limit=30):
    """Call this file with the training interpreter. Cached by the API."""
    from runtime import REPO_ROOT, train_python

    python = train_python()
    proc = subprocess.run(
        [str(python), "-u", str(Path(__file__).resolve()), "--limit", str(limit)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    text = (proc.stdout or "").strip().splitlines()
    if not text:
        detail = (proc.stderr or "W&B query produced no output").strip()
        return {"available": False, "detail": detail[-800:], "project": PROJECT, "runs": []}
    try:
        return json.loads(text[-1])
    except json.JSONDecodeError:
        detail = (proc.stderr or text[-1]).strip()
        return {"available": False, "detail": detail[-800:], "project": PROJECT, "runs": []}


if __name__ == "__main__":
    main()
