"""One-GPU training queue. Jobs run in their own tmux session, one at a time.

The spec written for each job is what ``run_job.py`` executes. Overrides are
checked here and again in that script. Nothing is passed through a shell.
"""

import json
import re
import shutil
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone

import yaml

from gpu import compute_busy, snapshot
from runtime import EXPLORER_ROOT, JOBS_DIR, REPO_ROOT, TrainEnvError, gpu_lock, train_python

JOBS_PATH = JOBS_DIR / "jobs.json"
_lock = threading.Lock()
_thread = None

KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
VALUE = re.compile(r"^[A-Za-z0-9_./+\-:,\[\]{}]+$")
# A directory name and a W&B display name. No spaces or slashes.
EXP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$")
PRECISIONS = ("bf16-mixed", "16-mixed", "32-true")
LOGGERS = ("wandb", "csv")
# Same names as few_shot_mode in configs/data/data.yaml.
FEW_SHOT_MODES = (
    "max_px_site_pheno",
    "max_px_site",
    "max_px_pheno",
    "max_px",
    "random",
    "all",
)
DATA_DEFAULT = "neophytes_split_cv3_train_1024"
MODEL_DEFAULT = "model_neophytes_mit_b2"
# Example command in a model config's header, e.g. the 512 schedule in
# model_neophytes_mit_b2_ndsm_dev.yaml. The form selects that data config.
DOCUMENTED_DATA = re.compile(r"#\s*python train\.py data=([A-Za-z0-9_]+)")


class JobError(ValueError):
    """The form or the machine cannot accept this job."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load():
    if not JOBS_PATH.is_file():
        return []
    try:
        return json.loads(JOBS_PATH.read_text()).get("jobs", [])
    except json.JSONDecodeError:
        return []


def _save(jobs):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = JOBS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"jobs": jobs}, indent=2))
    tmp.replace(JOBS_PATH)


def data_configs():
    names = []
    folder = REPO_ROOT / "configs" / "data"
    for path in sorted(folder.glob("neophytes_split_*.yaml")):
        stem = path.stem
        if "_test" in stem:
            continue
        if "train" in stem or "debug" in stem:
            names.append(stem)
    return names


def model_configs():
    names = []
    folder = REPO_ROOT / "configs" / "model"
    for path in sorted(folder.glob("model_*.yaml")):
        stem = path.stem
        if stem == "model":
            continue
        if stem == "model_debug" or stem.startswith("model_neophytes_"):
            names.append(stem)
    return names


def _model_schedule(stem):
    """Schedule fields a model config sets on top of configs/model/model.yaml.

    ``data`` is set only when the config's header documents a train.py command
    with a data config the form is allowed to launch. That is how
    model_neophytes_mit_b2_ndsm_dev asks for the 512 split instead of the
    form's 1024 default.
    """
    folder = REPO_ROOT / "configs" / "model"
    base = yaml.safe_load((folder / "model.yaml").read_text()) or {}
    path = folder / f"{stem}.yaml"
    text = path.read_text()
    doc = yaml.safe_load(text) or {}
    merged = dict(base)
    merged.update(doc)
    epochs = int(merged["max_epochs"])
    every = int(merged["check_val_every_n_epoch"])
    allowed = set(data_configs())
    data = next((name for name in DOCUMENTED_DATA.findall(text) if name in allowed), None)
    note = ""
    if data:
        span = "epoch" if every == 1 else f"{every} epochs"
        note = f"This config trains on {data}: {epochs} epochs, validation every {span}."
    return {
        "max_epochs": epochs,
        "lr": float(merged["lr"]),
        "precision": str(merged["precision"]),
        "check_val_every_n_epoch": every,
        "data": data,
        "note": note,
    }


def train_options():
    models = model_configs()
    default_model = _model_schedule(MODEL_DEFAULT)
    return {
        "data": data_configs(),
        "models": models,
        "model_forms": {name: _model_schedule(name) for name in models},
        "precisions": list(PRECISIONS),
        "loggers": list(LOGGERS),
        "defaults": {
            "data": DATA_DEFAULT,
            "model": MODEL_DEFAULT,
            "seed": 42,
            "max_epochs": default_model["max_epochs"],
            "check_val_every_n_epoch": default_model["check_val_every_n_epoch"],
            "lr": default_model["lr"],
            "batch_size_train": 8,
            "batch_size_val": 2,
            "precision": default_model["precision"],
            "logger": "wandb",
            "run_test": False,
            "few_shot_k": None,
            "few_shot_bg_k": 10,
            "few_shot_mode": "max_px_site_pheno",
            "val_few_shot_k": None,
            "val_few_shot_bg_k": 0,
            "val_few_shot_mode": "all",
        },
        "few_shot_modes": list(FEW_SHOT_MODES),
    }


def _count(value, name, low, high):
    """An optional tile count. None and '' mean the field was left blank."""
    if value is None or value == "":
        return None
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise JobError(f"{name} must be a whole number") from exc
    if not low <= count <= high:
        raise JobError(f"{name} must be from {low} to {high}")
    return count


def few_shot_overrides(body):
    """Hydra args for a per-species subset. Blank counts use the whole split.

    Background tiles are only sent together with a species count. Otherwise the
    training config would add its own default (1000 background tiles) on top of
    a small species subset.
    """
    args = []
    train_k = _count(body.get("few_shot_k"), "images per species", 1, 5000)
    if train_k is not None:
        bg = _count(body.get("few_shot_bg_k"), "background tiles", 0, 5000)
        mode = body.get("few_shot_mode") or "max_px_site_pheno"
        if mode not in FEW_SHOT_MODES:
            raise JobError("unknown tile selection")
        args += [
            f"data.few_shot_k={train_k}",
            f"data.few_shot_bg_k={0 if bg is None else bg}",
            f"data.few_shot_mode={mode}",
        ]
    val_k = _count(body.get("val_few_shot_k"), "val images per species", 1, 5000)
    if val_k is not None:
        bg = _count(body.get("val_few_shot_bg_k"), "val background tiles", 0, 5000)
        mode = body.get("val_few_shot_mode") or "all"
        if mode not in FEW_SHOT_MODES:
            raise JobError("unknown val tile selection")
        args += [
            f"data.val_few_shot_k={val_k}",
            f"data.val_few_shot_bg_k={0 if bg is None else bg}",
            f"data.val_few_shot_mode={mode}",
        ]
    return args


def _check_override(arg):
    if not isinstance(arg, str) or len(arg) > 240 or arg.count("=") < 1:
        raise JobError("Each override must be key=value with no spaces.")
    key, value = arg.split("=", 1)
    if not KEY.fullmatch(key) or not VALUE.fullmatch(value):
        raise JobError(f"Override not allowed: {arg}")
    return arg


def _command_text(argv):
    return " ".join(argv)


def training_active():
    with _lock:
        return any(job["status"] == "running" for job in _load())


def _session(job_id):
    return f"np-{job_id}"


def _session_alive(job_id):
    if shutil.which("tmux") is None:
        return False
    proc = subprocess.run(
        ["tmux", "has-session", "-t", _session(job_id)],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _reconcile(job):
    if job["status"] != "running":
        return job
    if _session_alive(job["id"]):
        return job
    exit_path = JOBS_DIR / f"{job['id']}.exit"
    job["finished"] = _now()
    if not exit_path.is_file():
        job["status"] = "failed"
        job["error"] = "tmux session ended without an exit code"
        return job
    try:
        code = int(exit_path.read_text().strip() or "1")
    except ValueError:
        code = 1
    job["exit_code"] = code
    job["status"] = "done" if code == 0 else "failed"
    return job


def list_jobs():
    with _lock:
        jobs = [_reconcile(job) for job in _load()]
        _save(jobs)
        return {"jobs": [_public(job) for job in jobs]}


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Lightning's tqdm line, plus test.py's Evaluating bar. Epoch is 0-based.
# The step counts sit after the bar.
_BAR = re.compile(
    r"^(?:Epoch (?P<epoch>\d+)"
    r"|(?P<phase>Validation|Testing|Sanity Checking) DataLoader \d+"
    r"|(?P<plain>Evaluating)):"
    r"[^|\n]*\|[^|\n]*\|\s*(?P<step>\d+)/(?P<total>\d+)\s+\[(?P<elapsed>[^<]*)<(?P<left>[^,\]]*)"
)
_PHASE = {
    "Validation": "validation",
    "Testing": "testing",
    "Sanity Checking": "sanity check",
    "Evaluating": "evaluating",
}


def _max_epochs(job):
    match = re.search(r"(?:^|\s)model\.max_epochs=(\d+)", job.get("command") or "")
    if not match:
        return None
    return int(match.group(1))


def _left(token):
    """Turn a tqdm remainder like 50:29 or 1:13:55 into a short phrase."""
    token = token.strip()
    if not token or "?" in token:
        return ""
    parts = token.split(":")
    if not all(part.isdigit() for part in parts):
        return ""
    nums = [int(part) for part in parts]
    if len(nums) == 2:
        seconds = nums[0] * 60 + nums[1]
    elif len(nums) == 3:
        seconds = nums[0] * 3600 + nums[1] * 60 + nums[2]
    else:
        return ""
    if seconds <= 0:
        return ""
    minutes = max(1, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m left"
    return f"{minutes}m left"


def _phrase(kind, epoch, epochs, step, total, left):
    bits = []
    if epoch is not None:
        shown = epoch + 1
        bits.append(f"epoch {shown}/{epochs}" if epochs else f"epoch {shown}")
    bits.append(kind)
    bits.append(f"{step}/{total}")
    if left:
        bits.append(left)
    return " · ".join(bits)


def _describe_progress(text, epochs):
    """One line for the latest training, validation, or test bar in a log tail."""
    epoch = None
    latest = None
    finished = False
    for line in re.split(r"[\r\n]+", text):
        line = line.strip()
        if line.startswith("Training finished"):
            finished = True
            latest = None
            continue
        match = _BAR.match(line)
        if not match:
            continue
        finished = False
        if match.group("epoch") is not None:
            epoch = int(match.group("epoch"))
            latest = ("training", epoch, match)
        else:
            kind = _PHASE[match.group("phase") or match.group("plain")]
            latest = (kind, epoch if kind == "validation" else None, match)
    if latest is None:
        return "training finished" if finished else ""
    kind, epoch_n, match = latest
    return _phrase(
        kind,
        epoch_n,
        epochs,
        match.group("step"),
        match.group("total"),
        _left(match.group("left")),
    )


def _progress_text(job):
    if job.get("status") != "running":
        return ""
    path = JOBS_DIR / f"{job['id']}.log"
    if not path.is_file():
        return ""
    size = path.stat().st_size
    if size <= 0:
        return ""
    take = min(size, 1_048_576)
    with open(path, "rb") as handle:
        handle.seek(size - take)
        data = handle.read()
    text = _ANSI.sub("", data.decode("utf-8", "replace"))
    return _describe_progress(text, _max_epochs(job))


def _public(job):
    return {
        "id": job["id"],
        "status": job["status"],
        "created": job["created"],
        "started": job.get("started"),
        "finished": job.get("finished"),
        "exp_name": job["exp_name"],
        "data": job["data"],
        "model": job["model"],
        "command": job["command"],
        "run_test": job["run_test"],
        "exit_code": job.get("exit_code"),
        "error": job.get("error"),
        "wait_reason": job.get("wait_reason") or "",
        "progress": _progress_text(job),
    }


def enqueue(body):
    """Validate a form submission and append a queued job."""
    try:
        python = train_python()
    except TrainEnvError as exc:
        raise JobError(str(exc)) from exc
    if shutil.which("tmux") is None:
        raise JobError("tmux is not installed")

    options = train_options()
    data = body["data"]
    model = body["model"]
    if data not in options["data"]:
        raise JobError("unknown data config")
    if model not in options["models"]:
        raise JobError("unknown model config")
    seed = int(body["seed"])
    epochs = int(body["max_epochs"])
    every = int(body.get("check_val_every_n_epoch") or 1)
    lr = float(body["lr"])
    batch_train = int(body["batch_size_train"])
    batch_val = int(body["batch_size_val"])
    precision = body["precision"]
    logger = body["logger"]
    run_test = bool(body.get("run_test"))
    if not 0 <= seed <= 2**31 - 1:
        raise JobError("seed is out of range")
    if not 1 <= epochs <= 500:
        raise JobError("max_epochs must be from 1 to 500")
    if not 1 <= every <= 500:
        raise JobError("check_val_every_n_epoch must be from 1 to 500")
    if not 1e-8 <= lr <= 1:
        raise JobError("lr must be from 1e-8 to 1")
    if not 1 <= batch_train <= 64 or not 1 <= batch_val <= 64:
        raise JobError("batch size must be from 1 to 64")
    if precision not in PRECISIONS:
        raise JobError("unknown precision")
    if logger not in LOGGERS:
        raise JobError("logger must be wandb or csv")
    overrides = [_check_override(item) for item in body.get("overrides") or []]
    if len(overrides) > 40:
        raise JobError("too many overrides")

    job_id = datetime.now().strftime("%Y%m%d%H%M%S") + f"{datetime.now().microsecond:06d}"
    exp_name = resolve_exp_name(body, data, model, seed)
    train_argv = [
        str(python),
        str(REPO_ROOT / "train.py"),
        f"data={data}",
        f"model={model}",
        f"pl_seed={seed}",
        f"model.max_epochs={epochs}",
        f"model.check_val_every_n_epoch={every}",
        f"model.lr={_format_lr(lr)}",
        f"data.batch_size_train={batch_train}",
        f"data.batch_size_val={batch_val}",
        f"model.precision={precision}",
        f"logger={logger}",
        *few_shot_overrides(body),
        *overrides,
        f"exp_name={exp_name}",
    ]
    test_argv = None
    if run_test:
        test_argv = [str(python), str(REPO_ROOT / "test.py"), f"exp_name={exp_name}"]
    command = _command_text(train_argv)
    if test_argv:
        command += "\n" + _command_text(test_argv)

    job = {
        "id": job_id,
        "status": "queued",
        "created": _now(),
        "started": None,
        "finished": None,
        "exp_name": exp_name,
        "data": data,
        "model": model,
        "command": command,
        "run_test": run_test,
        "exit_code": None,
        "error": None,
        "wait_reason": "",
    }
    spec = {"train": train_argv, "test": test_argv}
    with _lock:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        (JOBS_DIR / f"{job_id}.spec.json").write_text(json.dumps(spec))
        jobs = _load()
        jobs.append(job)
        _save(_trim(jobs))
    return _public(job)


def _trim(jobs):
    finished = [job for job in jobs if job["status"] not in {"queued", "running"}]
    keep = {job["id"] for job in finished[-50:]}
    keep.update(job["id"] for job in jobs if job["status"] in {"queued", "running"})
    return [job for job in jobs if job["id"] in keep]


def subset_tag(body):
    """Short name suffix for a subset, e.g. t10_v10_tbg10_vbg0.

    Species counts come first, then the background tiles sent with each of them.
    A side with no species count is left out.
    """
    parts = []
    train_k = _count(body.get("few_shot_k"), "images per species", 1, 5000)
    val_k = _count(body.get("val_few_shot_k"), "val images per species", 1, 5000)
    if train_k is not None:
        parts.append(f"t{train_k}")
    if val_k is not None:
        parts.append(f"v{val_k}")
    if train_k is not None:
        bg = _count(body.get("few_shot_bg_k"), "background tiles", 0, 5000)
        parts.append(f"tbg{0 if bg is None else bg}")
    if val_k is not None:
        bg = _count(body.get("val_few_shot_bg_k"), "val background tiles", 0, 5000)
        parts.append(f"vbg{0 if bg is None else bg}")
    return "_".join(parts)


def resolve_exp_name(body, data, model, seed):
    """The W&B name. Blank keeps the timestamped name the queue used to assign."""
    raw = str(body.get("exp_name") or "").strip()
    if not raw:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw = f"{stamp}_{data}_{model}_s{seed}"
        tag = subset_tag(body)
        if tag:
            raw = f"{raw}_{tag}"
    if not EXP_NAME.fullmatch(raw):
        raise JobError("W&B name must use letters, numbers, dots, hyphens, and underscores")
    return raw


def _format_lr(lr):
    text = f"{lr:.8f}".rstrip("0").rstrip(".")
    return text or "0"


def cancel(job_id):
    if not re.fullmatch(r"[0-9]{14,20}", job_id):
        raise JobError("unknown job")
    with _lock:
        jobs = _load()
        match = next((job for job in jobs if job["id"] == job_id), None)
        if match is None:
            raise JobError("unknown job")
        if match["status"] in {"done", "failed", "cancelled"}:
            raise JobError("job already finished")
        match["status"] = "cancelled"
        match["finished"] = _now()
        match["wait_reason"] = ""
        _save(jobs)
    if _session_alive(job_id):
        subprocess.run(
            ["tmux", "kill-session", "-t", _session(job_id)],
            capture_output=True,
            check=False,
        )
    return _public(match)


def read_log(job_id, offset, tail=None):
    if not re.fullmatch(r"[0-9]{14,20}", job_id):
        raise JobError("unknown job")
    path = JOBS_DIR / f"{job_id}.log"
    if not path.is_file():
        return {"offset": 0, "text": ""}
    size = path.stat().st_size
    if tail:
        return {"offset": size, "text": _tail_entries(path, tail, size)}
    offset = max(0, min(int(offset), size))
    with open(path, "rb") as handle:
        handle.seek(offset)
        chunk = handle.read(64_000)
    return {"offset": offset + len(chunk), "text": chunk.decode("utf-8", "replace")}


def _tail_entries(path, count, size):
    """Last ``count`` log records, split on newlines and progress-bar returns.

    Reads backward from the end so a long run does not have to be loaded whole.
    """
    count = max(1, min(int(count), 5000))
    if size <= 0:
        return ""
    window = 64_000
    pos = size
    data = b""
    with open(path, "rb") as handle:
        while True:
            parts = re.split(rb"[\r\n]+", data)
            if parts and parts[-1] == b"":
                parts.pop()
            # A window that starts mid-file begins with a partial record.
            complete = parts[1:] if pos > 0 and parts else parts
            if len(complete) >= count or pos == 0:
                chosen = (parts if pos == 0 else complete)[-count:]
                return "\n".join(part.decode("utf-8", "replace") for part in chosen)
            take = min(window, pos)
            pos -= take
            handle.seek(pos)
            data = handle.read(take) + data
            window = min(window * 2, 1_048_576)


def _start(job):
    proc = subprocess.run(
        [
            "tmux", "new-session", "-d",
            "-s", _session(job["id"]),
            "-c", str(REPO_ROOT),
            str(train_python()), "-u", str(EXPLORER_ROOT / "run_job.py"),
            "--job", job["id"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        job["status"] = "failed"
        job["finished"] = _now()
        job["error"] = (proc.stderr or "tmux failed to start").strip()
        job["wait_reason"] = ""
        return True
    job["status"] = "running"
    job["started"] = _now()
    job["wait_reason"] = ""
    return True


def _tick():
    import predict_client  # local: avoid a cycle while the client imports nothing from here

    with _lock:
        jobs = [_reconcile(job) for job in _load()]
        running = next((job for job in jobs if job["status"] == "running"), None)
        nxt = next((job for job in jobs if job["status"] == "queued"), None)
        if running is not None or nxt is None:
            _save(jobs)
            return
        if not gpu_lock.acquire(blocking=False):
            nxt["wait_reason"] = "A prediction is using the GPU"
            _save(jobs)
            return
        try:
            predict_client.stop()
            if compute_busy():
                nxt["wait_reason"] = "GPU is busy"
                _save(jobs)
                return
            _start(nxt)
            _save(jobs)
        finally:
            gpu_lock.release()


def _loop():
    while True:
        try:
            _tick()
        except Exception:
            path = JOBS_DIR / "scheduler.log"
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as handle:
                handle.write(_now() + "\n")
                traceback.print_exc(file=handle)
        time.sleep(2)


def start_scheduler():
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, name="train-queue", daemon=True)
    _thread.start()


def gpu_status():
    """Snapshot plus whether the queue is holding the GPU."""
    info = snapshot()
    info["training"] = training_active()
    return info
