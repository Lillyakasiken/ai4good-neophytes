"""Run one queued training job. Started by tmux, not by hand.

Reads ``cache/jobs/<id>.spec.json`` and execs ``train.py`` with that argv.
An optional ``test.py`` follows when the spec asks for it. The exit code is
written next to the log so the queue can mark the job done.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = Path(__file__).resolve().parent / "cache" / "jobs"
PYTHON = ROOT / ".venv" / "bin" / "python"
ALLOWED_SCRIPTS = {ROOT / "train.py", ROOT / "test.py"}
KEY = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
# A checkpoint is named best_epoch=29.ckpt. Hydra only accepts that '=' inside quotes.
VALUE = __import__("re").compile(r'^[A-Za-z0-9_./+\-:=,\[\]{}\\"]+$')


def _override_ok(arg):
    if "=" not in arg or len(arg) > 512:
        return False
    key, value = arg.split("=", 1)
    return bool(KEY.fullmatch(key) and VALUE.fullmatch(value))


def _refusal(argv):
    """The first argument that is not allowed, or an empty string."""
    if len(argv) < 2:
        return "missing command"
    python = Path(argv[0]).resolve()
    script = Path(argv[1]).resolve()
    if python != PYTHON.resolve() or script not in ALLOWED_SCRIPTS:
        return "command is not train.py or test.py"
    for arg in argv[2:]:
        if not _override_ok(arg):
            return arg
    return ""


def _argv_ok(argv):
    return _refusal(argv) == ""


def _ckpt(exp_name):
    """Best checkpoint of a finished run, else its last checkpoint."""
    if not __import__("re").fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", exp_name):
        raise RuntimeError("bad checkpoint run")
    folder = ROOT / "lightning_logs" / exp_name / "checkpoints"
    best = sorted(folder.glob("best_*.ckpt"), key=lambda path: path.stat().st_mtime)
    ckpt = best[-1] if best else folder / "last.ckpt"
    if not ckpt.is_file():
        raise RuntimeError(f"no checkpoint in {exp_name}")
    return ckpt


def _steps(spec):
    """Steps in order. The checkpoint is filled in when that step starts."""
    if spec.get("steps"):
        return spec["steps"]
    steps = [{"argv": spec["train"]}]
    if spec.get("test"):
        steps.append({"argv": spec["test"]})
    return steps


def _hydra_quote(value):
    """A Hydra override value. Quotes keep '=' in best_epoch=29.ckpt as one value."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _argv_for(step):
    """Argv for one step. A fine-tune loads the base weights only once they exist."""
    argv = list(step["argv"])
    if step.get("init_from"):
        argv.append("init_ckpt=" + _hydra_quote(_ckpt(step["init_from"])))
    return argv


def _commands(spec):
    return [_argv_for(step) for step in _steps(spec)]


def _note(log_path, message):
    with open(log_path, "ab", buffering=0) as log:
        log.write((str(message).rstrip() + "\n").encode())


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--job":
        sys.exit("usage: run_job.py --job <id>")
    job_id = sys.argv[2]
    if not job_id.isalnum() or len(job_id) > 32:
        sys.exit("bad job id")
    log_path = JOBS / f"{job_id}.log"
    exit_path = JOBS / f"{job_id}.exit"
    code = 1
    try:
        spec = json.loads((JOBS / f"{job_id}.spec.json").read_text())
        steps = _steps(spec)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        code = 0
        with open(log_path, "ab", buffering=0) as log:
            for step in steps:
                argv = _argv_for(step)
                if not _argv_ok(argv):
                    log.write(f"refusing job argv: {_refusal(argv)}\n".encode())
                    code = 1
                    break
                proc = subprocess.run(
                    argv,
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                )
                code = proc.returncode
                if code != 0:
                    break
    except Exception as exc:
        _note(log_path, exc)
        code = 1
    exit_path.write_text(str(code))
    sys.exit(code)


if __name__ == "__main__":
    main()
