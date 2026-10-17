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
VALUE = __import__("re").compile(r"^[A-Za-z0-9_./+\-:,\[\]{}]+$")


def _override_ok(arg):
    if "=" not in arg or len(arg) > 240:
        return False
    key, value = arg.split("=", 1)
    return bool(KEY.fullmatch(key) and VALUE.fullmatch(value))


def _argv_ok(argv):
    if len(argv) < 2:
        return False
    python = Path(argv[0]).resolve()
    script = Path(argv[1]).resolve()
    if python != PYTHON.resolve() or script not in ALLOWED_SCRIPTS:
        return False
    return all(_override_ok(arg) for arg in argv[2:])


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--job":
        sys.exit("usage: run_job.py --job <id>")
    job_id = sys.argv[2]
    if not job_id.isalnum() or len(job_id) > 32:
        sys.exit("bad job id")
    spec_path = JOBS / f"{job_id}.spec.json"
    spec = json.loads(spec_path.read_text())
    commands = [spec["train"]]
    if spec.get("test"):
        commands.append(spec["test"])
    for argv in commands:
        if not _argv_ok(argv):
            sys.exit("refusing job argv")

    log_path = JOBS / f"{job_id}.log"
    exit_path = JOBS / f"{job_id}.exit"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    code = 0
    with open(log_path, "ab", buffering=0) as log:
        for argv in commands:
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
    exit_path.write_text(str(code))
    sys.exit(code)


if __name__ == "__main__":
    main()
