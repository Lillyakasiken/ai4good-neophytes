"""Rebuild the tiles a training run used.

Imported by the API, this starts the repo-root interpreter. Run as a script,
it prints one JSON object on the last stdout line and does not import the API.

The lists are not stored with the run. They are rebuilt by loading the Hydra
config written at training time and repeating the same seed, transforms, and
dataset setup as ``train.py``. Labelled picks do not use the random number
generator. Background picks do, so anything between the seed and that draw has
to stay in the same order.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

EXPLORER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPLORER_ROOT.parent
LOG_DIR = REPO_ROOT / "lightning_logs"
_FOLD = re.compile(r"cv\d+")


def _is_test_run(overrides_text):
    """True when this Hydra dir is the test pass, not the training run."""
    for raw in overrides_text.splitlines():
        line = raw.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        if line.startswith("data=") and line.split("=", 1)[1].endswith("_test"):
            return True
    return False


def training_hydra_dir(exp_dir):
    """Newest Hydra output under the run that is not the test pass."""
    root = Path(exp_dir) / "hydra"
    if not root.is_dir():
        return None
    found = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        config = child / ".hydra" / "config.yaml"
        if not config.is_file():
            continue
        overrides = child / ".hydra" / "overrides.yaml"
        text = overrides.read_text() if overrides.is_file() else ""
        if _is_test_run(text):
            continue
        found.append(child)
    if not found:
        return None
    found.sort(key=lambda path: path.name)
    return found[-1]


def _names(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [] if text in {"", "null", "None"} else [text]
    names = []
    for item in value:
        text = str(item).strip()
        if text and text not in {"null", "None"}:
            names.append(text)
    return names


def fold_of(data_name):
    """``np_cv3`` -> ``cv3``. Anything else keeps the name without a leading ``np_``."""
    text = "" if data_name is None else str(data_name)
    match = _FOLD.search(text)
    if match:
        return match.group(0)
    if text.startswith("np_"):
        return text[3:]
    return text or None


def _score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value:  # NaN
        return None
    return float(value)


def _test_scores(exp_dir):
    """Newest finished test that wrote a site split. None when the run has no test yet."""
    best = None
    best_mtime = -1.0
    for path in Path(exp_dir).glob("test_*/quantitative/metrics_site_split.json"):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if mtime < best_mtime:
            continue
        same = doc.get("same_site") or {}
        held = doc.get("held_out") or {}
        best_mtime = mtime
        best = {
            "same_site_f1": _score(same.get("F1-avg-wo0")),
            "held_out_f1": _score(held.get("F1-avg-wo0")),
            "same_site_tiles": same.get("tiles"),
        }
    return best


def describe_run(exp_dir):
    """Fields the run picker needs. Reads the YAML itself, so it does not start torch."""
    import yaml

    hydra_dir = training_hydra_dir(exp_dir)
    if hydra_dir is None:
        return None
    config_path = hydra_dir / ".hydra" / "config.yaml"
    doc = yaml.safe_load(config_path.read_text()) or {}
    data = doc.get("data") or {}
    model = doc.get("model") or {}
    scores = _test_scores(exp_dir)
    described = {
        "exp_name": Path(exp_dir).name,
        "fold": fold_of(data.get("data_name")),
        "train_site": _names(data.get("train_site")),
        "k": data.get("few_shot_k"),
        "bg": data.get("few_shot_bg_k"),
        "seed": doc.get("pl_seed"),
        "mode": data.get("few_shot_mode"),
        "max_epochs": model.get("max_epochs"),
        "tested": scores is not None,
        "same_site_f1": None,
        "held_out_f1": None,
        "same_site_tiles": None,
        "mtime": config_path.stat().st_mtime,
    }
    if scores:
        described.update(scores)
    return described


def list_runs():
    """Runs that still have a training Hydra config, newest first."""
    if not LOG_DIR.is_dir():
        return []
    rows = []
    for exp_dir in LOG_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        try:
            described = describe_run(exp_dir)
        except Exception:
            continue
        if described is not None:
            rows.append(described)
    rows.sort(key=lambda row: row["mtime"], reverse=True)
    return rows


def _optional_int(value):
    if value is None:
        return None
    return int(value)


def build(exp_name):
    """Train and val image paths for one run. Imports torch. Call from the repo .venv."""
    exp_dir = LOG_DIR / exp_name
    if exp_dir.resolve().parent != LOG_DIR.resolve() or not exp_dir.is_dir():
        raise FileNotFoundError(f"no run named {exp_name}")
    hydra_dir = training_hydra_dir(exp_dir)
    if hydra_dir is None:
        raise FileNotFoundError(f"{exp_name} has no training config")

    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Same import and setup order as train.py up to the dataset. Imports that
    # happen before the seed cannot change the draws, but the five transform
    # builds sit after it and have to stay there.
    import matplotlib
    import torch
    import torch.multiprocessing
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything

    from datasets.neophyte_datamodule import NeophyteDataModule
    from utils.transform_utils import get_transforms

    cfg = OmegaConf.load(hydra_dir / ".hydra" / "config.yaml")
    OmegaConf.resolve(cfg)

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = not bool(cfg.deterministic_train)
    matplotlib.rcParams.update(OmegaConf.to_container(cfg.matplotlib_style, resolve=True))
    if cfg.pl_seed:
        seed_everything(int(cfg.pl_seed), workers=True)

    extra = cfg.data.transforms_additional_targets
    train_transforms = get_transforms(cfg.data.train_transforms, extra)
    val_transforms = get_transforms(cfg.data.val_transforms, extra)
    test_transforms = get_transforms(cfg.data.test_transforms, extra)
    dsm_transforms = get_transforms(cfg.data.dsm_transforms, toTensor=False)
    ndsm_transforms = get_transforms(cfg.data.ndsm_transforms, toTensor=False)

    data_module = NeophyteDataModule(
        cfg.data.image_dir,
        cfg.data.classes,
        train_transforms,
        val_transforms,
        test_transforms,
        dsm_transform=dsm_transforms,
        ndsm_transform=ndsm_transforms,
        dataset_stats=cfg.data.dataset_stats,
        weighted_sampling=cfg.data.weighted_sampling,
        cap_quantile=cfg.data.cap_quantile,
        lift_quantile=cfg.data.lift_quantile,
        bg_fraction=cfg.data.bg_fraction,
        min_strat_images=cfg.data.min_strat_images,
        value_mapping=cfg.data.value_mapping,
        ignore_index=cfg.data.ignore_index,
        load_auxiliary=cfg.data.load_auxiliary,
        load_dsm=cfg.data.load_dsm,
        load_dtm=cfg.data.load_dtm,
        load_ndsm=cfg.data.load_ndsm,
        few_shot_k=cfg.data.few_shot_k,
        few_shot_mode=cfg.data.few_shot_mode,
        few_shot_bg_k=cfg.data.few_shot_bg_k,
        few_shot_species=cfg.data.few_shot_species,
        train_site=cfg.data.train_site,
        train_stages=cfg.data.train_stages,
        stage_apply_val=cfg.data.stage_apply_val,
        max_nodata_frac=cfg.data.max_nodata_frac,
        val_few_shot_k=cfg.data.val_few_shot_k,
        val_few_shot_mode=cfg.data.val_few_shot_mode,
        val_few_shot_bg_k=cfg.data.val_few_shot_bg_k,
        val_subset=cfg.data.val_subset,
        load_into_ram=False,
        batch_size_train=cfg.data.batch_size_train,
        batch_size_val=cfg.data.batch_size_val,
        batch_size_test=cfg.data.batch_size_test,
        num_workers=0,
    )
    data_module.prepare_data()
    data_module.setup("fit")

    def paths(dataset):
        if dataset.stats_df is None:
            raise RuntimeError("dataset has no stats, so the chosen tiles cannot be named")
        return [str(path) for path in dataset.stats_df.index]

    return {
        "ok": True,
        "exp_name": exp_name,
        "seed": _optional_int(cfg.pl_seed),
        "train_site": _names(cfg.data.train_site),
        "few_shot_k": _optional_int(cfg.data.few_shot_k),
        "few_shot_bg_k": _optional_int(cfg.data.few_shot_bg_k),
        "few_shot_mode": None if cfg.data.few_shot_mode is None else str(cfg.data.few_shot_mode),
        "data_name": None if cfg.data.data_name is None else str(cfg.data.data_name),
        "train": paths(data_module.train_dataset),
        "val": paths(data_module.val_dataset),
    }


def fetch(exp_name):
    """Call this file with the training interpreter."""
    from runtime import REPO_ROOT as repo_root
    from runtime import train_python

    python = train_python()
    try:
        proc = subprocess.run(
            [str(python), "-u", str(Path(__file__).resolve()), "--exp", exp_name],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": "rebuilding the tile list took too long"}
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        detail = (proc.stderr or "tile rebuild produced no output").strip()
        return {"ok": False, "detail": detail[-800:]}
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        detail = (proc.stderr or lines[-1]).strip()
        return {"ok": False, "detail": detail[-800:]}
    if not payload.get("ok"):
        detail = payload.get("detail") or (proc.stderr or "tile rebuild failed").strip()
        payload["detail"] = str(detail)[-800:]
    return payload


def main():
    exp_name = ""
    if len(sys.argv) == 3 and sys.argv[1] == "--exp":
        exp_name = sys.argv[2]
    if not exp_name or "/" in exp_name or exp_name in {".", ".."}:
        payload = {"ok": False, "detail": "usage: run_tiles.py --exp <run directory name>"}
    else:
        try:
            payload = build(exp_name)
        except Exception as exc:
            payload = {"ok": False, "detail": str(exc)}
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
