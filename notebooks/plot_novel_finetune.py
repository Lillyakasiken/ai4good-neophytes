"""One figure per novel-species fine-tune.

Each run learns one new species from a single flight, then is scored on the
other flights. The top panel is that training flight against the pooled test.
The bottom panel is the same test, one cell per flight.

Runs are read from lightning_logs. A species is whatever appears in a
``_nv{species}_{flight}_ft_fl`` directory. For each species only the latest
stamp is drawn, so an earlier batch that collapsed is left out. Re-run this
as later flights finish; a flight with no test yet is left blank.
"""

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle

RUN_RE = re.compile(
    r"^(?P<stamp>\d{8}_\d{6}).*_nv(?P<species>[A-Za-z]+_[A-Za-z]+)_(?P<flight>.+)_ft_fl$"
)
BEST_RE = re.compile(r"New best score:\s*([0-9.]+)")
EPOCH_RE = re.compile(r"Epoch\s+\d+:\s+100%.*?val_f1_species=([0-9.]+).*?val_f1_novel=([0-9.]+)")


def repo_root():
    cwd = Path.cwd()
    if (cwd / "lightning_logs").is_dir():
        return cwd
    if (cwd.parent / "lightning_logs").is_dir():
        return cwd.parent
    raise FileNotFoundError("Cannot find lightning_logs/. Start in the repo root or in notebooks/.")


def load_json(path):
    return json.loads(path.read_text(), parse_constant=lambda token: float(token))


def species_label(name):
    genus, epithet = name.split("_", 1)
    return f"{genus}. {epithet}"


def flight_label(name):
    return str(name).replace("_", " ")


def _ink(value):
    """Readable text on viridis: dark on the bright end, white on the dark end."""
    return "black" if value > 0.6 else "white"


def best_validation(run):
    """Best val_f1_novel printed by the checkpoint callback, else the latest progress."""
    scores = []
    for log in sorted(run.glob("hydra/*/train_neophyte_semseg.log")):
        scores.extend(float(match) for match in BEST_RE.findall(log.read_text(errors="replace")))
    if scores:
        return max(scores)
    progress = run / "progress.json"
    if not progress.is_file():
        return None
    return load_json(progress).get("novel_f1")


def progress_of(run):
    path = run / "progress.json"
    if not path.is_file():
        return None
    return load_json(path)


def latest_test(run):
    tests = sorted(
        path for path in run.glob("test_*")
        if (path / "quantitative" / "metrics_novel.json").is_file()
    )
    return tests[-1] if tests else None


def novel_metrics(run):
    test = latest_test(run)
    if test is None:
        return None
    return load_json(test / "quantitative" / "metrics_novel.json")


def test_species_f1(run):
    """Mean species F1 on the test tiles, background excluded. ``F1-avg-wo0``."""
    test = latest_test(run)
    if test is None:
        return None
    path = test / "quantitative" / "metrics_semseg.json"
    if not path.is_file():
        return None
    return load_json(path).get("F1-avg-wo0")


def _epoch_scores(text):
    """val_f1_species at the epoch with the highest val_f1_novel."""
    pairs = [(float(species), float(novel)) for species, novel in EPOCH_RE.findall(text)]
    if not pairs:
        return None
    species, _novel = max(pairs, key=lambda item: item[1])
    return species


def _job_heads(root):
    """First slice of each queued-job log, read once for every flight."""
    cached = getattr(_job_heads, "cache", None)
    if cached is not None and cached[0] == root:
        return cached[1]
    jobs = root / "explorer" / "cache" / "jobs"
    heads = []
    if jobs.is_dir():
        for log in sorted(jobs.glob("*.log")):
            heads.append((log, log.read_text(errors="replace")[:40000]))
    _job_heads.cache = (root, heads)
    return heads


def val_species_f1(root, run, flight, species, stamp):
    """val_f1_species from the same epoch as the best val_f1_novel."""
    needle = stamp
    flight_name = f"_nv{species}_{flight}_ft_fl"
    texts = [log.read_text(errors="replace") for log in sorted(run.glob("hydra/*/train_neophyte_semseg.log"))]
    for log, head in _job_heads(root):
        if needle not in head or flight_name not in head:
            continue
        texts.append(log.read_text(errors="replace"))
    for text in texts:
        score = _epoch_scores(text)
        if score is not None:
            return score
    progress = run / "progress.json"
    if not progress.is_file():
        return None
    state = load_json(progress)
    values = [state.get("novel_f1")]
    values.extend((state.get("old_f1") or {}).values())
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def latest_batches(root):
    """Latest fine-tune stamp for each species that has a flight run."""
    stamps = {}
    for run in (root / "lightning_logs").iterdir():
        match = RUN_RE.search(run.name)
        if not match or not run.is_dir():
            continue
        species = match.group("species")
        stamp = match.group("stamp")
        if stamp > stamps.get(species, ""):
            stamps[species] = stamp
    return [(species, stamps[species]) for species in sorted(stamps)]


def finished_runs(root, species, stamp):
    records = []
    for run in sorted((root / "lightning_logs").iterdir()):
        match = RUN_RE.search(run.name)
        if not match or not run.is_dir():
            continue
        if match.group("species") != species or match.group("stamp") != stamp:
            continue
        flight = match.group("flight")
        metrics = novel_metrics(run)
        progress = progress_of(run)
        validation = best_validation(run)
        if validation is None and metrics is None:
            continue
        records.append({
            "flight": flight,
            "val_species": val_species_f1(root, run, flight, species, stamp),
            "validation": validation,
            "novel": None if metrics is None else metrics.get("novel_f1"),
            "same": None if metrics is None else metrics.get("same_site_f1"),
            "test_species": test_species_f1(run),
            "per_flight": [] if metrics is None else list(metrics.get("per_flight") or []),
            "tiles": None if metrics is None else metrics.get("eval_tiles"),
            "tiles_full": None if metrics is None else metrics.get("eval_tiles_full"),
            "other_per_flight": None if metrics is None else metrics.get("eval_other_per_flight"),
            "progress": progress,
        })
    return records


def still_training(record):
    progress = record["progress"] or {}
    epoch = progress.get("epoch")
    maximum = progress.get("max_epochs")
    if epoch is None or maximum is None:
        return False
    return int(epoch) < int(maximum)


def _finite(value):
    if value is None:
        return False
    return bool(np.isfinite(value))


def draw_summary(ax, records, label):
    measures = [
        ("val_species", "val_f1_species", "#1d3557", None),
        ("validation", "val_f1_novel", "#08519c", None),
        ("test_species", "test F1-avg-wo0", "#2a9d8f", None),
        ("novel", "novel_f1", "#f4a261", "////"),
        ("same", "same_site_f1", "#6baed6", None),
    ]
    positions = np.arange(len(records))
    height = 0.14
    offsets = np.linspace(-(len(measures) - 1) / 2, (len(measures) - 1) / 2, len(measures)) * 0.16
    for offset, (key, name, color, hatch) in zip(offsets, measures):
        for index, record in enumerate(records):
            value = record[key]
            y = positions[index] + offset
            if not _finite(value):
                ax.text(0.012, y, "—", va="center", ha="left", fontsize=8, color="0.45")
                continue
            ax.barh(
                y, value, height=height, color=color, hatch=hatch,
                edgecolor="0.25", linewidth=0.4,
            )
            ax.text(value + 0.015, y, f"{float(value):.3f}", va="center", ha="left", fontsize=7.5, color="0.15")
    labels = []
    for record in records:
        text = flight_label(record["flight"])
        progress = record["progress"] or {}
        if still_training(record):
            text = f"{text}\nepoch {int(progress['epoch'])}/{int(progress['max_epochs'])}"
        labels.append(text)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.set_xlim(0, 1.18)
    ax.set_ylim(len(records) - 0.6, -0.6)
    ax.set_xlabel("F1")
    ax.set_title("Validation and test", loc="left", fontsize=11, pad=72)
    ax.text(
        0, 1.02,
        "\n".join([
            "val_f1_species is the mean F1 of every species on the training flight.",
            f"val_f1_novel is the F1 of {label} on that same validation.",
            "test F1-avg-wo0 is the mean F1 of every species on the test.",
            f"novel_f1 is the F1 of {label} on flights at other sites.",
            f"same_site_f1 is the F1 of {label} on another date of the training site.",
        ]),
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="0.25", linespacing=1.35,
    )
    ax.grid(axis="x", color="#e4e9ee", zorder=0)
    ax.tick_params(length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Always list every measure. Labelling only the first flight drops a series
    # when that flight has no score (e.g. same_site_f1).
    handles = [
        Patch(facecolor=color, edgecolor="0.25", linewidth=0.4, hatch=hatch, label=name)
        for _, name, color, hatch in measures
    ]
    ax.legend(
        handles=handles, frameon=False, loc="lower right",
        bbox_to_anchor=(1.0, 1.02), fontsize=8, borderaxespad=0,
    )


def _row_rank(values):
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return (1, 0.0)
    return (0, -max(finite))


def draw_flights(ax, records, label):
    """Scored flight by training flight. Best row first. Colour is fixed at 0..1."""
    tested = [record for record in records if record["per_flight"]]
    columns = [record["flight"] for record in tested]
    names = []
    for record in tested:
        names.append(record["flight"])
        for cell in record["per_flight"]:
            names.append(cell["flight"])
    rows = list(dict.fromkeys(names))
    lookup = {
        (record["flight"], cell["flight"]): cell
        for record in tested
        for cell in record["per_flight"]
    }
    rows.sort(key=lambda flight: (
        _row_rank([lookup.get((train, flight), {}).get("f1") for train in columns]),
        flight,
    ))
    cmap = plt.cm.viridis
    for i, flight in enumerate(rows):
        for j, train in enumerate(columns):
            if flight == train:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#d9d9d9", edgecolor="white", linewidth=1.2,
                ))
                ax.text(j, i, "train", ha="center", va="center", fontsize=7.5, color="0.35")
                continue
            cell = lookup.get((train, flight))
            value = None if cell is None else cell.get("f1")
            if not _finite(value):
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#ececec", edgecolor="white", linewidth=1.2,
                ))
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="0.45")
                continue
            ax.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=cmap(float(value)), edgecolor="white", linewidth=1.2,
            ))
            ax.text(
                j, i, f"{float(value):.2f}", ha="center", va="center",
                fontsize=8, color=_ink(float(value)),
            )
    caps = [record.get("other_per_flight") for record in tested if record.get("other_per_flight") is not None]
    other_cap = int(caps[0]) if caps else 20
    ax.set_xticks(range(len(columns)), [flight_label(record["flight"]) for record in tested], fontsize=8)
    ax.set_yticks(range(len(rows)), [flight_label(name) for name in rows], fontsize=8)
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.text(
        0, 1.01,
        "\n".join([
            f"Per-flight F1 of {label}",
            f"F1 of {label} on each flight.",
            f"Every tile that has {label} in the mask.",
            f"Plus up to {other_cap} other tiles from that flight (mix of other species and empty), so false positives still count.",
        ]),
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8, fontweight="normal", color="0.25", linespacing=1.15,
        clip_on=False,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("auto")
    return rows, columns


def plot_species(root, species, stamp):
    label = species_label(species)
    records = finished_runs(root, species, stamp)
    if not records:
        print(f"skip {label}: no scored fine-tunes for {stamp}")
        return None
    tested = [record for record in records if record["per_flight"]]
    n_rows = 0
    if tested:
        names = set()
        for record in tested:
            names.add(record["flight"])
            names.update(cell["flight"] for cell in record["per_flight"])
        n_rows = len(names)
    width = max(8.4, 3.2 + 1.35 * max(1, len(tested)))
    height = 2.0 + 0.72 * len(records) + (0.34 * n_rows if n_rows else 0.4)
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(label, x=0.01, ha="left", fontsize=13)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.84, bottom=0.04)
    if n_rows:
        grid = fig.add_gridspec(
            2, 1, height_ratios=[0.85 * len(records) + 2.3, 0.34 * n_rows], hspace=0.30,
        )
        ax_bars = fig.add_subplot(grid[0])
        ax_heat = fig.add_subplot(grid[1])
        draw_summary(ax_bars, records, label)
        draw_flights(ax_heat, records, label)
        mappable = ScalarMappable(norm=Normalize(0, 1), cmap=plt.cm.viridis)
        heat_box = ax_heat.get_position()
        cax = fig.add_axes([heat_box.x1 + 0.012, heat_box.y0, 0.014, heat_box.height])
        bar = fig.colorbar(mappable, cax=cax)
        bar.set_label(f"{label} F1", fontsize=9)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8)
    else:
        ax_bars = fig.add_subplot(111)
        draw_summary(ax_bars, records, label)
    out = root / "notebooks" / "few_shot_site_figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{species.lower()}_finetune.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path} ({stamp}, {len(records)} flights)")
    for record in records:
        def shown(value):
            return "—" if value is None else f"{value:.3f}"
        print(
            f"{record['flight']}: val_f1_species {shown(record['val_species'])}, "
            f"val_f1_novel {shown(record['validation'])}, "
            f"test F1-avg-wo0 {shown(record['test_species'])}, "
            f"novel_f1 {shown(record['novel'])}, same_site_f1 {shown(record['same'])}"
        )
    return path


def main():
    root = repo_root()
    batches = latest_batches(root)
    if not batches:
        raise SystemExit("No novel-species flight fine-tunes found in lightning_logs/.")
    paths = []
    for species, stamp in batches:
        path = plot_species(root, species, stamp)
        if path is not None:
            paths.append(path)
    return paths


if __name__ == "__main__":
    main()
