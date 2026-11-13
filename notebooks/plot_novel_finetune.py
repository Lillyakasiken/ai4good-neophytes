"""One figure for the corrected S. inaequidens fine-tunes.

Each run learns the new species from a single flight, then is scored on the
other flights. The top panel is that training flight against the pooled test.
The bottom panel is the same test, one cell per flight.

The 20261109_152445 fine-tunes used a learning rate of 0.001 and collapsed.
This figure reads only the 20261110_183106 batch. Re-run it as later flights
finish; a flight with no test yet is left blank.
"""

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

STAMP = "20261110_183106"
SPECIES = "S_inaequidens"
FLIGHT_RE = re.compile(rf"^{STAMP}.*_nv{SPECIES}_(?P<flight>.+)_ft_fl$")
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


def val_species_f1(root, run, flight):
    """val_f1_species from the same epoch as the best val_f1_novel."""
    needle = f"{STAMP}"
    flight_name = f"_nv{SPECIES}_{flight}_ft_fl"
    texts = [log.read_text(errors="replace") for log in sorted(run.glob("hydra/*/train_neophyte_semseg.log"))]
    jobs = root / "explorer" / "cache" / "jobs"
    if jobs.is_dir():
        for log in sorted(jobs.glob("*.log")):
            head = log.read_text(errors="replace")[:40000]
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


def finished_runs(root):
    records = []
    for run in sorted((root / "lightning_logs").iterdir()):
        match = FLIGHT_RE.search(run.name)
        if not match or not run.is_dir():
            continue
        flight = match.group("flight")
        metrics = novel_metrics(run)
        progress = progress_of(run)
        validation = best_validation(run)
        if validation is None and metrics is None:
            continue
        records.append({
            "flight": flight,
            "val_species": val_species_f1(root, run, flight),
            "validation": validation,
            "novel": None if metrics is None else metrics.get("novel_f1"),
            "same": None if metrics is None else metrics.get("same_site_f1"),
            "test_species": test_species_f1(run),
            "per_flight": [] if metrics is None else list(metrics.get("per_flight") or []),
            "tiles": None if metrics is None else metrics.get("eval_tiles"),
            "tiles_full": None if metrics is None else metrics.get("eval_tiles_full"),
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


def draw_summary(ax, records):
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
    for offset, (key, label, color, hatch) in zip(offsets, measures):
        for index, record in enumerate(records):
            value = record[key]
            y = positions[index] + offset
            if not _finite(value):
                ax.text(0.012, y, "—", va="center", ha="left", fontsize=8, color="0.45")
                continue
            ax.barh(
                y, value, height=height, color=color, hatch=hatch,
                edgecolor="0.25", linewidth=0.4, label=label if index == 0 else None,
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
            "val_f1_novel is the F1 of S. inaequidens on that same validation.",
            "test F1-avg-wo0 is the mean F1 of every species on the test.",
            "novel_f1 is the F1 of S. inaequidens on flights at other sites.",
            "same_site_f1 is the F1 of S. inaequidens on another date of the training site.",
        ]),
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="0.25", linespacing=1.35,
    )
    ax.grid(axis="x", color="#e4e9ee", zorder=0)
    ax.tick_params(length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)


def _row_rank(values):
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return (1, 0.0)
    return (0, -max(finite))


def draw_flights(ax, records):
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
            same = cell.get("role") == "same_site"
            ax.text(
                j, i, f"{float(value):.2f}", ha="center", va="center",
                fontsize=8, color=_ink(float(value)),
                fontweight="bold" if same else "regular",
            )
    ax.set_xticks(range(len(columns)), [flight_label(record["flight"]) for record in tested], fontsize=8)
    ax.set_yticks(range(len(rows)), [flight_label(name) for name in rows], fontsize=8)
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_title("Per-flight F1 of S. inaequidens", loc="left", fontsize=11, pad=16)
    ax.text(
        0, 1.012,
        "F1 of this species on each flight. Grey is the training flight. Bold is another date of that site.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color="0.25",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("auto")
    return rows, columns


def main():
    root = repo_root()
    records = finished_runs(root)
    if not records:
        raise SystemExit(f"No {SPECIES} fine-tunes found for {STAMP}.")
    tested = [record for record in records if record["per_flight"]]
    n_rows = 0
    if tested:
        names = set()
        for record in tested:
            names.add(record["flight"])
            names.update(cell["flight"] for cell in record["per_flight"])
        n_rows = len(names)
    width = max(8.4, 3.2 + 1.35 * max(1, len(tested)))
    height = 1.7 + 0.72 * len(records) + (0.34 * n_rows if n_rows else 0.4)
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(species_label(SPECIES), x=0.01, ha="left", fontsize=13)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.84, bottom=0.04)
    if n_rows:
        grid = fig.add_gridspec(
            2, 1, height_ratios=[0.85 * len(records) + 2.3, 0.34 * n_rows], hspace=0.18,
        )
        ax_bars = fig.add_subplot(grid[0])
        ax_heat = fig.add_subplot(grid[1])
        draw_summary(ax_bars, records)
        draw_flights(ax_heat, records)
        mappable = ScalarMappable(norm=Normalize(0, 1), cmap=plt.cm.viridis)
        heat_box = ax_heat.get_position()
        cax = fig.add_axes([heat_box.x1 + 0.012, heat_box.y0, 0.014, heat_box.height])
        bar = fig.colorbar(mappable, cax=cax)
        bar.set_label("S. inaequidens F1", fontsize=9)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8)
    else:
        ax_bars = fig.add_subplot(111)
        draw_summary(ax_bars, records)
    out = root / "notebooks" / "few_shot_site_figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "s_inaequidens_finetune.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")
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


if __name__ == "__main__":
    main()
