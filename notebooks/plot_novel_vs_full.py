"""Full six-species model against the one-flight fine-tunes.

The navy score is the cv3 held-out-site F1 of U-Net + MiT-b2 trained on every
training site (512 crops, seed 42). The orange scores are the same architecture
after a five-species base and a fine-tune on one flight of the held-out species:
each dot is one choice of that flight, and the score is that species at the
other sites. The right panel is how many labelled tiles of the species each
setting trains on. Each fine-tune flight has its own line: the same colour
on both panels, and the name of that flight between them. The two F1s are
not the same test sites.
"""

from pathlib import Path
import csv
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import yaml

RUN_RE = re.compile(
    r"^(?P<stamp>\d{8}_\d{6}).*_nv(?P<species>[A-Za-z]+_[A-Za-z]+)_(?P<flight>.+)_ft_fl$"
)
SPECIES = [
    "A_altissima",
    "R_typhina",
    "B_davidii",
    "R_japonica",
    "B_orientalis",
    "S_inaequidens",
]
# Class order of metrics_semseg.json "F1", background first.
CLASS_INDEX = {
    "B_davidii": 1,
    "R_typhina": 2,
    "R_japonica": 3,
    "B_orientalis": 4,
    "S_inaequidens": 5,
    "A_altissima": 6,
}
# Test directory for this run was removed. These are its held-out-site F1s,
# docs/changes_2026-10-12.md section 4, best_epoch=29.
FULL_RUN = "20261013_002801.261406_np_cv3_Unet_mit_b2_512_s42"
FULL_F1_FALLBACK = {
    "A_altissima": 0.842,
    "R_typhina": 0.603,
    "B_davidii": 0.757,
    "R_japonica": 0.809,
    "B_orientalis": 0.609,
    "S_inaequidens": 0.377,
}
NAVY = "#08519c"
# One color per training flight, reused down the species rows. The same flight
# keeps this color, and the same height within its row, on both panels.
FLIGHT_COLORS = [
    "#d95f02", "#1b9e77", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
]


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


def colored_flights(rows):
    """Alphabetical flights. The same order and color is used on both panels."""
    rows = sorted(rows, key=lambda item: item["flight"])
    return [
        (row, FLIGHT_COLORS[i % len(FLIGHT_COLORS)])
        for i, row in enumerate(rows)
    ]


def full_f1(root):
    """Per-species held-out F1. The metrics file wins when the test dir is present."""
    run = root / "lightning_logs" / FULL_RUN
    files = sorted(run.glob("test_*/quantitative/metrics_semseg.json"))
    if not files:
        return dict(FULL_F1_FALLBACK), "documented"
    f1 = load_json(files[-1]).get("F1")
    if not isinstance(f1, list) or len(f1) < 7:
        return dict(FULL_F1_FALLBACK), "documented"
    return {name: float(f1[index]) for name, index in CLASS_INDEX.items()}, "metrics"


def latest_finetunes(root):
    """Latest fine-tune stamp per species, one record per training flight."""
    stamps = {}
    runs = []
    for run in (root / "lightning_logs").iterdir():
        match = RUN_RE.search(run.name)
        if not match or not run.is_dir():
            continue
        species = match.group("species")
        if species not in SPECIES:
            continue
        stamp = match.group("stamp")
        runs.append((stamp, species, match.group("flight"), run))
        if stamp > stamps.get(species, ""):
            stamps[species] = stamp
    chosen = []
    for stamp, species, flight, run in runs:
        if stamp != stamps.get(species):
            continue
        tests = sorted(
            path for path in run.glob("test_*")
            if (path / "quantitative" / "metrics_novel.json").is_file()
        )
        if not tests:
            continue
        metrics = load_json(tests[-1] / "quantitative" / "metrics_novel.json")
        score = metrics.get("novel_f1")
        if score is None:
            continue
        chosen.append({
            "species": species,
            "flight": flight,
            "stamp": stamp,
            "novel_f1": float(score),
        })
    return chosen


def _train_keys(root):
    """(year, flight, disk split) pairs the cv3 training set reads."""
    doc = yaml.safe_load((root / "configs" / "data" / "split_cv3.yaml").read_text())
    keys = set()
    for rel in doc["image_dir"]["train"]:
        parts = Path(rel).parts
        keys.add((parts[-3], parts[-2], parts[-1]))
    return keys


def label_counts(root, finetunes):
    """Labelled training tiles of each species: every cv3 training flight, and one flight."""
    keys = _train_keys(root)
    stats = root.parent / "data" / "Neophytes" / "stats_imagewise.csv"
    if not stats.is_file():
        stats = Path(yaml.safe_load((root / "configs" / "data" / "data.yaml").read_text())["dataset_stats"])
        if not stats.is_absolute():
            stats = (root / stats).resolve()
    full_tiles = {name: set() for name in SPECIES}
    full_flights = {name: set() for name in SPECIES}
    one_tiles = {}
    with stats.open(newline="") as handle:
        for row in csv.DictReader(handle):
            flight = row["site"]
            split = row["split"]
            if split == "val":
                continue
            for name in SPECIES:
                if int(float(row.get(f"{name}_px") or 0)) <= 0:
                    continue
                if (row["year"], flight, split) in keys:
                    full_tiles[name].add(row["img_path"])
                    full_flights[name].add(flight)
                one_tiles.setdefault((name, flight), set()).add(row["img_path"])
    for record in finetunes:
        tiles = one_tiles.get((record["species"], record["flight"]), set())
        record["tiles"] = len(tiles)
    summary = []
    for name in SPECIES:
        summary.append({
            "species": name,
            "flights": len(full_flights[name]),
            "tiles": len(full_tiles[name]),
        })
    return summary


def _style(ax):
    ax.grid(axis="x", color="#e4e9ee", zorder=0)
    ax.tick_params(length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _layout(by_species):
    """One line per flight, then a separate line for the full-model diamond."""
    step = 1.0
    gap = 1.15
    cursor = 0.0
    blocks = []
    for index, name in enumerate(SPECIES):
        flights = colored_flights(by_species[name])
        items = []
        for i, (row, color) in enumerate(flights):
            items.append((row, color, cursor + i * step))
        diamond = (items[-1][2] + 0.85) if items else cursor
        top = (items[0][2] if items else diamond) - 0.48
        band = (top, diamond + 0.48)
        center = (band[0] + band[1]) / 2
        blocks.append({
            "species": name,
            "index": index,
            "items": items,
            "center": center,
            "diamond": diamond,
            "band": band,
        })
        cursor = diamond + gap
    return blocks, cursor - gap


def plot(full, finetunes, counts):
    by_species = {name: [] for name in SPECIES}
    for record in finetunes:
        by_species[record["species"]].append(record)
    blocks, last = _layout(by_species)
    count_of = {row["species"]: row for row in counts}

    fig = plt.figure(figsize=(13.2, 11.4))
    fig.suptitle(
        "Six-species training against a one-flight fine-tune",
        x=0.012, y=0.985, ha="left", va="top", fontsize=14,
    )
    fig.text(
        0.012, 0.952,
        "Navy trains on every cv3 training site and is scored on the four held-out sites.\n"
        "Each coloured dot is one flight. Match it by colour and by its line in the middle. "
        "The two tests are not the same sites.",
        fontsize=8.5, color="0.25", linespacing=1.45, va="top",
    )
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.82, 1.45], wspace=0.018)
    ax_f1 = fig.add_subplot(grid[0, 0])
    ax_names = fig.add_subplot(grid[0, 1], sharey=ax_f1)
    ax_data = fig.add_subplot(grid[0, 2], sharey=ax_f1)
    panels = (ax_f1, ax_names, ax_data)

    for block in blocks:
        shade = "#e8eef3" if block["index"] % 2 == 0 else "#f7f9fb"
        y0, y1 = block["band"]
        for ax in panels:
            ax.axhspan(y0, y1, color=shade, zorder=0)
        for row, color, spot in block["items"]:
            ax_f1.scatter(
                row["novel_f1"], spot, s=42, color=color, edgecolor="0.15",
                linewidth=0.4, zorder=3,
            )
            if row["tiles"] > 0:
                ax_data.scatter(
                    row["tiles"], spot, s=42, color=color, edgecolor="0.15",
                    linewidth=0.4, zorder=3,
                )
            ax_names.scatter(
                0.06, spot, s=36, color=color, edgecolor="0.15",
                linewidth=0.4, zorder=3, clip_on=False,
            )
            ax_names.text(
                0.14, spot, flight_label(row["flight"]),
                va="center", ha="left", fontsize=8, color="0.15",
            )
        ax_f1.scatter(
            full[block["species"]], block["diamond"], s=54, marker="D", color=NAVY,
            edgecolor="white", linewidth=0.7, zorder=4,
        )
        ax_data.scatter(
            count_of[block["species"]]["tiles"], block["diamond"], s=54, marker="D",
            color=NAVY, edgecolor="white", linewidth=0.7, zorder=4,
        )
        ax_names.scatter(
            0.06, block["diamond"], s=42, marker="D", color=NAVY,
            edgecolor="white", linewidth=0.6, zorder=4, clip_on=False,
        )
        ax_names.text(
            0.14, block["diamond"], "all training sites",
            va="center", ha="left", fontsize=8, color=NAVY,
        )
        summary = count_of[block["species"]]
        ax_data.text(
            1.03, block["diamond"],
            f"{summary['flights']} flights, {summary['tiles']} tiles",
            transform=ax_data.get_yaxis_transform(),
            va="center", ha="left", fontsize=8, color=NAVY,
        )

    full_handle = ax_f1.scatter(
        [], [], s=54, marker="D", color=NAVY, edgecolor="white", linewidth=0.7,
        label="All training sites, held-out sites",
    )
    ax_f1.set_yticks(
        [block["center"] for block in blocks],
        [species_label(block["species"]) for block in blocks],
        fontsize=9,
    )
    ax_f1.set_xlabel("Species F1")
    ax_f1.set_xlim(-0.04, 1.05)
    ax_f1.set_ylim(last + 0.85, -0.85)
    ax_f1.set_title("Score", loc="left", fontsize=11, pad=8)
    _style(ax_f1)
    ax_f1.set_axisbelow(True)

    ax_names.set_xlim(0, 1)
    ax_names.set_xticks([])
    ax_names.tick_params(left=False, labelleft=False, length=0)
    ax_names.set_title("Training flight", loc="left", fontsize=11, pad=8)
    for spine in ax_names.spines.values():
        spine.set_visible(False)

    ax_data.set_xscale("log")
    ax_data.set_xlabel("Labelled training tiles of that species")
    ax_data.set_xlim(7, 700)
    ax_data.set_title("Labels the run was given", loc="left", fontsize=11, pad=8)
    ax_data.tick_params(left=False, labelleft=False, length=0)
    _style(ax_data)
    ax_data.set_axisbelow(True)

    fig.legend(
        [full_handle], [full_handle.get_label()],
        frameon=False, loc="lower left", bbox_to_anchor=(0.115, 0.885),
        fontsize=8.5, borderaxespad=0,
    )
    fig.subplots_adjust(left=0.10, right=0.84, top=0.86, bottom=0.07)
    return fig


def main():
    root = repo_root()
    full, source = full_f1(root)
    finetunes = latest_finetunes(root)
    if not finetunes:
        raise SystemExit("No one-flight fine-tunes found in lightning_logs/.")
    counts = label_counts(root, finetunes)
    fig = plot(full, finetunes, counts)
    out = root / "notebooks" / "few_shot_site_figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "novel_vs_full.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")
    print(f"full-model F1 source: {source} ({FULL_RUN})")
    for name in SPECIES:
        rows = [row for row in finetunes if row["species"] == name]
        scores = [row["novel_f1"] for row in rows]
        tiles = [row["tiles"] for row in rows]
        count = next(row for row in counts if row["species"] == name)
        print(
            f"{name}: full F1 {full[name]:.3f} on {count['flights']} flights / {count['tiles']} tiles; "
            f"one-flight F1 mean {np.mean(scores):.3f} "
            f"({min(scores):.3f}–{max(scores):.3f}, {len(rows)} flights, "
            f"{min(tiles)}–{max(tiles)} tiles)"
        )
    return path


if __name__ == "__main__":
    main()
