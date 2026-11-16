"""One figure for the six leave-one-species baselines.

Each baseline holds out one neophyte, trains on the other five, and scores those
five on validation. The held-out species has no head yet (novel_f1 stays 0).
Latest ``_nv{species}_base`` run per species is used.
"""

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

BASE_RE = re.compile(
    r"^(?P<stamp>\d{8}_\d{6}).*_nv(?P<species>[A-Za-z]+_[A-Za-z]+)_base$"
)
BEST_RE = re.compile(r"New best score:\s*([0-9.]+)")
SPECIES = [
    "A_altissima",
    "R_typhina",
    "B_davidii",
    "R_japonica",
    "B_orientalis",
    "S_inaequidens",
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


def best_val_f1_species(run):
    scores = []
    for log in sorted(run.glob("hydra/*/train_neophyte_semseg.log")):
        scores.extend(float(match) for match in BEST_RE.findall(log.read_text(errors="replace")))
    return max(scores) if scores else None


def latest_bases(root):
    latest = {}
    for run in (root / "lightning_logs").iterdir():
        match = BASE_RE.search(run.name)
        if not match or not run.is_dir():
            continue
        if not any((run / "checkpoints").glob("*.ckpt")):
            continue
        species = match.group("species")
        stamp = match.group("stamp")
        if stamp > latest.get(species, ("", None))[0]:
            latest[species] = (stamp, run)
    records = []
    for species in SPECIES:
        if species not in latest:
            continue
        stamp, run = latest[species]
        progress = load_json(run / "progress.json")
        old = progress.get("old_f1") or {}
        mean_old = float(np.mean(list(old.values()))) if old else None
        records.append({
            "species": species,
            "stamp": stamp,
            "run": run.name,
            "old_f1": {key: float(value) for key, value in old.items()},
            "mean_old": mean_old,
            "best_val": best_val_f1_species(run),
            "novel_f1": progress.get("novel_f1"),
        })
    return records


def _ink(value):
    return "black" if value > 0.6 else "white"


def plot_baselines(records):
    held = [record["species"] for record in records]
    fig = plt.figure(figsize=(10.2, 5.6))
    fig.suptitle(
        "Six leave-one-species baselines",
        x=0.01, ha="left", fontsize=13,
    )
    fig.text(
        0.01, 0.93,
        "Each baseline trains on five species and holds one out. "
        "Cells are validation F1 of the trained species; grey is the held-out plant.",
        fontsize=8.5, color="0.3",
    )
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 2.6], wspace=0.38)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])

    # Mean of the five trained species (prefer checkpoint monitor when logged).
    means = [
        record["best_val"] if record["best_val"] is not None else record["mean_old"]
        for record in records
    ]
    y = np.arange(len(records))
    ax_bar.barh(y, means, height=0.62, color="#08519c", edgecolor="0.25", linewidth=0.4)
    for index, value in enumerate(means):
        if value is None or not np.isfinite(value):
            ax_bar.text(0.02, index, "—", va="center", fontsize=8, color="0.45")
            continue
        ax_bar.text(value + 0.015, index, f"{value:.3f}", va="center", fontsize=8, color="0.15")
    ax_bar.set_yticks(y, [species_label(name) for name in held], fontsize=9)
    ax_bar.set_xlabel("val_f1_species (mean of 5)")
    ax_bar.set_xlim(0, 1.12)
    ax_bar.set_ylim(len(records) - 0.6, -0.6)
    ax_bar.set_title("Held-out species → baseline", loc="left", fontsize=10)
    ax_bar.grid(axis="x", color="#e4e9ee", zorder=0)
    ax_bar.tick_params(length=0)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    cmap = plt.cm.viridis
    for i, record in enumerate(records):
        for j, scored in enumerate(SPECIES):
            if scored == record["species"]:
                ax_heat.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#d9d9d9", edgecolor="white", linewidth=1.2,
                ))
                ax_heat.text(j, i, "held\nout", ha="center", va="center",
                             fontsize=7, color="0.4", linespacing=0.95)
                continue
            value = record["old_f1"].get(scored)
            if value is None or not np.isfinite(value):
                ax_heat.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#ececec", edgecolor="white", linewidth=1.2,
                ))
                ax_heat.text(j, i, "—", ha="center", va="center", fontsize=8, color="0.45")
                continue
            ax_heat.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=cmap(float(value)), edgecolor="white", linewidth=1.2,
            ))
            ax_heat.text(
                j, i, f"{float(value):.2f}", ha="center", va="center",
                fontsize=8, color=_ink(float(value)),
            )
    ax_heat.set_xticks(range(len(SPECIES)), [species_label(name) for name in SPECIES],
                       rotation=35, ha="right", fontsize=8)
    ax_heat.set_yticks(range(len(held)), [species_label(name) for name in held], fontsize=9)
    ax_heat.set_xlim(-0.5, len(SPECIES) - 0.5)
    ax_heat.set_ylim(len(held) - 0.5, -0.5)
    ax_heat.set_xlabel("Scored species")
    ax_heat.set_ylabel("Baseline (held-out species)")
    ax_heat.set_title("Per-species validation F1", loc="left", fontsize=10)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.set_aspect("equal")

    heat_box = ax_heat.get_position()
    cax = fig.add_axes([heat_box.x1 + 0.015, heat_box.y0, 0.015, heat_box.height])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax)
    bar.set_label("validation F1", fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.12, right=0.88, top=0.82, bottom=0.18)
    return fig


def main():
    root = repo_root()
    records = latest_bases(root)
    if len(records) < 6:
        missing = [name for name in SPECIES if name not in {r["species"] for r in records}]
        raise SystemExit(f"Need all six baselines; missing: {missing}")
    fig = plot_baselines(records)
    out = root / "notebooks" / "few_shot_site_figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "novel_baselines.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")
    for record in records:
        mean = record["best_val"] if record["best_val"] is not None else record["mean_old"]
        print(
            f"{record['species']}: val_f1_species {mean:.3f}  "
            f"stamp {record['stamp']}  novel_f1 {record['novel_f1']}"
        )
    return path


if __name__ == "__main__":
    main()
