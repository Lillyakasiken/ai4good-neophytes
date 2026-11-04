"""One figure for the one-site few-shot runs: where they sit, and how each plant fails by stage.

Positions are the median tile of each site in Swiss LV95 metres. Bars are mean
species F1, averaged over seeds. Solid bars sit on a training site and score
the rest of that site. Hatched bars sit on a held-out site and score that place.
"""

from pathlib import Path
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Text

RUN_RE = re.compile(
    r"_s(?P<seed>42|43)_site(?P<site>.+?)_t(?P<k>\d+)_tbg(?P<bg>\d+)$"
)
SPECIES = [
    ("A_altissima", "A. altissima"),
    ("R_typhina", "R. typhina"),
    ("B_davidii", "B. davidii"),
    ("R_japonica", "R. japonica"),
    ("B_orientalis", "B. orientalis"),
    ("S_inaequidens", "S. inaequidens"),
]
STAGES = ["non_flowering", "flowering", "fruiting"]
STAGE_LABEL = {"non_flowering": "non-flowering", "flowering": "flowering", "fruiting": "fruiting"}
K_COLOR = {1: "#d0d0d0", 3: "#6baed6", 10: "#08519c"}
# Card position in axes fraction. Basel, Dornach and Arlesheim share one
# neighbourhood, so those three cards sit in a row above the border.
CARD_AT = {
    "Basel_4": (0.18, 0.90),
    "Basel_2": (0.46, 0.90),
    "Dornach_1": (0.74, 0.90),
    "Olten_1": (0.34, 0.36),
    "Waltenschwil_1": (0.72, 0.34),
    "Ardon_1": (0.16, 0.12),
    "Rivera_1": (0.48, 0.10),
}


def draw_switzerland(ax):
    """Country outline in the same LV95 metres as the sites."""
    path = Path(__file__).resolve().parent / "switzerland_lv95.json"
    rings = json.loads(path.read_text())["rings"]
    ax.add_patch(Polygon(
        rings[0], closed=True, facecolor="#f3f6f8", edgecolor="#7d8b99",
        linewidth=0.9, zorder=0,
    ))
    for hole in rings[1:]:
        ax.add_patch(Polygon(
            hole, closed=True, facecolor="white", edgecolor="#7d8b99",
            linewidth=0.6, zorder=0,
        ))


def repo_root():
    cwd = Path.cwd()
    if (cwd / "lightning_logs").is_dir():
        return cwd
    if (cwd.parent / "lightning_logs").is_dir():
        return cwd.parent
    raise FileNotFoundError("Cannot find lightning_logs/. Start in the repo root or in notebooks/.")


def load_json(path):
    return json.loads(path.read_text(), parse_constant=lambda token: float(token))


def as_site(name):
    parts = str(name).split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-1])
    return str(name)


# Median tile origin of each site, Swiss LV95 metres, from explorer/cache.
CENTROIDS = {
    "Aarau_1": (2645170, 1248682),
    "Aarau_2": (2649346, 1248125),
    "Aarau_3": (2644752, 1248512),
    "Ardon_1": (2586595, 1116978),
    "Ardon_2": (2586669, 1117161),
    "Ardon_3": (2586896, 1117350),
    "Arlesheim_1": (2613040, 1261226),
    "Basel_2": (2616610, 1264807),
    "Basel_3": (2613677, 1265790),
    "Basel_4": (2613436, 1265796),
    "Chamoson_1": (2584728, 1115301),
    "Dietikon_1": (2672364, 1252258),
    "Dornach_1": (2612704, 1258090),
    "Dornach_2": (2612406, 1257083),
    "Olten_1": (2634691, 1244193),
    "Rivera_1": (2714718, 1109624),
    "Rivera_2": (2714702, 1109431),
    "Rueti_1": (2706560, 1236175),
    "Sion_1": (2594585, 1119573),
    "Waltenschwil_1": (2664825, 1242250),
}


def _num(value):
    if value is None:
        return np.nan
    return float(value)


def finished_runs(root):
    records = []
    flights = []
    pheno = []
    pairs = []
    detail = []
    for run in sorted((root / "lightning_logs").iterdir()):
        if not run.is_dir() or "20261030" not in run.name:
            continue
        match = RUN_RE.search(run.name)
        if not match or int(match.group("bg")) != 100:
            continue
        tests = sorted(
            p for p in run.glob("test_*")
            if "np_cv3" in p.name and "np_debug" not in p.name
            and (p / "quantitative" / "metrics_site_split.json").is_file()
        )
        if not tests:
            continue
        quant = tests[-1] / "quantitative"
        split = load_json(quant / "metrics_site_split.json")
        site = match.group("site")
        k = int(match.group("k"))
        seed = int(match.group("seed"))
        held_block = split["held_out"]
        records.append({
            "site": site,
            "k": k,
            "seed": seed,
            "same": split["same_site"]["F1-avg-wo0"],
            "held": held_block["F1-avg-wo0"],
        })
        for name, value in zip(held_block.get("class_names") or [], held_block.get("F1") or []):
            if name == "Background":
                continue
            pairs.append({
                "site": site, "k": k, "seed": seed,
                "species": name, "f1": _num(value),
            })
        same_block = split["same_site"]
        for name, value in zip(same_block.get("class_names") or [], same_block.get("F1") or []):
            if name == "Background":
                continue
            detail.append({
                "train": site, "k": k, "seed": seed, "place": "same",
                "species": name, "f1": _num(value),
            })
        site_path = quant / "metrics_semseg_site.json"
        if site_path.is_file():
            for flight, block in load_json(site_path).items():
                group = as_site(flight)
                flights.append({
                    "train": site, "k": k, "seed": seed,
                    "held_site": group, "f1": block.get("F1-avg-wo0"),
                })
                for name, value in zip(block.get("class_names") or [], block.get("F1") or []):
                    if name == "Background":
                        continue
                    detail.append({
                        "train": site, "k": k, "seed": seed, "place": group,
                        "species": name, "f1": _num(value),
                    })
        break_path = quant / "metrics_phenology_breakdown.json"
        if break_path.is_file():
            cells = load_json(break_path)["cells"]
            for stats, label in SPECIES:
                for stage in STAGES:
                    cell = cells.get(stats, {}).get(stage, {})
                    if cell.get("below_min_px") or cell.get("recall") is None:
                        continue
                    pheno.append({
                        "species": label, "stage": stage,
                        "recall": cell["recall"],
                        "miss": cell["miss_to_background"],
                        "confused": cell["confused"],
                    })
    return (
        pd.DataFrame(records), pd.DataFrame(flights), pd.DataFrame(pheno),
        pd.DataFrame(pairs), pd.DataFrame(detail),
    )


def seed_mean(frame, key):
    """Mean over seeds, then the value at each k. Missing k stays NaN."""
    if frame.empty:
        return {1: np.nan, 3: np.nan, 10: np.nan}
    grouped = frame.groupby("k")[key].mean()
    return {k: float(grouped[k]) if k in grouped.index else np.nan for k in (1, 3, 10)}


def held_by_k(flights, held_site):
    part = flights[flights["held_site"] == held_site] if not flights.empty else flights
    if part.empty:
        return {1: np.nan, 3: np.nan, 10: np.nan}
    # One score per training run, then the mean of those runs at each k.
    per_run = part.groupby(["train", "k", "seed"], as_index=False)["f1"].mean()
    return seed_mean(per_run, "f1")


def draw_bars(site, values, hatch):
    width, height = 118, 32
    area = DrawingArea(width, height, 0, 0)
    area.add_artist(Text(0, 24, site.replace("_", " "), fontsize=8, color="0.1", fontweight="medium"))
    for i, k in enumerate((1, 3, 10)):
        value = values[k]
        missing = value is None or (isinstance(value, float) and math.isnan(value))
        x = 2 + i * 38
        area.add_artist(Rectangle(
            (x, 4), 10, 12,
            facecolor="white" if missing else K_COLOR[k],
            edgecolor="0.4", linewidth=0.4,
            hatch="////" if hatch and not missing else None,
        ))
        area.add_artist(Text(
            x + 13, 10, "—" if missing else f"{float(value):.2f}",
            fontsize=7.5, va="center", ha="left", color="0.15",
        ))
    return area


def _ink(value):
    """Readable text on viridis: dark on the bright end, white on the dark end."""
    return "black" if value > 0.6 else "white"


def _show_matrix(ax, matrix):
    """viridis 0..1, grey for missing cells, white lines between cells, no frame."""
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#ececec")
    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    rows, cols = matrix.shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def heatmap(ax, frame, column, title, show_xticks=True):
    matrix = np.full((len(SPECIES), len(STAGES)), np.nan)
    if not frame.empty:
        for i, (_, label) in enumerate(SPECIES):
            for j, stage in enumerate(STAGES):
                part = frame[(frame["species"] == label) & (frame["stage"] == stage)]
                if len(part):
                    matrix[i, j] = part[column].mean()
    image = _show_matrix(ax, matrix)
    ax.set_xticks(range(len(STAGES)), [STAGE_LABEL[s] for s in STAGES] if show_xticks else [])
    ax.set_yticks(range(len(SPECIES)), [label for _, label in SPECIES])
    ax.set_title(title, loc="left", fontsize=11)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color=_ink(matrix[i, j]), fontsize=9)
    return image


def _sort_key(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return 1.0
    return -float(np.nanmean(values))


def plants_on_sites(root):
    """Sites that actually contain each plant, from the per-tile label table.

    A flight folder such as ``Basel_4_1`` is grouped to ``Basel_4``. Returns
    ``None`` when the table is missing, so the plot still draws.
    """
    candidates = [
        root.parent / "data" / "Neophytes" / "stats_imagewise.csv",
        Path("/zfs/ai4good/datasets/Neophytes/stats_imagewise.csv"),
    ]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        return None
    columns = ["site"] + [f"{key}_px" for key, _ in SPECIES]
    frame = pd.read_csv(path, usecols=columns)
    frame["site"] = frame["site"].map(as_site)
    present = set()
    for key, label in SPECIES:
        totals = frame.groupby("site")[f"{key}_px"].sum()
        for site, total in totals.items():
            if float(total) > 0:
                present.add((str(site), label))
    return present


def rank_heatmap(ax, pairs, present):
    """Training site by plant, held-out species F1. Best row and best column first.

    A plant with no labelled pixels on that training site is left blank. The
    sort ignores those cells, so a zero from a plant the site never had does
    not pull the site down. Returns the image and the site order.
    """
    plants = [label for _, label in SPECIES]
    sites = [] if pairs.empty else list(dict.fromkeys(pairs["site"]))
    grouped = None if pairs.empty else pairs.groupby(["site", "species"])["f1"].mean()
    matrix = np.full((len(plants), len(sites)), np.nan)
    for i, plant in enumerate(plants):
        for j, site in enumerate(sites):
            if present is not None and (site, plant) not in present:
                continue
            if grouped is not None:
                matrix[i, j] = grouped.get((site, plant), np.nan)
    plant_order = sorted(range(len(plants)), key=lambda i: _sort_key(matrix[i]))
    site_order = sorted(range(len(sites)), key=lambda j: _sort_key(matrix[:, j] if len(sites) else []))
    matrix = matrix[np.ix_(plant_order, site_order)] if len(sites) else matrix
    plants = [plants[i] for i in plant_order]
    sites = [sites[j] for j in site_order]
    n_runs = {} if pairs.empty else pairs.groupby("site")[["k", "seed"]].apply(
        lambda part: len(part.drop_duplicates())
    ).to_dict()
    image = _show_matrix(ax, matrix)
    ax.set_xticks(
        range(len(sites)),
        [f"{site.replace('_', ' ')}\n{n_runs.get(site, 0)} runs" for site in sites],
    )
    ax.set_yticks(range(len(plants)), plants)
    ax.set_title("Held-out F1 by training site and plant", loc="left", fontsize=11)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color=_ink(matrix[i, j]), fontsize=9)
    return image, sites


def seed_summary(detail):
    """Mean species F1 over k, then over the two seeds, plus |seed 42 − seed 43|."""
    columns = ["train", "place", "species", "mean", "gap"]
    if detail.empty:
        return pd.DataFrame(columns=columns)
    per_k = detail.groupby(["train", "seed", "place", "species", "k"], as_index=False)["f1"].mean()
    per_seed = per_k.groupby(["train", "seed", "place", "species"], as_index=False)["f1"].mean()
    rows = []
    for keys, part in per_seed.groupby(["train", "place", "species"]):
        train, place, species = keys
        by_seed = {int(row.seed): row.f1 for row in part.itertuples()}
        first, second = by_seed.get(42, np.nan), by_seed.get(43, np.nan)
        finite = [value for value in (first, second) if np.isfinite(value)]
        if not finite:
            continue
        rows.append({
            "train": train,
            "place": place,
            "species": species,
            "mean": float(np.mean(finite)),
            "gap": abs(first - second) if len(finite) == 2 else np.nan,
        })
    return pd.DataFrame(rows, columns=columns)


def pair_heatmap(ax, summary, present, train_sites, held_sites):
    """Each training site beside the held-out sites, one row per plant.

    The cell is the mean of the two seeds. The second line is |seed 42 − seed 43|.
    Grey means that plant has no labelled pixels on the site named by the column.
    """
    plants = [label for _, label in SPECIES]
    columns = []
    for site in train_sites:
        columns.append((site, "same"))
        for held in held_sites:
            columns.append((site, held))
    matrix = np.full((len(plants), len(columns)), np.nan)
    gaps = np.full_like(matrix, np.nan)
    lookup = {}
    if not summary.empty:
        for row in summary.itertuples():
            lookup[(row.train, row.place, row.species)] = (row.mean, row.gap)
    for i, plant in enumerate(plants):
        for j, (site, place) in enumerate(columns):
            scored_site = site if place == "same" else place
            if present is not None and (scored_site, plant) not in present:
                continue
            mean, gap = lookup.get((site, place, plant), (np.nan, np.nan))
            matrix[i, j] = mean
            gaps[i, j] = gap
    image = _show_matrix(ax, matrix)
    tick_labels = ["same site" if place == "same" else place.replace("_", " ") for _site, place in columns]
    ax.set_xticks(range(len(columns)), tick_labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=8)
    for tick, (_site, place) in zip(ax.get_xticklabels(), columns):
        if place == "same":
            tick.set_fontweight("bold")
    ax.set_yticks(range(len(plants)), plants)
    ax.set_title(
        "Per-plant F1 for each training site: own site first, then each held-out site",
        loc="left", fontsize=11, pad=38,
    )
    group = len(held_sites) + 1
    for boundary in range(group, len(columns), group):
        ax.axvline(boundary - 0.5, color="white", linewidth=6)
    for start, site in zip(range(0, len(columns), group), train_sites):
        middle = start + (group - 1) / 2
        ax.annotate(
            f"trained on {site.replace('_', ' ')}", xy=(middle, 1.075), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=9.5, fontweight="medium", color="0.15",
        )
        if present is not None:
            had = [label for label in plants if (site, label) in present]
            ax.annotate(
                "has " + ", ".join(had) if had else "no labelled plants",
                xy=(middle, 1.025), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=7.5, color="0.35", fontstyle="italic",
            )
        ax.plot(
            [start - 0.4, start + group - 0.6], [1.015, 1.015], color="0.35", linewidth=1,
            transform=ax.get_xaxis_transform(), clip_on=False,
        )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isfinite(matrix[i, j]):
                continue
            ink = _ink(matrix[i, j])
            if present is not None and (columns[j][0], plants[i]) not in present:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor="white", alpha=0.6, linewidth=0, zorder=2,
                ))
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor="0.6",
                    hatch="////", linewidth=0, zorder=2,
                ))
                ink = "0.2"
            has_gap = np.isfinite(gaps[i, j])
            ax.text(j, i - (0.13 if has_gap else 0), f"{matrix[i, j]:.2f}",
                    ha="center", va="center", color=ink, fontsize=8.5, zorder=3)
            if has_gap:
                ax.text(j, i + 0.22, f"Δ{gaps[i, j]:.2f}", ha="center", va="center",
                        color=ink, fontsize=6.5, alpha=0.8, zorder=3)
    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor="#ececec", edgecolor="none",
                      label="plant not on the scored site"),
            Rectangle((0, 0), 1, 1, facecolor="#e8e8e8", edgecolor="0.6", hatch="////", linewidth=0,
                      label="plant not on the training site, so never seen in training"),
        ],
        loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, frameon=False, fontsize=8.5,
    )
    return image


PLAN_RE = re.compile(
    r"_s(?P<seed>42|43)(?:_site(?P<site>.+?))?_t(?P<k>\d+)_tbg(?P<bg>\d+)$"
)
PLAN_SITES = [
    ("Basel_4", "Basel 4"),
    ("Olten_1", "Olten 1"),
    ("Ardon_1", "Ardon 1"),
    ("Arlesheim_1", "Arlesheim 1"),
    ("Dornach_2", "Dornach 2"),
    ("all", "all training sites"),
]
PLAN_SETTINGS = [
    (1, 100, "k = 1\nbg = 100"),
    (3, 100, "k = 3\nbg = 100"),
    (10, 100, "k = 10\nbg = 100"),
    (10, 10, "k = 10\nbg = 10"),
    (10, 1000, "k = 10\nbg = 1000"),
]
HELD_OUT_SITES = ["Basel_2", "Dornach_1", "Rivera_1", "Waltenschwil_1"]


def in_plan(site, k, bg):
    if site == "all":
        return k == 10 and bg == 100
    if bg == 100 and k in (1, 3, 10):
        return True
    return site == "Basel_4" and k == 10 and bg in (10, 1000)


def run_status(root):
    """planned / started / tested for each site, k, background count, and seed."""
    status = {}
    for run in (root / "lightning_logs").iterdir():
        if not run.is_dir() or "20261030" not in run.name:
            continue
        if "neophytes_split_cv3_train_512_model_neophytes_mit_b2" not in run.name:
            continue
        match = PLAN_RE.search(run.name)
        if not match:
            continue
        key = (
            match.group("site") or "all",
            int(match.group("k")),
            int(match.group("bg")),
            int(match.group("seed")),
        )
        tested = any(
            "np_cv3" in path.name and "np_debug" not in path.name
            and (path / "quantitative" / "metrics_site_split.json").is_file()
            for path in run.glob("test_*")
        )
        rank = {"planned": 0, "started": 1, "tested": 2}
        current = "tested" if tested else "started"
        if rank[current] >= rank.get(status.get(key, "planned"), 0):
            status[key] = current
    return status


def draw_run_plan(root, present):
    """One grid of the 36 runs: site, labelled tiles, empty tiles, and seed."""
    status = run_status(root)
    fig = plt.figure(figsize=(12.4, 7.6))
    ax = fig.add_subplot(111)
    face = {"tested": "#08519c", "started": "white", "planned": "#e6e6e6"}
    edge = {"tested": "#08519c", "started": "#08519c", "planned": "#bdbdbd"}
    ink = {"tested": "white", "started": "#08519c", "planned": "#666666"}
    for i, (site, _label) in enumerate(PLAN_SITES):
        for j, (k, bg, _text) in enumerate(PLAN_SETTINGS):
            if not in_plan(site, k, bg):
                continue
            ax.add_patch(Rectangle(
                (j - 0.48, i - 0.46), 0.96, 0.92,
                facecolor="#f4f7f8", edgecolor="none", zorder=0,
            ))
            for seed, shift in ((42, -0.18), (43, 0.18)):
                state = status.get((site, k, bg, seed), "planned")
                ax.scatter(
                    [j + shift], [i], s=420, facecolors=face[state], edgecolors=edge[state],
                    linewidths=1.2, zorder=2,
                )
                ax.text(
                    j + shift, i, str(seed), ha="center", va="center",
                    fontsize=7, color=ink[state], zorder=3,
                )
    def plant_line(site):
        if site == "all":
            return "every species, spread across sites"
        if not present:
            return ""
        names = [label for _key, label in SPECIES if (site, label) in present]
        return ", ".join(names)

    ylabels = [f"{label}\n{plant_line(site)}" for site, label in PLAN_SITES]
    ax.set_yticks(range(len(PLAN_SITES)), ylabels, fontsize=8)
    ax.set_xticks(range(len(PLAN_SETTINGS)), [text for _k, _bg, text in PLAN_SETTINGS], fontsize=8)
    ax.set_xlim(-0.6, len(PLAN_SETTINGS) - 0.4)
    ax.set_ylim(len(PLAN_SITES) - 0.55, -0.55)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "What was run\n"
        "Fold 3, 512 px crops, MiT-B2, 30 epochs, validation after epochs 10, 20 and 30.\n"
        "k is labelled tiles per species, the largest ones, spread over flights and stages. "
        "bg is extra empty tiles from the same site.\n"
        "Every run is tested on Basel 2, Dornach 1, Rivera 1 and Waltenschwil 1. "
        "A blank cell was not in the list.",
        x=0.02, ha="left", fontsize=10,
    )
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none", markersize=10,
            markerfacecolor=face[state], markeredgecolor=edge[state], label=label,
        )
        for state, label in (
            ("tested", "fold-3 test finished"),
            ("started", "started, no test yet"),
            ("planned", "not started"),
        )
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.78, bottom=0.16)
    out = root / "notebooks" / "few_shot_site_figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_plan.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main():
    root = repo_root()
    places = CENTROIDS
    runs, flights, pheno, pairs, detail = finished_runs(root)
    fig = plt.figure(figsize=(15.6, 18.4))
    grid = fig.add_gridspec(
        4, 2, width_ratios=[1.35, 1], height_ratios=[1, 1, 1.05, 1.5],
        wspace=0.3, hspace=0.42,
    )
    ax = fig.add_subplot(grid[0:2, 0])
    ax_recall = fig.add_subplot(grid[0, 1])
    ax_miss = fig.add_subplot(grid[1, 1])
    ax_rank = fig.add_subplot(grid[2, 0])
    ax_other = fig.add_subplot(grid[2, 1])
    ax_pairs = fig.add_subplot(grid[3, :])
    draw_switzerland(ax)

    trained = sorted({row.site for row in runs.itertuples()}) if not runs.empty else []
    held_sites = ["Basel_2", "Dornach_1", "Rivera_1", "Waltenschwil_1"]
    waiting = ["Arlesheim_1", "Dornach_2"]
    marked = set(CARD_AT) | set(waiting)
    others = [site for site in places if site not in marked]
    if others:
        xy = np.array([places[site] for site in others])
        ax.scatter(xy[:, 0], xy[:, 1], s=14, c="#d0d0d0", zorder=1, label="other study sites")

    for site in waiting:
        if site not in places:
            continue
        east, north = places[site]
        ax.scatter([east], [north], s=36, facecolors="white", edgecolors="0.35", zorder=3)
    for site in held_sites:
        if site not in places:
            continue
        east, north = places[site]
        ax.scatter([east], [north], s=42, marker="D", c="#f4a261", edgecolors="0.2", zorder=4)
    for site in trained:
        if site not in places:
            continue
        east, north = places[site]
        ax.scatter([east], [north], s=46, c="#1d3557", zorder=5)

    def add_bars(site, values, hatch):
        if site not in places or site not in CARD_AT:
            return
        east, north = places[site]
        box = AnnotationBbox(
            draw_bars(site, values, hatch), (east, north),
            xybox=CARD_AT[site],
            xycoords="data", boxcoords="axes fraction",
            frameon=True, pad=0.15,
            bboxprops=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#dddddd", lw=0.6),
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7),
        )
        ax.add_artist(box)

    for site in trained:
        add_bars(site, seed_mean(runs[runs["site"] == site], "same"), False)
    for site in held_sites:
        add_bars(site, held_by_k(flights, site), True)

    ax.set_xlim(2_470_000, 2_850_000)
    # Extra room above the border so the Basel cards sit clear of the sites.
    ax.set_ylim(1_065_000, 1_360_000)
    ax.set_aspect("equal")
    km = plt.FuncFormatter(lambda value, _pos: f"{value / 1000:.0f}")
    ax.xaxis.set_major_formatter(km)
    ax.yaxis.set_major_formatter(km)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("Swiss easting, LV95 (km)")
    ax.set_ylabel("Swiss northing, LV95 (km)")
    ax.set_title("Where the runs were trained and scored", loc="left")
    ax.tick_params(labelsize=8)
    ax.grid(color="#e4e9ee", zorder=0.4)

    for k, color in K_COLOR.items():
        ax.scatter([], [], marker="s", s=40, c=color, label=f"k = {k}")
    ax.scatter([], [], s=46, c="#1d3557", label="training site")
    ax.scatter([], [], s=42, marker="D", c="#f4a261", edgecolors="0.2", label="held-out site")
    ax.scatter([], [], s=36, facecolors="white", edgecolors="0.35", label="not run yet")
    ax.legend(
        frameon=True, loc="center left", fontsize=6.5, framealpha=0.94,
        bbox_to_anchor=(0.73, 0.55),
    )

    image = heatmap(ax_recall, pheno, "recall", "Named the right plant", show_xticks=False)
    heatmap(ax_miss, pheno, "miss", "Called background", show_xticks=False)
    heatmap(ax_other, pheno, "confused", "Named another plant")
    present = plants_on_sites(root)
    rank, trained_order = rank_heatmap(ax_rank, pairs, present)
    paired = pair_heatmap(ax_pairs, seed_summary(detail), present, trained_order, held_sites)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.915, bottom=0.07)

    def side_colorbar(image, top_ax, bottom_ax, label, gap=0.01):
        top, bottom = top_ax.get_position(), bottom_ax.get_position()
        cax = fig.add_axes([top.x1 + gap, bottom.y0, 0.011, top.y1 - bottom.y0])
        bar = fig.colorbar(image, cax=cax)
        bar.set_label(label, fontsize=9)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8)

    side_colorbar(image, ax_recall, ax_other, "share of that plant's pixels")
    side_colorbar(rank, ax_rank, ax_rank, "held-out species F1")
    side_colorbar(paired, ax_pairs, ax_pairs, "species F1")
    n = 0 if runs.empty else runs.groupby(["site", "k", "seed"]).ngroups
    fig.suptitle(
        f"One-site few-shot runs with a fold-3 test ({n} of 36). Bars average the two seeds.\n"
        "The three count panels are shares of pixels labelled as that plant and stage. They add to 1. "
        "Empty ground is not counted.\n"
        "The site-by-plant panel is held-out species F1. Best plant on top, best site on the left. "
        "Grey means the training site has no labelled pixels of that plant.\n"
        "Bottom: that same score on the training site, then on each held-out site. "
        "Δ is |seed 42 − seed 43|, missing when only one seed is tested. "
        "Grey means the plant is not on that column's site.",
        x=0.01, ha="left", fontsize=11,
    )
    out = root / "notebooks" / "few_shot_site_figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "site_map_overview.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")
    print(f"runs {len(runs)} pheno cells {len(pheno)}")
    draw_run_plan(root, present)
    return path


if __name__ == "__main__":
    main()
