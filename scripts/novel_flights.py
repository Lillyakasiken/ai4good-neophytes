"""Flights that a species-adaption run (task 2) can use, for one k.

A species' unseen flights are all flights of every site where it grows: Alexandru's
base for that species left those sites out of training. A flight qualifies for k when
its /test + /val folders hold at least k tiles that contain the species alone (the
candidates of few_shot_mode=random) and its /train folder (the scored rest of the
flight) holds at least MIN_SCORED tiles with the species.

    python scripts/novel_flights.py 10          # species year/flight candidates scored_tiles train_tiles

Output lines are interleaved over species (round robin), so a queue that stops early
still has results for every species.
"""

import argparse
import csv
from collections import Counter, defaultdict
from itertools import zip_longest

SPECIES = ["A_altissima", "R_typhina", "B_davidii", "R_japonica", "B_orientalis", "S_inaequidens"]
STATS = "../data/Neophytes/stats_imagewise.csv"
MIN_SCORED = 5


def site_group(flight):
    """``Basel_2_3`` -> ``Basel_2``."""
    head, _, tail = flight.rpartition("_")
    return head if tail.isdigit() else flight


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("k", type=int)
    parser.add_argument("--stats", default=STATS)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.stats)))
    year = {r["site"]: r["year"] for r in rows}
    per_species = []
    for sp in SPECIES:
        sites = {site_group(r["site"]) for r in rows if float(r[f"{sp}_px"] or 0) > 0}
        pool, scored, n_train = Counter(), Counter(), Counter()
        for r in rows:
            if site_group(r["site"]) not in sites:
                continue
            has = float(r[f"{sp}_px"] or 0) > 0
            if r["split"] == "train":
                n_train[r["site"]] += 1
                scored[r["site"]] += has
            elif has and float(r[f"only_{sp}"] or 0) == 1:
                pool[r["site"]] += 1
        flights = sorted(f for f in scored if scored[f] >= MIN_SCORED and pool[f] >= args.k)
        per_species.append([(sp, f"{year[f]}/{f}", pool[f], scored[f], n_train[f]) for f in flights])

    for group in zip_longest(*per_species):
        for item in group:
            if item:
                print(*item)


if __name__ == "__main__":
    main()
