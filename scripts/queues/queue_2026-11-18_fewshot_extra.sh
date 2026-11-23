#!/usr/bin/env bash
# Queue of Wed 2026-11-18: extra few-shot runs after queue_2026-11-16_fewshot_loso.sh, for
# the talk of Fri 2026-11-20. One lane, ordered by importance; kill it on Thursday morning
# whatever has not finished by then.
#
# The 2026-11-16 queue had 3 draws per point. PlantCLEF vs DINOv2 is already clear in a
# paired test (cv3 k = 10: PlantCLEF better in 16/18 pairs, Wilcoxon p = 0.0004), but the
# per-species ranges, and above all the single-flight draws (figure 4), rest on 3 draws.
# Draws 4-6 go into separate directories with the SAME method names, so summary.txt and
# the figures pool them with draws 1-3 (n = 6).
#
#   0. checks (CPU): what draws 4-6 get, random and single-flight
#   1. PlantCLEF, cv3, k = 10, one flight, draws 4-6     figure 4      ~70 min
#   2. PlantCLEF, cv3, k = 10, draws 4-6                 figure 1      ~70 min
#   3. DINOv2,    cv3, k = 10, draws 4-6                 figure 1      ~70 min
#   4. MiT-b2 ImageNet, cv3, k = 10, draws 1-3, all 6 species: the b2 side of figure 1
#      (only B. davidii so far). Needs ~40 GiB (35 GiB peak in October): runs here if
#      the GPU has room, otherwise at the end of the lane. Smoke test first.
#   5. PlantCLEF, cv3, k = 1, 3, 30, draws 4-6           figure 2      ~3.5 h
#   6. DINOv2,    cv3, k = 1, 3, 30, draws 4-6           figure 2      ~3.5 h
#
# Times: the DINOv2 jobs of 2026-11-17 that had the GPU to themselves took 68-70 min
# (0.48 s/batch with 18 heads); PlantCLEF is the same ViT-B/14. MiT-b2 with 18 heads is
# not measured (whole tiles, not a sliding window).
# Single-flight draws: A. altissima has only 4 flights with >= 10 tiles, so its draws 5
# and 6 reuse the flights of draws 1 and 2 (other tiles of that flight); every other
# species gets 3 new flights.
#
#     nvidia-smi
#     tmux new -s fs_extra
#     bash scripts/queues/queue_2026-11-18_fewshot_extra.sh
#
# Logs: lightning_logs/queue_2026-11-18_fewshot_extra/ (status.txt for the overview).
# Rerunning skips finished jobs (*.done); an interrupted job keeps its trained heads.

set -u

CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
QUEUE_DIR=lightning_logs/queue_2026-11-18_fewshot_extra
OUT=results/fewshot/loso
OLD_OUT=results/fewshot/cv3_bdavidii   # the runs of 2026-10-30
SEEDS="4 5 6"
PROBE_GIB=12   # ViT probes: 6 GiB peak with 18 heads
MIT_GIB=40     # MiT-b2 probes: 35 GiB peak in October (3 heads)

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/../.."
mkdir -p "$QUEUE_DIR" "$OUT"

log() { echo "$(date '+%F %T')  $*" | tee -a "$QUEUE_DIR/status.txt"; }
is_done() { [ -f "$QUEUE_DIR/$1.done" ]; }
mark_done() { touch "$QUEUE_DIR/$1.done"; }
stop() { log "STOP  $*"; exit 1; }
free_gib() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ') / 1024 )); }

# wait_for_gpu <GiB>
wait_for_gpu() {
    while [ "$(free_gib)" -lt "$1" ]; do
        echo "$(date '+%F %T')  GPU has $(free_gib) GiB free, waiting for $1 GiB ..."
        sleep 300
    done
}

# run_job <name> <min free GiB> <command...>: as in queue_2026-11-16_fewshot_loso.sh
run_job() {
    local name=$1 need=$2; shift 2
    if is_done "$name"; then
        log "SKIP  $name (done)"
        return 0
    fi
    wait_for_gpu "$need"
    log "START $name: $*"
    local t0=$SECONDS
    "$@" > "$QUEUE_DIR/$name.log" 2>&1 &
    local pid=$! peak=0 used
    while kill -0 "$pid" 2> /dev/null; do
        used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
               | awk -F', ' -v p="$pid" '$1 == p {print $2}')
        if [ -n "$used" ] && [ "$used" -gt "$peak" ]; then peak=$used; fi
        sleep 30
    done
    wait "$pid"
    local rc=$?
    echo "$peak" > "$QUEUE_DIR/$name.peak"
    log "END   $name: exit $rc, $(((SECONDS - t0) / 60)) min, peak GPU memory $((peak / 1024)) GiB"
    [ $rc -eq 0 ] && mark_done "$name"
    return $rc
}

# paired: PlantCLEF - DINOv2 on the same tiles (same fold, k, species, draw)
paired() {
    python - "$OUT/summary.csv" <<'EOF'
import sys
import pandas as pd
from scipy.stats import wilcoxon
d = pd.read_csv(sys.argv[1])
d = d[d.method.isin(['probe_pc24', 'probe_dinov2']) & (d.draw == 'random')]
p = d.pivot_table(index=['fold', 'k', 'species', 'seed'], columns='method', values=['F1', 'known_F1']).dropna()
print('PlantCLEF - DINOv2, linear probes, paired by draw (same tiles)')
for (fold, k), g in p.groupby(level=[0, 1]):
    diff = g['F1']['probe_pc24'] - g['F1']['probe_dinov2']
    known = g['known_F1']['probe_pc24'] - g['known_F1']['probe_dinov2']
    pv = wilcoxon(diff).pvalue if (diff != 0).any() else float('nan')
    print(f'{fold} k={k:>2}: new-species F1 {diff.mean():+.3f}, PlantCLEF better in {int((diff > 0).sum())}/{len(diff)} '
          f'pairs (Wilcoxon p = {pv:.2g}); known-species F1 {known.mean():+.3f}')
EOF
}

# report: summary table, paired test and figures (CPU, ~1 min)
report() {
    { python fewshot_probe.py summary --root $OUT \
        && paired > $OUT/paired_pc24_vs_dinov2.txt \
        && python scripts/plot_fewshot.py --roots $OLD_OUT $OUT --out $OUT/figures; } > "$QUEUE_DIR/report.log" 2>&1 \
        && log "report: $OUT/summary.txt, $OUT/paired_pc24_vs_dinov2.txt, $OUT/figures/" \
        || log "NOTE  report failed, see $QUEUE_DIR/report.log"
}

# probe_grid <job> <backbone> <k> [args...]: all 6 species x the draws at one k (cv3)
probe_grid() {
    local name=$1 backbone=$2 k=$3; shift 3
    run_job "$name" "$PROBE_GIB" python fewshot_probe.py probe --backbone "$backbone" --fold cv3 \
        --new_species all --k "$k" --seeds $SEEDS --method "probe_$backbone" --out "$OUT/$name" "$@" \
        || log "NOTE  $name failed, see $QUEUE_DIR/$name.log"
    grep -h "scoring speed" "$QUEUE_DIR/$name.log" 2> /dev/null | tail -n 1 | while read -r line; do log "$name: $line"; done
    report
}

# mit_b2 [now]: smoke test, then the 18 MiT-b2 heads (draws 1-3, the draws of the
# October B. davidii heads). With "now" it only runs if the GPU has the room right away.
mit_b2() {
    is_done cv3_mit_b2_k10 && return 0
    if [ "${1:-}" = now ] && [ "$(free_gib)" -lt "$MIT_GIB" ]; then
        log "NOTE  cv3_mit_b2_k10 deferred to the end of the lane: GPU has $(free_gib) GiB free, needs $MIT_GIB"
        return 0
    fi
    run_job smoke_mit_b2 "$MIT_GIB" python fewshot_probe.py probe --backbone mit_b2 --fold cv3 \
        --new_species B_davidii R_typhina --k 10 --seeds 1 --max_steps 20 --max_test_batches 12 \
        --method probe_mit_b2 --out "$QUEUE_DIR/smoke_mit_b2" \
        || { log "NOTE  smoke_mit_b2 failed, see $QUEUE_DIR/smoke_mit_b2.log"; return 1; }
    log "smoke_mit_b2: $(grep -h 'scoring speed' "$QUEUE_DIR/smoke_mit_b2.log" | tail -n 1)"
    run_job cv3_mit_b2_k10 "$MIT_GIB" python fewshot_probe.py probe --backbone mit_b2 --fold cv3 \
        --new_species all --k 10 --seeds 1 2 3 --method probe_mit_b2 --out "$OUT/cv3_mit_b2_k10" \
        || log "NOTE  cv3_mit_b2_k10 failed, see $QUEUE_DIR/cv3_mit_b2_k10.log"
    grep -h "scoring speed" "$QUEUE_DIR/cv3_mit_b2_k10.log" 2> /dev/null | tail -n 1 | while read -r line; do log "cv3_mit_b2_k10: $line"; done
    report
}

log "queue start"

# %% 0. checks (CPU): the tiles of draws 4-6
{ python fewshot_probe.py check --fold cv3 --new_species all --k 1 3 10 30 --seeds $SEEDS \
    && python fewshot_probe.py check --fold cv3 --new_species all --k 10 --seeds $SEEDS --draw flight; } \
    > "$QUEUE_DIR/check.log" 2>&1 || stop "dataset check failed, see $QUEUE_DIR/check.log"
short=$(grep -c "<-- only" "$QUEUE_DIR/check.log")
[ "$short" -eq 0 ] || log "NOTE  $short draws get fewer than k tiles (grep '<-- only' $QUEUE_DIR/check.log)"
grep "(flight)" "$QUEUE_DIR/check.log" | sed 's/^ *//' | paste -d' ' - - - | while read -r line; do log "flights: $line"; done

# %% 1.-3. k = 10, draws 4-6
probe_grid cv3_pc24_k10_flight_s456 pc24 10 --draw flight
probe_grid cv3_pc24_k10_s456 pc24 10
probe_grid cv3_dinov2_k10_s456 dinov2 10

# %% 4. MiT-b2 ImageNet, all species (if the GPU has room now)
mit_b2 now

# %% 5.-6. k sweep, draws 4-6
for k in 1 3 30; do probe_grid cv3_pc24_k${k}_s456 pc24 $k; done
for k in 1 3 30; do probe_grid cv3_dinov2_k${k}_s456 dinov2 $k; done

# %% 4 again, if it was deferred
mit_b2

log "queue finished"
