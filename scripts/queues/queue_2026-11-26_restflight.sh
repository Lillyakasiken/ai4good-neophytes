#!/usr/bin/env bash
# Queue of 26 Nov: score the cover runs of 25 Nov on the REST OF THEIR ADAPTED FLIGHT.
# The cover runs (task 1 adapt_cv3_cover_tp05, task 2 adapt_cv3_novel_cover_tp05) drew
# their tiles from all three folders of the flight and were scored only on the other
# flights of the site. Here each checkpoint is scored on the whole adapted flight minus
# the tiles of the flight's task 1 train_tiles.txt (data.exclude_tiles). The task 2 runs
# trained on a subset of those tiles, so all models are scored on identical tiles:
#   - received cv3 model (s42, best checkpoint): the baseline, 4 flights
#   - task 1 cover runs (final checkpoint): 4 flights
#   - task 2 cover runs, head/decoder lr 5e-4 (final checkpoint): 9 species x flight cells
# One confusion matrix per flight (adapt_eval: eval_site, no AUC, RGB only).
#
#   tmux new -s rest
#   AFTER=lightning_logs/queue_2026-11-25_folds/status.txt bash scripts/queues/queue_2026-11-26_restflight.sh
#
# AFTER: a queue status file to wait for ("queue finished" line) before the first job.
# Each finished job is recorded in map.txt (job, run, test dir) and skipped on a rerun.
# Logs, status and map: lightning_logs/queue_2026-11-26_restflight/. ~2 h in total.

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate /zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
set -u

AFTER=${AFTER:-}
MIN_FREE_GIB=${MIN_FREE_GIB:-20}
EXPLORER_API=${EXPLORER_API:-http://127.0.0.1:8010}
RECEIVED=20261013_002801.261406_np_cv3_Unet_mit_b2_512_s42

cd "$(dirname "$0")/../.."
QUEUE_DIR=lightning_logs/queue_2026-11-26_restflight
mkdir -p "$QUEUE_DIR"
touch "$QUEUE_DIR/map.txt"

log_status() { echo "$(date '+%F %T')  $*" | tee -a "$QUEUE_DIR/status.txt"; }

if [ -n "$AFTER" ]; then
    log_status "waiting for $AFTER to finish"
    until grep -q "queue finished" "$AFTER" 2>/dev/null; do sleep 120; done
fi

wait_for_gpu() {
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
        if [ "$free_mib" -ge $((MIN_FREE_GIB * 1024)) ]; then return; fi
        curl -s -m 30 -X POST "$EXPLORER_API/api/predict/release" > /dev/null 2>&1 && sleep 10 && continue
        echo "$(date '+%F %T')  GPU has $((free_mib / 1024)) GiB free, waiting for ${MIN_FREE_GIB} GiB ..."
        sleep 300
    done
}

# # score_job <job> <run dir name> <year> <flight> <exclude file> <ckpt_type>
score_job() {
    local job=$1 run=$2 year=$3 flight=$4 exclude=$5 ckpt=$6
    if grep -q "^$job " "$QUEUE_DIR/map.txt"; then log_status "SKIP  $job (done)"; return; fi
    if [ ! -f "$exclude" ]; then log_status "NO EXCLUDE FILE for $job: $exclude"; return; fi
    local d=../data/Neophytes/$year/$flight
    wait_for_gpu
    log_status "START $job: $run"
    python test.py +experiment=adapt_eval exp_name="$run" ckpt_type="$ckpt" \
        data=neophytes_split_cv3_test data.data_name=np_cv3_rest \
        "data.image_dir.test=['$d/train','$d/test','$d/val']" data.exclude_tiles="$exclude" \
        < /dev/null > "$QUEUE_DIR/$job.log" 2>&1
    local rc=$?
    log_status "END   $job: exit $rc"
    if [ $rc -eq 0 ]; then
        local tdir=$(ls -td lightning_logs/"$run"/test_*_np_cv3_rest_* 2>/dev/null | head -1)
        echo "$job $run ${tdir##*/}" >> "$QUEUE_DIR/map.txt"
    fi
}

# # flight -> year; task 2 species per flight (same 9 cells as queue_2026-11-25_novel_cover)
declare -A YEAR=([Basel_2_3]=2024 [Rivera_1_1]=2025 [Waltenschwil_1_1]=2024 [Dornach_1_1]=2025)
declare -A NOVEL=([Basel_2_3]="A_altissima B_davidii S_inaequidens" [Rivera_1_1]="B_davidii"
                  [Waltenschwil_1_1]="R_japonica" [Dornach_1_1]="B_orientalis B_davidii R_japonica A_altissima")

for fl in Basel_2_3 Rivera_1_1 Waltenschwil_1_1 Dornach_1_1; do
    t1=$(ls -td lightning_logs/*_np_cv3_adapt_${fl}_k10cover_tp05_steps500_*_s1 2>/dev/null | head -1)
    if [ -z "$t1" ]; then log_status "NO TASK 1 RUN for $fl"; continue; fi
    excl=$t1/train_tiles.txt
    score_job "received_${fl}" "$RECEIVED" "${YEAR[$fl]}" "$fl" "$excl" best
    score_job "cover_${fl}" "${t1##*/}" "${YEAR[$fl]}" "$fl" "$excl" final
    for sp in ${NOVEL[$fl]}; do
        nv=$(ls -td lightning_logs/*_np_nv${sp}_adapt_${fl}_k10list_tp05_hlr0.0005_dlr0.0005_steps500_*_s1 2>/dev/null | head -1)
        if [ -z "$nv" ]; then log_status "NO TASK 2 RUN for $sp $fl"; continue; fi
        score_job "novel_${sp}_${fl}" "${nv##*/}" "${YEAR[$fl]}" "$fl" "$excl" final
    done
done

log_status "queue finished"
