#!/usr/bin/env bash
# Queue of Wed 2026-10-14 (see docs/changes_2026-10-12.md, section 5, decision 4): the
# first Phase 1 change, a larger encoder. MiT-b4 instead of MiT-b2 on cv3 with seeds 42
# and 43, so that each run pairs with the 512 b2 run of the same seed.
#
# 0. Memory probe: b2 vs b4, a few steps on random inputs                   ~2 min
# 1. 512 crops, cv3, seed 42, mit_b4, then test.py                          ~11-13 + 1.5 h (not measured)
# 2. 512 crops, cv3, seed 43, mit_b4, then test.py                          ~11-13 + 1.5 h (not measured)
# 3. evaluate_cv.py for b2 and b4 on cv3 (CPU, reads result files)          ~1 min
#
# Same settings as the b2 runs of queue_2026-10-13_512.sh (30 epochs, validation every
# 3rd epoch, batch 8); only the encoder changes, so the comparison with b2 is paired.
#
# Memory: b4 is larger than b2 (34 GiB at 512), and the L40S has 45 GiB.
#   - Job 0 estimates the b4 peak as 34 GiB + (b4 - b2) of the probe. If the estimate
#     does not fit on the GPU, the queue stops before any training.
#   - Job 1 waits until the GPU has the estimate + 2 GiB free (at least 36), and the
#     real peak of its process is sampled every 30 s and written to status.txt.
#   - Job 2 waits for the measured peak + 2 GiB. If job 1 runs out of memory, the
#     queue stops, because job 2 would fail the same way.
#
#     tmux new -s queue_b4
#     bash scripts/queues/queue_2026-10-14_b4.sh
#
# Logs and exit codes go to lightning_logs/queue_2026-10-14/.

set -u

CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
QUEUE_DIR=lightning_logs/queue_2026-10-14
B2_PEAK_GIB=34      # measured peak of the 512 b2 runs
MARGIN_GIB=2
MIN_FREE_FLOOR=36   # never wait for less than the 512 b2 queue did

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/../.."
mkdir -p "$QUEUE_DIR"

log() { echo "$(date '+%F %T')  $*" | tee -a "$QUEUE_DIR/status.txt"; }

# wait_for_gpu <GiB>
wait_for_gpu() {
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
        if [ "$free_mib" -ge $(($1 * 1024)) ]; then return; fi
        echo "$(date '+%F %T')  GPU has $((free_mib / 1024)) GiB free, waiting for $1 GiB ..."
        sleep 300
    done
}

# run_job <name> <min free GiB> <command...>: waits for the GPU, logs output, exit
# code and the peak GPU memory of the job's process (MiB, in $QUEUE_DIR/<name>.peak)
run_job() {
    local name=$1 need=$2; shift 2
    wait_for_gpu "$need"
    log "START $name: $*"
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
    log "END   $name: exit $rc, peak GPU memory $((peak / 1024)) GiB"
    return $rc
}

# newest run directory of a 512 b4 run: run_dir_b4 <fold> <seed>
run_dir_b4() {
    ls -d lightning_logs/*_np_$1_Unet_mit_b4_512_s$2 2>/dev/null | sort | tail -n 1
}

# test_b4 <fold> <seed>: test.py on the best checkpoint of the newest matching run
test_b4() {
    local run_dir; run_dir=$(run_dir_b4 "$1" "$2")
    if [ -z "$run_dir" ]; then
        log "SKIP  test_b4_$1_s$2: no run directory"
        return 1
    fi
    run_job "test_b4_$1_s$2" "$MIN_FREE_FLOOR" \
        python test.py exp_name="$(basename "$run_dir")" ckpt_type=best compute_auc=False
}

# train_b4 <fold> <seed> <min free GiB>: the 512 development setting with mit_b4
train_b4() {
    run_job "train_b4_$1_s$2" "$3" python train.py data=neophytes_split_$1_train_512 \
        model=model_neophytes_mit_b4 model.max_epochs=30 model.check_val_every_n_epoch=3 \
        pl_seed=$2
}

# %% 0. memory probe
total_gib=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | tr -d ' ') / 1024 ))
run_job probe_mem "$MIN_FREE_FLOOR" python scripts/queues/probe_gpu_mem.py \
    --encoders mit_b2 mit_b4 --b2-peak-gib "$B2_PEAK_GIB" \
    || { log "STOP  memory probe failed, see $QUEUE_DIR/probe_mem.log"; exit 1; }
est_gib=$(tail -n 1 "$QUEUE_DIR/probe_mem.log" | awk '{printf "%d", $1 + 0.999}')
log "b4 estimated peak ${est_gib} GiB of ${total_gib} GiB"
if [ "$est_gib" -ge "$total_gib" ]; then
    log "STOP  b4 at batch 8 does not fit; nothing was trained"
    exit 1
fi
need=$(( est_gib + MARGIN_GIB )); [ "$need" -lt "$MIN_FREE_FLOOR" ] && need=$MIN_FREE_FLOOR
[ "$need" -gt "$total_gib" ] && need=$total_gib

# %% 1. cv3 s42
train_b4 cv3 42 "$need"
rc=$?
if [ $rc -ne 0 ] && grep -q "out of memory" "$QUEUE_DIR/train_b4_cv3_s42.log"; then
    log "STOP  train_b4_cv3_s42 ran out of memory; s43 would too"
    exit 1
fi
[ $rc -eq 0 ] && test_b4 cv3 42

peak_mib=$(cat "$QUEUE_DIR/train_b4_cv3_s42.peak")
if [ "$peak_mib" -gt 0 ]; then
    need=$(( (peak_mib + 1023) / 1024 + MARGIN_GIB ))
    [ "$need" -lt "$MIN_FREE_FLOOR" ] && need=$MIN_FREE_FLOOR
    [ "$need" -gt "$total_gib" ] && need=$total_gib
fi

# %% 2. cv3 s43
train_b4 cv3 43 "$need" && test_b4 cv3 43

# %% 3. per-run tables for the paired comparison b2 vs b4 (cv_runs.csv)
for enc in b2 b4; do
    python evaluate_cv.py --runs "*np_cv3_Unet_mit_${enc}_512_s*" --ckpt-type best \
        --out "results/cv/512_cv3_${enc}" >> "$QUEUE_DIR/evaluate_cv.log" 2>&1
    log "evaluate_cv $enc: exit $?, tables in results/cv/512_cv3_${enc}"
done

log "queue finished"
