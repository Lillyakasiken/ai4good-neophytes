#!/usr/bin/env bash
# Queue of Tue 2026-10-13 (see docs/changes_2026-10-12.md, section 6): the remaining
# Phase 0 development runs at 512 crops, so that cv3 and cv1 each have seeds 42 and 43.
#
# 0. test.py on the running 512 cv3 s42 run                                 ~1.25 h
# 1. 512 crops, cv3, seed 43  (pairs with s42 -> seed noise), then test.py  ~7.5 + 1.25 h
# 2. 512 crops, cv1, seed 42, then test.py                                  ~7.5 + 1.25 h
# 3. 512 crops, cv1, seed 43, then test.py                                  ~7.5 + 1.25 h
#
# Same settings as train_512_cv3_s42 of the Oct 12 queue (30 epochs, validation every
# 3rd epoch); only fold and seed change. The quick test pass at the end of train.py
# only logs to the logger. evaluate_cv.py reads the metrics_semseg.json of test.py, so
# every run also gets a test.py pass (best checkpoint, compute_auc=False as in the
# Oct 12 part 2 queue). test.py picks the test split of the fold from the run name.
#
# The queue first waits until scripts/queue_2026-10-12_cv3_part2.sh has exited (its
# last job is train_512_cv3_s42), so it can be started now in a second tmux window:
#
#     tmux new -s queue2
#     bash scripts/queue_2026-10-13_512.sh
#
# Every job then waits until the GPU has MIN_FREE_GIB free. The 512 run holds about
# 34 GiB, so the default is 36 instead of the 40 of the 1024 queues. Logs and exit
# codes go to lightning_logs/queue_2026-10-13/. A failed job does not stop the queue.

set -u

MIN_FREE_GIB=${MIN_FREE_GIB:-36}
CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
PREV_QUEUE=queue_2026-10-12_cv3_part2.sh
QUEUE_DIR=lightning_logs/queue_2026-10-13

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/.."
mkdir -p "$QUEUE_DIR"

wait_for_prev_queue() {
    while pgrep -f "^bash .*$PREV_QUEUE" > /dev/null; do
        echo "$(date '+%F %T')  $PREV_QUEUE still running, waiting ..."
        sleep 300
    done
    echo "$(date '+%F %T')  $PREV_QUEUE has exited; last lines of its status:" | tee -a "$QUEUE_DIR/status.txt"
    tail -n 2 lightning_logs/queue_2026-10-12/status.txt | tee -a "$QUEUE_DIR/status.txt"
}

wait_for_gpu() {
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
        if [ "$free_mib" -ge $((MIN_FREE_GIB * 1024)) ]; then return; fi
        echo "$(date '+%F %T')  GPU has $((free_mib / 1024)) GiB free, waiting for ${MIN_FREE_GIB} GiB ..."
        sleep 300
    done
}

# run_job <name> <command...>: waits for the GPU, logs output and exit code
run_job() {
    local name=$1; shift
    wait_for_gpu
    echo "$(date '+%F %T')  START $name: $*" | tee -a "$QUEUE_DIR/status.txt"
    "$@" > "$QUEUE_DIR/$name.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T')  END   $name: exit $rc" | tee -a "$QUEUE_DIR/status.txt"
    return $rc
}

# newest run directory of a 512 run: run_dir_512 <fold> <seed>
run_dir_512() {
    ls -d lightning_logs/*_np_$1_Unet_mit_b2_512_s$2 2>/dev/null | sort | tail -n 1
}

# test_512 <fold> <seed>: test.py on the best checkpoint of the newest matching run
test_512() {
    local run_dir; run_dir=$(run_dir_512 "$1" "$2")
    if [ -z "$run_dir" ]; then
        echo "$(date '+%F %T')  SKIP  test_512_$1_s$2: no run directory" | tee -a "$QUEUE_DIR/status.txt"
        return 1
    fi
    run_job "test_512_$1_s$2" python test.py exp_name="$(basename "$run_dir")" ckpt_type=best compute_auc=False
}

# train_512 <fold> <seed>: the 512 development setting of train_512_cv3_s42, then test.py
train_512() {
    run_job "train_512_$1_s$2" python train.py data=neophytes_split_$1_train_512 \
        model.max_epochs=30 model.check_val_every_n_epoch=3 pl_seed=$2 \
        && test_512 "$1" "$2"
}

wait_for_prev_queue

# %% 0. test.py on the 512 cv3 s42 run of the Oct 12 queue
test_512 cv3 42

# %% 1. cv3 s43: second seed of cv3
train_512 cv3 43

# %% 2.-3. cv1 s42 and s43: second development fold
train_512 cv1 42
train_512 cv1 43

echo "$(date '+%F %T')  queue finished" | tee -a "$QUEUE_DIR/status.txt"
