#!/usr/bin/env bash
# Overnight queue of Mon 2026-10-12 (see docs/changes_2026-10-12.md). Jobs run one
# after the other; start it once in tmux from the repo root:
#
#     tmux new -s queue
#     bash scripts/queue_2026-10-12_cv3.sh
#
# 1. smoke test of today's code changes on the debug split (~10 min, CSV logger):
#    one epoch, then a resume to two epochs. If it fails, the queue stops here,
#    before the GPU is spent on a broken pipeline.
# 2. test.py on both checkpoints of the 1024 baseline (baseline_run1 in W&B), ~1-1.5 h each
# 3. 512-crop baseline on cv3, seed 42, 30 epochs, validation every 3rd epoch (~10-15 h, not measured)
#
# Every job waits until the GPU has MIN_FREE_GIB free (the 1024 baseline OOMed at step 1
# when another process held 4.6 GiB). Logs and exit codes go to lightning_logs/queue_2026-10-12/.

set -u

MIN_FREE_GIB=${MIN_FREE_GIB:-40}
CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
BASELINE_EXP=20261008_194342.853965_np_cv3_Unet_mit_b2_1024_s42
BASELINE_CKPTS=/zfs/ai4good/groups/neophytes/zhixuan_li/checkpoints
SMOKE_EXP=smoke_20261012_np_debug_resnet18
QUEUE_DIR=lightning_logs/queue_2026-10-12

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/.."
mkdir -p "$QUEUE_DIR"

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

# %% 1. smoke test: train 1 epoch, then resume the same run to 2 epochs
rm -rf "lightning_logs/$SMOKE_EXP"
SMOKE="python train.py data=neophytes_split_debug_1024 model=model_debug logger=csv exp_name=$SMOKE_EXP"
if ! run_job smoke_train $SMOKE model.max_epochs=1 \
   || ! run_job smoke_resume $SMOKE model.max_epochs=2; then
    echo "Smoke test failed, queue stopped. See $QUEUE_DIR/smoke_*.log" | tee -a "$QUEUE_DIR/status.txt"
    exit 1
fi
grep -q "Resuming training from" "$QUEUE_DIR/smoke_resume.log" \
    && echo "smoke: resume OK" | tee -a "$QUEUE_DIR/status.txt" \
    || echo "smoke: WARNING, resume message missing" | tee -a "$QUEUE_DIR/status.txt"

# %% 2. test.py on both checkpoints of the 1024 baseline
mkdir -p "lightning_logs/$BASELINE_EXP/checkpoints"
for ckpt in "best_epoch=38.ckpt" "last.ckpt"; do
    ln -sfn "$BASELINE_CKPTS/$ckpt" "lightning_logs/$BASELINE_EXP/checkpoints/$ckpt"
done
run_job test_baseline_best python test.py exp_name=$BASELINE_EXP ckpt_type=best
run_job test_baseline_last python test.py exp_name=$BASELINE_EXP ckpt_type=last

# %% 3. 512-crop baseline, cv3, seed 42 (train.py runs a test pass with the best checkpoint at the end)
run_job train_512_cv3_s42 python train.py data=neophytes_split_cv3_train_512 \
    model.max_epochs=30 model.check_val_every_n_epoch=3 pl_seed=42

echo "$(date '+%F %T')  queue finished" | tee -a "$QUEUE_DIR/status.txt"
