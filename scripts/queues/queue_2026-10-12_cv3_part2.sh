#!/usr/bin/env bash
# Continuation of scripts/queues/queue_2026-10-12_cv3.sh after its smoke test had passed.
# Jobs 2 and 3 of that queue, with one change: test.py runs with compute_auc=False.
#
# Why: with AUROC on, test.py needed 4.6 s per batch (~6 h per checkpoint), because
# it updates four 200-threshold AUROC metrics per batch (global + per fold, multiclass +
# binary) on 2 x 2048^2 px. The failure analysis needs the confusion matrices and the
# per-class / per-site / per-month F1, not the AUROC (RQ4); run that later on an idle GPU.
#
#     tmux send-keys -t queue "bash scripts/queues/queue_2026-10-12_cv3_part2.sh" Enter

set -u

MIN_FREE_GIB=${MIN_FREE_GIB:-40}
CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
BASELINE_EXP=20261008_194342.853965_np_cv3_Unet_mit_b2_1024_s42
BASELINE_CKPTS=/zfs/ai4good/groups/neophytes/zhixuan_li/checkpoints
QUEUE_DIR=lightning_logs/queue_2026-10-12

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/../.."
mkdir -p "$QUEUE_DIR"

wait_for_gpu() {
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
        if [ "$free_mib" -ge $((MIN_FREE_GIB * 1024)) ]; then return; fi
        echo "$(date '+%F %T')  GPU has $((free_mib / 1024)) GiB free, waiting for ${MIN_FREE_GIB} GiB ..."
        sleep 300
    done
}

run_job() {
    local name=$1; shift
    wait_for_gpu
    echo "$(date '+%F %T')  START $name: $*" | tee -a "$QUEUE_DIR/status.txt"
    "$@" > "$QUEUE_DIR/$name.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T')  END   $name: exit $rc" | tee -a "$QUEUE_DIR/status.txt"
    return $rc
}

# %% 2. test.py on both checkpoints of the 1024 baseline, without AUROC
mkdir -p "lightning_logs/$BASELINE_EXP/checkpoints"
for ckpt in "best_epoch=38.ckpt" "last.ckpt"; do
    ln -sfn "$BASELINE_CKPTS/$ckpt" "lightning_logs/$BASELINE_EXP/checkpoints/$ckpt"
done
run_job test_baseline_best python test.py exp_name=$BASELINE_EXP ckpt_type=best compute_auc=False
run_job test_baseline_last python test.py exp_name=$BASELINE_EXP ckpt_type=last compute_auc=False

# %% 3. 512-crop baseline, cv3, seed 42 (train.py runs a test pass with the best checkpoint at the end)
run_job train_512_cv3_s42 python train.py data=neophytes_split_cv3_train_512 \
    model.max_epochs=30 model.check_val_every_n_epoch=3 pl_seed=42

echo "$(date '+%F %T')  queue finished" | tee -a "$QUEUE_DIR/status.txt"
