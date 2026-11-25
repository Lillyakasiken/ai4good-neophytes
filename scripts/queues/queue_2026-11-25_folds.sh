#!/usr/bin/env bash
# Queue of 25 Nov: the received model (course recipe: U-Net MiT-b2, Focal, inverse-image
# weighted sampling, 512 crops, 30 epochs, val every 3) for the folds that have none yet,
# cv2, cv4, cv5, seed 42, each followed by its test pass (per-site/flight confusion
# matrices in metrics_semseg_site.json: the baselines of tasks 1 and 2 at these folds).
# Same overrides as the received cv1/cv3 runs of 13 Oct (zhixuan_li/.../lightning_logs);
# the composed config matches theirs apart from our new options, which are all off.
#
#   tmux new -s folds
#   AFTER=lightning_logs/queue_2026-11-25_novel_cover/status.txt bash scripts/queues/queue_2026-11-25_folds.sh
#
# AFTER: a queue status file to wait for ("queue finished" line) before the first job, so
# two queues never share the GPU. A finished training (test table in its log) or test
# (metrics_semseg_site.json in the run) is skipped, so rerunning resumes the queue.
# Logs and status: lightning_logs/queue_2026-11-25_folds/. ~8-9 h per fold.

# # ~/.bashrc (WANDB_API_KEY) and conda use unset variables: source them before set -u
source ~/.bashrc
source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate /zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
set -u

FOLDS=${FOLDS:-"cv2 cv4 cv5"}
SEED=${SEED:-42}
AFTER=${AFTER:-}
MIN_FREE_GIB=${MIN_FREE_GIB:-36}
EXPLORER_API=${EXPLORER_API:-http://127.0.0.1:8010}

cd "$(dirname "$0")/../.."
QUEUE_DIR=lightning_logs/queue_2026-11-25_folds
mkdir -p "$QUEUE_DIR"

if [ -n "$AFTER" ]; then
    echo "$(date '+%F %T')  waiting for $AFTER to finish" | tee -a "$QUEUE_DIR/status.txt"
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

log_status() { echo "$(date '+%F %T')  $*" | tee -a "$QUEUE_DIR/status.txt"; }

for cv in $FOLDS; do
    name="train_${cv}_s${SEED}"
    if grep -q "Test metric" "$QUEUE_DIR/$name.log" 2>/dev/null; then
        log_status "SKIP  $name (done)"
    else
        wait_for_gpu
        log_status "START $name"
        python train.py data=neophytes_split_${cv}_train_512 model.max_epochs=30 \
            model.check_val_every_n_epoch=3 pl_seed="$SEED" < /dev/null > "$QUEUE_DIR/$name.log" 2>&1
        log_status "END   $name: exit $?"
    fi

    run=$(ls -td lightning_logs/*_np_${cv}_Unet_mit_b2_512_s${SEED} 2>/dev/null | head -1)
    if [ -z "$run" ]; then log_status "NO RUN DIR for $cv"; continue; fi
    name="test_${cv}_s${SEED}"
    if ls "$run"/test_*/quantitative/metrics_semseg_site.json > /dev/null 2>&1; then
        log_status "SKIP  $name (done)"
    else
        wait_for_gpu
        log_status "START $name: ${run##*/}"
        python test.py exp_name="${run##*/}" data=neophytes_split_${cv}_test ckpt_type=best \
            compute_auc=False < /dev/null > "$QUEUE_DIR/$name.log" 2>&1
        log_status "END   $name: exit $?"
    fi
done

log_status "queue finished"
