#!/usr/bin/env bash
# Few-shot site adaption on cv3 (task 1, phase 1): fine-tune the received cv3 model on
# k labelled tiles per species of one held-out flight, then score it with test.py.
# Scoring and settings: configs/data/neophytes_split_cv3_adapt_512.yaml,
# configs/model/model_neophytes_mit_b2_adapt.yaml, adapt_report.py.
#
#   MODE=baseline   test.py on the received model (k = 0), whole cv3 test set, with the
#                   same evaluation settings as the adapted runs. Also times one full pass.
#   MODE=calibrate  one flight (CAL_FLIGHT), k=3, one seed, for each step count in
#                   CAL_STEPS: picks the number of fine-tuning steps (rest-of-flight F1).
#   MODE=grid       FLIGHTS x KS x SEEDS at MAX_STEPS. Every run is scored on its own site
#                   only (rest of flight + other flights of the site). Optional: FULL_SEED=1
#                   also scores seed 1 on the whole cv3 test set (other sites, ~1 h per run).
#
#   tmux new -s adapt
#   MODE=baseline bash scripts/queues/queue_2026-11-24_adapt_cv3.sh
#   MODE=calibrate bash scripts/queues/queue_2026-11-24_adapt_cv3.sh
#   MODE=grid MAX_STEPS=500 bash scripts/queues/queue_2026-11-24_adapt_cv3.sh
#
# Every job waits until the GPU has MIN_FREE_GIB free. Logs and exit codes go to
# lightning_logs/queue_2026-11-24_adapt/. A failed job does not stop the queue.

set -u

MODE=${MODE:?set MODE=baseline, calibrate or grid}
MIN_FREE_GIB=${MIN_FREE_GIB:-36}
MAX_STEPS=${MAX_STEPS:-500}
CAL_FLIGHT=${CAL_FLIGHT:-2025/Rivera_1_1}
CAL_STEPS=${CAL_STEPS:-"250 500 1000 2000"}
FLIGHTS=${FLIGHTS:-"2024/Basel_2_3 2025/Dornach_1_1 2025/Rivera_1_1 2024/Waltenschwil_1_1"}
KS=${KS:-"1 3 10"}
SEEDS=${SEEDS:-"1 2 3"}
FULL_SEED=${FULL_SEED:-0}   # 0 = never; a seed (e.g. 1) also scores that seed on the whole fold

# # the received cv3 model (512 crops, seed 42) and its run name
RECEIVED=20261013_002801.261406_np_cv3_Unet_mit_b2_512_s42
RECEIVED_CKPT=/zfs/ai4good/groups/neophytes/lightning_logs/$RECEIVED/checkpoints/best_epoch=29.ckpt

CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
QUEUE_DIR=lightning_logs/queue_2026-11-24_adapt

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/../.."
mkdir -p "$QUEUE_DIR"

# # test.py settings shared by baseline and adapted runs: per-flight confusion matrices,
# # no AUC / month / phenology breakdowns and no elevation rasters (the model is RGB only)
EVAL_ARGS=(eval_site=True eval_cv=False eval_month=False eval_auxiliary=False compute_auc=False
           data.load_auxiliary=False data.load_dsm=False data.load_dtm=False data.load_ndsm=False)

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

# site_test_dirs <year/flight>: the cv3 test folders (/train) of all flights of its site, as a Hydra list
site_test_dirs() {
    local year=${1%%/*} flight=${1##*/}
    local dirs; dirs=$(ls -d ../data/Neophytes/"$year"/"${flight%_*}"_*/train | sed "s/.*/'&'/" | paste -sd, -)
    echo "[$dirs]"
}

# adapt <year/flight> <k> <seed> <steps> <full 0|1>: fine-tune, then score on the site (and the fold)
adapt() {
    local year=${1%%/*} flight=${1##*/} k=$2 seed=$3 steps=$4 full=$5
    local exp; exp=$(date +%Y%m%d_%H%M%S.%6N)_np_cv3_adapt_${flight}_k${k}_steps${steps}_Unet_mit_b2_512_s${seed}
    run_job "train_${flight}_k${k}_steps${steps}_s${seed}" python train.py exp_name="$exp" logger=csv \
        data=neophytes_split_cv3_adapt_512 model=model_neophytes_mit_b2_adapt \
        data.adapt_year="$year" data.adapt_flight="$flight" data.few_shot_k="$k" \
        model.max_steps="$steps" pl_seed="$seed" init_ckpt="$RECEIVED_CKPT" || return 1
    run_job "test_site_${flight}_k${k}_steps${steps}_s${seed}" python test.py exp_name="$exp" ckpt_type=final \
        "${EVAL_ARGS[@]}" "data.image_dir.test=$(site_test_dirs "$1")"
    if [ "$full" = 1 ]; then
        run_job "test_fold_${flight}_k${k}_steps${steps}_s${seed}" python test.py exp_name="$exp" ckpt_type=final \
            "${EVAL_ARGS[@]}"
    fi
}

case "$MODE" in
    baseline)
        # # test.py reads checkpoints from lightning_logs/<exp_name>/: link the received one there
        mkdir -p "lightning_logs/$RECEIVED/checkpoints"
        ln -sf "$RECEIVED_CKPT" "lightning_logs/$RECEIVED/checkpoints/"
        run_job "test_received_cv3" python test.py exp_name="$RECEIVED" ckpt_type=best "${EVAL_ARGS[@]}"
        ;;
    calibrate)
        for steps in $CAL_STEPS; do adapt "$CAL_FLIGHT" 3 1 "$steps" 0; done
        ;;
    grid)
        for flight in $FLIGHTS; do for k in $KS; do for seed in $SEEDS; do
            adapt "$flight" "$k" "$seed" "$MAX_STEPS" $([ "$seed" = "$FULL_SEED" ] && echo 1 || echo 0)
        done; done; done
        ;;
    *) echo "unknown MODE=$MODE"; exit 1 ;;
esac

echo "$(date '+%F %T')  queue finished ($MODE)" | tee -a "$QUEUE_DIR/status.txt"
