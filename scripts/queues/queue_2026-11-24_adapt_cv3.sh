#!/usr/bin/env bash
# Few-shot site adaption on cv3 (task 1, phase 1): fine-tune the received cv3 model on
# k labelled tiles per species of one held-out flight, then score it with test.py.
# Settings: configs/experiment/adapt_cv3_finetune.yaml (training) and
# configs/experiment/adapt_eval.yaml (scoring); results: adapt_report.py.
#
#   MODE=quick      one run (QUICK_FLIGHT, QUICK_K, seed 1, MAX_STEPS) and the received model,
#                   both scored on the flight's site (rest of the flight + the site's other
#                   flights), then the report.
#   MODE=curve      one run (QUICK_FLIGHT, QUICK_K, seed 1, MAX_STEPS), training only: the
#                   train and rest-of-flight F1 curves go to W&B (validation every 100 steps).
#   MODE=baseline   the received model (k = 0) scored on the whole cv3 test set.
#   MODE=calibrate  one flight (QUICK_FLIGHT), k=3, seed 1, for each step count in
#                   CAL_STEPS: picks the number of fine-tuning steps (rest-of-flight F1).
#   MODE=grid       FLIGHTS x KS x SEEDS at MAX_STEPS, each scored on its own site
#                   (rest of the flight + the other flights of the site).
#
#   tmux new -s adapt
#   MODE=quick bash scripts/queues/queue_2026-11-24_adapt_cv3.sh
#   MODE=grid MAX_STEPS=500 bash scripts/queues/queue_2026-11-24_adapt_cv3.sh
#
# Every job waits until the GPU has MIN_FREE_GIB free. Logs and exit codes go to
# lightning_logs/queue_2026-11-24_adapt/. A failed job does not stop the queue.

set -u

MODE=${MODE:?set MODE=curve, quick, baseline, calibrate or grid}
MIN_FREE_GIB=${MIN_FREE_GIB:-36}
MAX_STEPS=${MAX_STEPS:-500}
QUICK_FLIGHT=${QUICK_FLIGHT:-2025/Rivera_1_1}
QUICK_K=${QUICK_K:-10}
CAL_STEPS=${CAL_STEPS:-"250 500 1000 2000"}
FLIGHTS=${FLIGHTS:-"2024/Basel_2_3 2025/Dornach_1_1 2025/Rivera_1_1 2024/Waltenschwil_1_1"}
KS=${KS:-"1 3 10"}
SEEDS=${SEEDS:-"1 2 3"}
EXPERIMENT=${EXPERIMENT:-adapt_cv3_finetune}   # training config in configs/experiment/

# # the received cv3 model (512 crops, seed 42), as named in adapt_cv3_finetune.yaml
RECEIVED=20261013_002801.261406_np_cv3_Unet_mit_b2_512_s42
RECEIVED_CKPT=/zfs/ai4good/groups/neophytes/lightning_logs/$RECEIVED/checkpoints/best_epoch=29.ckpt

CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
QUEUE_DIR=lightning_logs/queue_2026-11-24_adapt

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

# test_dirs <year/flight> <flight|site>: the cv3 test folders (/train) of the flight or of
# all flights of its site, as a Hydra list override of the test set
test_dirs() {
    local year=${1%%/*} flight=${1##*/}
    local pattern=$flight; [ "$2" = site ] && pattern="${flight%_*}_*"
    local dirs; dirs=$(ls -d ../data/Neophytes/"$year"/$pattern/train | sed "s/.*/'&'/" | paste -sd, -)
    echo "data.image_dir.test=[$dirs]"
}

# test_received <year/flight> <flight|site|fold>: the received model on the same tiles
test_received() {
    # # test.py reads checkpoints from lightning_logs/<exp_name>/: link the received one there
    mkdir -p "lightning_logs/$RECEIVED/checkpoints"
    ln -sf "$RECEIVED_CKPT" "lightning_logs/$RECEIVED/checkpoints/"
    local scope=(); [ "$2" != fold ] && scope=("$(test_dirs "$1" "$2")")
    run_job "test_received_${1##*/}_$2" python test.py +experiment=adapt_eval exp_name="$RECEIVED" \
        ckpt_type=best "${scope[@]}"
}

# adapt <year/flight> <k> <seed> <steps> <flight|site>: fine-tune, then score
adapt() {
    local year=${1%%/*} flight=${1##*/} k=$2 seed=$3 steps=$4 scope=$5
    local name=${flight}_k${k}_${EXPERIMENT}_steps${steps}_s${seed}
    run_job "train_$name" python train.py +experiment="$EXPERIMENT" \
        data.adapt_year="$year" data.adapt_flight="$flight" data.few_shot_k="$k" \
        model.max_steps="$steps" pl_seed="$seed" || return 1
    local run_dir; run_dir=$(ls -d lightning_logs/*_np_cv3_adapt_${flight}_k${k}_*steps${steps}_Unet_mit_b2_512_s${seed} | sort | tail -n 1)
    run_job "test_$name" python test.py +experiment=adapt_eval exp_name="$(basename "$run_dir")" \
        "$(test_dirs "$1" "$scope")"
}

case "$MODE" in
    curve)
        run_job "curve_${QUICK_FLIGHT##*/}_k${QUICK_K}_${EXPERIMENT}_steps${MAX_STEPS}_s1" python train.py \
            +experiment="$EXPERIMENT" data.adapt_year="${QUICK_FLIGHT%%/*}" data.adapt_flight="${QUICK_FLIGHT##*/}" \
            data.few_shot_k="$QUICK_K" model.max_steps="$MAX_STEPS" pl_seed=1
        ;;
    quick)
        adapt "$QUICK_FLIGHT" "$QUICK_K" 1 "$MAX_STEPS" site
        test_received "$QUICK_FLIGHT" site
        python adapt_report.py --out "$QUEUE_DIR/report_quick" \
            --runs "*np_cv3_adapt_${QUICK_FLIGHT##*/}_k${QUICK_K}_*steps${MAX_STEPS}_*_s1" \
            --baseline "$(ls -d lightning_logs/$RECEIVED/test_*/quantitative | sort | tail -n 1)/metrics_semseg_site.json"
        ;;
    baseline)
        test_received "$QUICK_FLIGHT" fold
        ;;
    calibrate)
        for steps in $CAL_STEPS; do adapt "$QUICK_FLIGHT" 3 1 "$steps" flight; done
        ;;
    grid)
        for flight in $FLIGHTS; do for k in $KS; do for seed in $SEEDS; do
            adapt "$flight" "$k" "$seed" "$MAX_STEPS" site
        done; done; done
        ;;
    *) echo "unknown MODE=$MODE"; exit 1 ;;
esac

echo "$(date '+%F %T')  queue finished ($MODE)" | tee -a "$QUEUE_DIR/status.txt"
