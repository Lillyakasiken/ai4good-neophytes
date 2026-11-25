#!/usr/bin/env bash
# Queue of 25 Nov, task 2 at the cv3 held-out sites (adapt_cv3_novel_cover_tp05): for each
# species x flight of task 1's table, the base without the species is fine-tuned on the
# tiles of task 1's cover run of that flight that contain the species, and scored on the
# other flights of the site. Needs the task 1 cover runs (queue_2026-11-25_cover.sh).
#
#   PART=lrtest bash scripts/queues/queue_2026-11-25_novel_cover.sh
#       learning-rate check on two cells that never learned the new class overnight
#       (A. altissima at Basel_2_3, B. orientalis at Dornach_1_1): base lr everywhere,
#       head 10x, head + decoder 10x
#   PART=all HEAD_LR=0.0005 DECODER_LR=0.0005 bash scripts/queues/queue_2026-11-25_novel_cover.sh
#       the 9 cells; head + decoder at 10x the base lr, as in Alexandru's new-species
#       fine-tune (set beforehand, not tuned on the scored flights)
#
# A job whose log already holds the test results is skipped, so rerunning the script
# resumes the queue. Logs and status: lightning_logs/queue_2026-11-25_novel_cover/.

# # ~/.bashrc (WANDB_API_KEY) and conda use unset variables: source them before set -u
source ~/.bashrc
source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate /zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
set -u

PART=${PART:-lrtest}
HEAD_LR=${HEAD_LR:-null}
DECODER_LR=${DECODER_LR:-null}
K=${K:-10}
SEED=${SEED:-1}
MIN_FREE_GIB=${MIN_FREE_GIB:-36}
EXPLORER_API=${EXPLORER_API:-http://127.0.0.1:8010}
# # species x flight cells of the task 1 table (>= 10 tiles in the flight, >= 5 scored);
# # the two species that never learned the new class overnight first, as an early check
CELLS_ALL="A_altissima:2024/Basel_2_3 B_orientalis:2025/Dornach_1_1
B_davidii:2024/Basel_2_3 S_inaequidens:2024/Basel_2_3 B_davidii:2025/Rivera_1_1 R_japonica:2024/Waltenschwil_1_1
B_davidii:2025/Dornach_1_1 R_japonica:2025/Dornach_1_1 A_altissima:2025/Dornach_1_1"
CELLS_LRTEST="A_altissima:2024/Basel_2_3 B_orientalis:2025/Dornach_1_1"

cd "$(dirname "$0")/../.."
QUEUE_DIR=lightning_logs/queue_2026-11-25_novel_cover
mkdir -p "$QUEUE_DIR"

wait_for_gpu() {
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
        if [ "$free_mib" -ge $((MIN_FREE_GIB * 1024)) ]; then return; fi
        curl -s -m 30 -X POST "$EXPLORER_API/api/predict/release" > /dev/null 2>&1 && sleep 10 && continue
        echo "$(date '+%F %T')  GPU has $((free_mib / 1024)) GiB free, waiting for ${MIN_FREE_GIB} GiB ..."
        sleep 300
    done
}

run_job() {
    local name=$1; shift
    if grep -q "Test metric" "$QUEUE_DIR/$name.log" 2>/dev/null; then
        echo "$(date '+%F %T')  SKIP  $name (done)" | tee -a "$QUEUE_DIR/status.txt"
        return 0
    fi
    wait_for_gpu
    echo "$(date '+%F %T')  START $name: $*" | tee -a "$QUEUE_DIR/status.txt"
    "$@" < /dev/null > "$QUEUE_DIR/$name.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T')  END   $name: exit $rc" | tee -a "$QUEUE_DIR/status.txt"
    return $rc
}

# run_cell <species:year/flight> <head_lr> <decoder_lr>
run_cell() {
    local sp=${1%%:*} f=${1#*:}
    local flight=${f##*/} year=${f%%/*}
    local tiles
    tiles=$(ls -d lightning_logs/*_np_cv3_adapt_${flight}_k${K}cover_tp05_*_s${SEED} 2>/dev/null | tail -1)/train_tiles.txt
    if [ ! -f "$tiles" ]; then
        echo "$(date '+%F %T')  MISSING task 1 tiles for $flight ($tiles)" | tee -a "$QUEUE_DIR/status.txt"
        return 1
    fi
    run_job "novel_${sp}_${flight}_k${K}_hlr${2}_dlr${3}_s${SEED}" python train.py +experiment=adapt_cv3_novel_cover_tp05 \
        data.novel_species="$sp" data.adapt_year="$year" data.adapt_flight="$flight" data.few_shot_k="$K" \
        data.few_shot_tiles="$tiles" model.head_lr="$2" model.decoder_lr="$3" pl_seed="$SEED"
}

case "$PART" in
    lrtest)
        for cell in $CELLS_LRTEST; do
            run_cell "$cell" null null
            run_cell "$cell" 0.0005 null
            run_cell "$cell" 0.0005 0.0005
        done
        ;;
    all)
        for cell in $CELLS_ALL; do run_cell "$cell" "$HEAD_LR" "$DECODER_LR"; done
        ;;
    *) echo "unknown PART $PART" ;;
esac

echo "$(date '+%F %T')  queue finished ($PART)" | tee -a "$QUEUE_DIR/status.txt"
