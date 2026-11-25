#!/usr/bin/env bash
# Queue of 25 Nov, task 1 "cover" runs: the generic fine-tune (adapt_cv3_cover_tp05) at
# the four held-out cv3 sites. Exactly k=10 tiles per species from the whole adapted
# flight (tiles may hold several species); each run ends with the train.py test pass on
# the OTHER flights of its site, whose per-class F1 / precision / recall the report
# (reports/scripts/cover_report.py in the workspace) reads from the job log.
#
#   tmux new -s cover
#   bash scripts/queues/queue_2026-11-25_cover.sh
#
# A job whose log already holds the test results is skipped, so rerunning the script
# resumes the queue. A failed job does not stop the queue. Logs and status:
# lightning_logs/queue_2026-11-25_cover/.

# # ~/.bashrc (WANDB_API_KEY) and conda use unset variables: source them before set -u
source ~/.bashrc
source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate /zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
set -u

K=${K:-10}
SEED=${SEED:-1}
MIN_FREE_GIB=${MIN_FREE_GIB:-36}
FLIGHTS=${FLIGHTS:-"2024/Basel_2_3 2025/Rivera_1_1 2024/Waltenschwil_1_1 2025/Dornach_1_1"}
# # Max's tile explorer (explorer branch) on this node: its prediction worker keeps the
# # model on the GPU; asking it to unload frees the memory (saved predictions stay)
EXPLORER_API=${EXPLORER_API:-http://127.0.0.1:8010}

cd "$(dirname "$0")/../.."
QUEUE_DIR=lightning_logs/queue_2026-11-25_cover
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

# run_job <name> <command...>: skips a finished job, waits for the GPU, logs output and exit code
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

for f in $FLIGHTS; do
    run_job "cover_${f##*/}_k${K}_s${SEED}" python train.py +experiment=adapt_cv3_cover_tp05 \
        data.adapt_year="${f%%/*}" data.adapt_flight="${f##*/}" data.few_shot_k="$K" pl_seed="$SEED"
done

echo "$(date '+%F %T')  queue finished" | tee -a "$QUEUE_DIR/status.txt"
