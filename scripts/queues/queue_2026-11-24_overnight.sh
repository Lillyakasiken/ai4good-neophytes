#!/usr/bin/env bash
# Overnight queue, 24 Nov. One training run per job; each run ends with the train.py
# test pass on the rest of its flight (the flight's /train tiles), whose per-class
# F1 / precision / recall the report reads from the job log.
#
#   1. Task 1, LoRA: adapt_cv3_lora_nf (k=10 non-flowering-first tiles per species,
#      LoRA rank 8 in the encoder + head) at the four held-out cv3 flights, seed 1.
#   2. Task 2, species adaption: adapt_novel_finetune_tp05 for every (species, flight)
#      that scripts/novel_flights.py lists, k=10 first, then k=5, seed 1.
#
#   tmux new -s overnight
#   bash scripts/queues/queue_2026-11-24_overnight.sh
#
# A job whose log already holds the test results is skipped, so rerunning the script
# resumes the queue. A failed job does not stop the queue. Logs and status:
# lightning_logs/queue_2026-11-24_overnight/.

# # ~/.bashrc (WANDB_API_KEY) and conda use unset variables: source them before set -u
source ~/.bashrc
source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate /zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
set -u

PARTS=${PARTS:-"lora novel"}
NOVEL_KS=${NOVEL_KS:-"10 5"}
SEED=${SEED:-1}
MIN_FREE_GIB=${MIN_FREE_GIB:-36}
LORA_FLIGHTS=${LORA_FLIGHTS:-"2024/Basel_2_3 2025/Rivera_1_1 2024/Waltenschwil_1_1 2025/Dornach_1_1"}
# # Max's tile explorer (explorer branch) on this node: its prediction worker keeps the
# # model on the GPU; asking it to unload frees the memory (saved predictions stay)
EXPLORER_API=${EXPLORER_API:-http://127.0.0.1:8010}

cd "$(dirname "$0")/../.."
QUEUE_DIR=lightning_logs/queue_2026-11-24_overnight
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

for part in $PARTS; do
    case "$part" in
        lora)
            for f in $LORA_FLIGHTS; do
                run_job "lora_${f##*/}_k10_s${SEED}" python train.py +experiment=adapt_cv3_lora_nf \
                    data.adapt_year="${f%%/*}" data.adapt_flight="${f##*/}" data.few_shot_k=10 pl_seed="$SEED"
            done
            ;;
        novel)
            for k in $NOVEL_KS; do
                python scripts/novel_flights.py "$k" > "$QUEUE_DIR/novel_flights_k$k.txt"
                while read -r sp f _; do
                    run_job "novel_${sp}_${f##*/}_k${k}_s${SEED}" python train.py +experiment=adapt_novel_finetune_tp05 \
                        data.novel_species="$sp" data.adapt_year="${f%%/*}" data.adapt_flight="${f##*/}" \
                        data.few_shot_k="$k" pl_seed="$SEED"
                done < "$QUEUE_DIR/novel_flights_k$k.txt"
            done
            ;;
        *) echo "unknown part $part" ;;
    esac
done

echo "$(date '+%F %T')  queue finished" | tee -a "$QUEUE_DIR/status.txt"
