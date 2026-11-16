#!/usr/bin/env bash
# Queue of Fri 2026-10-30: redo the fine-tuning arms of queue_2026-10-29_fewshot.sh.
# There, b2 fine-tuned on the 10 B. davidii tiles (lr 5e-5, 30 epochs) never predicted
# B. davidii, not even on its training crops (train_f1_class_1 = 0 in every epoch):
# the base run trained the B. davidii output channel to lose everywhere (mean logit
# -14 vs -7 for the known species on base A's head), and AdamW steps of ~lr cannot undo
# that in 1740 steps. The linear probes, with a fresh head and lr 1e-3, learnt it.
#
# 0. Wait until base C of the first queue is done (its ft_C steps are skipped there)
# 1. Screen on A, draw 1 (~4 min each, no test pass): which recipe learns the new class
#      r1  reset_head_classes=[1], lr 1e-4    fresh B. davidii channel, base LR
#      r3  reset_head_classes=[1], lr 3e-4
#      n3  no reset,               lr 3e-4    is the LR alone enough?
#    The recipe with the highest train_f1_class_1 in the last epoch goes on (stop if
#    none reaches 0.3). Picking on the training crops uses no test or validation data.
# 2. ft_A with that recipe, draws 2 and 3 (+ draw 1 from the screen), one test pass
# 3. ft_C (b2 + nDSM) with the same recipe, draws 1-3, one test pass
# 4. Summary: results/fewshot/cv3_bdavidii/summary.txt
#
# About 8 h after base C (estimated: the two test passes take ~3.2 h each).
#
#     tmux new -s queue_ft
#     bash scripts/queues/queue_2026-10-30_fewshot_ft.sh
#
# Logs go to lightning_logs/queue_2026-10-30_fewshot_ft/; a rerun skips finished steps.

set -u

CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
QUEUE_DIR=lightning_logs/queue_2026-10-30_fewshot_ft
FIRST_QUEUE=lightning_logs/queue_2026-10-29_fewshot
OUT=results/fewshot/cv3_bdavidii
FEWSHOT_DATA=neophytes_split_cv3_train_512_fewshot_bdavidii
FT_ARGS=(model.max_epochs=30 model.check_val_every_n_epoch=1000 test_ckpt=null logger=csv)
declare -A RECIPE=([r1]="reset_head_classes=1 model.lr=0.0001"
                   [r3]="reset_head_classes=1 model.lr=0.0003"
                   [n3]="model.lr=0.0003")
MIN_TRAIN_F1=0.3
TRAIN_GIB=36   # b2 512 fine-tuning and the 3-model test passes both peak at ~33 GiB

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/../.."
mkdir -p "$QUEUE_DIR" "$OUT"

[ -f "$QUEUE_DIR/names.env" ] || echo "TS=$(date +%Y%m%d_%H%M%S).000000" > "$QUEUE_DIR/names.env"
source "$QUEUE_DIR/names.env"
TS1=$(sed -n 's/^TS=//p' "$FIRST_QUEUE/names.env")  # run names of the base runs
declare -A BASE_EXP=([A]=${TS1}_np_cv3_ho1_Unet_mit_b2_512_s42 [C]=${TS1}_np_cv3_ho1_Unet_mit_b2_nDSM_512_s42)
declare -A BASE_MODEL=([A]=model_neophytes_mit_b2 [C]=model_neophytes_mit_b2_ndsm)
declare -A BASE_NAME=([A]=Unet_mit_b2 [C]=Unet_mit_b2_nDSM)

log() { echo "$(date '+%F %T')  $*" | tee -a "$QUEUE_DIR/status.txt"; }
is_done() { [ -f "$QUEUE_DIR/$1.done" ]; }
mark_done() { touch "$QUEUE_DIR/$1.done"; }
stop() { log "STOP  $*"; exit 1; }

# wait_for_gpu <GiB>
wait_for_gpu() {
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
        if [ "$free_mib" -ge $(($1 * 1024)) ]; then return; fi
        echo "$(date '+%F %T')  GPU has $((free_mib / 1024)) GiB free, waiting for $1 GiB ..."
        sleep 300
    done
}

# run_job <name> <min free GiB> <command...>: as in queue_2026-10-29_fewshot.sh
run_job() {
    local name=$1 need=$2; shift 2
    if is_done "$name"; then
        log "SKIP  $name (done)"
        return 0
    fi
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
    [ $rc -eq 0 ] && mark_done "$name"
    return $rc
}

best_ckpt() { ls lightning_logs/"$1"/checkpoints/best_*.ckpt 2>/dev/null | sort | tail -n 1; }
ft_exp() { echo "${TS}_np_cv3_fs1_${BASE_NAME[$1]}_512_s$2_from$1_$3"; }  # <A|C> <seed> <recipe>

# ft <A|C> <seed> <recipe>: one fine-tuning run (final weights, no test pass)
ft() {
    local init; init=$(best_ckpt ${BASE_EXP[$1]})
    run_job "ft_$1_s$2_$3" "$TRAIN_GIB" python train.py data=$FEWSHOT_DATA model=${BASE_MODEL[$1]} \
        "init_ckpt='$init'" data.few_shot_seed=$2 pl_seed=$2 ${RECIPE[$3]} "${FT_ARGS[@]}" \
        exp_name=$(ft_exp "$1" "$2" "$3")
}

# train_f1_new <exp>: train_f1_class_1 of the last epoch
train_f1_new() {
    python - "$1" <<'EOF'
import glob, sys
import pandas as pd
d = pd.read_csv(sorted(glob.glob(f'lightning_logs/{sys.argv[1]}/version_*/metrics.csv'))[-1])
print(f"{d['train_f1_class_1'].dropna().iloc[-1]:.3f}")
EOF
}

# finetune_all <A|C> <recipe>: draws 1-3, then the 3 models in one test pass
finetune_all() {
    local tag=$1 recipe=$2 ckpts=() names=()
    [ -n "$(best_ckpt ${BASE_EXP[$tag]})" ] || { log "SKIP  ft_$tag: no checkpoint of ${BASE_EXP[$tag]}"; return; }
    for s in 1 2 3; do
        ft "$tag" "$s" "$recipe" && { ckpts+=("lightning_logs/$(ft_exp "$tag" "$s" "$recipe")/checkpoints/final.ckpt"); names+=("s$s"); }
    done
    [ ${#ckpts[@]} -gt 0 ] || return
    run_job "eval_ft_${tag}_$recipe" "$TRAIN_GIB" python fewshot_probe.py eval --ckpts "${ckpts[@]}" \
        --names "${names[@]}" --out $OUT/ft_${tag}_$recipe
}

log "queue start (TS=$TS, first queue TS=$TS1)"

# %% 0. wait for base C of the first queue (done, or that queue stopped)
until [ -f "$FIRST_QUEUE/base_C.done" ] || grep -q "STOP" "$FIRST_QUEUE/status.txt"; do sleep 300; done
[ -f "$FIRST_QUEUE/base_C.done" ] || log "NOTE  base C did not finish (see $FIRST_QUEUE/status.txt): A only"
[ -n "$(best_ckpt ${BASE_EXP[A]})" ] || stop "no checkpoint of ${BASE_EXP[A]}"

# %% 1. screen the recipes on A, draw 1
best=""; best_f1=-1
for r in r1 r3 n3; do
    ft A 1 "$r" || { log "NOTE  ft_A_s1_$r failed"; continue; }
    f1=$(train_f1_new "$(ft_exp A 1 "$r")")
    log "screen $r (${RECIPE[$r]}): train_f1_class_1 = $f1"
    if python -c "import sys; sys.exit(not $f1 > $best_f1)"; then best=$r; best_f1=$f1; fi
done
[ -n "$best" ] && python -c "import sys; sys.exit(not $best_f1 >= $MIN_TRAIN_F1)" \
    || stop "no recipe learns B. davidii on its training crops (best train_f1_class_1 $best_f1)"
log "recipe: $best (${RECIPE[$best]}), train_f1_class_1 $best_f1"

# %% 2.-3. all draws with that recipe
finetune_all A "$best"
finetune_all C "$best"

# %% 4. summary
python fewshot_probe.py summary --root $OUT 2>&1 | tee -a "$QUEUE_DIR/status.txt"
log "queue finished"
