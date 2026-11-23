#!/usr/bin/env bash
# Queue of Thu 2026-10-29 (docs/plantclef2024_fewshot_en.md, section 1.3):
# the few-shot benchmark on one held-out species. B. davidii (cv3: the most
# test pixels and test sites of all species, full-data b2 F1 0.84) is background in the
# base training of the five other species; then each method gets k = 10 labelled
# B. davidii tiles (+ a replay of the known species), three draws (few_shot_seed 1/2/3,
# the same tiles for every method), and is scored on the cv3 test sites.
#
# 0. Checks: weights, configs, dataset stats of the held-out / few-shot subsets (CPU)
# 1. Smoke tests: fine-tuning from a checkpoint, eval, probes, nDSM        ~20-30 min
# 2. A    b2 base training (5 species) + train.py's test (5 species)    ~7.6 h + 1.2 h
# 3. Linear probes on frozen features, 3 draws each, one test pass each  ~1 h each
#      probe_pc24    PlantCLEF 2024 ViT-B/14            (8.3.6: the frozen plant features)
#      probe_dinov2  DINOv2 ViT-B/14                    (control: plant photos vs generic SSL)
#      probe_mit_b2  MiT-b2, ImageNet                   (control: same probe, b2 features)
#      probe_A       encoder of A                       (features learnt on the other species)
# 4. ft_A   A fine-tuned on the 10 tiles (lr 5e-5, 30 epochs), 3 draws + one test pass
# 5. C    b2 + nDSM (stem) base training, same held-out setting          ~8 h + 1.2 h
#         (with A also the paired RQ2 signal on the five known species, section 9 step 3)
# 6. ft_C   C fine-tuned like ft_A: does height help with a new woody species?
# 7. Summary table: results/fewshot/cv3_bdavidii/summary.txt
#
# About 26 h in total (estimated, not measured). After step 4 (~13 h) section 9 step 4
# has its answer (b2 fine-tuning vs the frozen-feature probes); steps 5-6 add nDSM.
# Every model is scored by fewshot_probe.py (bf16, whole tiles; the ViT probes on 512
# windows): per-class F1 at the argmax and the threshold-free AP, which separates
# "features cannot find it" from "the threshold is off" (section 8.2).
#
# Not tuned: fine-tuning lr/epochs, probe lr. Neither the fine-tuning nor the probes
# validate (a new species has no validation labels); they use the final weights.
#
#     tmux new -s queue_fewshot
#     bash scripts/queues/queue_2026-10-29_fewshot.sh
#
# Logs go to lightning_logs/queue_2026-10-29_fewshot/ (status.txt for the overview).
# Rerunning the script skips the finished steps (*.done) and resumes an interrupted
# base training from its last.ckpt.

set -u

CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
QUEUE_DIR=lightning_logs/queue_2026-10-29_fewshot
OUT=results/fewshot/cv3_bdavidii
HOLDOUT_DATA=neophytes_split_cv3_train_512_holdout_bdavidii
FEWSHOT_DATA=neophytes_split_cv3_train_512_fewshot_bdavidii
PC24_WEIGHTS=/zfs/ai4good/groups/neophytes/zhixuan_li/pretrained/plantclef2024/vitd2pc24all_backbone_ema.pth
# # any trained cv3 512 b2 run, only for the smoke tests
SMOKE_CKPT="lightning_logs/20261013_002801.261406_np_cv3_Unet_mit_b2_512_s42/checkpoints/best_epoch=29.ckpt"
SEEDS="1 2 3"
FT_ARGS=(model.lr=0.00005 model.max_epochs=30 model.check_val_every_n_epoch=1000 test_ckpt=null logger=csv)
TRAIN_GIB=36   # b2 512 training (34-40 GiB peak), as the 512 queues
INFER_GIB=12   # probes and test passes

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/../.."
mkdir -p "$QUEUE_DIR" "$OUT"

# # run names are fixed on the first start, so a restart resumes the same runs
if [ ! -f "$QUEUE_DIR/names.env" ]; then
    echo "TS=$(date +%Y%m%d_%H%M%S).000000" > "$QUEUE_DIR/names.env"
fi
source "$QUEUE_DIR/names.env"
A_EXP=${TS}_np_cv3_ho1_Unet_mit_b2_512_s42
C_EXP=${TS}_np_cv3_ho1_Unet_mit_b2_nDSM_512_s42
declare -A BASE_EXP=([A]=$A_EXP [C]=$C_EXP)
declare -A BASE_MODEL=([A]=model_neophytes_mit_b2 [C]=model_neophytes_mit_b2_ndsm)
declare -A BASE_NAME=([A]=Unet_mit_b2 [C]=Unet_mit_b2_nDSM)

log() { echo "$(date '+%F %T')  $*" | tee -a "$QUEUE_DIR/status.txt"; }
is_done() { [ -f "$QUEUE_DIR/$1.done" ]; }
mark_done() { touch "$QUEUE_DIR/$1.done"; }

# wait_for_gpu <GiB>
wait_for_gpu() {
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
        if [ "$free_mib" -ge $(($1 * 1024)) ]; then return; fi
        echo "$(date '+%F %T')  GPU has $((free_mib / 1024)) GiB free, waiting for $1 GiB ..."
        sleep 300
    done
}

# run_job <name> <min free GiB> <command...>: waits for the GPU, logs output, exit
# code and the peak GPU memory of the job's process (MiB, in $QUEUE_DIR/<name>.peak).
# A job that succeeds is marked done and skipped on a rerun.
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

stop() { log "STOP  $*"; exit 1; }

# best_ckpt <exp>: the val_f1_species checkpoint of a base run
best_ckpt() { ls lightning_logs/"$1"/checkpoints/best_*.ckpt 2>/dev/null | sort | tail -n 1; }

# base <A|C>: held-out base training; the test pass at the end scores the 5 known species.
# An error after "Training finished" (in that test pass) does not block the few-shot steps.
base() {
    local tag=$1 name="base_$1"
    run_job "$name" "$TRAIN_GIB" python train.py data=$HOLDOUT_DATA model=${BASE_MODEL[$tag]} \
        model.max_epochs=30 model.check_val_every_n_epoch=3 pl_seed=42 exp_name=${BASE_EXP[$tag]}
    if [ $? -ne 0 ]; then
        grep -q "Training finished" "$QUEUE_DIR/$name.log" && [ -n "$(best_ckpt ${BASE_EXP[$tag]})" ] \
            || stop "$name failed, see $QUEUE_DIR/$name.log"
        log "NOTE  $name: training finished, its test pass failed; going on with the best checkpoint"
        mark_done "$name"
    fi
}

# finetune <A|C>: fine-tune the base run on each draw, then score the 3 models in one pass
finetune() {
    local tag=$1 init ckpts=() names=()
    init=$(best_ckpt ${BASE_EXP[$tag]})
    [ -n "$init" ] || { log "SKIP  ft_$tag: no checkpoint of base_$tag"; return; }
    for s in $SEEDS; do
        local exp=${TS}_np_cv3_fs1_${BASE_NAME[$tag]}_512_s${s}_from${tag}
        run_job "ft_${tag}_s$s" "$TRAIN_GIB" python train.py data=$FEWSHOT_DATA model=${BASE_MODEL[$tag]} \
            "init_ckpt='$init'" data.few_shot_seed=$s pl_seed=$s "${FT_ARGS[@]}" exp_name=$exp \
            && { ckpts+=("lightning_logs/$exp/checkpoints/final.ckpt"); names+=("s$s"); }
    done
    [ ${#ckpts[@]} -gt 0 ] || { log "SKIP  eval_ft_$tag: no fine-tuned model"; return; }
    run_job "eval_ft_$tag" "$INFER_GIB" python fewshot_probe.py eval --ckpts "${ckpts[@]}" \
        --names "${names[@]}" --out $OUT/ft_$tag
}

# probe <tag> <backbone>: 3 heads (one per draw) + one test pass; a failure is logged only
probe() {
    run_job "probe_$1" "$INFER_GIB" python fewshot_probe.py probe --backbone "$2" --seeds $SEEDS \
        --out $OUT/probe_$1 || log "NOTE  probe_$1 failed, see $QUEUE_DIR/probe_$1.log"
}

log "queue start, run names from $QUEUE_DIR/names.env (TS=$TS)"

# %% 0. checks (CPU)
[ -f "$PC24_WEIGHTS" ] || stop "$PC24_WEIGHTS missing; extract it with scripts/extract_pc24_backbone.py"
[ -f "$SMOKE_CKPT" ] || stop "$SMOKE_CKPT missing (smoke tests only; point SMOKE_CKPT at any cv3 512 b2 run)"
python -c "import timm; timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True)" \
    > "$QUEUE_DIR/weights_dinov2.log" 2>&1 \
    || stop "DINOv2 weights not available, see $QUEUE_DIR/weights_dinov2.log"
python fewshot_probe.py check --seeds $SEEDS > "$QUEUE_DIR/check.log" 2>&1 \
    || stop "dataset check failed, see $QUEUE_DIR/check.log"
log "check: $(grep -E 'B_davidii_px left|seed [0-9]:|shared' "$QUEUE_DIR/check.log" | tr '\n' ';')"

# %% 1. smoke tests (GPU, minutes): every code path the long jobs use
SMOKE=$QUEUE_DIR/smoke
run_job smoke_ft "$TRAIN_GIB" python train.py data=$FEWSHOT_DATA "init_ckpt='$SMOKE_CKPT'" \
    data.few_shot_seed=1 "${FT_ARGS[@]}" model.max_epochs=1 exp_name=smoke_${TS}_np_cv3_fs1_ft \
    || stop "smoke_ft failed"
run_job smoke_eval "$INFER_GIB" python fewshot_probe.py eval \
    --ckpts lightning_logs/smoke_${TS}_np_cv3_fs1_ft/checkpoints/final.ckpt --names s1 \
    --max_test_batches 3 --out $SMOKE/eval || stop "smoke_eval failed"
run_job smoke_probe_pc24 "$INFER_GIB" python fewshot_probe.py probe --backbone pc24 --seeds 1 \
    --max_steps 20 --max_test_batches 3 --out $SMOKE/probe_pc24 || stop "smoke_probe_pc24 failed"
run_job smoke_probe_ckpt "$INFER_GIB" python fewshot_probe.py probe --backbone "ckpt:$SMOKE_CKPT" --seeds 1 \
    --max_steps 20 --max_test_batches 3 --out $SMOKE/probe_ckpt || stop "smoke_probe_ckpt failed"
run_job smoke_ndsm "$TRAIN_GIB" python train.py data=neophytes_split_debug_1024 model=model_neophytes_mit_b2_ndsm \
    model.max_epochs=1 logger=csv exp_name=smoke_${TS}_np_debug_mit_b2_ndsm || stop "smoke_ndsm failed"
run_job smoke_eval_ndsm "$INFER_GIB" python fewshot_probe.py eval \
    --ckpts "$(best_ckpt smoke_${TS}_np_debug_mit_b2_ndsm)" --names s1 \
    --max_test_batches 3 --out $SMOKE/eval_ndsm || stop "smoke_eval_ndsm failed"

# %% 2. A: b2 base training
base A

# %% 3. linear probes on frozen features
probe pc24   pc24
probe dinov2 dinov2
probe mit_b2 mit_b2
probe A      "ckpt:$(best_ckpt $A_EXP)"

# %% 4. b2 fine-tuning
finetune A
python fewshot_probe.py summary --root $OUT > /dev/null 2>&1 && log "summary after step 4: $OUT/summary.txt"

# %% 5.-6. b2 + nDSM
base C
finetune C

# %% 7. summary
python fewshot_probe.py summary --root $OUT 2>&1 | tee -a "$QUEUE_DIR/status.txt"
log "queue finished"
