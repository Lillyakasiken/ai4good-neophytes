#!/usr/bin/env bash
# Queue of Tue 2026-10-20 (see docs/plantclef2024_vit_notes_en.md, section 8): the
# PlantCLEF 2024 ViT (ViTD2PC24All) as the encoder of the baseline U-Net (RQ1), with
# the same DINOv2 ViT without the PlantCLEF fine-tuning as the control.
#
# Reference: U-Net + MiT-b2, 512 crops, cv3 and cv1 x s42 and s43 (already trained and
# tested). Every run here uses the same data, sampler, loss, 30 epochs, validation every
# 3rd epoch and batch 8; only the encoder changes (+ LP-FT and layer-wise LR decay,
# which the ViT needs, see section 8.3). Each run pairs with the b2 run of the same
# fold and seed.
#
# 0. Weights check (PlantCLEF backbone file, DINOv2 from the HF cache)
# 1. Smoke test: debug split, 2 epochs (1 frozen + 1 fine-tuned), csv logger   ~5 min
# 2. pc24   cv3 s42 + test.py   first reading on the iteration fold
# 3. pc24   cv1 s42 + test.py   second development fold
# 4. dinov2 cv3 s42 + test.py   control: what the plant photos add
# 5. dinov2 cv1 s42 + test.py
# 6. pc24   cv3 s43 + test.py   completes pc24 on 2 folds x 2 seeds (adoption rule)
# 7. pc24   cv1 s43 + test.py
# 8. evaluate_cv.py for b2 / pc24 / dinov2 and a paired table vs b2 (CPU)    ~1 min
#
# Each training run is roughly 10-15 h + 1.5-2 h test.py (not measured; a ViT-B run
# was budgeted at 1.5-2x a b2 run, which takes 7.6 h), so the whole queue takes about
# 3.5 days. After job 3 (~1 day) both folds have a first pc24 number.
#
# Memory: as in queue_2026-10-14_b4.sh, a job waits until the GPU has the expected
# peak + 2 GiB free (at least 36 GiB, the b2 threshold), and the peak of each job's
# process is sampled every 30 s. If a training run runs out of memory, it is retried
# once with gradient checkpointing, which then stays on for the remaining runs; a
# second out-of-memory error stops the queue. An out-of-memory error after "Training
# finished" (in train.py's own test pass on whole tiles) is no reason to retrain:
# gradient checkpointing does not help inference, and the best checkpoint is there,
# so the queue goes on with test.py.
#
#     tmux new -s queue_vit
#     bash scripts/queues/queue_2026-10-20_vit_pc24.sh
#
# Logs and exit codes go to lightning_logs/queue_2026-10-20/.

set -u

CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
QUEUE_DIR=lightning_logs/queue_2026-10-20
PC24_WEIGHTS=/zfs/ai4good/groups/neophytes/zhixuan_li/pretrained/plantclef2024/vitd2pc24all_backbone_ema.pth
MARGIN_GIB=2
MIN_FREE_FLOOR=36   # never wait for less than the 512 b2 queue did
NEED_GIB=$MIN_FREE_FLOOR
EXTRA=""            # set to model.grad_checkpointing=True after an out-of-memory error

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/../.."
mkdir -p "$QUEUE_DIR"

log() { echo "$(date '+%F %T')  $*" | tee -a "$QUEUE_DIR/status.txt"; }

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
# code and the peak GPU memory of the job's process (MiB, in $QUEUE_DIR/<name>.peak)
run_job() {
    local name=$1 need=$2; shift 2
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
    return $rc
}

# update NEED_GIB from the measured peak of a job: update_need <name>
update_need() {
    local peak_mib; peak_mib=$(cat "$QUEUE_DIR/$1.peak" 2>/dev/null || echo 0)
    [ "$peak_mib" -gt 0 ] || return
    local total_gib=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | tr -d ' ') / 1024 ))
    NEED_GIB=$(( (peak_mib + 1023) / 1024 + MARGIN_GIB ))
    [ "$NEED_GIB" -lt "$MIN_FREE_FLOOR" ] && NEED_GIB=$MIN_FREE_FLOOR
    [ "$NEED_GIB" -gt "$total_gib" ] && NEED_GIB=$total_gib
}

# newest run directory: run_dir <tag> <fold> <seed>, tag = pc24 or dinov2
run_dir() {
    ls -d lightning_logs/*_np_$2_Unet_vitb14_$1_512_s$3 2>/dev/null | sort | tail -n 1
}

# test_vit <tag> <fold> <seed>: test.py on the best checkpoint of the newest matching run
test_vit() {
    local dir; dir=$(run_dir "$1" "$2" "$3")
    if [ -z "$dir" ]; then
        log "SKIP  test_$1_$2_s$3: no run directory"
        return 1
    fi
    run_job "test_$1_$2_s$3" "$MIN_FREE_FLOOR" \
        python test.py exp_name="$(basename "$dir")" ckpt_type=best compute_auc=False
}

# training_finished <job name>: trainer.fit returned (train.py prints this before its test pass)
training_finished() {
    grep -q "Training finished" "$QUEUE_DIR/$1.log"
}

# oom_in_training <job name>: out of memory before the end of training (retry is worth it)
oom_in_training() {
    grep -q "out of memory" "$QUEUE_DIR/$1.log" && ! training_finished "$1"
}

# train_vit <tag> <fold> <seed>: the 512 development setting with a ViT encoder, then test.py
train_vit() {
    local tag=$1 fold=$2 seed=$3 name="train_$1_$2_s$3"
    local cmd=(python train.py data=neophytes_split_${fold}_train_512 model=model_neophytes_vit_${tag}
               model.max_epochs=30 model.check_val_every_n_epoch=3 pl_seed=$seed)
    run_job "$name" "$NEED_GIB" "${cmd[@]}" $EXTRA
    local rc=$?
    if [ $rc -ne 0 ] && oom_in_training "$name"; then
        if [ -n "$EXTRA" ]; then
            log "STOP  $name ran out of memory with gradient checkpointing"
            exit 1
        fi
        EXTRA="model.grad_checkpointing=True"
        log "RETRY $name with gradient checkpointing (kept for the remaining runs)"
        name="${name}_gc"
        run_job "$name" "$NEED_GIB" "${cmd[@]}" $EXTRA
        rc=$?
        if [ $rc -ne 0 ] && oom_in_training "$name"; then
            log "STOP  $name ran out of memory with gradient checkpointing"
            exit 1
        fi
    fi
    update_need "$name"
    if [ $rc -eq 0 ]; then
        test_vit "$tag" "$fold" "$seed"
    elif training_finished "$name"; then
        log "NOTE  $name: training finished, train.py's own test pass failed (exit $rc); running test.py"
        test_vit "$tag" "$fold" "$seed"
    fi
}

# %% 0. weights
if [ ! -f "$PC24_WEIGHTS" ]; then
    log "STOP  $PC24_WEIGHTS missing; extract it with scripts/extract_pc24_backbone.py"
    exit 1
fi
python -c "import timm; timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True)" \
    > "$QUEUE_DIR/weights_dinov2.log" 2>&1 \
    || { log "STOP  DINOv2 weights not available, see $QUEUE_DIR/weights_dinov2.log"; exit 1; }

# %% 1. smoke test: model build, weight loading, LP -> FT switch, whole-tile val/test.
# # 1024 crops with a fine-tuned ViT need well over 8 GiB, so wait for the full floor
# # rather than stop the queue on an out-of-memory error on a shared GPU.
run_job smoke_pc24 "$MIN_FREE_FLOOR" python train.py data=neophytes_split_debug_1024 model=model_neophytes_vit_pc24 \
    logger=csv exp_name="smoke_$(date +%Y%m%d_%H%M%S)_np_debug_vitb14_pc24" \
    model.max_epochs=2 model.freeze_encoder_epochs=1 model.check_val_every_n_epoch=1 \
    || { log "STOP  smoke test failed, see $QUEUE_DIR/smoke_pc24.log"; exit 1; }

# %% 2.-7. development runs, ordered so that every finished job adds a paired comparison
train_vit pc24   cv3 42
train_vit pc24   cv1 42
train_vit dinov2 cv3 42
train_vit dinov2 cv1 42
train_vit pc24   cv3 43
train_vit pc24   cv1 43

# %% 8. per-run tables (cv_runs.csv) and the paired differences to b2
for enc in mit_b2 vitb14_pc24 vitb14_dinov2; do
    python evaluate_cv.py --runs "*np_cv[13]_Unet_${enc}_512_s*" --ckpt-type best \
        --out "results/cv/512_dev_${enc}" >> "$QUEUE_DIR/evaluate_cv.log" 2>&1
    log "evaluate_cv $enc: exit $?, tables in results/cv/512_dev_${enc}"
done
python - <<'EOF' 2>&1 | tee -a "$QUEUE_DIR/status.txt"
import pandas as pd
cols = ['F1-avg-wo0'] + [c for c in pd.read_csv('results/cv/512_dev_mit_b2/cv_runs.csv').columns
                         if c.startswith('F1_') and c != 'F1_Background']
ref = pd.read_csv('results/cv/512_dev_mit_b2/cv_runs.csv').set_index(['fold', 'seed'])[cols]
for enc in ['vitb14_pc24', 'vitb14_dinov2']:
    try:
        runs = pd.read_csv(f'results/cv/512_dev_{enc}/cv_runs.csv').set_index(['fold', 'seed'])[cols]
    except FileNotFoundError:
        continue
    delta = (runs - ref).dropna()
    print(f'\n{enc} - mit_b2, paired species F1 (n={len(delta)}):')
    print(delta.round(3).to_string())
    print('mean ' + ' '.join(f'{v:+.3f}' for v in delta.mean()))
EOF

log "queue finished"
