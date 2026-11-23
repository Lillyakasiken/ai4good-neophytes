#!/usr/bin/env bash
# Queue of Mon 2026-11-16 (docs/plantclef2024_fewshot_en.md): the PlantCLEF few-shot
# experiments for the talk of Fri 2026-11-20. The runs of 2026-10-30 had ONE new species
# (B. davidii), one fold, k = 10: PlantCLEF ViT + linear head F1 0.593, b2 base +
# fine-tune 0.208. This queue asks whether that holds for every species, how it depends
# on k and on where the tiles come from, and whether it is the frozen features or the
# probe protocol.
#
# Two lanes, one tmux session each; they run side by side (6 + 9 GiB of GPU memory):
#
#   probes   frozen ViT + linear head, each of the 6 species as the new one, 3 draws.
#            One job = 18 heads + ONE pass over the test split (~1.3-1.7 h, estimated).
#     0. checks (CPU) and smoke tests (minutes)
#     1. PlantCLEF, cv3, k = 10          (its B. davidii heads repeat the 2026-10-30 runs:
#                                         same tiles, F1 should be 0.659 / 0.640 / 0.481)
#     2. DINOv2,    cv3, k = 10          control: plant photos vs generic SSL
#     3. PlantCLEF, cv3, k = 10, all tiles of a draw from ONE flight (teammates' setting)
#     4. PlantCLEF, cv3, k = 1, 3, 30    how many tiles does a new species need
#     5. PlantCLEF, cv1, k = 10          second fold
#     6. DINOv2,    cv3, k = 1, 3, 30
#     7. DINOv2,    cv1, k = 10
#
#   unet     B. davidii only: the PlantCLEF U-Net through the b2 protocol of 2026-10-30
#     0. smoke tests (minutes)
#     1. base P: PlantCLEF U-Net without B. davidii             ~7.4 h + 1 h test
#     2. ft_P_r3: P fine-tuned on the 10 tiles (fresh B. davidii channel, lr 3e-4,
#        30 epochs = the recipe of ft_A_r3), 3 draws            ~10 min each
#     3. probe_P: linear head on the ViT of P                   ~0.6 h
#     4. test pass of the 3 ft_P_r3 models                      ~3 h
#     5. ft_P_frozen: as 2 with the ViT frozen, + test pass     ~3.5 h
#
# The times are estimates from the October runs, not measured; with both lanes on the
# GPU each is slower. After every job: results/fewshot/loso/summary.txt and figures/.
# The probes use the full test split at full resolution, like the October runs.
#
#     tmux new -s fs_probes
#     bash scripts/queues/queue_2026-11-16_fewshot_loso.sh probes
#     tmux new -s fs_unet
#     bash scripts/queues/queue_2026-11-16_fewshot_loso.sh unet
#
# Logs go to lightning_logs/queue_2026-11-16_fewshot_loso/ (status.txt for the overview).
# Rerunning a lane skips its finished jobs (*.done); an interrupted probe job keeps its
# trained heads, an interrupted base training resumes from its last.ckpt.

set -u

LANE=${1:-}
case "$LANE" in probes|unet) ;; *) echo "usage: $0 probes|unet"; exit 1 ;; esac

CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
QUEUE_DIR=lightning_logs/queue_2026-11-16_fewshot_loso
OUT=results/fewshot/loso
OLD_OUT=results/fewshot/cv3_bdavidii   # the runs of 2026-10-30
HOLDOUT_DATA=neophytes_split_cv3_train_512_holdout_bdavidii
FEWSHOT_DATA=neophytes_split_cv3_train_512_fewshot_bdavidii
VIT_MODEL=model_neophytes_vit_pc24
PC24_WEIGHTS=/zfs/ai4good/groups/neophytes/zhixuan_li/pretrained/plantclef2024/vitd2pc24all_backbone_ema.pth
# # any trained cv3 512 PlantCLEF U-Net, only for the smoke tests of the unet lane
SMOKE_EXP=20261020_121112.499060_np_cv3_Unet_vitb14_pc24_512_s42
SEEDS="1 2 3"
FT_ARGS=(reset_head_classes=1 model.lr=0.0003 model.max_epochs=30 model.check_val_every_n_epoch=1000
         test_ckpt=null logger=csv)
PROBE_GIB=12   # ViT probes peaked at 6 GiB in October (3 heads)
TRAIN_GIB=14   # PlantCLEF U-Net at 512: 9 GiB
EVAL_GIB=16    # 3 ViT U-Nets on whole tiles: 7 GiB for one in test.py

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/../.."
mkdir -p "$QUEUE_DIR" "$OUT"

# # run names are fixed on the first start (of either lane), so a restart resumes the same runs
( set -o noclobber; echo "TS=$(date +%Y%m%d_%H%M%S).000000" > "$QUEUE_DIR/names.env" ) 2> /dev/null
source "$QUEUE_DIR/names.env"
P_EXP=${TS}_np_cv3_ho1_Unet_vitb14_pc24_512_s42

log() { echo "$(date '+%F %T')  [$LANE] $*" | tee -a "$QUEUE_DIR/status.txt"; }
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
    local t0=$SECONDS
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
    log "END   $name: exit $rc, $(((SECONDS - t0) / 60)) min, peak GPU memory $((peak / 1024)) GiB"
    [ $rc -eq 0 ] && mark_done "$name"
    return $rc
}

# report: summary table + figures from everything scored so far (CPU, ~1 min)
report() {
    python fewshot_probe.py summary --root $OUT > "$QUEUE_DIR/report_$LANE.log" 2>&1 \
        && python scripts/plot_fewshot.py --roots $OLD_OUT $OUT --out $OUT/figures >> "$QUEUE_DIR/report_$LANE.log" 2>&1 \
        && log "report: $OUT/summary.txt, $OUT/figures/" \
        || log "NOTE  report failed, see $QUEUE_DIR/report_$LANE.log"
}

# best_ckpt <exp>: the val_f1_species checkpoint of a run
best_ckpt() { ls lightning_logs/"$1"/checkpoints/best_*.ckpt 2>/dev/null | sort | tail -n 1; }

# ============================================================================
# lane "probes"
# ============================================================================

# probe_grid <job> <backbone> <fold> <k> [args...]: all 6 species x the draws at one k,
# 18 heads + one test pass, into $OUT/<job>. A failure is logged, the lane goes on.
probe_grid() {
    local name=$1 backbone=$2 fold=$3 k=$4; shift 4
    run_job "$name" "$PROBE_GIB" python fewshot_probe.py probe --backbone "$backbone" --fold "$fold" \
        --new_species all --k "$k" --seeds $SEEDS --method "probe_$backbone" --out "$OUT/$name" "$@"
    local rc=$?
    [ $rc -eq 0 ] || log "NOTE  $name failed, see $QUEUE_DIR/$name.log"
    grep -h "scoring speed" "$QUEUE_DIR/$name.log" 2> /dev/null | tail -n 1 | while read -r line; do log "$name: $line"; done
    report
    return $rc
}

# sanity: the B. davidii k = 10 heads of step 1 against probe_pc24 of 2026-10-30
sanity_bdavidii() {
    python - "$OUT/cv3_pc24_k10" "$OLD_OUT/probe_pc24" <<'EOF'
import json, sys
new_dir, old_dir = sys.argv[1:]
for s in (1, 2, 3):
    new, old = (json.load(open(p)) for p in (f'{new_dir}/B_davidii_k10_s{s}.json', f'{old_dir}/s{s}.json'))
    same_tiles = sorted(new['train_tiles']) == sorted(old['train_tiles'])
    diff = new['F1'][1] - old['F1'][1]
    flag = '' if same_tiles and abs(diff) <= 0.03 else '  <-- CHECK'
    print(f"sanity B. davidii k=10 draw {s}: F1 {new['F1'][1]:.3f}, 2026-10-30 {old['F1'][1]:.3f} ({diff:+.3f}), "
          f"same training tiles: {same_tiles}{flag}")
EOF
}

lane_probes() {
    # %% 0. checks (CPU)
    [ -f "$PC24_WEIGHTS" ] || stop "$PC24_WEIGHTS missing; extract it with scripts/extract_pc24_backbone.py"
    python -c "import timm; timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True)" \
        > "$QUEUE_DIR/weights_dinov2.log" 2>&1 \
        || stop "DINOv2 weights not available, see $QUEUE_DIR/weights_dinov2.log"
    # # how many candidate tiles each species has, and what every draw gets
    { python fewshot_probe.py check --fold cv3 --new_species all --k 1 3 10 30 --seeds $SEEDS \
        && python fewshot_probe.py check --fold cv3 --new_species all --k 10 --seeds $SEEDS --draw flight \
        && python fewshot_probe.py check --fold cv1 --new_species all --k 10 --seeds $SEEDS; } \
        > "$QUEUE_DIR/check.log" 2>&1 || stop "dataset check failed, see $QUEUE_DIR/check.log"
    grep -E "candidate tiles" "$QUEUE_DIR/check.log" | cut -c 1-160 | while read -r line; do log "check: $line"; done
    short=$(grep -c "<-- only" "$QUEUE_DIR/check.log")
    [ "$short" -eq 0 ] || log "NOTE  $short draws get fewer than k tiles (grep '<-- only' $QUEUE_DIR/check.log)"

    # %% 0. smoke tests (GPU, minutes): the grid, the flight draws, summary and figures
    SMOKE=$QUEUE_DIR/smoke_probes
    run_job smoke_grid "$PROBE_GIB" python fewshot_probe.py probe --backbone pc24 --fold cv3 \
        --new_species B_davidii S_inaequidens --k 1 10 --seeds 1 --max_steps 20 --max_test_batches 12 \
        --method probe_pc24 --out $SMOKE/cv3_pc24 || stop "smoke_grid failed, see $QUEUE_DIR/smoke_grid.log"
    run_job smoke_flight "$PROBE_GIB" python fewshot_probe.py probe --backbone dinov2 --fold cv1 \
        --new_species A_altissima --k 10 --seeds 2 --draw flight --max_steps 20 --max_test_batches 12 \
        --method probe_dinov2 --out $SMOKE/cv1_dinov2_flight || stop "smoke_flight failed, see $QUEUE_DIR/smoke_flight.log"
    { python fewshot_probe.py summary --root $SMOKE \
        && python scripts/plot_fewshot.py --roots $SMOKE --out $SMOKE/figures; } > "$QUEUE_DIR/smoke_report.log" 2>&1 \
        || stop "smoke summary / figures failed, see $QUEUE_DIR/smoke_report.log"
    log "smoke tests passed (4 heads: $(grep -h 'scoring speed' "$QUEUE_DIR/smoke_grid.log" | tail -n 1))"

    # %% 1. PlantCLEF, cv3, k = 10: the main table
    probe_grid cv3_pc24_k10 pc24 cv3 10 || stop "cv3_pc24_k10 failed"
    sanity_bdavidii 2>&1 | while read -r line; do log "$line"; done

    # %% 2. DINOv2 control
    probe_grid cv3_dinov2_k10 dinov2 cv3 10

    # %% 3. all k tiles from one flight (seeds 1, 2, 3 = three flights)
    probe_grid cv3_pc24_k10_flight pc24 cv3 10 --draw flight

    # %% 4. k sweep, PlantCLEF
    for k in 1 3 30; do probe_grid cv3_pc24_k$k pc24 cv3 $k; done

    # %% 5. second fold
    probe_grid cv1_pc24_k10 pc24 cv1 10

    # %% 6.-7. DINOv2: k sweep, second fold
    for k in 1 3 30; do probe_grid cv3_dinov2_k$k dinov2 cv3 $k; done
    probe_grid cv1_dinov2_k10 dinov2 cv1 10
}

# ============================================================================
# lane "unet"
# ============================================================================

ft_exp() { echo "${TS}_np_cv3_fs1_Unet_vitb14_pc24_512_s$2_fromP_$1"; }  # <r3|frozen> <seed>

# train_f1_new <exp>: train_f1_class_1 of the last epoch (did it learn B. davidii on its crops?)
train_f1_new() {
    python - "$1" <<'EOF'
import glob, sys
import pandas as pd
d = pd.read_csv(sorted(glob.glob(f'lightning_logs/{sys.argv[1]}/version_*/metrics.csv'))[-1])
print(f"{d['train_f1_class_1'].dropna().iloc[-1]:.3f}")
EOF
}

# finetune_P <r3|frozen> [train.py overrides...]: P fine-tuned on each draw (final
# weights, no test pass in train.py)
finetune_P() {
    local variant=$1 init s; shift
    init=$(best_ckpt $P_EXP)
    for s in $SEEDS; do
        run_job "ft_P_${variant}_s$s" "$TRAIN_GIB" python train.py data=$FEWSHOT_DATA model=$VIT_MODEL \
            "init_ckpt='$init'" data.few_shot_seed=$s pl_seed=$s "${FT_ARGS[@]}" "$@" \
            exp_name=$(ft_exp "$variant" "$s") \
            && log "ft_P_${variant}_s$s: train_f1_class_1 = $(train_f1_new "$(ft_exp "$variant" "$s")" 2>&1 | tail -n 1)" \
            || log "NOTE  ft_P_${variant}_s$s failed, see $QUEUE_DIR/ft_P_${variant}_s$s.log"
    done
}

# eval_P <r3|frozen>: the fine-tuned models of all draws in one test pass
eval_P() {
    local variant=$1 ckpts=() names=() s ckpt
    for s in $SEEDS; do
        ckpt=lightning_logs/$(ft_exp "$variant" "$s")/checkpoints/final.ckpt
        is_done "ft_P_${variant}_s$s" && [ -f "$ckpt" ] && { ckpts+=("$ckpt"); names+=("s$s"); }
    done
    [ ${#ckpts[@]} -gt 0 ] || { log "SKIP  eval_ft_P_$variant: no fine-tuned model"; return; }
    run_job "eval_ft_P_$variant" "$EVAL_GIB" python fewshot_probe.py eval --ckpts "${ckpts[@]}" \
        --names "${names[@]}" --method "ft_P_$variant" --new_species B_davidii --k 10 --fold cv3 \
        --out "$OUT/cv3_ft_P_$variant" || log "NOTE  eval_ft_P_$variant failed, see $QUEUE_DIR/eval_ft_P_$variant.log"
    report
}

lane_unet() {
    # %% 0. checks and smoke tests (GPU, minutes): every code path of the long jobs
    [ -f "$PC24_WEIGHTS" ] || stop "$PC24_WEIGHTS missing; extract it with scripts/extract_pc24_backbone.py"
    SMOKE_CKPT=$(best_ckpt $SMOKE_EXP)
    [ -n "$SMOKE_CKPT" ] || stop "no checkpoint of $SMOKE_EXP (smoke tests only; point SMOKE_EXP at any cv3 512 PlantCLEF U-Net run)"
    SMOKE=$QUEUE_DIR/smoke_unet
    # # 2 epochs, the ViT frozen in the first: fine-tuning from a checkpoint, the new
    # # B. davidii channel and the unfreezing
    run_job smoke_ft "$TRAIN_GIB" python train.py data=$FEWSHOT_DATA model=$VIT_MODEL "init_ckpt='$SMOKE_CKPT'" \
        data.few_shot_seed=1 "${FT_ARGS[@]}" model.max_epochs=2 model.freeze_encoder_epochs=1 \
        exp_name=smoke_${TS}_np_cv3_fs1_ft_P || stop "smoke_ft failed, see $QUEUE_DIR/smoke_ft.log"
    run_job smoke_eval "$EVAL_GIB" python fewshot_probe.py eval \
        --ckpts lightning_logs/smoke_${TS}_np_cv3_fs1_ft_P/checkpoints/final.ckpt --names s1 \
        --method ft_P_r3 --new_species B_davidii --k 10 --max_test_batches 5 --out $SMOKE/cv3_ft_P \
        || stop "smoke_eval failed, see $QUEUE_DIR/smoke_eval.log"
    run_job smoke_probe_vitckpt "$PROBE_GIB" python fewshot_probe.py probe --backbone "vitckpt:$SMOKE_CKPT" \
        --fold cv3 --new_species B_davidii --k 10 --seeds 1 --max_steps 20 --max_test_batches 5 \
        --method probe_P --out $SMOKE/cv3_probe_P || stop "smoke_probe_vitckpt failed, see $QUEUE_DIR/smoke_probe_vitckpt.log"
    python fewshot_probe.py summary --root $SMOKE > "$QUEUE_DIR/smoke_unet_report.log" 2>&1 \
        || stop "smoke summary failed, see $QUEUE_DIR/smoke_unet_report.log"
    log "smoke tests passed ($(grep -h 'scoring speed' "$QUEUE_DIR/smoke_eval.log" | tail -n 1))"

    # %% 1. base P: PlantCLEF U-Net on the five other species (B. davidii tiles dropped).
    # An error after "Training finished" (in train.py's test pass) does not block the rest.
    run_job base_P "$TRAIN_GIB" python train.py data=$HOLDOUT_DATA model=$VIT_MODEL \
        model.max_epochs=30 model.check_val_every_n_epoch=3 pl_seed=42 exp_name=$P_EXP
    if [ $? -ne 0 ]; then
        grep -q "Training finished" "$QUEUE_DIR/base_P.log" && [ -n "$(best_ckpt $P_EXP)" ] \
            || stop "base_P failed, see $QUEUE_DIR/base_P.log"
        log "NOTE  base_P: training finished, its test pass failed; going on with the best checkpoint"
        mark_done base_P
    fi
    [ -n "$(best_ckpt $P_EXP)" ] || stop "no checkpoint of $P_EXP"

    # %% 2. ft_P_r3: full fine-tuning (ViT frozen in epochs 0-5, as in the base training)
    finetune_P r3

    # %% 3. probe_P: linear head on the ViT of P (features after the base training)
    run_job probe_P "$PROBE_GIB" python fewshot_probe.py probe --backbone "vitckpt:$(best_ckpt $P_EXP)" \
        --fold cv3 --new_species B_davidii --k 10 --seeds $SEEDS --method probe_P --out $OUT/cv3_probe_P \
        || log "NOTE  probe_P failed, see $QUEUE_DIR/probe_P.log"
    report

    # %% 4. test pass of ft_P_r3
    eval_P r3

    # %% 5. ft_P_frozen: only the pyramid, the decoder and the head learn the new species
    finetune_P frozen model.freeze_encoder_epochs=1000
    eval_P frozen
}

log "lane start (TS=$TS)"
lane_$LANE
report
log "lane finished"
