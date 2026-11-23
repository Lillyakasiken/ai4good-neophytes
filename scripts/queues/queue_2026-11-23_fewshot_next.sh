#!/usr/bin/env bash
# Queue of Mon 2026-11-23: the next steps and open limitations of
# docs/plantclef2024_fewshot_en.md (sections 5 and 6), after the 2026-11-16 / 11-18 queues.
# Results go to the same root (results/fewshot/loso), so summary.txt and the figures pool
# them with the earlier runs.
#
# Three lanes, one tmux session each:
#
#   probes   linear probes, ~6-12 GiB, ~1-1.7 h per job                       ~22 h
#     0. checks (CPU) and smoke tests (minutes)
#     1. calibration without validation labels (6.4, limitation 5): the cv3 k = 10
#        PlantCLEF heads of draws 1-6, reused, with a bias of the new species chosen on
#        the few-shot subset + a bias sweep on the test split
#     2. PlantCLEF + MiT-b2 fusion probe, cv3, k = 10, draws 1-3 (6.3: S. inaequidens)
#     3. labelling budget (6.5): 10 tiles from 5 flights, from 2 flights, and 30 tiles
#        from one flight, draws 1-6 (10 tiles all sites / one flight exist)
#     4. more folds (limitation 3): PlantCLEF + DINOv2 on cv2, cv4, cv5 (draws 1-3)
#        and cv1 draws 4-6
#
#   vitbase  per held-out species: PlantCLEF U-Net base P (~12 h, 9 GiB)    ~80 h
#     (6.1, limitation 2: probe_P and the 2x2 so far only for B. davidii)
#     1. for each species: base P -> probe_P (3 draws)
#     2. then for each species: ft_P_r3 (3 draws, ~10 min each) -> one test pass (~2.5 h)
#
#   b2base   per held-out species: MiT-b2 U-Net base A (~7.5 h, 33 GiB)     ~75 h
#     0. combined model for B. davidii (6.2) with the October base A: known species from
#        A, new species from the PlantCLEF heads of draws 1-6 (no training)
#     1. for each species: base A -> probe_A (3 draws) -> combined model (6 draws)
#     2. then for each species: ft_A_r3 (3 draws) -> one test pass (~3 h)
#
# The species order of vitbase / b2base (default below) puts first the species where the
# teammates' fine-tuned MiT-b2 is about as good as the PlantCLEF probe (section 3.5); give
# other species, or fewer, as arguments:
#     bash scripts/queues/queue_2026-11-23_fewshot_next.sh vitbase A_altissima B_orientalis
#
# GPU (44 GiB, shared): probes + vitbase fit side by side (~12 + 9 GiB). b2base needs
# ~33-36 GiB: start it when the probes lane is done (vitbase 9 GiB + b2base 33 GiB is
# close to the limit); every job waits until the GPU has its memory free.
#
#     nvidia-smi
#     tmux new -s fs_probes   ; bash scripts/queues/queue_2026-11-23_fewshot_next.sh probes
#     tmux new -s fs_vitbase  ; bash scripts/queues/queue_2026-11-23_fewshot_next.sh vitbase
#     tmux new -s fs_b2base   ; bash scripts/queues/queue_2026-11-23_fewshot_next.sh b2base
#
# Logs: lightning_logs/queue_2026-11-23_fewshot_next/ (status.txt for the overview).
# Rerunning a lane skips its finished jobs (*.done); an interrupted probe job keeps its
# trained heads, an interrupted base training resumes from its last.ckpt.
# After each job: results/fewshot/loso/{summary.txt, analysis.txt, paired_pc24_vs_dinov2.txt, figures/}.

set -u

LANE=${1:-}
case "$LANE" in probes|vitbase|b2base) ;; *) echo "usage: $0 probes|vitbase|b2base [species ...]"; exit 1 ;; esac
shift
HO_SPECIES=${*:-A_altissima B_orientalis R_japonica R_typhina S_inaequidens}

CONDA_ENV=/zfs/ai4good/groups/neophytes/miniforge3/envs/ai4good-neophytes
QUEUE_DIR=lightning_logs/queue_2026-11-23_fewshot_next
OUT=results/fewshot/loso
OLD_OUT=results/fewshot/cv3_bdavidii   # the runs of 2026-10-30
PC24_WEIGHTS=/zfs/ai4good/groups/neophytes/zhixuan_li/pretrained/plantclef2024/vitd2pc24all_backbone_ema.pth
# # October base A (MiT-b2 U-Net without B. davidii) and a cv3 512 PlantCLEF U-Net for smoke tests
BASE_A_BDAVIDII=lightning_logs/20261029_185537.000000_np_cv3_ho1_Unet_mit_b2_512_s42
SMOKE_VIT_EXP=lightning_logs/20261020_121112.499060_np_cv3_Unet_vitb14_pc24_512_s42
ALL_SPECIES="B_davidii R_typhina R_japonica B_orientalis S_inaequidens A_altissima"
declare -A CLS=([B_davidii]=1 [R_typhina]=2 [R_japonica]=3 [B_orientalis]=4 [S_inaequidens]=5 [A_altissima]=6)
declare -A SHORT=([B_davidii]=bdavidii [R_typhina]=rtyphina [R_japonica]=rjaponica [B_orientalis]=borientalis
                  [S_inaequidens]=sinaequidens [A_altissima]=aaltissima)
FT_ARGS=(model.lr=0.0003 model.max_epochs=30 model.check_val_every_n_epoch=1000 test_ckpt=null logger=csv)
PROBE_GIB=12    # ViT probes: 6 GiB peak with 18 heads
FUSION_GIB=24   # not measured; the smoke test logs its peak
VIT_GIB=14      # PlantCLEF U-Net at 512: 9 GiB (base and fine-tuning)
VIT_EVAL_GIB=16 # 3 ViT U-Nets on whole tiles: 7 GiB
B2_GIB=34       # MiT-b2 U-Net at 512: 33 GiB (base, fine-tuning, 3-model test pass)
B2_PROBE_GIB=36 # MiT-b2 probes / combined model on whole tiles: 35 GiB in October

source /zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$(dirname "$0")/../.."
mkdir -p "$QUEUE_DIR" "$OUT"

# # run names are fixed on the first start (of any lane), so a restart resumes the same runs
( set -o noclobber; echo "TS=$(date +%Y%m%d_%H%M%S).000000" > "$QUEUE_DIR/names.env" ) 2> /dev/null
source "$QUEUE_DIR/names.env"

log() { echo "$(date '+%F %T')  [$LANE] $*" | tee -a "$QUEUE_DIR/status.txt"; }
is_done() { [ -f "$QUEUE_DIR/$1.done" ]; }
mark_done() { touch "$QUEUE_DIR/$1.done"; }
stop() { log "STOP  $*"; exit 1; }
free_gib() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ') / 1024 )); }

# wait_for_gpu <GiB>
wait_for_gpu() {
    while [ "$(free_gib)" -lt "$1" ]; do
        echo "$(date '+%F %T')  GPU has $(free_gib) GiB free, waiting for $1 GiB ..."
        sleep 300
    done
}

# run_job <name> <min free GiB> <command...>: as in queue_2026-11-16_fewshot_loso.sh
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

# paired: PlantCLEF - DINOv2 on the same tiles (as in queue_2026-11-18_fewshot_extra.sh)
paired() {
    python - "$OUT/summary.csv" <<'EOF'
import sys
import pandas as pd
from scipy.stats import wilcoxon
d = pd.read_csv(sys.argv[1])
d = d[d.method.isin(['probe_pc24', 'probe_dinov2']) & (d.draw == 'random')]
p = d.pivot_table(index=['fold', 'k', 'species', 'seed'], columns='method', values=['F1', 'known_F1']).dropna()
print('PlantCLEF - DINOv2, linear probes, paired by draw (same tiles); see analysis.txt section C for the')
print('test with the species as the unit')
for (fold, k), g in p.groupby(level=[0, 1]):
    diff = g['F1']['probe_pc24'] - g['F1']['probe_dinov2']
    known = g['known_F1']['probe_pc24'] - g['known_F1']['probe_dinov2']
    pv = wilcoxon(diff).pvalue if (diff != 0).any() else float('nan')
    print(f'{fold} k={k:>2}: new-species F1 {diff.mean():+.3f}, PlantCLEF better in {int((diff > 0).sum())}/{len(diff)} '
          f'pairs (Wilcoxon p = {pv:.2g}); known-species F1 {known.mean():+.3f}')
EOF
}

# report: summary, paired test, analyses and figures (CPU, ~1 min); one lane at a time
report() {
    (
        flock 9
        { python fewshot_probe.py summary --root $OUT \
            && paired > $OUT/paired_pc24_vs_dinov2.txt \
            && python scripts/analyze_fewshot.py --root $OUT \
            && python scripts/plot_fewshot.py --roots $OLD_OUT $OUT --out $OUT/figures; } > "$QUEUE_DIR/report_$LANE.log" 2>&1 \
            && log "report: $OUT/summary.txt, analysis.txt, figures/" \
            || log "NOTE  report failed, see $QUEUE_DIR/report_$LANE.log"
    ) 9> "$QUEUE_DIR/report.lock"
}

speed() { grep -h "scoring speed" "$QUEUE_DIR/$1.log" 2> /dev/null | tail -n 1 | while read -r line; do log "$1: $line"; done; }

# best_ckpt <exp dir>: the val_f1_species checkpoint of a run
best_ckpt() { ls "$1"/checkpoints/best_*.ckpt 2>/dev/null | sort | tail -n 1; }

# fs_args <species> <seed>: train.py overrides of the few-shot subset, the same tiles as
# fewshot_probe.py probe --new_species <species> --k 10 --seeds <seed> (checked in the smoke tests)
fs_args() {
    local sp=$1 seed=$2 few="" s
    for s in $ALL_SPECIES; do few+="${few:+,}$s:$([ "$s" = "$sp" ] && echo 10 || echo 50)"; done
    FS_ARGS=(data=neophytes_split_cv3_train_512 "data.few_shot_k={$few}" data.few_shot_mode=random
             data.few_shot_bg_k=200 data.few_shot_seed="$seed" data.cap_quantile=null
             data.data_name=np_cv3_fs${CLS[$sp]} reset_head_classes=${CLS[$sp]})
}

# train_f1_new <exp> <class>: train F1 of the new class in the last epoch (learnt on its crops?)
train_f1_new() {
    python - "$1" "$2" <<'EOF'
import glob, sys
import pandas as pd
d = pd.read_csv(sorted(glob.glob(f'lightning_logs/{sys.argv[1]}/version_*/metrics.csv'))[-1])
print(f"{d[f'train_f1_class_{sys.argv[2]}'].dropna().iloc[-1]:.3f}")
EOF
}

# ============================================================================
# lane "probes"
# ============================================================================

# probe_grid <job> <backbone> <fold> <k> <seeds> [args...]: all 6 species, one test pass
probe_grid() {
    local name=$1 backbone=$2 fold=$3 k=$4 seeds=$5 need=$PROBE_GIB; shift 5
    [ "$backbone" = fusion ] && need=$FUSION_GIB
    run_job "$name" "$need" python fewshot_probe.py probe --backbone "$backbone" --fold "$fold" \
        --new_species all --k "$k" --seeds $seeds --method "probe_$backbone" --out "$OUT/$name" "$@" \
        || log "NOTE  $name failed, see $QUEUE_DIR/$name.log"
    speed "$name"
    report
}

# calibrate <job> <heads dir> <seeds>: the existing heads, a calibrated bias + bias sweep
calibrate() {
    local name=$1 heads=$2 seeds=$3
    run_job "$name" "$PROBE_GIB" python fewshot_probe.py probe --backbone pc24 --fold cv3 --new_species all \
        --k 10 --seeds $seeds --heads_from "$OUT/$heads" --calibrate --method probe_pc24_cal --out "$OUT/$name" \
        || log "NOTE  $name failed, see $QUEUE_DIR/$name.log"
    speed "$name"
    grep -h "calibration bias" "$QUEUE_DIR/$name.log" 2> /dev/null | cut -c 1-120 | while read -r line; do log "$name: $line"; done
    report
}

lane_probes() {
    # %% 0. checks (CPU): what the new draws get
    [ -f "$PC24_WEIGHTS" ] || stop "$PC24_WEIGHTS missing; extract it with scripts/extract_pc24_backbone.py"
    { python fewshot_probe.py check --fold cv3 --new_species all --k 10 --seeds 1 2 3 4 5 6 --draw flights5 \
        && python fewshot_probe.py check --fold cv3 --new_species all --k 10 --seeds 1 2 3 4 5 6 --draw flights2 \
        && python fewshot_probe.py check --fold cv3 --new_species all --k 30 --seeds 1 2 3 4 5 6 --draw flight \
        && python fewshot_probe.py check --fold cv2 --new_species all --k 10 --seeds 1 2 3 \
        && python fewshot_probe.py check --fold cv4 --new_species all --k 10 --seeds 1 2 3 \
        && python fewshot_probe.py check --fold cv5 --new_species all --k 10 --seeds 1 2 3 \
        && python fewshot_probe.py check --fold cv1 --new_species all --k 10 --seeds 4 5 6; } \
        > "$QUEUE_DIR/check.log" 2>&1 || stop "dataset check failed, see $QUEUE_DIR/check.log"
    short=$(grep -c "<-- only" "$QUEUE_DIR/check.log")
    [ "$short" -eq 0 ] || log "NOTE  $short draws get fewer than k tiles (grep '<-- only' $QUEUE_DIR/check.log)"
    grep -E "^cv[245] .*candidate tiles" "$QUEUE_DIR/check.log" | cut -c 1-110 | while read -r line; do log "check: $line"; done

    # %% 0. smoke tests (GPU, minutes): calibration, fusion, a multi-flight draw, another fold
    SMOKE=$QUEUE_DIR/smoke_probes
    run_job smoke_calib "$PROBE_GIB" python fewshot_probe.py probe --backbone pc24 --fold cv3 \
        --new_species B_davidii S_inaequidens --k 10 --seeds 1 --calibrate --max_steps 20 --max_test_batches 12 \
        --method probe_pc24_cal --out $SMOKE/cv3_pc24_cal || stop "smoke_calib failed, see $QUEUE_DIR/smoke_calib.log"
    run_job smoke_fusion "$FUSION_GIB" python fewshot_probe.py probe --backbone fusion --fold cv3 \
        --new_species S_inaequidens --k 10 --seeds 1 --max_steps 20 --max_test_batches 12 \
        --method probe_fusion --out $SMOKE/cv3_fusion || stop "smoke_fusion failed, see $QUEUE_DIR/smoke_fusion.log"
    run_job smoke_flights "$PROBE_GIB" python fewshot_probe.py probe --backbone pc24 --fold cv5 \
        --new_species A_altissima --k 10 --seeds 2 --draw flights5 --max_steps 20 --max_test_batches 12 \
        --method probe_pc24 --out $SMOKE/cv5_pc24_flights5 || stop "smoke_flights failed, see $QUEUE_DIR/smoke_flights.log"
    { python fewshot_probe.py summary --root $SMOKE && python scripts/analyze_fewshot.py --root $SMOKE \
        && python scripts/plot_fewshot.py --roots $SMOKE --out $SMOKE/figures; } > "$QUEUE_DIR/smoke_report.log" 2>&1 \
        || stop "smoke summary / analysis / figures failed, see $QUEUE_DIR/smoke_report.log"
    log "smoke tests passed; fusion peak $(( $(cat "$QUEUE_DIR/smoke_fusion.peak") / 1024 )) GiB, $(grep -h 'scoring speed' "$QUEUE_DIR/smoke_fusion.log" | tail -n 1)"

    # %% 1. calibration of the existing cv3 k = 10 PlantCLEF heads (draws 1-3, 4-6)
    calibrate cv3_pc24_k10_cal cv3_pc24_k10 "1 2 3"
    calibrate cv3_pc24_k10_cal_s456 cv3_pc24_k10_s456 "4 5 6"

    # %% 2. PlantCLEF + MiT-b2 fusion probe
    probe_grid cv3_fusion_k10 fusion cv3 10 "1 2 3"

    # %% 3. labelling budget: 5 flights, 2 flights (10 tiles); one flight (30 tiles)
    probe_grid cv3_pc24_k10_flights5 pc24 cv3 10 "1 2 3" --draw flights5
    probe_grid cv3_pc24_k10_flights5_s456 pc24 cv3 10 "4 5 6" --draw flights5
    probe_grid cv3_pc24_k10_flights2 pc24 cv3 10 "1 2 3" --draw flights2
    probe_grid cv3_pc24_k10_flights2_s456 pc24 cv3 10 "4 5 6" --draw flights2
    probe_grid cv3_pc24_k30_flight pc24 cv3 30 "1 2 3" --draw flight
    probe_grid cv3_pc24_k30_flight_s456 pc24 cv3 30 "4 5 6" --draw flight

    # %% 4. more folds: cv2, cv4, cv5 (draws 1-3), cv1 draws 4-6
    for fold in cv2 cv4 cv5; do
        probe_grid ${fold}_pc24_k10 pc24 $fold 10 "1 2 3"
        probe_grid ${fold}_dinov2_k10 dinov2 $fold 10 "1 2 3"
    done
    probe_grid cv1_pc24_k10_s456 pc24 cv1 10 "4 5 6"
    probe_grid cv1_dinov2_k10_s456 dinov2 cv1 10 "4 5 6"
}

# ============================================================================
# lanes "vitbase" and "b2base": one base U-Net per held-out species
# ============================================================================

# base_exp <P|A> <species>
base_exp() {
    local model=Unet_vitb14_pc24
    [ "$1" = A ] && model=Unet_mit_b2
    echo "lightning_logs/${TS}_np_cv3_ho${CLS[$2]}_${model}_512_s42"
}
ft_exp() {  # <P|A> <species> <seed>
    local model=Unet_vitb14_pc24
    [ "$1" = A ] && model=Unet_mit_b2
    echo "${TS}_np_cv3_fs${CLS[$2]}_${model}_512_s$3_from$1_r3"
}

# train_base <P|A> <species>: the U-Net on the five other species (30 epochs, as base P / A
# of B. davidii). An error after "Training finished" (train.py's test pass) does not block.
train_base() {
    local tag=$1 sp=$2 model=model_neophytes_vit_pc24 need=$VIT_GIB exp name=base_$1_$2
    [ "$tag" = A ] && { model=model_neophytes_mit_b2; need=$B2_GIB; }
    exp=$(base_exp "$tag" "$sp")
    python fewshot_probe.py check --holdout_data neophytes_split_cv3_train_512_holdout_${SHORT[$sp]} --holdout_only \
        > "$QUEUE_DIR/check_holdout_$sp.log" 2>&1 || { log "NOTE  holdout check of $sp failed, see $QUEUE_DIR/check_holdout_$sp.log"; return 1; }
    run_job "$name" "$need" python train.py data=neophytes_split_cv3_train_512_holdout_${SHORT[$sp]} model=$model \
        model.max_epochs=30 model.check_val_every_n_epoch=3 pl_seed=42 exp_name=$(basename "$exp")
    if [ $? -ne 0 ]; then
        if grep -q "Training finished" "$QUEUE_DIR/$name.log" && [ -n "$(best_ckpt "$exp")" ]; then
            log "NOTE  $name: training finished, its test pass failed; going on with the best checkpoint"
            mark_done "$name"
        else
            log "NOTE  $name failed, see $QUEUE_DIR/$name.log; skipping $sp in this lane"
            return 1
        fi
    fi
    [ -n "$(best_ckpt "$exp")" ] || { log "NOTE  no checkpoint in $exp"; return 1; }
}

# probe_base <P|A> <species>: linear head on the encoder of the base (draws 1-3)
probe_base() {
    local tag=$1 sp=$2 backbone need=$PROBE_GIB
    backbone="vitckpt:$(best_ckpt "$(base_exp P "$sp")")"
    [ "$tag" = A ] && { backbone="ckpt:$(best_ckpt "$(base_exp A "$sp")")"; need=$B2_PROBE_GIB; }
    run_job "probe_${tag}_$sp" "$need" python fewshot_probe.py probe --backbone "$backbone" --fold cv3 \
        --new_species "$sp" --k 10 --seeds 1 2 3 --method "probe_$tag" --out "$OUT/cv3_probe_${tag}_$sp" \
        || log "NOTE  probe_${tag}_$sp failed, see $QUEUE_DIR/probe_${tag}_$sp.log"
    report
}

# combo <species> <base ckpt>: known species from base A, new species from the PlantCLEF
# heads of draws 1-6 (no training)
combo() {
    local sp=$1 ckpt=$2
    run_job "combo_A_pc24_$sp" "$B2_PROBE_GIB" python fewshot_probe.py combo --base_ckpt "$ckpt" --backbone pc24 \
        --heads_from "$OUT/cv3_pc24_k10" "$OUT/cv3_pc24_k10_s456" --fold cv3 --new_species "$sp" --k 10 \
        --seeds 1 2 3 4 5 6 --method combo_A_pc24 --out "$OUT/cv3_combo_A_pc24_$sp" \
        || log "NOTE  combo_A_pc24_$sp failed, see $QUEUE_DIR/combo_A_pc24_$sp.log"
    speed "combo_A_pc24_$sp"
    report
}

# finetune <P|A> <species>: the base fine-tuned on each draw (recipe r3: fresh channel of the
# new species, lr 3e-4, 30 epochs), then the three models in one test pass
finetune() {
    local tag=$1 sp=$2 model=model_neophytes_vit_pc24 need=$VIT_GIB eval_need=$VIT_EVAL_GIB init s exp
    local ckpts=() names=()
    [ "$tag" = A ] && { model=model_neophytes_mit_b2; need=$B2_GIB; eval_need=$B2_GIB; }
    init=$(best_ckpt "$(base_exp "$tag" "$sp")")
    [ -n "$init" ] || { log "SKIP  ft_${tag}_r3_$sp: no base checkpoint"; return; }
    for s in 1 2 3; do
        exp=$(ft_exp "$tag" "$sp" "$s")
        fs_args "$sp" "$s"
        run_job "ft_${tag}_r3_${sp}_s$s" "$need" python train.py "${FS_ARGS[@]}" model=$model "init_ckpt='$init'" \
            pl_seed=$s "${FT_ARGS[@]}" exp_name=$exp \
            && log "ft_${tag}_r3_${sp}_s$s: train_f1_class_${CLS[$sp]} = $(train_f1_new "$exp" "${CLS[$sp]}" 2>&1 | tail -n 1)" \
            || log "NOTE  ft_${tag}_r3_${sp}_s$s failed, see $QUEUE_DIR/ft_${tag}_r3_${sp}_s$s.log"
        [ -f "lightning_logs/$exp/checkpoints/final.ckpt" ] && { ckpts+=("lightning_logs/$exp/checkpoints/final.ckpt"); names+=("s$s"); }
    done
    [ ${#ckpts[@]} -gt 0 ] || { log "SKIP  eval_ft_${tag}_r3_$sp: no fine-tuned model"; return; }
    run_job "eval_ft_${tag}_r3_$sp" "$eval_need" python fewshot_probe.py eval --ckpts "${ckpts[@]}" \
        --names "${names[@]}" --method "ft_${tag}_r3" --new_species "$sp" --k 10 --fold cv3 \
        --out "$OUT/cv3_ft_${tag}_r3_$sp" || log "NOTE  eval_ft_${tag}_r3_$sp failed, see $QUEUE_DIR/eval_ft_${tag}_r3_$sp.log"
    report
}

# smoke_base <P|A>: fine-tuning with the per-species overrides, its test pass (minutes)
smoke_base() {
    local tag=$1 model=model_neophytes_vit_pc24 ckpt need=$VIT_GIB
    ckpt=$(best_ckpt $SMOKE_VIT_EXP)
    [ "$tag" = A ] && { model=model_neophytes_mit_b2; ckpt=$(best_ckpt $BASE_A_BDAVIDII); need=$B2_GIB; }
    [ -n "$ckpt" ] || stop "no checkpoint for the smoke tests of lane $LANE"
    fs_args A_altissima 1
    run_job smoke_ft_$tag "$need" python train.py "${FS_ARGS[@]}" model=$model "init_ckpt='$ckpt'" pl_seed=1 \
        "${FT_ARGS[@]}" model.max_epochs=2 exp_name=smoke_${TS}_np_cv3_fs6_ft_$tag \
        || stop "smoke_ft_$tag failed, see $QUEUE_DIR/smoke_ft_$tag.log"
    grep -q "Few-shot subset" "$QUEUE_DIR/smoke_ft_$tag.log" || stop "smoke_ft_$tag: no few-shot subset in the log"
    run_job smoke_eval_$tag "$need" python fewshot_probe.py eval \
        --ckpts lightning_logs/smoke_${TS}_np_cv3_fs6_ft_$tag/checkpoints/final.ckpt --names s1 \
        --method ft_${tag}_r3 --new_species A_altissima --k 10 --max_test_batches 5 \
        --out $QUEUE_DIR/smoke_$LANE/cv3_ft_$tag || stop "smoke_eval_$tag failed, see $QUEUE_DIR/smoke_eval_$tag.log"
}

lane_vitbase() {
    [ -f "$PC24_WEIGHTS" ] || stop "$PC24_WEIGHTS missing; extract it with scripts/extract_pc24_backbone.py"
    smoke_base P
    log "smoke tests passed; species: $HO_SPECIES"
    # %% 1. base P and probe_P, species by species
    for sp in $HO_SPECIES; do
        train_base P "$sp" && probe_base P "$sp"
    done
    # %% 2. fine-tuning of every base P
    for sp in $HO_SPECIES; do finetune P "$sp"; done
}

lane_b2base() {
    [ -f "$PC24_WEIGHTS" ] || stop "$PC24_WEIGHTS missing; extract it with scripts/extract_pc24_backbone.py"
    smoke_base A
    run_job smoke_combo "$B2_PROBE_GIB" python fewshot_probe.py combo --base_ckpt "$(best_ckpt $BASE_A_BDAVIDII)" \
        --backbone pc24 --heads_from "$OUT/cv3_pc24_k10" --fold cv3 --new_species B_davidii --k 10 --seeds 1 \
        --max_test_batches 12 --method combo_A_pc24 --out $QUEUE_DIR/smoke_b2base/cv3_combo \
        || stop "smoke_combo failed, see $QUEUE_DIR/smoke_combo.log"
    log "smoke tests passed; combo peak $(( $(cat "$QUEUE_DIR/smoke_combo.peak") / 1024 )) GiB; species: $HO_SPECIES"

    # %% 0. combined model for B. davidii (October base A)
    combo B_davidii "$(best_ckpt $BASE_A_BDAVIDII)"
    # %% 1. base A, probe_A, combined model, species by species
    for sp in $HO_SPECIES; do
        train_base A "$sp" && probe_base A "$sp" && combo "$sp" "$(best_ckpt "$(base_exp A "$sp")")"
    done
    # %% 2. fine-tuning of every base A
    for sp in $HO_SPECIES; do finetune A "$sp"; done
}

for sp in $HO_SPECIES; do [ -n "${CLS[$sp]:-}" ] || stop "unknown species $sp; choose from $ALL_SPECIES"; done
log "lane start (TS=$TS)"
lane_$LANE
report
log "lane finished"
