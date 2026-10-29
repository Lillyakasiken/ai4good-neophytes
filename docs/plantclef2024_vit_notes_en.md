# Notes on the PlantCLEF 2024 ViT (ViTD2PC24All) for Drone-Based Neophyte Identification

> Compiled: 2026-10-20
> Context: The project uses drone RGB imagery to locate six neophyte species, several of which may occur in the same image.
> Imaging conditions (selected plots): PhaseOne high-resolution RGB camera, flight altitude of 50 m, GSD of approximately 2–3 mm.

---

## 0. Summary

ViTD2PC24All is a **species classifier trained on photographs of individual plants**, and its weights are publicly available. The papers on "zero-shot", "multi-species", and "Bayesian prior" inference with this model all address the same question: how to apply it to images that contain several species, a setting it was never trained on.

At our GSD of 2–3 mm, a zero-shot test is worth running, but the model will still see roughly 5–12 times less detail than in the images it is familiar with. Its most practical value is therefore **as a feature-extraction backbone**, combined with a small set of our own annotations for a linear probe or fine-tuning.

---

## 1. The model

From a statistical perspective, the model consists of two parts:

- **Backbone (feature extractor):** A Vision Transformer that maps an image to a 768-dimensional vector (embedding). The backbone can be viewed as a learned, high-dimensional, nonlinear transformation, that is, a set of automatically constructed covariates.
- **Head (classifier):** A multinomial logistic regression (softmax) on the 768-dimensional embedding, which outputs probabilities for 7,806 species.

Origin:

- Architecture: ViT-B/14 with four register tokens, approximately 86 M parameters, a 518 × 518 input (37 × 37 patches of 14 pixels), and a 768-dimensional CLS embedding as output.
- Pre-training: the self-supervised DINOv2 method on LVD-142M (approximately 142 million images).
- Fine-tuning: the PlantCLEF 2024 training set, a subset of Pl@ntNet data that mainly covers the flora of south-western Europe and contains approximately 1.4 million single-plant photographs of 7,806 species.

The organizers provide two versions:

| Short name | Full name | Training | Statistical analogy |
|---|---|---|---|
| ViTD2PC24OC | `vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier` | Frozen backbone; only the head is trained | Fixed covariates; only the multinomial logit is fitted |
| ViTD2PC24All | `vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all` | Initialized from OC; backbone and head are fully fine-tuned (92 epochs, low learning rate of 8e-5) | The covariates themselves are also learned |

- On the single-plant test set, ViTD2PC24All reaches a top-1/top-5 accuracy of approximately 75.91%/92.82% (ViTD2PC24OC: 63.69%/83.88%).
- Most subsequent teams use the All version because its embeddings are richer and it classifies more accurately.
- Both models were trained with a cross-entropy loss (one species per image) and an exponential moving average (EMA) of the weights. Because the All version also fine-tunes the backbone, its embeddings are no longer aligned with those of the original self-supervised DINOv2.

---

## 2. Key terms

### 2.1 Task

In PlantCLEF 2024–2026, participants receive a high-resolution top-down photograph of a vegetation quadrat and must list every species it contains.

The core difficulty is the **distribution shift between training and testing**: the training data are single-label photographs of individual plants, whereas the test data are multi-label images of multi-species quadrats.

### 2.2 "Zero-shot"

Here, the term does **not** mean recognizing unseen species, as it does for CLIP-like models. Instead, it means that the single-plant classifier is **applied directly to quadrat images without any training on multi-species quadrat images**. In statistical terms, this is direct extrapolation under covariate shift.

### 2.3 "Multi-species": tiling

Because the classifier predicts only one species per input, the image is divided into a grid of tiles; each tile is classified separately, and the tile predictions are then aggregated (by union, top-K, or voting).

- DS@GT (second place in PlantCLEF 2025) uses a 4 × 4 grid, so that each tile matches the 518 × 518 network input.
- NEUON AI (second place in PlantCLEF 2024) uses an 8 × 8 grid and combines the predictions by Bayesian model averaging.
- Rationale: each small tile most likely contains only one or two species, which approximately reduces the task to a single-label problem.

### 2.4 "Bayesian prior adaptation": correcting prior probability shift

This step is a classical **correction for label shift (prior probability shift)**:

- The softmax outputs p_train(y | x), which implicitly contains the class frequencies of the training set, π_train(y); on Pl@ntNet, common species are photographed more often.
- The species frequencies in the test setting, π_test(y), differ from these training frequencies.
- Bayes' rule corrects for this difference:

  **p_test(y | x) ∝ p_train(y | x) · π_test(y) / π_train(y)**

DS@GT applies this correction in three steps:

1. Cluster the test images by visual similarity with PaCMAP and K-Means.
2. Filter by geolocation, setting the prior to zero for species that cannot occur in the region.
3. Aggregate the tile predictions by voting, and reweight them with the Bayesian prior of each cluster.

This pipeline achieves a macro F1 of 0.348 on the private leaderboard without any additional training.

**Expectations:** This score earned second place (the winner reached 0.365), and it was obtained on close-range top-down photographs of quadrats. Zero-shot inference therefore works, but it is far from solving the problem.

---

## 3. Available weights

- **Zenodo:** DOI `10.5281/zenodo.10848263` (PlantCLEF 2024 pretrained models); a HuggingFace version has also been reported.
- Training environment: timm 0.9.16, torch 2.2.1.
- Reference code: DS@GT's complete pipeline, including tiling and prior correction, is available at <https://github.com/dsgt-arc/plantclef-2025>.
- License: CC BY 4.0, which permits commercial use with attribution; confirm the terms on the Zenodo page before use.

Loading example (the file names follow the README in the Zenodo archive; each model subdirectory contains `model_best.pth.tar`, `args.yaml`, and `class_mapping.txt`, and the archive also includes the example script `basic_usage_pretrained_model.py`):

```python
import timm, torch

model = timm.create_model(
    "vit_base_patch14_reg4_dinov2.lvd142m",
    pretrained=False,
    num_classes=7806,
    checkpoint_path="model_best.pth.tar",
)
model.eval()

# To obtain the 768-dimensional embedding instead of species probabilities:
feats = model.forward_head(model.forward_features(x), pre_logits=True)
```

---

## 4. Scale gap between our imagery and the model's training data

| Data | Approximate resolution |
|---|---|
| Single-plant training images (close-up phone photographs of leaves or flowers) | Far finer than 1 mm/pixel |
| PlantCLEF quadrat test images (quadrat of approximately 0.5 m × 0.5 m; on the order of 3,000 pixels per side) | Approximately 0.2–0.3 mm/pixel |
| PlantCLEF 4 × 4 tile resized to 518 pixels | Approximately 0.25–0.4 mm/pixel |
| **Our imagery (PhaseOne, 50 m)** | **2–3 mm/pixel**; one native 518 × 518 tile covers approximately 1.0–1.6 m on the ground |

*Note:* The PlantCLEF values are estimates. The exact resolution depends on how much ground each photograph covers beyond the 0.5 m quadrat frame; a 4 × 4 tile of a 3,000-pixel image (750 pixels) is downsampled to 518 pixels, which coarsens its resolution by a factor of approximately 1.45.

Key points:

- The model sees roughly 5–12 times less detail than in a PlantCLEF tile.
- A 5 cm leaf spans approximately 20 pixels, that is, only 1–2 ViT patches (14 pixels each). The model can therefore resolve mainly canopy structure, leaf outlines, inflorescences, and color, but not fine details such as leaf venation or pubescence.
- Encouragingly, the PlantCLEF quadrat images are themselves **vertical top-down photographs**, and the model has been shown to work from this viewpoint. Our imagery therefore differs from these images mainly in scale, not in viewpoint.
- For comparison, conventional drone surveys (30–50 m altitude, 1–2 cm GSD) render leaf texture essentially invisible, and zero-shot performance is expected to be poor; our 2–3 mm GSD is roughly 3–10 times finer.

---

## 5. Expected feasibility by species type

(The six target species are *A. altissima*, *R. typhina*, *B. davidii*, *R. japonica*, *B. orientalis* and *S. inaequidens*; the grouping below also lists other common Central European neophytes as examples.)

- **Large-leaved, tall, stand-forming species** (e.g., *Reynoutria*, *Heracleum mantegazzianum*, *Ailanthus*): their leaves and umbels span tens to more than a hundred pixels, so these species are relatively promising.
- **Species recognized by inflorescence or color** (e.g., flowering *Solidago* and *Impatiens glandulifera*): a single flower covers only a few to a dozen or so pixels, but dense stands show a distinctive color and inflorescence shape. **Phenology determines success**, so flights should be scheduled during the flowering period.
- **Small herbs** (e.g., *Senecio inaequidens*, *Ambrosia artemisiifolia*): these species occupy very few pixels within mixed vegetation, so zero-shot identification is difficult, and training on our own annotations will most likely be necessary.

---

## 6. Recommended methods

### 6.1 Approaches (in increasing order of effort)

1. **Zero-shot pilot:** Select drone images known to contain target species, tile them, run the model, and inspect the probabilities and ranks of the target species.
2. **Linear probe (recommended main approach):** Annotate a set of tiles (presence or absence of each of the six species), extract the 768-dimensional embeddings, and fit one logistic regression per species (multi-label, i.e., six independent sigmoids rather than one softmax). A few hundred to a few thousand annotated tiles usually yield good results.
3. **Fine-tuning:** If the linear probe is insufficient, fine-tune the backbone (fully or with LoRA). Because this backbone already encodes plant features, it requires fewer annotations than a general-purpose model would.

### 6.2 Define tiles by physical size, then upsample them to 518 pixels

- Test several ground sizes, such as 25 cm, 50 cm, and 1 m (approximately 100, 200, and 400 pixels at a GSD of 2.5 mm), and upsample all tiles to 518 pixels before inference.
- Upsampling adds no information, but it brings the apparent size of leaves and flowers closer to what the model saw during training; ViT features are sensitive to scale.
- The optimal tile size may differ between species and can be determined experimentally.
- Cost: smaller tiles increase the computational load.

### 6.3 Degradation-curve experiment (assessing feasibility without annotated drone images)

1. Photograph the six species in the field with a phone, with a ruler or another reference object of known size in the frame.
2. Blur and downsample the photographs to equivalent GSDs of 1, 2, 3, and 5 mm.
3. Run the model, and record how the top-1/top-5 hit rate of each species changes with GSD.

This low-cost experiment yields a feasibility curve for each species. The downsampled photographs can later serve as scale-matched training data for fine-tuning.

### 6.4 Use the original single images, not only the orthomosaic

- Orthomosaicking resamples the images and introduces seamlines, both of which remove detail.
- Drone surveys usually have 60–80% image overlap, so each plant appears in several photographs from slightly different viewpoints. Merging tile predictions by geographic coordinates (e.g., by averaging logits) combines repeated measurements of the same object, similar to how the winning PlantCLEF 2024 team (Atlantic) merged repeated photographs of the same plot taken at different times.
- These observations are not independent, so the merged probabilities must be recalibrated.

### 6.5 Handling multiple species in one image

- Treat each tile as one observation, and classify it into the six species plus a background/other class; reassembling the tile predictions yields a distribution map.
- **Do not apply a softmax over the six classes alone:** such a softmax would force every tile into one of the neophyte classes and produce many false positives. Instead, keep an "other" class, or use independent sigmoids with thresholds. In the zero-shot stage, keep all 7,806 classes and use the summed probability of the remaining classes as "other".
- **Image-level presence:** Aggregate the tile probabilities with the maximum or with noisy-OR, that is, 1 − ∏(1 − p_ik). Because adjacent tiles are highly correlated, however, noisy-OR overestimates presence and requires calibration.
- **Cross-validation must use spatial blocks (spatial block CV):** random splits severely overestimate performance because of spatial autocorrelation.
- **Set thresholds by the cost of missed detections:** in the early detection of invasive species, a missed detection usually costs more than a false alarm, so the operating point on the PR curve can favor high recall, followed by manual review.
- **Habitat priors:** Known habitat preferences (riverbanks, railway margins, forest edges, etc.) can serve as prior layers and enter the prediction through the formula in §2.4.

### 6.6 Computational cost

- PhaseOne aerial cameras typically have 100–150 MP sensors (the dual-camera iXM-RS280F produces 280 MP images). With a 100 MP sensor (approximately 11,600 × 8,700 pixels) at a GSD of 2.5 mm, one photograph covers approximately 29 m × 22 m.
- Native 518-pixel tiles amount to approximately 370 tiles per photograph; 50 cm physical tiles (approximately 200 pixels, then upsampled) amount to approximately 2,500 tiles per photograph.
- Each tile requires a full ViT-B forward pass at 518 × 518 input. Large survey areas therefore require a capable GPU or an inexpensive screening step (an RGB vegetation index or a small model) that first removes clearly irrelevant tiles.

### 6.7 Choose the backbone empirically

ViTD2PC24All is optimized for close-range plant photographs; at our scale, the embeddings of the original DINOv2 (without PlantCLEF fine-tuning; timm `vit_base_patch14_reg4_dinov2.lvd142m` with `pretrained=True`) may perform better. The linear-probe stage should therefore include both backbones and compare their per-species PR curves under spatial block cross-validation.

---

## 7. Next steps

1. Check whether the six target species are among the model's 7,806 classes, using `class_mapping.txt` in the Zenodo archive, which maps output indices to the species IDs in `PlantCLEF2024singleplanttrainingdata.csv`. If a species is missing, the classification head is useless for it, but the backbone remains usable. *Done (Oct 20): all six species and Centaurea stoebe are in the head (output index: A. altissima 1595, R. typhina 4437, B. davidii 2737, R. japonica 4427, B. orientalis 2923, S. inaequidens 4670, C. stoebe 3182).*
2. Select a few drone images known to contain the target species, run zero-shot inference at several physical tile sizes, and inspect the ranks of the target species in the softmax output (keep all classes and do not renormalize over the six target classes).
3. In parallel, run the degradation-curve experiment with phone photographs.
4. Based on the results of steps 2 and 3, decide which species zero-shot inference can handle, which require annotated tiles for a linear probe, and which may require fine-tuning.

---

## 8. Experiment plan: the PlantCLEF ViT as the segmentation encoder (Oct 20)

Sections 1–7 treat the model as a tile classifier. We already have pixel labels, so the most direct test of its value is the one in PROJECT_PLAN Phase 2 (RQ1): **put the backbone into our segmentation model, keep everything else fixed, and measure the cross-site species F1.** Zero-shot tiling remains a stretch goal (Centaurea stoebe).

### 8.1 Questions

1. Does the PlantCLEF ViT (ViTD2PC24All) raise cross-site species F1 over the U-Net + MiT-b2 baseline?
2. Is any gain due to the **plant photos**, or only to a larger, DINOv2-pretrained ViT? This is answered by the same ViT with the original DINOv2 weights (`vit_base_patch14_reg4_dinov2.lvd142m`), whose only difference is the missing PlantCLEF fine-tuning.

### 8.2 Reference: U-Net + MiT-b2, 512 crops, cv3 and cv1 × s42 and s43

Test species F1 (`F1-avg-wo0`, best checkpoint) of the six baseline runs at 512 crops (30 epochs, validation every 3rd epoch, batch 8):

| Encoder | cv3 s42 | cv3 s43 | cv1 s42 | cv1 s43 | Train time |
| --- | --- | --- | --- | --- | --- |
| MiT-b2 (24 M encoder parameters) | 0.666 | 0.623 | 0.741 | 0.703 | 7.6–9 h |
| MiT-b4 (61 M) | 0.633 | 0.644 | – | – | 13.5 h |

We compare against **MiT-b2 on both folds and both seeds**:

- It is the project baseline that every RQ is measured against, and the only setting with the full development design of PROJECT_PLAN (2 folds × 2 seeds), so every ViT run has a paired b2 run on the same fold and seed.
- MiT-b4 is not the stronger baseline. On cv3 the paired mean is 0.639 against 0.644 for b2 (Δ −0.006; s42 −0.033, s43 +0.021), well within the seed noise of b2 (0.666 vs 0.623), at 1.8× the training time. b4 serves only as a capacity reference: if the ViT-B (86 M) beats b2 but b4 (61 M) does not, the gain is unlikely to be model size alone.
- The 1024 run (cv3 s42, 0.594) is not comparable; it has a single seed and a different crop size.

Seed noise is large (cv3: 0.666 vs 0.623), so a single run decides nothing; only the paired mean over the four (fold, seed) pairs counts (PROJECT_PLAN, evaluation rules 2–4).

### 8.3 Model

| Part | Baseline | PlantCLEF ViT |
| --- | --- | --- |
| Encoder | MiT-b2, ImageNet | ViT-B/14 + 4 registers, ViTD2PC24All EMA weights (backbone only, head dropped) |
| Encoder → decoder | 4 hierarchical stages, strides 4–32 | ViTDet simple feature pyramid on blocks 3, 6, 9, 12 → strides 4, 8, 16, 32 with the **same channels as MiT-b2** (64, 128, 320, 512) |
| Decoder, head | U-Net (256, 128, 64, 32, 16) | identical (same shapes, same initialisation) |
| Data, sampler, augmentation, Focal loss, 30 epochs, AdamW 1e-4, cosine | – | identical |
| Fine-tuning | all parameters at one LR | **LP-FT**: ViT frozen for 6 epochs, then everything with **layer-wise LR decay 0.75** (top block 7.5e-5, patch embedding 2.4e-6) |
| Validation / test | whole 2048 px tiles | whole 2048 px tiles (input padded to a multiple of 14, position embedding interpolated) |

- The pyramid adds 6.8 M parameters. The ViT plus pyramid has 93 M parameters, against 24 M for the MiT-b2 encoder.
- LP-FT and layer decay are the one deliberate change to the recipe. A ViT fine-tuned at full LR from a random decoder tends to overwrite its pretrained features. This change confounds the comparison with b2 slightly, but it does not touch the comparison between PlantCLEF and DINOv2, where both use the same recipe.
- The validation after epoch 6 is the last one with a frozen ViT. It thus gives the frozen-encoder screen of PROJECT_PLAN at no extra cost (validation sites only, so it ranks features but is not a result).
- Code: [models/vit_encoder.py](../models/vit_encoder.py) and the configs `model_neophytes_vit_pc24` / `model_neophytes_vit_dinov2`. The weights are extracted by [scripts/extract_pc24_backbone.py](../scripts/extract_pc24_backbone.py) to `/zfs/ai4good/groups/neophytes/zhixuan_li/pretrained/plantclef2024/vitd2pc24all_backbone_ema.pth`. The weights are under CC BY 4.0, and the paper must cite the Zenodo record.

### 8.4 Runs and order

Queue: [scripts/queues/queue_2026-10-20_vit_pc24.sh](../scripts/queues/queue_2026-10-20_vit_pc24.sh)

| # | Run | Pairs with | Why at this position |
| --- | --- | --- | --- |
| 0 | Smoke test (debug split, 1 frozen + 1 fine-tuned epoch) | – | Catches build, weight-loading and memory errors in minutes |
| 1 | PlantCLEF cv3 s42 | b2 cv3 s42 | First reading on the iteration fold |
| 2 | PlantCLEF cv1 s42 | b2 cv1 s42 | Second fold before a second seed: fold differences are larger than seed differences |
| 3 | DINOv2 cv3 s42 | PlantCLEF cv3 s42 | Control: does the gain come from the plant photos? |
| 4 | DINOv2 cv1 s42 | PlantCLEF cv1 s42 | |
| 5 | PlantCLEF cv3 s43 | b2 cv3 s43 | Completes 2 folds × 2 seeds, needed for the adoption rule |
| 6 | PlantCLEF cv1 s43 | b2 cv1 s43 | |
| 7 | `evaluate_cv.py` + paired table vs b2 | – | `results/cv/512_dev_*`, Δ per (fold, seed) in `status.txt` |

- **Cost:** about 10–15 h per training run plus 1.5–2 h of `test.py`. This is an estimate, not a measurement (PROJECT_PLAN budgets a ViT run at 1.5–2× a b2 run). The whole queue takes about 3.5 days, so started on Oct 20 it finishes around Oct 24, before the Oct 30 talk. Runs 1–2 land after about one day.
- **Memory:** not measured yet. The queue records the peak of every job. On an out-of-memory error, it retries once with gradient checkpointing.
- DINOv2 gets only s42. If PlantCLEF and DINOv2 differ by less than b2's seed noise on both folds, the s43 runs of DINOv2 come next. Otherwise the two s42 pairs already show the direction.

### 8.5 How we read the result

The adoption rule is fixed in advance (PROJECT_PLAN, rule 4). We keep the PlantCLEF encoder if the mean paired ΔF1 against b2 over the four pairs is positive and no species loses more than 3 F1 points. The win in at least 4 of 5 folds is checked later, in the 5-fold confirmation.

| PlantCLEF vs b2 | PlantCLEF vs DINOv2 | Conclusion |
| --- | --- | --- |
| better | better | Plant-specific pretraining helps (the RQ1 hypothesis); go on to the confirmation runs and to nDSM via FiLM on this encoder |
| better | ≈ | The gain comes from DINOv2 / the ViT, not from the plant photos; still adopt the encoder, but RQ1 is reported as "general SSL features suffice" |
| ≈ or worse | – | Negative result for RQ1 at our GSD (2–3 mm/px, see section 4); MiT stays the encoder (PROJECT_PLAN fallback of Nov 4) |

Per class, we look first at the two known failure modes of the baseline:

- **R. typhina precision (0.31 in the 1024 baseline run):** 61% of its predicted pixels are A. altissima. Both are compound-leaved trees, and the PlantCLEF head tells them apart in close-up photos.
- **S. inaequidens recall:** 93% of it is missed as background. This is the hardest case for a ViT, because one 14 px patch is about 3.5 cm on the ground.

A gain for R. typhina without a gain for Senecio would match the scale argument of section 4.

### 8.6 Risks

| Risk | Sign | Fallback |
| --- | --- | --- |
| The position embedding, trained on a 37 × 37 grid (512 crops), does not transfer to the 147 × 147 grid of a whole 2048 tile | Validation F1 far below training F1 from the first validation on | Sliding-window inference with 512 windows (`patch_2_img_size`); this is inference-only, but `test.py` currently forces whole tiles and needs a 2-line option |
| A patch of 14 px (3.5 cm) is too coarse for Senecio | Senecio F1 below b2 while other species gain | Upsample the input ×2 (4× the tokens); a separate ablation |
| Out of memory | Exit code and log of the queue | Gradient checkpointing (automatic), then batch 4 |
| Runs take longer than 15 h | Time per epoch in the log of run 1 | Drop the DINOv2 cv1 run, or validate every 5th epoch |

### 8.7 Not in this queue

- DINOv3 ViT-B/16 and the ImageNet-supervised ViT-B of PROJECT_PLAN Phase 2: these run only if PlantCLEF and DINOv2 differ, because they answer the same question with another pretraining source.
- nDSM: `fusion_mode=stem` already works with this encoder (FiLM on the pyramid) and runs after the encoder decision.
- The 5-fold × 3-seed confirmation and the zero-shot tiling of sections 2 and 6.

### 8.8 Commands

```bash
# once: extract the backbone (already done for the path in the config)
python scripts/extract_pc24_backbone.py \
    /zfs/ai4good/groups/neophytes/zhixuan_li/pretrained/plantclef2024/pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar \
    /zfs/ai4good/groups/neophytes/zhixuan_li/pretrained/plantclef2024/vitd2pc24all_backbone_ema.pth

# the queue (in tmux)
tmux new -s queue_vit
bash scripts/queues/queue_2026-10-20_vit_pc24.sh

# a single run by hand
python train.py data=neophytes_split_cv3_train_512 model=model_neophytes_vit_pc24 \
    model.max_epochs=30 model.check_val_every_n_epoch=3 pl_seed=42
```

---

## References

- Goëau H., Espitalier V., Bonnet P., Joly A. *Overview of PlantCLEF 2024: multi-species plant identification in vegetation plot images.* CLEF 2024 Working Notes. <https://arxiv.org/pdf/2509.15768>
- Martellucci G., Goëau H., Bonnet P., Vinatier F., Joly A. *Overview of PlantCLEF 2025: Multi-Species Plant Identification in Vegetation Quadrat Images.* CLEF 2025 Working Notes. <https://arxiv.org/pdf/2509.17602>
- Gustineli M., Miyaguchi A., Cheung A., Khattak D. *Tile-Based ViT Inference with Visual-Cluster Priors for Zero-Shot Multi-Species Plant Identification.* CLEF 2025. <https://arxiv.org/pdf/2507.06093> · Code: <https://github.com/dsgt-arc/plantclef-2025>
- Erten A., Gustineli M., Cheung A. *Multi-Scale ViT Inference with Habitat-Fit Priors and kNN Retrieval for Multi-Species Plant Identification.* CLEF 2026 (third place in PlantCLEF 2026). <https://arxiv.org/pdf/2607.14509>
- Dourado Filho L. A., da Silva Neto A. M., David R. P., Calumby R. T. *Zero-Shot Segmentation through Prototype-Guidance for Multi-Label Plant Species Identification.* CLEF 2025 (fifth place in PlantCLEF 2025). <https://arxiv.org/pdf/2512.19957>
- PlantCLEF 2024 pretrained models (Zenodo). <https://doi.org/10.5281/zenodo.10848263>
- PlantCLEF 2024 task page. <https://www.imageclef.org/PlantCLEF2024>
