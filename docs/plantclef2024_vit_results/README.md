# PlantCLEF 2024 ViT as the segmentation encoder: results (Oct 22)

Results of [queue_2026-10-20_vit_pc24.sh](../../scripts/queues/queue_2026-10-20_vit_pc24.sh) (plan: [plantclef2024_vit_notes_en.md](../plantclef2024_vit_notes_en.md), section 8). U-Net at 512 crops, 30 epochs, only the encoder differs. Test F1 = `F1-avg-wo0`, best checkpoint.

| Encoder | cv3 s42 | cv3 s43 | cv1 s42 | cv1 s43 |
| --- | --- | --- | --- | --- |
| MiT-b2 (baseline) | 0.666 | 0.623 | 0.741 | 0.703 |
| PlantCLEF ViT-B/14 | 0.529 | 0.540 | 0.679 | 0.698 |
| DINOv2 ViT-B/14 (control) | 0.503 | – | 0.677 | – |

- **PlantCLEF vs b2:** mean paired ΔF1 −0.072 over the 4 pairs; A. altissima −0.187. The adoption rule fails, so MiT-b2 stays the encoder.
- **PlantCLEF vs DINOv2 (s42):** +0.026 on cv3, +0.002 on cv1, within b2's seed noise (~0.04).
- **Frozen features:** at the last validation with a frozen ViT (epoch 5), PlantCLEF reaches 0.63–0.68 and DINOv2 0.46–0.47. Fine-tuning closes this gap.
- **Val vs test:** PlantCLEF nearly matches b2 on validation (best 0.79–0.81 vs 0.81–0.82) but drop more on the test sites.
- **Error type:** almost all ViT errors are species pixels predicted as background (lower recall). Confusion between species is near zero for all three encoders.
- Each training run took 6.2–7.5 h and peaked at 8–9 GiB of GPU memory.

## Figures

- `1_test_f1_per_species.png`: test F1 per species (bars = mean, markers = single runs)
- `2_val_f1_curves.png`: validation species F1 per epoch; the ViT is frozen in epochs 0–5
- `3_missed_as_background.png`: share of each species' test pixels predicted as background, all runs pooled

## F1 NaN rule (`metrics_from_confusion`)

With the proposed fix (NaN only if a class is absent from both reference and prediction, instead of absent from the reference), the **test F1 above does not change**: both test folds contain all six species. The per-site breakdown does change (mean over sites, `summary.csv`): b2 0.624 → 0.274, PlantCLEF 0.475 → 0.200. b2 drops more because it predicts more absent species. Per-month metrics change as well.

Reproduce (CPU, from the repo root): `python scripts/plot_vit_512.py`
