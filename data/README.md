# Data

Do not commit datasets.

Expected MVTec AD layout:

```text
data/raw/mvtec_ad/<category>/
  train/good/*.png
  test/<defect_type>/*.png
  ground_truth/<defect_type>/*_mask.png
```

Core categories:
- metal_nut
- screw

Few-shot protocol:
1. Sample a small support subset from real defect images, stratified by defect type.
2. Lock every remaining real defect as held-out evaluation.
3. Save split manifests for reproducibility.
4. Never evaluate on any image used as support.
