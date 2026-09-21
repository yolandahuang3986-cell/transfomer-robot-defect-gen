# transfomer-robot-defect-gen

CA6127 group project: **Robot-View Rare-Defect Generation for Embodied Industrial Inspection**.

## Core research question
Can synthetic rare-defect images improve robotic visual inspection when real defect data is scarce?

## Core pipeline
```text
MVTec AD
-> few-shot real-defect support split
-> GAN vs Diffusion defect generation
-> fixed downstream inspection model
-> held-out real-defect evaluation
-> robot-view robustness
-> optional ManiSkill inspect-and-sort demo
```

## Data-integrity rule
MVTec AD standard training data are normal-only. Real defects are sampled from the official defect pool into a small **support** set, while all remaining real defects are locked as **held-out evaluation**. No support image may appear in held-out evaluation.

## MVP categories
- metal_nut
- screw

Optional:
- transistor

## First commands
```bash
pip install -r requirements.txt

python scripts/audit_mvtec.py \
  --root data/raw/mvtec_ad \
  --categories metal_nut screw \
  --out results/metrics/dataset_audit.csv

python scripts/build_fewshot_splits.py \
  --root data/raw/mvtec_ad \
  --categories metal_nut screw \
  --n-support 5 \
  --seed 42 \
  --out-dir data/splits
```

## Scope lock
Core:
1. MVTec AD subset
2. GAN-based defect synthesis
3. Diffusion-based defect synthesis
4. Fixed downstream inspection model
5. Real held-out evaluation
6. Data-scarcity and synthetic-ratio ablations

Extension:
- robot-view proxy robustness
- optional ManiSkill inspect-and-sort demo

Do not expand into full VLA / humanoid foundation-model training before the core experiment matrix is complete.
