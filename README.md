# transfomer-robot-defect-gen

CA6127 group project: **Rare-Defect Data Generation under Data Scarcity for Industrial Inspection**.

## Core research question
Can synthetic rare-defect data improve industrial inspection when real defect data is scarce, and when do more expensive generative methods justify their cost?

## Core pipeline
```text
MVTec AD
-> frozen few-shot real-defect support split
-> Real only / Procedural / GAN / Diffusion
-> fixed downstream inspection model
-> held-out real-defect evaluation
-> quality / utility / cost analysis
-> optional robot-view robustness
```

## Core comparison
1. Real only
2. Real + Procedural synthesis
3. Real + GAN
4. Real + Diffusion

## Data-integrity rule
MVTec AD standard training data are normal-only. Real defects are sampled from the official defect pool into a small **support** set, while all remaining real defects are locked as **held-out evaluation**. No support image may appear in held-out evaluation.

The split is frozen after Phase 0 and shared by all synthesis methods.

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
2. Frozen few-shot support / held-out real-defect split
3. Procedural synthetic-anomaly baseline
4. GAN-based defect synthesis
5. Diffusion-based defect synthesis
6. Fixed downstream inspection model
7. Unified evaluator and JSON result schema
8. Synthetic-ratio, hyperparameter, quality, utility, and cost analysis

Optional extension:
- robot-view proxy robustness

Do not expand into robot control, full VLA, humanoid foundation-model training, or simulator work before the four core experiment chains are complete.

## Team collaboration
See [TEAM_COLLABORATION.md](TEAM_COLLABORATION.md) for:
- three-person role ownership;
- skill-based B/C role swap rule;
- generator interface contract;
- frozen-split governance;
- phase gates and acceptance criteria;
- diffusion fallback rule;
- collaboration and QA workflow.
