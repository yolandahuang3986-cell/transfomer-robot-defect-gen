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

## Current status — 6 October 2026

- **Member A Phase 0:** complete. The `5-shot per defect type, seed 42`
  manifest and freeze attestation are tracked under `data/splits/`; the
  evaluator, result schema, dummy contract, and Gate 0 runner are in place.
- **Shared synthetic QC and comparison scope:** implemented on the current PR
  branch. All methods use the same QC thresholds. The core comparison excludes
  `metal_nut/flip`; the frozen split itself is unchanged. `color` remains in
  scope pending a support-mask compatibility audit before reviewing generated
  outputs.
- **Member B:** Procedural and GAN adapters/results are not yet present in this
  repository; record feasibility only after the assigned member runs them.
- **Member C:** the latest team update reported all nine support-trained mask
  models complete and the anomaly model still training. Sample generation,
  shared-QC results, and the project adapter remain to be added here.
- **Next project stage:** finish B/C adapters and contract checks, run the
  real-only baseline, then run the matched synthetic-data comparisons and
  quality/cost analyses.

Dataset images and weights stay outside Git. The frozen split manifest is the
approved protocol artifact; do not recreate or overwrite it for routine runs.

## Reproduce Phase 0

Install the declared project dependencies in the environment used for testing:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

To audit a mounted dataset and run Gate 0 on TC2, use the commands below.
Dataset archives and expanded images belong under `~/datasets/mvtec_ad`, never
inside this repository.

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

## TC2 command reference

Run on TC2 from the repository root using the isolated `robot-phase0` environment.
Dataset archives and expanded images belong under `~/datasets/mvtec_ad` and are
never copied into this repository.

```bash
~/envs/robot-phase0/bin/python -m pytest -q
~/envs/robot-phase0/bin/python scripts/audit_mvtec.py --root ~/datasets/mvtec_ad --categories metal_nut screw --out ~/datasets/mvtec_ad/dataset_audit.csv
srun --partition=MGPU-TC2 --qos=normal --gres=gpu:nvidia:1 --cpus-per-task=2 --mem=16G --time=00:10:00 ~/envs/robot-phase0/bin/python scripts/run_gate0.py --device cuda --root ~/datasets/mvtec_ad --manifest data/splits/fewshot_n5_seed42.json --category metal_nut --out ~/datasets/mvtec_ad/gate0_metal_nut.json
```

The approved split is frozen and tracked at
`data/splits/fewshot_n5_seed42.json`; its review record is
`data/splits/fewshot_n5_seed42.freeze.json`. Its protocol metadata is
`5-shot per defect type, seed 42`. The dummy generator makes constant fixture
patches solely for the Gate 0 smoke test. Gate 0 performs one optimization step
to prove the loader, U-Net, evaluator, and JSON contract; its metrics are not a
benchmark.

Split creation refuses to overwrite an existing manifest. Any future protocol
change requires a new manifest filename and an explicit review record.

The full real-only baseline is a project experiment after the Phase 0 smoke
test. The trainer checks the freeze attestation and writes model weights only
to the ignored checkpoint path:

```bash
srun --partition=MGPU-TC2 --qos=normal --gres=gpu:nvidia:1 --cpus-per-task=8 --mem=28G --time=06:00:00 \
  ~/envs/robot-phase0/bin/python scripts/train_inspector.py \
  --root ~/datasets/mvtec_ad --manifest data/splits/fewshot_n5_seed42.json \
  --category metal_nut --method real_only --epochs 50 --batch-size 8 \
  --image-size 256 --device cuda \
  --out results/metrics/real_only_metal_nut_seed42.json \
  --checkpoint checkpoints/real_only_metal_nut_seed42.pt
```
