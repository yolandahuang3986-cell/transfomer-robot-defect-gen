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

## Current status — 17 October 2026

- Phase 0 is complete; the frozen split and Gate 0 are tracked.
- B's Procedural/GAN implementation is merged. The 75/125 Procedural candidate pools
  have been received and independently rechecked: all 200 pairs pass shared QC and
  mask/change alignment checks. Technical and Codex-assisted visual reviews are
  complete. Metal_nut's original 75-pair pool has been accepted, and all three E1
  runs completed on TC2 A40 (job 43178, exit 0). Screw remains on hold for
  shadow/background placement.
  See the [fixed review sheets and findings](docs/reviews/b_delivery_20261017/README.md).
  GAN has documented quality failures. C's validation-stage [PR #5](https://github.com/yolandahuang3986-cell/transfomer-robot-defect-gen/pull/5)
  remains under review; its formal delivery is pending.
- A's primary evaluation infrastructure supports accepted synthetic bundles, shared
  QC rechecks, equal optimization steps, three training seeds and result summaries.
  Code validation is not evidence of downstream improvement.
- Six E0 baseline runs completed on TC2 A40 (job 40765, exit 0).
  [E0 results and limitations](docs/member_a_delivery.md) and the
  [completed metal_nut E1 comparison](results/metrics/e1_metal_nut_20261017/README.md)
  are recorded. E1 increases mean pixel AUROC, while thresholded Dice decreases
  and normal-image FPR remains 1.0. The remaining comparisons await accepted pools.
- The approved [formal protocol](docs/formal_evaluation_protocol.md) excludes only
  flip and retains color. Follow the [execution runbook](docs/formal_evaluation_runbook.md)
  for artifact requirements, acceptance, training and reporting.

Full datasets, generated batches and weights stay outside Git. The eight fixed
review illustrations linked above include their own image attribution and license.
The frozen split manifest is the approved protocol artifact; do not recreate or
overwrite it for routine runs.

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

The formal real-only baseline uses the frozen optimization-step budget after
the Phase 0 smoke test. Run seeds 42/43/44 for both categories from a clean commit.
The trainer checks the freeze attestation and writes weights to an ignored path:

```bash
srun --partition=MGPU-TC2 --qos=normal --gres=gpu:nvidia:1 --cpus-per-task=8 --mem=28G --time=06:00:00 \
  ~/envs/robot-phase0/bin/python scripts/train_inspector.py \
  --root ~/datasets/mvtec_ad --manifest data/splits/fewshot_n5_seed42.json \
  --category metal_nut --method real_only --formal --seed 42 --batch-size 8 \
  --image-size 256 --device cuda \
  --out results/metrics/real_only_metal_nut_seed42.json \
  --checkpoint checkpoints/real_only_metal_nut_seed42.pt
```
