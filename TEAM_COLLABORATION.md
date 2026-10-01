# Team Collaboration Plan — CA6127 Rare-Defect Generation

## 1. Project Goal

Build a reproducible technical-review project that evaluates whether synthetic rare-defect data is useful for industrial inspection under data scarcity.

Core comparison:

```text
Real only
vs
Real + Procedural Synthesis
vs
Real + GAN
vs
Real + Diffusion
```

Primary questions:

1. How do procedural, GAN, and diffusion methods differ in realism, diversity, controllability, and cost?
2. Does synthetic data improve performance on held-out real defects?
3. When does a more expensive generative model provide enough downstream utility to justify its cost?

Robot-view robustness is an optional extension only after the four core experiment chains are complete.

---

## 2. Team Structure

The team has three members.

The work is intentionally **not** split as "one person = one model." The project needs a shared experimental standard, so Member A acts as both integration owner and quality gatekeeper.

### Member A — Data, Evaluation, Integration, Quality Owner

**Owns**
- dataset audit;
- leakage-safe few-shot split;
- split freeze and version control;
- downstream inspection model;
- evaluator;
- common experiment runner;
- common JSON result schema;
- generator interface contract;
- contract tests;
- Phase Gate acceptance.

**Does not own**
- procedural synthesis;
- GAN implementation;
- diffusion implementation;
- routine code review of every line written by B/C.

**Implementation principle**

A should reuse a stable segmentation implementation, e.g. a library U-Net, instead of building the downstream model from scratch. A's scarce time is reserved for experiment integrity and integration quality.

**Primary deliverables**
- `scripts/audit_mvtec.py`
- frozen split manifests under `data/splits/`
- downstream training/evaluation entry point
- common runner
- result JSON schema
- generator contract test
- Phase 1 acceptance record

---

### Member B — Procedural Synthesis + GAN

**Owns**
- procedural synthetic anomaly baseline (CutPaste/DRAEM-style);
- GAN reproduction and integration;
- GAN generation experiments;
- GAN hyperparameter analysis;
- GAN failure analysis.

**Support obligation**

Once the GAN path is stable, B should help C reproduce or debug the diffusion path if C is blocked.

**Primary deliverables**
- procedural generator adapter;
- GAN generator adapter;
- generated image/mask samples;
- GAN experiment configs;
- GAN hyperparameter results;
- GAN failure-case gallery.

---

### Member C — Diffusion + Cost Analysis

**Owns**
- AnomalyDiffusion feasibility check;
- environment and GPU requirement audit;
- reproduction/integration;
- sampling-step analysis;
- generation latency / compute-cost analysis;
- diffusion failure analysis.

**Primary deliverables**
- AnomalyDiffusion environment notes;
- diffusion generator adapter;
- generated image/mask samples;
- sampling-step experiment;
- latency/cost results;
- diffusion failure cases.

---

## 3. Skill-Based Role Swap Rule

Before implementation starts, compare B and C on:

- PyTorch debugging ability;
- CUDA/environment troubleshooting;
- experience reproducing research repositories;
- available GPU memory and runtime access.

If B is materially stronger than C, swap the method ownership:

```text
B -> Diffusion
C -> Procedural + GAN
```

The rest of the collaboration model stays unchanged.

Reason: diffusion is the highest-risk reproduction path and should be owned by the member best able to handle environment and GPU issues.

A remains Data/Evaluation/Integration/Quality Owner regardless of the B/C swap.

---

## 4. Interface Contract

A defines the generator contract before B/C integrate their methods.

Every generator must expose an equivalent interface:

```python
generate(
    normal_images,
    num_samples,
    defect_type=None,
    seed=42,
    **kwargs
) -> GeneratedBatch
```

Minimum output contract:

```text
GeneratedBatch
├── images
├── masks
└── metadata
```

Each generated sample must include metadata equivalent to:

```json
{
  "method": "gan | diffusion | procedural",
  "category": "metal_nut",
  "defect_type": "scratch",
  "seed": 42,
  "source_image_id": "...",
  "generation_config": {},
  "generation_seconds": 0.0
}
```

B/C are free to use different upstream repositories internally, but they must normalize outputs to the same contract.

---

## 5. Contract Test

A maintains a short contract test.

A generator is **not accepted** merely because it can display an image in a notebook.

Minimum checks:

1. accepts the agreed function arguments;
2. returns the requested number of samples;
3. image and mask counts match;
4. image/mask spatial dimensions are valid;
5. masks are binary or consistently normalized;
6. metadata exists for every sample;
7. outputs are reproducible under a fixed seed where the method supports determinism;
8. outputs can be consumed directly by the downstream evaluator.

If the contract test fails, the method is not considered integrated.

---

## 6. Data Integrity and Split Freeze

The split is the most important experiment-integrity artifact.

### Rules

- support and held-out real defects must be disjoint;
- the same support split is used by procedural, GAN, and diffusion methods;
- held-out real defects are never used to fit a generator or downstream training condition;
- generated data never enters the final real test set;
- all experiments use the same frozen split manifest.

### Freeze process

At the end of Phase 0:

1. A generates the split manifest;
2. the team reviews counts and class/defect-type coverage;
3. the file is committed to the repository;
4. it is declared **frozen**.

Any later split change requires A's approval and a documented reason.

Do not silently regenerate splits during experimentation.

---

## 7. Common Result Schema

All experiments must save machine-readable results.

Minimum example:

```json
{
  "experiment_id": "E3_metal_nut_diffusion_real10_syn100_seed42",
  "category": "metal_nut",
  "method": "diffusion",
  "real_train_count": 10,
  "synthetic_count": 100,
  "seed": 42,
  "generator": {
    "sampling_steps": 50,
    "generation_seconds_per_image": 0.0
  },
  "inspection": {
    "pixel_auroc": 0.0,
    "iou": 0.0,
    "dice": 0.0,
    "defect_recall": 0.0
  }
}
```

No final result should exist only inside a notebook cell.

---

## 8. Phase Plan and Acceptance Gates

### Phase 0 — Protocol and Feasibility Lock

**A**
- audit MVTec subset;
- define split protocol;
- freeze split;
- define generator interface;
- define JSON schema;
- implement dummy generator;
- run dummy data through the downstream pipeline end-to-end.

**B**
- inspect procedural baseline implementation path;
- inspect GAN upstream repository;
- verify environment and expected outputs.

**C**
- inspect AnomalyDiffusion upstream repository;
- record CUDA/PyTorch requirements;
- record expected GPU memory/runtime;
- identify checkpoint/fine-tuning path;
- identify likely blockers.

**Phase 0 outputs**
- frozen split;
- interface contract;
- contract test;
- dummy generator;
- downstream/evaluator pipeline smoke test;
- GAN feasibility note;
- diffusion feasibility note;
- fallback decision written down.

The source audit and per-method environment-validation template are tracked in
[`docs/member_bc_feasibility.md`](docs/member_bc_feasibility.md). Keep the
executor-only environment, GPU, VRAM, runtime, and reproduction fields TBD until
the assigned member records an actual run.

**Gate 0**
A confirms that a dummy generated sample can pass:

```text
generator -> standardized output -> downstream loader -> evaluator -> JSON result
```

Do not wait for GAN or diffusion before validating this path.

---

### Phase 1 — Method Reproduction and Integration

**B acceptance target**
- procedural method generates valid samples;
- GAN generates 20–50 reviewable samples with masks/metadata;
- both pass the contract test.

**C acceptance target**
- diffusion generates 20–50 reviewable samples with masks/metadata;
- passes the contract test.

**A acceptance target**
- visually audits sample/mask pairs;
- verifies metadata;
- verifies no split leakage;
- confirms downstream evaluator can consume all methods.

**Gate 1**
A decides whether each method is accepted.

"Runs without error" is not sufficient.

Minimum acceptance evidence:
- 20–50 generated samples;
- corresponding masks;
- metadata;
- contract-test pass;
- manual sample audit by A.

---

### Diffusion Fallback Rule

Diffusion is the highest-risk component.

During Phase 0, C must already define a fallback path.

If no usable diffusion samples exist by the end of Phase 1, A triggers the agreed fallback rather than extending the deadline indefinitely.

Fallback priority:

1. official released checkpoint/inference path;
2. lower-resolution or reduced-category fine-tuning;
3. reduced sample count for the technical comparison;
4. alternative reproducible diffusion-based industrial anomaly generator.

The fallback must preserve the research question: compare a diffusion-based generative method against procedural/GAN alternatives.

---

### Phase 2 — Core Experiment Matrix

Core chains:

```text
E0 Real only
E1 Real + Procedural
E2 Real + GAN
E3 Real + Diffusion
```

Run on the agreed MVTec categories using the same frozen support/test protocol.

Core outputs:
- downstream Recall;
- Dice/F1;
- IoU;
- AUROC where appropriate;
- generation runtime;
- qualitative image/mask grids.

**Gate 2**
All four chains must run end-to-end and save standardized results.

Robot-view robustness is still out of scope at this point.

---

### Phase 3 — Analysis

**A**
- synthetic-ratio analysis;
- cross-method downstream utility comparison;
- data-integrity/reproducibility check.

**B**
- GAN hyperparameter analysis;
- procedural vs GAN comparison;
- GAN failure taxonomy.

**C**
- diffusion sampling-step analysis;
- quality/cost trade-off;
- diffusion failure taxonomy.

**Shared**
- quality vs downstream utility;
- cost vs utility;
- failure-case comparison;
- discussion of when simple procedural synthesis is sufficient.

Recommended figures:
- Real / Procedural / GAN / Diffusion sample grid;
- image + mask overlays;
- synthetic-ratio vs Recall/F1;
- generation cost vs downstream utility;
- quality vs downstream utility;
- failure-case gallery.

---

### Phase 4 — Optional Extension

Only start after Gate 2 is complete.

Possible extension:
- rotation;
- perspective;
- scale;
- brightness/contrast;
- partial occlusion.

Purpose:
- test robot-view-like robustness;
- connect the course project to future industrial/robotics commercialization.

Ownership:
- whoever has capacity after the core project is stable.

Do not pre-assign this extension.

Do not let it block the report.

---

## 9. Quality Assurance Mechanism

A should control quality through mechanisms, not continuous code supervision.

### QA-1 — Contract

B/C must integrate through the shared generator contract.

### QA-2 — Frozen split

The experiment split is immutable after Phase 0 unless A approves a documented change.

### QA-3 — Acceptance criteria

Every deliverable has observable evidence. "It runs" is never a sufficient acceptance statement.

### QA-4 — Dummy-first integration

A validates the entire evaluation pipeline using a dummy generator in Week 1.

This prevents A from becoming an integration bottleneck after B/C finish their work.

### QA-5 — Reproducibility

Final experiments must record:
- random seed;
- upstream repository commit;
- PyTorch/CUDA versions;
- generation configuration;
- training configuration;
- runtime;
- result JSON.

---

## 10. Pull Request and Collaboration Rules

Recommended workflow:

```text
main
├── feature/data-eval
├── feature/procedural-gan
└── feature/diffusion
```

Rules:
- do not commit datasets, checkpoints, or generated bulk images;
- keep large outputs outside Git;
- each PR must state how to reproduce the result;
- generated metrics must follow the shared schema;
- B/C should not modify frozen split files directly;
- changes to shared interfaces require A review;
- method-internal implementation changes do not require A to inspect every line if the contract and tests pass.

---

## 11. Workload Balance

Expected relative workload after this split:

```text
A: high coordination + medium implementation
B: procedural + GAN + diffusion support
C: high-risk diffusion + cost analysis
```

This is intentionally more balanced than assigning A both the evaluation platform and a full synthesis baseline.

If diffusion reproduction becomes substantially heavier than expected, B supports C after the GAN path is stable.

If B/C skill levels strongly differ, apply the role-swap rule before Phase 1.

---

## 12. Definition of Done

The project is core-complete when:

- [ ] dataset audit is reproducible;
- [ ] split is frozen and leakage-safe;
- [ ] dummy generator passes the full pipeline;
- [ ] procedural generator passes contract tests;
- [ ] GAN generator passes contract tests;
- [ ] diffusion generator passes contract tests;
- [ ] all four core experiment chains save standardized JSON results;
- [ ] held-out real defects are used for final evaluation;
- [ ] at least one GAN hyperparameter analysis is complete;
- [ ] at least one diffusion sampling/cost analysis is complete;
- [ ] synthetic-ratio analysis is complete;
- [ ] quantitative and qualitative comparisons exist;
- [ ] failure cases are categorized;
- [ ] README contains reproducible commands;
- [ ] final code package excludes datasets and trained weights.

Optional only:
- [ ] robot-view robustness;
- [ ] simulator/demo.

---

## 13. Immediate Next Actions

### A
1. finalize split semantics;
2. implement dummy generator;
3. finalize generator interface and contract test;
4. make the downstream U-Net pipeline run end-to-end;
5. publish the result schema.

### B
1. reproduce procedural baseline;
2. reproduce GAN inference/training path;
3. normalize both methods to the shared interface.

### C
1. audit AnomalyDiffusion requirements;
2. record GPU/CUDA/PyTorch constraints;
3. test official inference/fine-tuning path;
4. produce the first diffusion sample as early as possible.

### Team checkpoint
Before large-scale experimentation, verify:

```text
same split
same downstream model
same evaluator
same result schema
different generation method
```

That is the experimental contract for the entire project.
