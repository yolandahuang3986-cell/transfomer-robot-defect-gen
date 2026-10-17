# Quality and cost ledger

## Procedural candidate pools

| Category | Attempts | Shared-QC passes | Pass rate | Formal decision | CPU generation (s) | CPU QC (s) | Generation + QC per QC-passing pair (s) |
|---|---:|---:|---:|---|---:|---:|---:|
| metal_nut | 83 | 75 | 90.36% | accepted_and_evaluated | 1.1789 | 0.3581 | 0.02049 |
| screw | 136 | 125 | 91.91% | on_hold_placement_quality | 2.3096 | 0.5145 | 0.02259 |

All eight metal_nut and eleven screw rejections were too-small masks. These are
reported CPU timings from B's generation environment, with rejected shared-QC
attempts included. They are not controlled GPU inference measurements. Normal
loading and resizing occur before the exporter's generation timer; serialization,
compression and review-sheet production occur afterwards. The measured generation
block is 1.6730 seconds for metal_nut and 2.9849 seconds for screw; its timing scope
is broader than the helper's generation-only field. Use the full-precision source
values in the JSON ledger rather than adding overlapping timer fields.

The generator has no learned fitting stage. That fact is recorded as inapplicable
fitting time, not a fabricated measured zero. A formal cost per accepted screw
pair is undefined because none of the batch is admitted to training. Its 125 QC
passes remain a candidate-pool statistic. No per-defect semantic coverage claim
is made for generic synthetic_patch.

## Downstream program runtimes

| Condition | Category | Steps per run | Mean program runtime (s) | Three-run total (s) |
|---|---|---:|---:|---:|
| real_only | metal_nut | 1500 | 80.79 | 242.36 |
| real_only | screw | 2200 | 120.05 | 360.16 |
| procedural | metal_nut | 1500 | 68.59 | 205.76 |

These timers start inside the trainer and include checks, dataset setup, training
and evaluation up to result assembly. They exclude Python import startup,
checkpoint writing and queue time. They are not pure GPU kernel timings. E0's
six-run Slurm allocation took 696 seconds; metal_nut E1's three-run allocation took
285 seconds. Neither duration includes upstream generation or the external
acceptance process, and their different scope prevents direct whole-method cost
comparison.

The E1 training-only probe measured 0.04701 seconds per update after five warmup
updates over twenty measured updates. Peak PyTorch allocated memory was
551,783,424 bytes (about 0.514 GiB). This is not total process VRAM and is not a
GAN/Diffusion generator measurement. Do not attribute differences in the program
timers to augmentation itself.

## Remaining method costs

GAN has no accepted formal pool. Its preview and bounded-ablation evidence is
recorded separately; formal accepted-pair cost, comparable fitting wall time and
a scoped peak-memory measurement remain unavailable. Diffusion's formal pool
and associated cost/provenance handoff are pending. The ledger leaves these
measurements null, with reasons. Existing development timings cannot substitute
for formal accounting or justify a controlled speed ranking.

[Full-precision ledger and source hashes](quality_cost_ledger_20261017.json).
