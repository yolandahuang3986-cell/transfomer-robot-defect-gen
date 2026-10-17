# GAN failure evidence and bounded ablation

The received B package contains generation diagnostics rather than an accepted
formal GAN pool. These records support `not_run_quality_gate` for E2; they do not
supply a zero downstream score or a successful GAN comparison.

## Original 300-epoch previews

Five fixed normal training sources were used per defect type. The in-scope
preview sets contain 15 metal_nut and 25 screw pairs. B's shared-QC report records
5/5 passing for metal_nut scratch and color and 0/5 for bent. All five screw types
have 0/5 passing. The screw probability-area report contains 23 empty masks out
of 25 at threshold 0.5; the two nonempty masks have only 2 and 8 pixels.

These are small, existing preview sets, not complete generation attempts under
the 10x budget. Some metal_nut outputs pass geometric QC yet show deformation or
poor correspondence. Do not equate these counts with whole-pool acceptance or
extrapolate them to every normal source. Historical flip diagnostics are excluded
from the formal scope.

## Matched bent ablation

The default background-loss weight 0.1 and a zero-weight variant each trained
for 50 epochs at recorded seed 42 on the same 220 train/good images and five
frozen bent support pairs. Packaged provenance matches for those inputs, upstream
commit, epoch/decay settings, GPU and common source override. Package-version
lists are byte-identical; both logs reach epoch 50. A independently rethresholded
the supplied probability arrays and checked their area/empty counts against the
preview summaries. Shared-QC counts below remain attributed to B's packaged
report; A did not rerun GAN training.

| Variant | Epoch | Preview count | Empty masks, independently checked | B-reported shared-QC passes |
|---|---:|---:|---:|---:|
| lambda_background=0.1 | 25 | 5 | 0 | 5/5 |
| lambda_background=0.1 | 50 | 5 | 3 | 1/5 |
| lambda_background=0.0 | 25 | 5 | 0 | 0/5 |
| lambda_background=0.0 | 50 | 5 | 4 | 0/5 |

The default arm's geometric passes fall from 5/5 at epoch 25 to 1/5 at epoch 50.
The zero-background arm has 0/5 passes at both checkpoints: low foreground overlap
at epoch 25, then four empty masks and one too-small mask at epoch 50. Inspection
of the four contact sheets also shows broad texture/shape changes and weak
image-mask correspondence. Disabling this loss did not provide a stable repair
in this bounded experiment. Neither the default epoch-25 preview nor an altered
inference mode is substituted for a formal accepted pool.

## What the evidence does and does not establish

The loss formulation can create an incentive to shrink predicted masks, and the
recorded BatchNorm diagnostic is sensitive to running statistics. These are
possible contributors, not isolated explanations. One seed, one defect type,
five preview sources and 50 epochs cannot establish a unique cause or prove
that the published method fails generally. The comparison also does not match
the original 300-epoch budget, so it cannot replace those baseline outputs.

E2 remains unrun because no complete, provenance-verified pool passed the quality
gate. The failure evidence is retained as a result about feasibility under the
project conditions. No extra GAN training was started for this analysis.

[Compact evidence and source hashes](gan_failure_evidence_20261017.json) link each
claim to the unchanged received package. Original logs, probability arrays,
weights and review images remain outside Git.
