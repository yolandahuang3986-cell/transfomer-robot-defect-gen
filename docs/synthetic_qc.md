# Shared synthetic-sample quality control

This QC applies to every generated sample from Procedural, GAN, and Diffusion
before it enters downstream training. The Real-only condition has no generated
samples and does not use this filter. A owns this common policy and its
executable helper in `src/generation/qc.py`; every generator inherits
`generate_with_qc` from the shared base interface. B and C must return source
IDs that map to training-normal inputs and use this entry point; they must not
add method-specific acceptance thresholds.

## Allowed inputs

QC may read only the generated image, its generated binary mask, and the
training-normal source image identified by a relative
`<category>/train/good/<image>` `source_image_id`; the helper rejects IDs from
other partitions. The shared
foreground estimator is deterministic and uses the same border-color/MAD
algorithm for every method. Its inputs and any parameters may be calibrated only on
training-normal images. Never read support-defect or held-out images to derive
foreground masks or tune QC thresholds. In particular, do not use test-normal
images; they belong to the frozen held-out evaluation partition.

The current estimator takes the median RGB value in a border band 5% of the
shorter image side wide as background color. It marks pixels as foreground
when their RGB distance from that color is greater than
`max(0.08, border_median_distance + 6 × 1.4826 × border_MAD)`, then keeps the largest 8-connected component. These
parameters are part of QC policy v1 and are applied to the training-normal
source image before evaluating its generated pair. If a generator changes
resolution without changing image geometry or aspect ratio, resize the
foreground mask to the generated output grid with nearest-neighbor sampling
before computing ratios. Cropping or changing aspect ratio is rejected because
the source foreground can no longer be aligned without a method-specific
transform.

Reject the sample if image/mask dimensions do not match, either contains
non-finite values, the mask is not binary after the shared contract's
normalization, or the training-normal foreground estimator returns an empty
object mask. Do not substitute a method-specific foreground estimate.

## Shared rejection rules

Let `M` be the generated binary defect mask, `O` the object foreground mask
estimated from its training-normal source image, `A_M = |M|`, and `A_O = |O|`.
Reject a sample if any of these conditions holds:

1. `A_M == 0` (empty mask).
2. `A_M / A_O < 0.001` or `A_M / A_O > 0.20` (mask covers less than 0.1% or
   more than 20% of the object foreground).
3. `|M ∩ O| / A_M < 0.90` (less than 90% of mask pixels lie on the object).

The thresholds are fixed for all three synthetic methods and both MVP
categories. Compare counts on the native output grid before any training-time
resize. A must version this policy and its foreground-estimator implementation;
threshold changes require rerunning QC for every method and recording the
reason before downstream evaluation.

## Fair comparison and audit record

Use the same accepted synthetic count per category/defect type and cap each
method at 10 attempts per requested accepted sample for that stratum. Do not
lower the thresholds, top up selectively, or choose samples based on held-out
metrics. If a method cannot reach the target within the 10x budget, record an
underfilled condition rather than relaxing QC. Count rejected attempts in
generation time and cost.

For each method, category, and defect type, record the QC version, seed, target
accepted count, attempts, accepted count, rejection counts by rule, acceptance
rate, and total generation/QC time. Keep a diagnostic image-mask overlay audit
for each method; visual review may trigger a shared policy revision, but must
not silently remove individual samples under additional method-specific rules.
