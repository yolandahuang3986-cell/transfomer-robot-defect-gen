# B delivery review sheets and verification results

17 October 2026 — Procedural candidate pools. The full delivery package has been
received and independently rechecked under A's TC2 account. This page contains
fixed review sheets; the complete training pools, dataset archive and weights
remain in external project storage.

| Category | Delivered pairs | Generation attempts | A's shared-QC recheck | Review outcome |
|---|---:|---:|---:|---|
| metal_nut | 75 | 83 | 75/75 | Accepted as the complete 75-pair pool; E1 complete: three seeds; job 43178, exit 0 |
| screw | 125 | 136 | 125/125 | Technical review complete; batch on hold for placement problems |

The archive SHA-256 matches the delivery record, and all 500 package-file checksums
pass. The frozen manifest, original-data hashes, and source/donor membership have
been verified. For all 200 pairs, masks match the pixels changed in the canonical
float32 images, and PNG exports match the canonical data. Hashes of the three
generation source files match the delivery provenance.

## How to read the sheets

Each sheet has five rows. From left to right, each row shows the **normal source
image, generated image, mask, and mask overlay**. Source images come from the
frozen manifest's train/good partition; no held-out images are displayed.

The 20 pairs per category follow A's existing audit script: seed 42, shuffle, then
successive pop operations. No samples were removed or replaced. B's supplementary
review set uses a different sampling procedure and is recorded separately.

The generated label is generic `synthetic_patch`; it does not establish named
defect semantics such as bent or scratch. Passing automated QC and pixel-alignment
checks does not establish realistic defect quality or downstream benefit.

## Findings and acceptance status

- **metal_nut:** No clear systematic placement onto the background was observed
  in A's fixed 20 pairs. Some patches are small, low contrast, or close to an
  object edge. The original complete 75-pair pool has been accepted for E1 following the
  user's instruction to start formal training on 17 October 2026.
- **screw:** B's supplementary samples 0009 and 0082 show shadow/background
  patches. A's samples 0077 and 0065 also raise edge/shadow concerns. The entire
  batch remains on hold. Do not remove individual samples, refill the pool, or
  relax shared QC. Samples 0009 and 0082 are not in A's fixed set below.
- **Review completed:** Package integrity, provenance checks, mask alignment,
  shared-QC rechecks, and Codex-assisted visual review have been completed.
- **Acceptance recorded:** A has issued the hash-bound legacy-purpose review
  and whole-batch acceptance for metal_nut. This delegated decision follows the
  user's instruction to start training and the documented review findings. The
  visual review was assisted by Codex; it is not an additional independent human
  rating or a claim that the user personally rated each image. The original
  complete batch is unchanged. See the
  [A execution runbook](../../formal_evaluation_runbook.md#用途校验与历史-b-bundle)
  for legacy-purpose handling.

## E1 completion — 17 October 2026

TC2 job **43178** completed on **NVIDIA A40**, TC2N02, with **exit code 0:0**
and total elapsed time **4:45**. Clean main commit:
`77ebf7f5870cc21a97d98373c2f03ac83871219b`. All **88 CPU tests** and CUDA Gate 0
passed. Three formal runs completed at **seeds 42/43/44, 1,500 optimizer steps
per seed, batch 8, 256-pixel inputs**, using the same accepted 75-pair pool.
Existing E0 runs were preserved; screw remains on hold.

[Results, paired E0 comparison and execution evidence](../../../results/metrics/e1_metal_nut_20261017/README.md)
show mean pixel AUROC of **0.8202 ± 0.0995**, versus E0 **0.6559 ± 0.1177**.
Mean Dice decreases to **0.0591 ± 0.0644**, and normal-image FPR remains **1.0**.
These results do not establish improved segmentation or usable image-level
detection. No two-category Procedural macro score is reported while screw is
on hold. No threshold, seed or training-budget changes were made after observing
these results.

Detailed acceptance and execution records are retained in the OneDrive project
review folder and A's TC2 storage. Their hashes are recorded in the
[machine-readable summary](review_manifest.json).

## Metal nut

### Sheet 1

Sample IDs: 0008, 0065, 0036, 0066, 0013.

![metal_nut fixed review sheet 1](images/metal_nut_1.png)

### Sheet 2

Sample IDs: 0014, 0053, 0012, 0047, 0064.

![metal_nut fixed review sheet 2](images/metal_nut_2.png)

### Sheet 3

Sample IDs: 0035, 0019, 0022, 0046, 0041.

![metal_nut fixed review sheet 3](images/metal_nut_3.png)

### Sheet 4

Sample IDs: 0060, 0011, 0071, 0038, 0006.

![metal_nut fixed review sheet 4](images/metal_nut_4.png)

## Screw

### Sheet 1

Sample IDs: 0008, 0077, 0065, 0097, 0036.

![screw fixed review sheet 1](images/screw_1.png)

### Sheet 2

Sample IDs: 0066, 0013, 0014, 0107, 0053.

![screw fixed review sheet 2](images/screw_2.png)

### Sheet 3

Sample IDs: 0012, 0109, 0047, 0064, 0035.

![screw fixed review sheet 3](images/screw_3.png)

### Sheet 4

Sample IDs: 0019, 0022, 0041, 0116, 0011.

![screw fixed review sheet 4](images/screw_4.png)

## Sources and verification records

- [Machine-readable summary and image hashes](review_manifest.json)
- [Image attribution and license](ATTRIBUTION.md)
- [B code-delivery record](../../member_b_delivery.md)
- [Formal evaluation protocol](../../formal_evaluation_protocol.md)

Generation code: `bf798af`. Independent verification code: `fd12294`.
The full batches and original logs remain in external project storage. These
eight review sheets do not replace the full delivery or formal acceptance record.
