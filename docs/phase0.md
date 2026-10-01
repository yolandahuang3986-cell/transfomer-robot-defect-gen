# Member A Phase 0 contract

- Dataset root stays outside Git. Audit validates readable files, normal-only
  training, complete binary defect masks, category coverage, and records
  relative paths and SHA-256 hashes.
- The seeded split samples exactly `n_support` images per category/defect type
  and requires at least one held-out image for every stratum. Paths and content
  hashes are checked for leakage. Manifest paths are portable and the whole
  manifest is checksummed.
- Split generation uses exclusive creation. A generated manifest has status
  `candidate`. `freeze_split.py` rechecks all data hashes and writes a separate
  review attestation without changing the candidate. The reviewer must actually
  inspect and approve the displayed counts/coverage before running it.
- Generators exchange float32 RGB NCHW images in `[0,1]`, float32 binary N1HW
  masks, and one metadata record per image. Contract tests validate all fields.
- U-Net uses `segmentation-models-pytorch` with an untrained ResNet-18 encoder;
  no model download or weights are needed. Threshold is fixed at 0.5. AUROC and
  defect recall are null when their denominators are undefined. Empty-mask IoU
  and Dice are 1 only when both prediction and target are empty.
- Gate 0 runs the dummy through the same batch validator/loader, takes one
  training step, evaluates either independent fixtures or held-out real images,
  validates the versioned JSON schema, and creates output exclusively. It is a
  smoke test, not a quality result.
