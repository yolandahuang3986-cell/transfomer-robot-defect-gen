# Diffusion analysis scripts (Member C)

Evidence and diagnostics behind the internal rules and the validation report.
None of these change generation or shared QC. Outputs go outside Git.

| Script | Env | What it answers |
|---|---|---|
| `check_anomaly_embeddings.py` | Anomalydiffusion | Which defect names in the anomaly `embeddings.pt` were actually trained (flip must not be). |
| `mask_probe.py` | Anomalydiffusion, GPU | Raw stage-1 mask outputs per mask embedding; counts valid / empty / inverted / dirty background. Basis of `mask_filter_v1`. |
| `support_latent_cells.py` | Anomalydiffusion | Support-mask area and 32x32 latent cells per defect type. Basis of the `mask_filter_v2` latent rule. |
| `fg_probe.py` | defectgen | Shared-QC foreground vs tighter variants on train/good images. Basis of `placement_v2`. |
| `latent_mask_check.py` | Anomalydiffusion | Latent cells of saved masks, next to the visibility ratio. |
| `qc_check_worker_output.py` | defectgen | Shared QC on a worker output dir, printed separately from internal stage-1 counts; old visibility ratio (reporting only). |
| `formal_qc_recheck.py` | defectgen | Shared QC with the formal loader's convention (bilinear 256 source, foreground on that grid). |
| `edit_region_analysis.py` | Anomalydiffusion | Change inside the saved mask vs inside its latent footprint vs far background; compares old/new visibility ratios against human labels (AUC). |
| `zoom_review.py` | any | x4 crops for blind human labelling (ratio hidden unless `--show_ratio`). |
| `make_contact_sheet.py` | any | Source / mask / generated / outline sheet for a worker output dir. |
| `bundle_contact_sheet.py` | any | Same sheet, read-only from an exported bundle (`batch.npz` + `metadata.json`): exactly the bundle's pairs, in bundle order. |
| `rejects_sheet.py` | any | First k rejected stage-1 masks per reason (raw + judged mask); needs `save_rejects`. |
| `run_summary.py` | any | Cost, stage-1 yield and shared-QC outcome of one exported bundle. |

Scripts that read the support list take it as an argument; pass the frozen list used for training.
