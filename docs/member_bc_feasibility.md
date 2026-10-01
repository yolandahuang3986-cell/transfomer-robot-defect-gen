# Member B/C Phase 0 feasibility record

Scope: audit the already selected official sources and validate an actual local
runtime. Do not compare or select replacement models. No method is implemented
in this record. Environment-specific fields remain **TBD** until the assigned
member runs the source on the project environment.

## Source audit

| Owner | Agreed method | Primary source | Official implementation evidence / Phase 0 observation |
|---|---|---|---|
| B | Procedural, CutPaste/DRAEM-style | [CutPaste CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf); [DRAEM ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf); [DRAEM authors' code](https://github.com/VitjanZ/DRAEM) | CutPaste trains representations to distinguish normal from cut-and-paste augmentation. DRAEM simulates anomalies on normal images and supplies pixel masks for a discriminative segmentation task. DRAEM repository documents MVTec + DTD inputs and a PyTorch training command. The team's procedural adapter remains CutPaste/DRAEM-style. |
| B | GAN, SyntheticDefectGeneration | [Official PyTorch repository](https://github.com/pasqualecoscia/SyntheticDefectGeneration); [ICIP 2023 paper](https://doi.org/10.1109/ICIP49359.2023.10222874) | Repository README identifies itself as the official implementation. It uses unpaired image-to-image translation to synthesize defective images and segmentation masks; README describes per-product/defect data preparation and CUDA training/evaluation commands. |
| C | Diffusion, AnomalyDiffusion | [Official authors' repository](https://github.com/sjtuplayer/anomalydiffusion); [AAAI 2024 paper](https://ojs.aaai.org/index.php/AAAI/article/view/28696) | Repository and paper identify the few-shot anomaly image-generation method. README specifies Python 3.8, CUDA 11.8, GCC 7.5, a latent-diffusion checkpoint, and separate mask/anomaly generation steps. The repository's stated environment must be compared with the assigned member's isolated setup. |

## Actual local feasibility — assigned member to complete

Record the exact repo commit, environment and command. Do not infer GPU model,
VRAM, runtime, or reproduction status from the cluster's general inventory or
from another method's run.

| Field | B: Procedural | B: GAN | C: Diffusion |
|---|---|---|---|
| Executor / date | TBD | TBD | TBD |
| Source commit | TBD | TBD | TBD |
| Python / PyTorch | TBD | TBD | TBD |
| CUDA / compiler | TBD | TBD | TBD |
| GPU model / VRAM | TBD | TBD | TBD |
| Minimal import / setup result | TBD | TBD | TBD |
| First sample command / exit status | TBD | TBD | TBD |
| End-to-end runtime | TBD | TBD | TBD |
| Checkpoints / additional data required | TBD | TBD | TBD |
| Output image-mask pair inspected | TBD | TBD | TBD |
| Blockers and next action | TBD | TBD | TBD |

The already observed TC2 base account allows one GPU, 10 CPUs, 30 GB RAM and a
six-hour job limit. This is cluster-account capacity, not a measured per-method
VRAM or runtime result. The Phase 0 U-Net Gate 0 ran on an A40; that does not
establish B/C method feasibility.
