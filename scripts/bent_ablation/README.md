# Bent background-loss ablation

Purpose: diagnose mask failure, not prove downstream improvement.
Two fresh 50-epoch runs on identical 220 normal / 5 support inputs. Seed42, batch1,256px,decay25. Only lambda_background differs (0.1 versus0). Both use the same newly installed isolated environment, pinned official source, and deterministic settings. All other official losses remain unchanged. Upstream benchmark=True is overridden in memory for both arms; source files are not edited. warn_only deterministic algorithms may still warn about nondeterministic operations.

This is a matched new comparison, not directly comparable to historical300-epoch runs with unknown dependency versions and no seed. One seed is exploratory evidence, not a robust statistical conclusion. Previews at25 and50 use eval mode, identical5 normal training images, masks at0.5 plus diagnostics at0.2. Do not tune with heldout images. Neither nonempty masks nor lower losses establish quality: inspect images, alignment, diversity and then independent downstream utility.

SLURM requests one GPU,2CPU,16GB,4h. Estimated training ~1h total from previous throughput; setup/queue adds time. Runs sequentially and stops on failure; no requeue and no changes to current Screw jobs. Outputs and new venv live under ~/defect_runs/bent_ablation_JOBID. This temporarily uses additional disk space. Original weights/data/environment stay intact.

Preflight checks pinned clean upstream, exact frozen manifest file checksum, hashes of all225 source images and5 masks, support membership and pinned albumentations1.3.1. Data staging is immutable, outputs refuse overwrite. Full dependencies and arguments saved for each arm. Job has not run on TC2 at package creation.

Local validation: Python and shell syntax, actual frozen data preflight, preview exporter tested using existing epoch300 checkpoint fixtures (these fixtures are NOT new25/50-epoch results). GPU training remains untested until submitted.

Success criteria for diagnosis: compare both arms using fixed inputs, retained probabilities, source/image/mask contact sheets at both checkpoints; observe whether removal changes mask collapse AND visual correspondence. Preserve failed results. No automatic claim of a fix or promotion to production synthesis.
