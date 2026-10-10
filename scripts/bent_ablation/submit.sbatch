#!/bin/bash
#SBATCH --job-name=bent-ablation
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
export MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
PACKAGE="$SLURM_SUBMIT_DIR"
RUN="$HOME/defect_runs/bent_ablation_$SLURM_JOB_ID"
mkdir -p "$RUN"
# Isolate both environment and tracked upstream source from running Screw jobs.
"$HOME/envs/defect-b/bin/python" -m venv "$RUN/env"
PY="$RUN/env/bin/python"
"$PY" -m pip install -r "$PACKAGE/requirements.txt"
git clone --no-hardlinks "$HOME/projects/sdg" "$RUN/upstream"
git -C "$RUN/upstream" checkout --detach 5a8099204bc0858da314f8e0b5e8b6770c298e98
for arm in control no_background; do
 weight=0.1
 if [ "$arm" = no_background ]; then weight=0; fi
 "$PY" -u "$PACKAGE/run_experiment.py" --upstream "$RUN/upstream" \
  --manifest "$HOME/projects/defect-gen/data/splits/fewshot_n5_seed42.json" \
  --data "$HOME/datasets/mvtec_ad" --out "$RUN/$arm" --weight "$weight" \
  > "$RUN/${arm}.log" 2>&1
 "$PY" -u "$PACKAGE/preview.py" --upstream "$RUN/upstream" --run "$RUN/$arm" \
  --manifest "$HOME/projects/defect-gen/data/splits/fewshot_n5_seed42.json" --data "$HOME/datasets/mvtec_ad" \
  > "$RUN/${arm}_preview.log" 2>&1
done
printf 'ABLATION_COMPLETE\n%s\n' "$RUN"
