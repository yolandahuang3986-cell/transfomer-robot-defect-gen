#!/bin/bash
#SBATCH --job-name=b-gan-smoke
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=28G
#SBATCH --time=02:00:00
set -euo pipefail
# Submit only after copying the B code overlay into this project checkout.
module load anaconda
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/defect-gen}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/mvtec_ad}"
GAN_ENV="${GAN_ENV:-$HOME/envs/defect-gan}"
UPSTREAM_DIR="${UPSTREAM_DIR:-$HOME/projects/SyntheticDefectGeneration}"
RUN_ROOT="$HOME/defect_runs/gan-smoke-${SLURM_JOB_ID}"
cd "$PROJECT_DIR"
if [ ! -x "$GAN_ENV/bin/python" ]; then
  conda create -y --override-channels -c conda-forge -p "$GAN_ENV" python=3.11 pip
fi
PY="$GAN_ENV/bin/python"
"$PY" -m pip install -r requirements-gan.txt
if [ ! -d "$UPSTREAM_DIR/.git" ]; then
  git clone https://github.com/pasqualecoscia/SyntheticDefectGeneration.git "$UPSTREAM_DIR"
fi
# Worker checks exact upstream revision and rejects modified tracked source.
test "$(git -C "$UPSTREAM_DIR" rev-parse HEAD)" = "5a8099204bc0858da314f8e0b5e8b6770c298e98"
mkdir -p "$RUN_ROOT"
"$PY" scripts/prepare_gan_support.py --root "$DATA_ROOT" --manifest data/splits/fewshot_n5_seed42.json --category metal_nut --defect-type scratch --out "$RUN_ROOT/support"
"$PY" scripts/train_gan.py --root "$DATA_ROOT" --manifest data/splits/fewshot_n5_seed42.json --stage "$RUN_ROOT/support" --upstream "$UPSTREAM_DIR" --python "$PY" --device cuda --max-steps 10 --image-size 256 --out "$RUN_ROOT/train"
"$PY" scripts/generate_gan.py --root "$DATA_ROOT" --manifest data/splits/fewshot_n5_seed42.json --upstream "$UPSTREAM_DIR" --python "$PY" --checkpoint "$RUN_ROOT/train/generator_A2B.pt" --category metal_nut --defect-type scratch --device cuda --num-samples 20 --allow-smoke --out "$RUN_ROOT/samples"
"$PY" -m pip freeze > "$RUN_ROOT/packages.txt"
echo "GAN_GPU_SMOKE_COMPLETE: $RUN_ROOT"
