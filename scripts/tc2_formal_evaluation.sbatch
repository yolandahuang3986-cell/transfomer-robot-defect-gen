#!/bin/bash
#SBATCH --job-name=robot-a-eval
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
set -euo pipefail

# Usage: sbatch --output=/external/path/job.log this.sbatch CODE_ROOT DATA_ROOT RUN_ROOT [DELIVERIES_JSON]
CODE_ROOT=${1:?Pass the clean committed checkout}
DATA_ROOT=${2:?Pass the frozen MVTec root}
RUN_ROOT=${3:?Pass a new external run directory}
ROBOT_PYTHON=${ROBOT_PYTHON:-$HOME/envs/robot-phase0/bin/python}
cd "$CODE_ROOT"
test -z "$(git status --porcelain)"
mkdir "$RUN_ROOT"
"$ROBOT_PYTHON" -m pytest -q
"$ROBOT_PYTHON" scripts/run_gate0.py --device cuda --out "$RUN_ROOT/gate0.json"
"$ROBOT_PYTHON" scripts/benchmark_inspector.py --root "$DATA_ROOT" --device cuda --out "$RUN_ROOT/timing.json"
DELIVERY_ARGS=()
if [ "$#" -ge 4 ]; then
    DELIVERY_ARGS=(--deliveries "$4")
fi
"$ROBOT_PYTHON" scripts/run_formal_evaluation.py --root "$DATA_ROOT" \
    --out "$RUN_ROOT/primary" --device cuda --execute "${DELIVERY_ARGS[@]}"
