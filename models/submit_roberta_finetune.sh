#!/bin/bash
# Submit 25 RoBERTa-base finetuning jobs: 4 inner + 1 final for each of 5 outer folds.

set -euo pipefail

JOB_PREFIX="roberta_cv"
TIME_LIMIT="00:30:00"
MEMORY="24G"
CPUS_PER_TASK=4
PARTITION="gpu"

# Adjust these two paths for your HPC setup.
VENV="/gxfs_work/cau/sunpn1134/wordrel4lcp/wordrel_env"
PROJECT_DIR="/gxfs_work/cau/sunpn1134/wordrel4lcp"

SCRIPT_PATH="${PROJECT_DIR}/models/finetune_roberta_cv.py"
LOG_DIR="${PROJECT_DIR}/models/logs"
MODELS_DIR="${PROJECT_DIR}/models/roberta_models"
FOLDS_DIR="${PROJECT_DIR}/data/feature_enriched/folds"

mkdir -p "$LOG_DIR"

submit_job () {
  local OUTER_FOLD="$1"
  local RUN_TYPE="$2"
  local INNER_FOLD="$3"

  local NAME_SUFFIX="outer${OUTER_FOLD}_${RUN_TYPE}"
  local INNER_ARG=""
  if [[ "$RUN_TYPE" == "inner" ]]; then
    NAME_SUFFIX="${NAME_SUFFIX}${INNER_FOLD}"
    INNER_ARG="--inner-fold ${INNER_FOLD}"
  fi

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_PREFIX}_${NAME_SUFFIX}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=1
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/${NAME_SUFFIX}_%j.out
#SBATCH --error=${LOG_DIR}/${NAME_SUFFIX}_%j.err

set -euo pipefail
source ${VENV}/bin/activate
cd ${PROJECT_DIR}

python ${SCRIPT_PATH} \
  --outer-fold ${OUTER_FOLD} \
  --run-type ${RUN_TYPE} \
  ${INNER_ARG} \
  --folds-dir ${FOLDS_DIR} \
  --models-dir ${MODELS_DIR} \
  --seed 667
EOF
}

for OUTER_FOLD in 1 2 3 4 5; do
  for INNER_FOLD in 1 2 3 4; do
    echo "Submitting outer_fold=${OUTER_FOLD}, inner_fold=${INNER_FOLD}"
    submit_job "$OUTER_FOLD" "inner" "$INNER_FOLD"
    sleep 1
  done

  echo "Submitting outer_fold=${OUTER_FOLD}, final"
  submit_job "$OUTER_FOLD" "final" "0"
  sleep 1
done

echo "All RoBERTa CV jobs submitted."

