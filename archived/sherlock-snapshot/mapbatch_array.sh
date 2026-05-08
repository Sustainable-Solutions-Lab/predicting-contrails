#!/usr/bin/bash
#SBATCH --job-name=warmingmap_partial
#SBATCH --output=warmingmap_partial.%A_%a.out
#SBATCH --error=warmingmap_partial.%A_%a.err
#SBATCH --time=3:00:00
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16GB

set -euo pipefail

cd /home/groups/sjdavis/silas/Watershed/
source setup.sh

INPUT_GLOB="${INPUT_GLOB:-./adjustedEFs/*.pq}"
PARTIAL_DIR="${PARTIAL_DIR:-./partials}"
mkdir -p "${PARTIAL_DIR}"

mapfile -t INPUT_FILES < <(ls ${INPUT_GLOB} 2>/dev/null | sort)
if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
  echo "No input files found for INPUT_GLOB=${INPUT_GLOB}"
  exit 1
fi

IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
if (( IDX < 0 || IDX >= ${#INPUT_FILES[@]} )); then
  echo "Array index ${IDX} out of range for ${#INPUT_FILES[@]} files"
  exit 1
fi

INPUT_FILE="${INPUT_FILES[$IDX]}"
OUT_FILE="${PARTIAL_DIR}/part_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.npz"

echo "Input: ${INPUT_FILE}"
echo "Output: ${OUT_FILE}"
/usr/bin/time -v python3 warmingmap_partial.py --input "${INPUT_FILE}" --output "${OUT_FILE}"
