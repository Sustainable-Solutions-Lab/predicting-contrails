#!/usr/bin/bash
set -euo pipefail

cd /home/groups/sjdavis/silas/Watershed/

INPUT_GLOB="${INPUT_GLOB:-./betterEFs/*.pq}"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
PARTIAL_DIR="${PARTIAL_DIR:-./partials}"
MAP_TAG="${MAP_TAG:-all}"

mapfile -t INPUT_FILES < <(ls ${INPUT_GLOB} 2>/dev/null | sort)
NUM_FILES="${#INPUT_FILES[@]}"
if [[ "${NUM_FILES}" -eq 0 ]]; then
  echo "No input files found for INPUT_GLOB=${INPUT_GLOB}"
  exit 1
fi

mkdir -p "${PARTIAL_DIR}"
echo "Found ${NUM_FILES} input parquet files."
echo "Submitting array with max concurrency ${MAX_CONCURRENT}."

ARRAY_JOB_ID="$(
  sbatch --parsable \
    --array="0-$((NUM_FILES - 1))%${MAX_CONCURRENT}" \
    --export=ALL,INPUT_GLOB="${INPUT_GLOB}",PARTIAL_DIR="${PARTIAL_DIR}" \
    mapbatch_array.sh
)"
echo "Submitted array job: ${ARRAY_JOB_ID}"

REDUCE_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --export=ALL,PARTIAL_GLOB="${PARTIAL_DIR}/part_${ARRAY_JOB_ID}_*.npz",MAP_TAG="${MAP_TAG}" \
    mapbatch_reduce.sh
)"
echo "Submitted reduce job: ${REDUCE_JOB_ID}"
echo "Track with: squeue -u \$USER"
