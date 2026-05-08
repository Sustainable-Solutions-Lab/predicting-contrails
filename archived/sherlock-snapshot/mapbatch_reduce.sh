#!/usr/bin/bash
#SBATCH --job-name=warmingmap_reduce
#SBATCH --output=warmingmap_reduce.%j.out
#SBATCH --error=warmingmap_reduce.%j.err
#SBATCH --time=6:00:00
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=32GB

set -euo pipefail

cd /home/groups/sjdavis/silas/Watershed/
source setup.sh

PARTIAL_GLOB="${PARTIAL_GLOB:-./partials/part_*.npz}"
MAP_TAG="${MAP_TAG:-all}"

/usr/bin/time -v python3 warmingmap_reduce.py --partials-glob "${PARTIAL_GLOB}" --tag "${MAP_TAG}"
