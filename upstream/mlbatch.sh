#!/usr/bin/bash
#SBATCH --job-name=jul12ml
#SBATCH --output=jul12ml.%j.out
#SBATCH --error=jul12ml.%j.err
#SBATCH --time=5:00:00
#SBATCH -p normal
#SBATCH -c 10
#SBATCH --mem=75GB


cd /home/groups/sjdavis/silas/Watershed/
source setup.sh
/usr/bin/time -v python3 contrail_avoidance_regression_ml_jul12.py 