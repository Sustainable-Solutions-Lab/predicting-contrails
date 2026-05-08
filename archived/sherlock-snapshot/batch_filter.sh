#!/usr/bin/bash
#SBATCH --job-name=features
#SBATCH --output=f17.%j.out
#SBATCH --error=f17.%j.err
#SBATCH --time=10:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=20GB


cd /home/groups/sjdavis/silas/Watershed/
source setup.sh
python3 features_jul17.py $1
