#!/usr/bin/bash
#SBATCH --job-name=warmingmap
#SBATCH --output=warmingmap.%j.out
#SBATCH --error=warmingmap.%j.err
#SBATCH --time=6:00:00
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=75GB


cd /home/groups/sjdavis/silas/Watershed/
source setup.sh
/usr/bin/time -v python3 warmingmap.py all
