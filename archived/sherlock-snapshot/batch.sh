#!/usr/bin/bash
#SBATCH --job-name=featurescreation
#SBATCH --output=featurescreation.%j.out
#SBATCH --error=featurescreation.%j.err
#SBATCH --time=3:30:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8GB


cd /home/groups/sjdavis/silas/Watershed/
source setup.sh
/usr/bin/time -v python3 features_jul2.py 20190102


"""Copy to mac:Scp s2df5@sh04-ln01:/home/groups/sjdavis/silas/Watershed/batchfeatures_20190102_gdf.pq ."""
"""copy to cluster: scp allbatch.sh s2df5@sh03-ln04:/home/groups/sjdavis/silas/Watershed """