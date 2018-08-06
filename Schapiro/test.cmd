#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=18:00:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=noamm@princeton.edu

echo "test"

. /home/noamm/anaconda3/envs/leabra7/etc/profile.d/conda.sh

conda activate base

cd /home/noamm/GitHub/Leabra7-CLS-Modeling/Schapiro

./Schapiro_Model.py 10 1000
