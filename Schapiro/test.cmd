#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=noamm@princeton.edu

. /home/noamm/anaconda3/envs/leabra7/etc/profile.d/conda.sh

conda activate base

cd /home/noamm/GitHub/Leabra7-CLS-Modeling/Schapiro

./Schapiro_Model.py 1 1000
