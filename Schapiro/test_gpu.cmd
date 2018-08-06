#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=noamm@princeton.edu

echo "test gpu"

. /home/noamm/anaconda3/envs/leabra7/etc/profile.d/conda.sh

conda activate base

cd /home/noamm/GitHub/Leabra7-CLS-Modeling/Schapiro

./Schapiro_Model.py 1 1000
