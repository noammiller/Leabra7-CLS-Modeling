#!/bin/bash
# parallel job using 500 processors. and runs for 4 hours (max)
#SBATCH -N 20
#SBATCH --ntasks-per-node=4
#SBATCH -t 1:00:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user= noamm@princeton.edu

conda activate leabra7
cd /home/noamm/GitHub/Leabra7-CLS-Modeling/Schapiro
./Schapiro_Model.py 20 1000
