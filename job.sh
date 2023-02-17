#!/bin/bash

#SBATCH --job-name=spipy-qualitative

#SBATCH --output=job-spipy-qualitative-%j.log
#SBATCH --partition=vulture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=500MB

module purge
module load matplotlib
# yields python-3.9
# and packages
# numpy, scipy, pandas, matplotlib
module load networkx
module load scikit-learn
module load tqdm
module load GCC
module load ngspice

srun python3 src/main.py --experiment none --qualitative --signal_generator spipy --target damp_chirp
