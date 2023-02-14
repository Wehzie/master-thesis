#!/bin/bash

# RUNNING
# sbatch jobscript.sh

#SBATCH --job-name=python_sweep_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=r.tappe.maestro@student.rug.nl
#SBATCH --output=test-job-%j.log

# regular, short, vulture
#SBATCH --partition=vulture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# gauge memory usage with top
# RES column indicates RAM usage in bytes

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

srun python3 src/main.py
