#!/bin/bash

# run with:
# sbatch jobscript.sh

#SBATCH --job-name=python_sweep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=r.tappe.maestro@student.rug.nl
#SBATCH --output=job-%j.log

# partitions:
# regular, short, vulture

#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# gauge memory usage with top
# RES column indicates RAM usage in bytes

#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2GB

module purge
# yields python-3.9
# and packages
# numpy, scipy, pandas, matplotlib
module load networkx
module load scikit-learn
module load tqdm
module load GCC
module load ngspice
module load matplotlib

srun python3 src/main.py --production