#!/bin/bash
#SBATCH --job-name=k_sweep
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=rafael.tappe.maestro@student.rug.nl
#SBATCH --output=job-%j.log

# regular, short, vulture
#SBATCH --partition=vulture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# gauge using top
#SBATCH --time=00:30:00
# RES column indicates RAM usage in bytes
#SBATCH --mem-per-cpu=1GB

module load matplotlib
# yields python-3.9
# and packages
# numpy, scipy, pandas, matplotlib
module load networkx
module load scikit-learn
module load tqdm
module load GCC

mkdir data

srun python3 src/search_module.py