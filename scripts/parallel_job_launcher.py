import subprocess
from typing import List

import argparse

def build_job_script(command: str, time: str, mem: str) -> str:
    return f"""
#!/bin/bash

# SBATCH --job-name=experiment
# SBATCH --mail-type=ALL
# SBATCH --mail-user=r.tappe.maestro@student.rug.nl
# SBATCH --output=job-%j.log

# regular, short, vulture
# SBATCH --partition=regular
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=1

# gauge memory usage with top
# RES column indicates RAM usage in bytes

#SBATCH --time={time}
#SBATCH --mem-per-cpu={mem}

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

{command}
"""

parser = argparse.ArgumentParser(description="Launch multiple Slurm processes.")

parser.add_argument("--production", action="store_true", help="Launch scripts with production parameters.", default=False, required=False)

args = parser.parse_args()
print(f"Running with {args}")

legal_python_experiments = [
    "target",
    "n_osc",
    "z_ops",
    "samples",
    "frequency",
    "weight",
    "offset",
    "phase",
    "amplitude",
    "all",
    "none"
]

legal_hybrid_experiments = [
    "target",
    "n_osc",
    "z_ops",
    "duration",
    "resistor",
    "weight",
    "offset",
    "phase",
    "all",
    "none"
]

def build_job_commands():
    base_call = ["srun", "python3", "src/main.py"]

    invocations = []
    for e in legal_python_experiments:
        extension_args = ["--signal_generator", "python", f"--experiment {e}", "--target magpie"]
        if args.production:
            extension_args.append("--production")
        invocations.append(base_call + extension_args)

    for e in legal_hybrid_experiments:
        extension_args = ["--signal_generator", "spipy", f"--experiment {e}", "--target sine"]
        if args.production:
            extension_args.append("--production")
        invocations.append(base_call + extension_args)

    return invocations


def ask_for_confirmation(srun_commands: List[str]):
    print("The following commands will be executed:")
    for command in srun_commands:
        print(command)
    proceed = input("Continue? (y/n)")
    if proceed != "y":
        print("Aborting.")
        exit()
    print("Proceeding...")

def main():
    memory = "500MB" if args.production else "2GB"
    time = "00:01:00" if args.production else "03:00:00"
    srun_commands = build_job_commands()

    ask_for_confirmation(srun_commands)

    for command in srun_commands:
        script = build_job_script(command, time, memory)
        with open("job.sh", "w") as f:
            f.write(script)
        subprocess.run(["sbatch", "job.sh"])

if __name__ == "__main__":
    main()