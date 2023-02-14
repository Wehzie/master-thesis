import subprocess
from typing import List

import argparse

send_mail_config = (
f"""#SBATCH --mail-type=ALL
#SBATCH --mail-user=r.tappe.maestro@student.rug.nl"""
)


def build_job_script(command: str, time: str, mem: str, partition: str, name: str, mail: str) -> str:
    return (
f"""#!/bin/bash

#SBATCH --job-name={name}
{mail}
#SBATCH --output=job-{name}-%j.log
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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
""")

parser = argparse.ArgumentParser(description="Launch multiple Slurm processes.")

parser.add_argument("--production", action="store_true", help="Launch scripts with production parameters.", default=False, required=False)

args = parser.parse_args()
print(f"Running with {args}")

python_experiments = [
    "target",
    "n_osc",
    "z_ops",
    "samples",
    "frequency",
    "weight",
    "offset",
    "phase",
    "amplitude",
]

hybrid_experiments = [
    "target",
    "n_osc",
    "z_ops",
    "duration",
    "resistor",
    "weight",
    "offset",
    "phase",
]

def build_job_commands():
    base_call = ["srun", "python3", "src/main.py"]

    invocations = []
    names = []
    for e in python_experiments:
        extension_args = ["--signal_generator", "python", f"--experiment {e}", "--target magpie"]
        if args.production:
            extension_args.append("--production")
        invocations.append(base_call + extension_args)
        names.append(f"python-{e}")

    for e in hybrid_experiments:
        extension_args = ["--signal_generator", "spipy", f"--experiment {e}", "--target sine"]
        if args.production:
            extension_args.append("--production")
        invocations.append(base_call + extension_args)
        names.append(f"spipy-{e}")

    return invocations, names


def ask_for_confirmation(srun_commands: List[str], time: str, memory: str, partition: str):
    print("The following commands will be executed:")
    for command in srun_commands:
        print(command)
    print(f"Running with partition {partition}, memory {memory}, time {time}.")
    proceed = input("Continue? (y/n)")
    if proceed != "y":
        print("Aborting.")
        exit()
    print("Proceeding...")

def main():
    partition = "regular" if args.production else "vulture"
    memory = "2GB" if args.production else "300MB"
    time = "03:00:00" if args.production else "00:01:00"
    mail = send_mail_config if args.production else ""

    srun_commands, names = build_job_commands()

    ask_for_confirmation(srun_commands, time, memory, partition)

    counter = 0
    for command, name in zip(srun_commands, names):
        joined_command = " ".join(command)
        script = build_job_script(joined_command, time, memory, partition, name, mail)
        with open("job.sh", "w") as f:
            f.write(script)
        subprocess.run(["sbatch", "job.sh"])
        counter += 1
        if counter == 2 and not args.production:
            break


if __name__ == "__main__":
    main()