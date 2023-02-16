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
    if args.production:
        base_call.append("--production")

    invocations = []
    names = []
    for e in python_experiments:
        extension_args = ["--signal_generator", "python", f"--experiment {e}", "--target", "magpie"]
        invocations.append(base_call + extension_args)
        names.append(f"python-{e}")

    for e in hybrid_experiments:
        extension_args = ["--signal_generator", "spipy", f"--experiment {e}", "--target", "sine"]
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

def run_jobs(srun_commands: List[List[str]], names: List[str], time: str, memory: str, partition: str, mail: str):
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

def launch_experiments():
    partition = "regular" if args.production else "vulture"
    memory = "2GB" if args.production else "300MB"
    time = "03:00:00" if args.production else "00:01:00"
    mail = send_mail_config if args.production else ""

    srun_commands, names = build_job_commands()

    ask_for_confirmation(srun_commands, time, memory, partition)

    run_jobs(srun_commands, names, time, memory, partition, mail)


def launch_quantitative():

    def build_job_commands():
        base_call = ["srun", "python3", "src/main.py", "--experiment", "none", "--qualitative"]
        if args.production:
                base_call.append("--production")
        
        python_extension = ["--signal_generator", "python", "--target", "magpie"]
        python_args = base_call + python_extension

        spipy_extension = ["--signal_generator", "spipy", "--target", "damp_chirp"]
        spipy_args = base_call + spipy_extension
        return [python_args, spipy_args]
    
    partition = "regular" if args.production else "vulture"
    memory = "8GB" if args.production else "500MB"
    time = "00:30:00" if args.production else "00:01:00"
    mail = send_mail_config if args.production else ""

    srun_commands = build_job_commands()
    names = ["python-qualitative", "spipy-qualitative"]

    ask_for_confirmation(srun_commands, time, memory, partition)

    run_jobs(srun_commands, names, time, memory, partition, mail)

if __name__ == "__main__":
    launch_experiments()
    launch_quantitative()