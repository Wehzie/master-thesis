from dataclasses import dataclass
import subprocess
from typing import List
from pathlib import Path

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
# yields python-3.9
# and packages
# numpy, scipy, pandas, matplotlib
module load networkx
module load scikit-learn
module load tqdm
module load GCC
module load ngspice
module load matplotlib

{command}
""")

parser = argparse.ArgumentParser(description="Launch multiple Slurm processes.")

parser.add_argument("--production", action="store_true", help="Launch scripts with production parameters.", default=False, required=False)

args = parser.parse_args()
print(f"Running with {args}")

python_experiments = [
    # "target",
    # "n_osc",
    # "z_ops",
    # "samples",
    # "frequency",
    # "weight",
    # "offset",
    # "phase",
    # "amplitude",
    # "gain_dist",
    "freq_dist",
]

hybrid_experiments = [
    # "target",
    # "target_freq",
    # "n_osc",
    # "z_ops",
    # "duration",
    # "resistor",
    # "weight",
    # "offset",
    # "phase",
    # "gain_dist",
    "freq_dist",
]

@dataclass
class Job:
    name: str
    command: str
    time: str
    memory: str
    partition: str
    mail: str

    def assign_special_time(self):
        """durations are given for m=10 and z=20000 and n=100"""
        if not args.production:
            return
        if "duration" in self.name: # set m to 3
            self.time = "12:00:00"
        if "target" in self.name: # set m to 3
            self.time = "12:00:00"
        if "target_freq" in self.name: # set m to 3
            self.time = "12:00:00"
        if "frequency" in self.name:
            self.time = "05:00:00"
        if "n_osc" in self.name:
            self.time = "05:00:00"
        if "resistor" in self.name:
            self.time = "04:00:00"
        if "z_ops" in self.name:
            self.time = "03:00:00"
        if "offset" in self.name:
            self.time = "03:00:00"
        if "phase" in self.name:
            self.time = "03:00:00"
        if "weight" in self.name:
            self.time = "03:00:00"
        if "amplitude" in self.name:
            self.time = "02:30:00"
        if "samples" in self.name:
            self.time = "02:00:00"
        if "gain_dist" in self.name:
            self.time = "00:30:00"
        if "freq_dist" in self.name:
            self.time = "00:30:00"
        
    def assign_special_memory(self):
        if not args.production:
            return
        if "spipy-duration" in self.name:
            self.memory = "8GB"
        if "gain_dist" in self.name:
            self.memory = "8GB"
        if "freq_dist" in self.name:
            self.memory = "8GB"
        if "n_osc" in self.name:
            self.memory = "3GB"
        if "python-duration" in self.name:
            self.memory = "2GB"

def build_job_commands(time: str, memory: str, partition: str, mail: str) -> List[Job]:
    base_call = ["srun", "python3", "src/main.py"]
    if args.production:
        base_call.append("--production")
        base_call.append("--algo all")

    jobs = []
    for e in python_experiments:
        extension_args = ["--signal_generator", "python", f"--experiment {e}", "--target", "magpie"]
        job = Job(f"python-{e}", " ".join(base_call + extension_args), time, memory, partition, mail)
        job.assign_special_time()
        job.assign_special_memory()
        jobs.append(job)

    for e in hybrid_experiments:
        extension_args = ["--signal_generator", "spipy", f"--experiment {e}", "--target", "sine"]
        job = Job(f"spipy-{e}", " ".join(base_call + extension_args), time, memory, partition, mail)
        job.assign_special_time()
        job.assign_special_memory()
        jobs.append(job)

    return jobs

def ask_for_confirmation(jobs: List[Job]):
    print("The following commands will be executed:")
    for job in jobs:
        print(job.command, job.time, job.memory, job.partition)
    proceed = input("Continue? (y/n)")
    if proceed != "y":
        print("Aborting.")
        exit()
    print("Proceeding...")

def run_jobs(jobs: List[Job]):
    counter = 0
    for job in jobs:
        script = build_job_script(job.command, job.time, job.memory, job.partition, job.name, job.mail)
        with open("job.sh", "w") as f:
            f.write(script)
        subprocess.run(["sbatch", "job.sh"])
        counter += 1
        if counter == 2 and not args.production:
            break

def launch_experiments():
    partition = "regular" if args.production else "regular"
    memory = "1GB" if args.production else "300MB"
    time = "03:00:00" if args.production else "00:01:00"
    mail = send_mail_config if args.production else ""

    jobs = build_job_commands(time, memory, partition, mail)

    ask_for_confirmation(jobs)

    run_jobs(jobs)


def launch_qualitative():

    def build_qual_job_commands():
        base_call = ["srun", "python3", "src/main.py", "--experiment", "none", "--qualitative"]
        if args.production:
                base_call.append("--production")
        
        python_extension = ["--signal_generator", "python", "--target", "magpie"]
        python_args = base_call + python_extension

        spipy_extension = ["--signal_generator", "spipy", "--target", "damp_chirp"]
        spipy_args = base_call + spipy_extension

        names = ["python-qualitative", "spipy-qualitative"]
        return names, [python_args, spipy_args]

    def ask_for_confirmation(commands: List[List[str]], time: str, memory: str, partition: str):
        print("The following commands will be executed:")
        for command in commands:
            print(command, time, memory, partition)
        proceed = input("Continue? (y/n)")
        if proceed != "y":
            print("Aborting.")
            exit()
        print("Proceeding...")

    def run_jobs(commands: List[List[str]], names: List[str], time: str, memory: str, partition: str, mail: str):
        counter = 0
        for command, name in zip(commands, names):
            joined_command = " ".join(command)
            script = build_job_script(joined_command, time, memory, partition, name, mail)
            with open("job.sh", "w") as f:
                f.write(script)
            subprocess.run(["sbatch", "job.sh"])
            counter += 1
            if counter == 2 and not args.production:
                break

    def build_qual_various_targets():
        names = ["sine-1Hz", "sine-10Hz", "sine-100Hz", "sine-1000Hz", "sine-10kHz" "sine-100kHz", "sine-500kHz", "sine-1MHz"]
        targets = [
                    ["--target", "sine", "--target_freq", "1",],
                    ["--target", "sine", "--target_freq", "10",],
                    ["--target", "sine", "--target_freq", "100",],
                    ["--target", "sine", "--target_freq", "1000",],
                    ["--target", "sine", "--target_freq", "10000",],
                    ["--target", "sine", "--target_freq", "100000",],
                    ["--target", "sine", "--target_freq", "500000",],
                    ["--target", "sine", "--target_freq", "1000000",],
                ]

        base_call = ["srun", "python3", "src/main.py", "--experiment", "none", "--qualitative", "--algo", "best"]
        if args.production:
                base_call.append("--production")
        return [base_call + target for target in targets], names
    
    partition = "regular" if args.production else "regular"
    memory = "8GB" if args.production else "500MB"
    time = "00:15:00" if args.production else "00:01:00"
    mail = send_mail_config if args.production else ""

    srun_commands, names = build_qual_various_targets()

    # run spipy only
    if False:
        srun_commands = [srun_commands[1]]
        names = [names[1]]

    ask_for_confirmation(srun_commands, time, memory, partition)

    run_jobs(srun_commands, names, time, memory, partition, mail)

def clean():
    Path("job.sh").unlink(missing_ok=True)

if __name__ == "__main__":
    launch_experiments()
    #launch_qualitative()
    clean()