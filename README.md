# Function generation from a Sum of Oscillator Signals

## Abstract

Energy demand in data-intensive applications is an ever-growing concern.
Training a recent large language model consumes energy in the order of hundreds of US households per year.
Remarkably, the human brain is orders of magnitudes more energy efficient than modern day digital computers.
Taking inspiration from the brain, the neuromorphic paradigm aims to build more efficient computing systems using analog devices. 

In this work, an ensemble of oscillators is designed and simulated with the goal of arbitrary time-series approximation.
Each oscillator-neuron is formed by a vanadium dioxide memristor in series with a resistor-capacitor (RC) circuit.
Multiple gradient-free optimization algorithms are explored to perturb the oscillator ensemble, in order to change their frequency, gain, phase and offset.
A range of real-world and synthetic target functions are tested against the system.

We show that the vanadium-dioxide oscillator ensemble is suitable for function generation across a range of algorithms when the target's frequencies lie within the frequency band of the oscillator-neurons.
The system benefits from broad phase and frequency diversity, in particular from the addition of slower oscillators.
In contrast, a wide dynamic range leads to exponential loss growth for a majority of algorithms. Furthermore, an increase in the number of oscillators tends to increase loss linearly.

## Requirements

Install required libraries.

    pip install -r requirements.txt

Pydot is used with NetworkX for drawing graphs.
Pydot requires GraphViz to be installed.
Installation instructions are found [here](https://graphviz.org/download/).

    # Debian/Ubuntu
    sudo apt install graphviz

## Running

To start the simulation execute

    python src/main.py

For advice on parameters run.

    python src/main.py -h

### Detaching a terminal

Detaching a terminal is useful in order to close a terminal temporarily but being able to resume the terminal later to monitor progress of a simulation.
Some options to do this are `tmux` and `screen`, where `tmux` is more modern.
Instructions are geared towards Debian and Ubuntu hosts.

Install requirements

    sudo apt install tmux

Start tmux session

    tmux

Run desired commands, start simulation

    python src/main.py

Detach tmux

    # press Ctrl + b
    # press d

Now the terminal running tmux can be closed or an SSH connection to a host can be terminated

To reattach a session

    tmux ls # to list sessions
    tmux attach-session -t <session-name>

Multiple sessions can be run by detaching with

    # press Ctrl + b
    # press $
    # enter session name

### Deployment on Slurm (High Performance Computing cluster)

First prepare a job according to <https://wiki.hpc.rug.nl/habrok/job_management/running_jobs>.
Complete job files are in the `scripts` directory.

Start a job; make sure the script is executed from the same directory as done during testing.

    cd path/to/project
    sbatch peregrine_job.sh

List active jobs.

    squeue -u $USER
    
Get information about a running job.

    jobinfo JobID

## Testing

Static type checking.

    mypy src/

Static code analysis.

    # easy-going
    pyflakes src/
    # pedantic
    pylint --errors-only src/

Formatting.

    black src/

Unit tests.

    pytest src/tests