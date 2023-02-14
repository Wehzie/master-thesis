# Circuit Generator

- build graphs in Python using networkX
- convert graphs to SPICE netlists 
- run SPICE netlists with ngspice
- solve non-linear transformation (NLT) tasks. for example sinus to square wave.


## DevOps

- Before running a full sweep run a test sweep with m-averages=1 with the final parameters
- Before running a full sweep run a test sweep with the test parameters on the target hardware

### Testing

Unit tests are executed with

    pytest src/tests

## In order of Priority: IDEAS and TODOS before final experiment can run

In order of priority:

### High priority

- Parallelization: Write script to launch quantitative experiments as separate slurm jobs.

- show sampling rate and duration, not number of samples in the sweep plots

- Normal distribution plots not saved, fix or drop normal distribution plots.

- z-ops plot double entry at same value as default value

- x-label for weight sweep plot is not clear enough

- hybrid generator not listening to n-osc from hybrid_params_test

### Medium priority

- Python: increase sampling rate to avoid aliasing. Verify with individual oscillator plots.

- Map R to Freq in analysis for Hybrid signal generator

- Plot improvement: Reuse algorithm bar chart with z-ops sweep.

- Plot Improvement: For oscillators vs. rmse. Show up to 1k oscillators, but draw a dotted line into the plot at around 300 oscillators to indicate what's technologically feasible.

- Plot Improvement: For number of samples vs. rmse; Convert this to signal duration vs. rmse.

- Plot Improvement: save mean, mode, stddev to frequency diversity plot title.

- Reproducibility issue: Maffezzoni et al. report different oscillator frequencies than I have.

### Low priority

- Profile Python's memory usage.

- Plotting custom legend for mask with multiple members in a group. Such that a group shares a color and one entry in the legend (individual algorithm names shouldn't be visible).

- SPICE: Add phase via delayed voltage delivery.

- Plot Improvement: Histogram over the frequencies in an ensemble of oscillators currently doesn't accurately measure the frequencies, and the bins are too large.

- Plot: Frequency diversity (band) (y-axis) vs. number of oscillators (x-axis) for n algorithms. Hypothesis: As the number of oscillators increases, the frequency diversity increases.

- Algorithm: Algorithm finds the best combination of parameters iteratively.

- Implement different signal generation function for Python, more similar to spice (instead of sawtooth).

- Implement custom extrapolator function to accelerate SPICE signal generation.

- Implement MP for SpiPy

- Implement time in seconds vs. rmse based on samples vs. rmse

- Renaming:
    - algo_monte_carlo -> algo_monte_carlo_greedy
    - algo_mcmc -> algo_monte_carlo_ergodic
    - param_util -> param_funcs
    - param_test_py -> param_py_test
    - param_test_spipy -> param_hybrid_test
    - param -> param_py, param_hybrid

- Make phase a separate vector of an ensample similar to weights and allow separate phase tuning without replacing the underlying time series

## Requirements

Pydot is used with NetworkX for drawing graphs.
Pydot requires GraphViz to be installed.
Installation instructions are found [here](https://graphviz.org/download/).

    # Debian/Ubuntu
    sudo apt install graphviz

## Running

To start the simulation execute

    python src/main.py

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

### Deployment on Slurm (Peregrine HPC)

First prepare a job according to <https://wiki.hpc.rug.nl/peregrine/job_management/start>.
Complete job files are in the `scripts` directory.

Start a job; make sure the script is executed from the same directory as done during testing.

    cd path/to/project
    sbatch peregrine_job.sh

List active jobs.

    squeue -u $USER
    
Get information about a running job.

    jobinfo JobID

#### Notes on Runtime

Results except target sweep
    - hardware = Peregrine
    - multiprocessor = False
    - m_averages=7
    - z_ops=1e4
    - n_osc = 100
    - samples = 300
    - wall time: 6 h 30 m
    - max memory: 1.2 GB

Target sweep
    - hardware = Peregrine
    - multiprocessor = False
    - generator=Python
    - z_ops = 5e4
    - n_osc = 100
    - samples = 300
    - wall time. 6 h 18 m
    - max memory: 133.29 MB

Test Sweep with
    - hardware = Laptop
    - multiprocessor = False
    - generator=Hybrid
    - test parameters

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