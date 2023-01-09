# Circuit Generator

- build graphs in Python using networkX
- convert graphs to SPICE netlists 
- run SPICE netlists with ngspice
- solve non-linear transformation (NLT) tasks. for example sinus to square wave.


## DevOps

- Before running a full sweep run a test sweep with m-averages=1 with the final parameters
- Before running a full sweep run a test sweep with the test parameters on the target hardware

## In order of Priority: IDEAS and TODOS before final experiment can run

In order of priority:

### High priority

- Group algorithms for plots without re-running the entire experiment.

- Python: increase sampling rate to avoid aliasing. Verify with individual oscillator plots.

- SPICE: Fix bias issue.
    1.1 Make bias distribution narrower.
    1.2 Optimize bias separately (e.g. at start and end of runtime)
    1.3 Choose normal distribution for bias instead of uniform.

- Reproducibility issue: Maffezzoni et al. report different oscillator frequencies than I have.

- Automate qualitative run against a list of targets (sine, beat, chirp, magpie, yes, okay)

## Medium priority

- Plot Improvement: For oscillators vs. rmse. Show up to 1k oscillators, but draw a dotted line into the plot at around 300 oscillators to indicate what's technologically feasible.

- Plot Improvement: For number of samples vs. rmse; Convert this to signal duration vs. rmse.

- Plot Improvement: save mean, mode, stddev to frequency diversity plot title.

## Low priority

- SPICE: Add phase via delayed voltage delivery.

- Plot Improvement: Histogram over the frequencies in an ensemble of oscillators currently doesn't accurately measure the frequencies, and the bins are too large.

- Plot: Frequency band (y-axis) vs. number of oscillators (x-axis)

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
    