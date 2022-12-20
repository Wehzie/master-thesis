# Circuit Generator

- build graphs in Python using networkX
- convert graphs to SPICE netlists 
- run SPICE netlists with ngspice
- solve non-linear transformation (NLT) tasks. for example sinus to square wave.


## DevOps

- Before running a full sweep run a test sweep with m-averages=1 with the final parameters
- Before running a full sweep run a test sweep with the test parameters on the target hardware

## In order of Priority: IDEAS and TODOS before final experiment can run

1. Python: increase sampling rate to avoid aliasing. Verify with individual oscillator plots.
2. SPICE: Fix bias issue.
    1.1 Make bias distribution narrower.
    1.2 Optimize bias separately (e.g. at start and end of runtime)
    1.3 Choose normal distribution for bias instead of uniform.
3. Plot Improvement: Remove sample index from all figures. Add arbitrary unit [a.u.] notes to Python times. Add sampling rate.
4. Plot Improvement: add units to the individual oscillator plot.
5. Reproducibility issue: Maffezzoni et al. report different oscillator frequencies than I have.
6. Algorithm: Implement Metropolis-Hastings and Simulated Annealing with acceptance probability instead of variable number of weights.
7. SPICE: Add phase via delayed voltage delivery.
8. Plot Improvement: Histogram over the frequencies in an ensemble of oscillators currently doesn't accurately measure the frequencies, and the bins are too large.
9. Plot: Frequency band (y-axis) vs. number of oscillators (x-axis)
10. Algorithm: Algorithm finds the best combination of parameters iteratively.


- Plot Improvement: For oscillators vs. rmse. Show up to 1k oscillators, but draw a dotted line into the plot at around 300 oscillators to indicate what's technologically feasible.
- Plot Improvement: For number of samples vs. rmse; Convert this to signal duration vs. rmse.
- Plot Improvement: save mean, mode, stddev to frequency diversity plot title.