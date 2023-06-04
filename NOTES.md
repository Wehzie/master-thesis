## In order of Priority: IDEAS and TODOS

In order of priority:

### High priority

- Set legend false on algo ranking bar chart

- Do target sweep with linear regression only or 1 or 2 more good algorithms.

### Medium priority

- Implement Las Vegas initialization options, in particular one with a full set of oscillators.

- Plot Improvement: custom legend for mask with multiple members in a group. Such that a group shares a color and one entry in the legend (individual algorithm names shouldn't be visible).

- Plot Improvement: save mean, mode, stddev to frequency diversity plot title.

- Add magpie to target sweep.

- Plot improvement: Reuse algorithm bar chart with z-ops sweep.

- Python: increase sampling rate to avoid aliasing. Verify with individual oscillator plots.

- Map R to Freq in analysis for Hybrid signal generator

- Plot Improvement: For oscillators vs. rmse. Show up to 1k oscillators, but draw a dotted line into the plot at around 300 oscillators to indicate what's technologically feasible.

- Reproducibility issue: Maffezzoni et al. report different oscillator frequencies than I have.

### Low priority

- Profile Python's memory usage.

- SPICE: Add phase via delayed voltage delivery.

- Plot: Frequency diversity (band) (y-axis) vs. number of oscillators (x-axis) for n algorithms. Hypothesis: As the number of oscillators increases, the frequency diversity increases.

- Algorithm: Algorithm finds the best combination of parameters iteratively.

- Implement different signal generation function for Python, more similar to spice (instead of sawtooth).

- Renaming:
    - algo_monte_carlo -> algo_monte_carlo_greedy
    - algo_mcmc -> algo_monte_carlo_ergodic
    - param_util -> param_funcs
    - param_test_py -> param_py_test
    - param_test_spipy -> param_hybrid_test
    - param -> param_py, param_hybrid

- Make phase a separate vector of an ensample similar to weights and allow separate phase tuning without replacing the underlying time series

- solve non-linear transformation (NLT) tasks. for example sinus to square wave.
