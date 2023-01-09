import copy
from pathlib import Path

import const
import sample
import meta_target
import data_analysis
import data_io
import data_preprocessor
import experimenteur
import experiment_analysis as expan
if const.TEST_PARAMS:
    import params_test_py as params
else:
    import params
import param_types as party
import params_test_spipy
import param_util
import sweep_types as sweety
import gen_signal_spipy

import numpy as np

def qualitative_algo_sweep(algo_sweep: sweety.AlgoSweep, m_target: meta_target.UnionMetaTarget, visual: bool = False) -> None:
    """algo sweep without averaging or collecting results.
    plots the best sample for each algorithm against the target."""
    for awa in algo_sweep.algo_with_args:
        awa: sweety.AlgoWithArgs
        search_alg = awa.Algo(awa.algo_args)
        best_sample, z_ops = search_alg.search()
        if visual: sample.evaluate_prediction(best_sample, m_target, z_ops, search_alg.__class__.__name__)

@data_analysis.print_time
def produce_all_results(algo_sweep: sweety.AlgoSweep, target: np.ndarray, base_rand_args: party.PythonSignalRandArgs) -> None:
    """run all experiments and plot results"""
    show_all = False
    exp = experimenteur.Experimenteur()

    results = exp.run_rand_args_sweep(algo_sweep, params.n_osc_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    experiment_description = expan.plot_n_vs_rmse(df, len(target), show=show_all)
    data_io.hoard_experiment_results(experiment_description, results, df)

    results = exp.run_z_ops_sweep(algo_sweep, params.z_ops_sweep)
    df = expan.conv_results_to_pd(results)
    experiment_description = expan.plot_z_vs_rmse(df, len(target), show=show_all)
    data_io.hoard_experiment_results(experiment_description, results, df)

    results = exp.run_sampling_rate_sweep(params.sampling_rate_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    experiment_description = expan.plot_samples_vs_rmse(df, show=show_all)
    data_io.hoard_experiment_results(experiment_description, results, df)

    freq_sweeps = [params.freq_sweep_from_zero, params.freq_sweep_around_vo2]
    freq_sweep_names = ["freq_range_from_zero", "freq_range_around_vo2"]
    for freq_sweep, freq_sweep_name in zip(freq_sweeps, freq_sweep_names):
        results = exp.run_rand_args_sweep(algo_sweep, freq_sweep, base_rand_args)
        df = expan.conv_results_to_pd(results)
        experiment_description = expan.plot_freq_range_vs_rmse(df, len(target), freq_sweep_name, show=show_all)
        data_io.hoard_experiment_results(experiment_description, results, df)
        
    results = exp.run_rand_args_sweep(algo_sweep, params.weight_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    experiment_description = expan.plot_weight_range_vs_rmse(df, len(target), show=show_all)
    data_io.hoard_experiment_results(experiment_description, results, df)

    results = exp.run_rand_args_sweep(algo_sweep, params.phase_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    experiment_description = expan.plot_phase_range_vs_rmse(df, len(target), show=show_all)
    data_io.hoard_experiment_results(experiment_description, results, df)

    results = exp.run_rand_args_sweep(algo_sweep, params.offset_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    experiment_description = expan.plot_offset_range_vs_rmse(df, len(target), show=show_all)
    data_io.hoard_experiment_results(experiment_description, results, df)

    results = exp.run_rand_args_sweep(algo_sweep, params.amplitude_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    experiment_description = expan.plot_amplitude_vs_rmse(df, len(target), show=show_all)
    data_io.hoard_experiment_results(experiment_description, results, df)

def run_multi_directional_experiment():
    """experiments whose parameters are based on other experiments"""
    # TODO: take best algorithm
    # then also increase the z_ops to see if the weight-range-to-rmse curve flattens
    # same for rmse vs n_osc
    # also for frequency band


# TODO: address running multiple targets automatically
# TODO: check that multiprocessing works
# TODO: consider different signal generation function for Python, more similar to spice
# TODO: fix aliasing for Python
@data_analysis.print_time
def main():
    # SpiPy
    if False:
        rand_args = params_test_spipy.spice_rand_args_uniform
        m_target = meta_target.MetaTargetTime(rand_args)

        # scale the number of samples in the target to the number of samples produced by spice
        signal_generator = gen_signal_spipy.SpipySignalGenerator()
        spice_samples = signal_generator.estimate_number_of_samples(rand_args)
        m_target.adjust_samples(spice_samples)

        if False:
            # synthetic target signal
            from scipy import signal
            t = np.linspace(0, 1, spice_samples)
            duration = rand_args.time_stop-rand_args.time_start
            m_target.signal = signal.sawtooth(2 * np.pi * 5 * t) * 10
            #m_target.signal = signal.chirp(t, f0=1, f1=100, t1=1, method="linear") * 10
            #m_target.signal = const.RNG.normal(-1, 1, spice_samples)*5 + const.RNG.uniform(-1, 1, spice_samples)*2
            m_target.time = np.linspace(0, duration, spice_samples+1)[0:-1]
        
        data_analysis.plot_signal(m_target.signal, m_target.time, show=True)

        algo_sweep = param_util.init_algo_sweep(m_target.signal, rand_args, sig_generator=signal_generator, max_z_ops=5e3, m_averages=1)

        qualitative_algo_sweep(algo_sweep, m_target, visual=True)

    # Python
    if True:
        rand_args = params.py_rand_args_uniform
        m_target = meta_target.MetaTargetSample(rand_args)
        
        if False: # synthetic target signal
            from scipy import signal
            t = np.linspace(0, 1, rand_args.samples)
            m_target.signal = signal.sawtooth(2 * np.pi * 5 * t) * 10
            m_target.signal = const.RNG.normal(-1, 1, rand_args.samples)*5 + const.RNG.uniform(-1, 1, rand_args.samples)*2
            data_analysis.plot_signal(m_target.signal, show=True)

        algo_sweep_test = param_util.init_algo_sweep(m_target.signal, rand_args, max_z_ops=1e2, m_averages=2)

        # qualitative_algo_sweep(algo_sweep_test, m_target, visual=False)
        produce_all_results(algo_sweep_test, m_target.signal, rand_args)

if __name__ == "__main__":
    main()

# IDEA: maybe it would be cool to write an algorithm for self-adjustment of the correct number of oscillators