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
import matplotlib.pyplot as plt



def post_main(best_sample: sample.Sample, m_target: meta_target.MetaTarget,
    z_ops: int, alg_name: str, plot_time: bool = True, plot_freq: bool = False) -> None:
    m_target.signal = m_target.y_signal
    # TODO: x_time
    # normalize target to range 0 1
    target_norm = data_preprocessor.norm1d(m_target.signal)

    # find best sample and save
    print(f"signal_sum mean: {np.mean(best_sample.weighted_sum)}")
    best_sample.save_sample()
    data_io.save_signal_to_wav(best_sample.weighted_sum, m_target.sampling_rate, m_target.dtype, Path("data/best_sample.wav"))

    norm_sample = sample.Sample.norm_sample(best_sample, target_norm)

    # compute regression against target
    reg_sample = sample.Sample.regress_sample(best_sample, m_target.signal)
    data_io.save_signal_to_wav(reg_sample.weighted_sum, m_target.sampling_rate, m_target.dtype, Path("data/fit.wav"))

    # norm regression after fit (good enough)
    norm_reg_sample = sample.Sample.norm_sample(reg_sample, target_norm)

    # plots
    if plot_time: # time-domain
        #hist_rmse(rmse_list, title="sum distribution")
        #hist_rmse(rmse_norm_list, title="norm-sum distribution")
        data_analysis.plot_pred_target(best_sample.weighted_sum, m_target.signal, title=f"{alg_name}, sum")
        data_analysis.plot_pred_target(reg_sample.weighted_sum, m_target.signal, title=f"{alg_name}, regression")
        data_analysis.plot_pred_target(norm_sample.weighted_sum, target_norm, title=f"{alg_name}, norm-sum")
        data_analysis.plot_pred_target(norm_reg_sample.weighted_sum, target_norm, title=f"{alg_name}, norm after fit")
    if plot_freq: # frequency-domain
        data_analysis.plot_fourier(m_target.signal, title=f"{alg_name}, target")
        data_analysis.plot_fourier(best_sample.weighted_sum, title=f"{alg_name}, sum")
        data_analysis.plot_fourier(reg_sample.weighted_sum, title=f"{alg_name}, regression")

    print(f"{alg_name}")
    print(f"z_ops: {z_ops}")
    print(f"best_sample.rmse_sum {best_sample.rmse}")
    print(f"best_sample.rmse_sum-norm {norm_sample.rmse}")
    print(f"best_sample.rmse_fit {reg_sample.rmse}")
    print(f"best_sample.rmse_fit-norm {norm_reg_sample.rmse}")

    plt.show()

def qualitative_algo_sweep(algo_sweep: sweety.AlgoSweep, m_target: meta_target.MetaTarget, visual: bool = False) -> None:
    """algo sweep without averaging or collecting results.
    plots the best sample for each algorithm against the target."""
    for awa in algo_sweep.algo_with_args:
        awa: sweety.AlgoWithArgs
        search_alg = awa.Algo(awa.algo_args)
        best_sample, z_ops = search_alg.search()
        if visual: post_main(best_sample, m_target, z_ops, search_alg.__class__.__name__)

@data_analysis.print_time
def produce_all_results(algo_sweep: sweety.AlgoSweep, target: np.ndarray, base_rand_args: party.PythonSignalRandArgs) -> None:
    """run all experiments and plot results"""
    show_all = False
    exp = experimenteur.Experimenteur()

    results = exp.run_rand_args_sweep(algo_sweep, params.n_osc_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_n_vs_rmse(df, len(target), show=show_all)

    results = exp.run_z_ops_sweep(algo_sweep, params.z_ops_sweep)
    df = expan.conv_results_to_pd(results)
    expan.plot_z_vs_rmse(df, len(target), show=show_all)

    results = exp.run_sampling_rate_sweep(params.sampling_rate_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_samples_vs_rmse(df, show=show_all)

    freq_sweeps = [params.freq_sweep_from_zero, params.freq_sweep_around_vo2]
    freq_sweep_names = ["freq_range_from_zero", "freq_range_around_vo2"]
    for freq_sweep, freq_sweep_name in zip(freq_sweeps, freq_sweep_names):
        results = exp.run_rand_args_sweep(algo_sweep, freq_sweep, base_rand_args)
        df = expan.conv_results_to_pd(results)
        expan.plot_freq_range_vs_rmse(df, len(target), freq_sweep_name, show=show_all)
        
    results = exp.run_rand_args_sweep(algo_sweep, params.weight_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_weight_range_vs_rmse(df, len(target), show=show_all)

    results = exp.run_rand_args_sweep(algo_sweep, params.phase_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_phase_range_vs_rmse(df, len(target), show=show_all)

    results = exp.run_rand_args_sweep(algo_sweep, params.offset_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_offset_range_vs_rmse(df, len(target), show=show_all)

    results = exp.run_rand_args_sweep(algo_sweep, params.amplitude_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_amplitude_vs_rmse(df, len(target), show=show_all)

def run_multi_directional_experiment():
    """experiments whose parameters are based on other experiments"""
    # TODO: take best algorithm
    # then also increase the z_ops to see if the weight-range-to-rmse curve flattens
    # same for rmse vs n_osc
    # also for frequency band


# TODO: pickle figures after save to png
# TODO: store intermediate pickle of results
@data_analysis.print_time
def main():
    rand_args = params_test_spipy.spice_rand_args_uniform
    m_target = meta_target.MetaTargetTime(rand_args)

    # scale the number of samples in the target to the number of samples produced by spice
    signal_generator = gen_signal_spipy.SpipySignalGenerator()
    spice_samples = signal_generator.estimate_number_of_samples(rand_args)
    m_target.adjust_samples(spice_samples)

    # synthetic target signal
    # from scipy import signal
    # t = np.linspace(0, 1, spice_samples)
    # m_target.y_signal = signal.sawtooth(2 * np.pi * 5 * t) * 10
    # data_analysis.plot_signal(m_target.y_signal, show=True)

    algo_sweep = param_util.init_algo_sweep(m_target.y_signal, rand_args, sig_generator=signal_generator, max_z_ops=30, m_averages=1)

    qualitative_algo_sweep(algo_sweep, m_target, visual=True)

    exit()
    rand_args = params.py_rand_args_uniform
    m_target = meta_target.MetaTarget(rand_args)
    algo_sweep = param_util.init_algo_sweep(m_target.signal, rand_args)

    qualitative_algo_sweep(algo_sweep, m_target, visual=True)
    produce_all_results(algo_sweep, m_target.signal, rand_args)

if __name__ == "__main__":
    main()

# IDEA: maybe it would be cool to write an algorithm for self-adjustment of the correct number of oscillators