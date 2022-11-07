import copy
from functools import wraps
from pathlib import Path
from typing import Callable, Final, List, Tuple

import data_analysis
import data_io
import test_params as params
import sample
import data_preprocessor
import param_types as party
import experimenteur
import result_types as resty
import experiment_analysis as expan
import meta_target
import const


import numpy as np
import matplotlib.pyplot as plt

from sweep_types import AlgoSweep


def post_main(best_sample: sample.Sample, m_target: meta_target.MetaTarget,
    z_ops: int, alg_name: str, plot_time: bool = True, plot_freq: bool = False) -> None:
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

def qualitative_algo_sweep(algo_sweep: AlgoSweep, m_target: meta_target.MetaTarget, visual: bool = False) -> None:
    """algo sweep without averaging or collecting results.
    plots the best sample for each algorithm against the target."""
    for Algo, algo_args in zip(algo_sweep.algo, algo_sweep.algo_args):
        search_alg = Algo(algo_args)
        best_sample, z_ops = search_alg.search()
        if visual: post_main(best_sample, m_target, z_ops, search_alg.__class__.__name__)

def produce_all_results(algo_sweep: AlgoSweep, target: np.ndarray, base_rand_args: party.PythonSignalRandArgs) -> None:
    """run all experiments and plot results"""
    show_all = True
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

    for freq_sweep in [params.freq_sweep_from_zero, params.freq_sweep_around_vo2]:
        results = exp.run_rand_args_sweep(algo_sweep, freq_sweep, base_rand_args)
        df = expan.conv_results_to_pd(results)
        expan.plot_freq_range_vs_rmse(df, len(target), show=show_all)

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


@data_analysis.print_time
def main():
    rand_args = params.py_rand_args_uniform
    m_target = meta_target.MetaTarget(rand_args)
    algo_sweep = params.init_algo_sweep(m_target.signal, rand_args)

    # qualitative_algo_sweep(algo_sweep, m_target, visual=True)
    # exit()
    # produce_all_results(algo_sweep, m_target.signal, rand_args)
    exp = experimenteur.Experimenteur()
    # results = exp.run_algo_sweep(algo_sweep)
    # results = exp.run_rand_args_sweep(algo_sweep, params.freq_sweep_from_zero, rand_args)
    # results = exp.run_rand_args_sweep(algo_sweep, params.expo_time_sweep, rand_args)
    # results = exp.run_sampling_rate_sweep(params.sampling_rate_sweep, rand_args)
    # results = exp.run_z_ops_sweep(algo_sweep, params.z_ops_sweep)

    results = exp.run_z_ops_sweep(algo_sweep, params.z_ops_sweep)
    df = expan.conv_results_to_pd(results)
    expan.plot_z_vs_rmse(df, len(m_target.signal), show=True)
    exit()

    for r in results:
        print(f"{r}\n")
    data_io.pickle_results(results, Path("data/results.pickle"))

    df = expan.conv_results_to_pd(results)
    df.to_csv(Path("data/experiment.csv"))

    print(df.describe())
    print(df["samples"])
    print(f"df.columns: {df.columns}")

    expan.plot_samples_vs_rmse(df, show=True)

if __name__ == "__main__":
    main()

# IDEA: maybe it would be cool to write an algorithm for self-adjustment of the correct number of oscillators