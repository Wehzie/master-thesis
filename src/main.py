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


import numpy as np
import matplotlib.pyplot as plt

from sweep_types import AlgoSweep


def post_main(best_sample: sample.Sample, sampling_rate: int, target: np.ndarray, raw_dtype: np.dtype,
    z_ops: int, plot_time: bool = True, plot_freq: bool = False) -> None:
    # normalize target to range 0 1
    target_norm = data_preprocessor.norm1d(target)

    # find best sample and save
    print(f"signal_sum mean: {np.mean(best_sample.weighted_sum)}")
    best_sample.save_sample()
    data_io.save_signal_to_wav(best_sample.weighted_sum, sampling_rate, raw_dtype, Path("data/best_sample.wav"))

    norm_sample = sample.Sample.norm_sample(best_sample, target_norm)
    
    # compute regression against target
    reg_sample = sample.Sample.regress_sample(best_sample, target)
    data_io.save_signal_to_wav(reg_sample.weighted_sum, sampling_rate, raw_dtype, Path("data/fit.wav"))
    
    # norm regression after fit (good enough)
    norm_reg_sample = sample.Sample.norm_sample(reg_sample, target_norm)

    # plots
    if plot_time: # time-domain
        #hist_rmse(rmse_list, title="sum distribution")
        #hist_rmse(rmse_norm_list, title="norm-sum distribution")
        data_analysis.plot_pred_target(best_sample.weighted_sum, target, title="sum")
        data_analysis.plot_pred_target(reg_sample.weighted_sum, target, title="regression")
        data_analysis.plot_pred_target(norm_sample.weighted_sum, target_norm, title="norm-sum")
        data_analysis.plot_pred_target(norm_reg_sample.weighted_sum, target_norm, title="norm after fit")
    if plot_freq: # frequency-domain
        data_analysis.plot_fourier(target, title="target")
        data_analysis.plot_fourier(best_sample.weighted_sum, title="sum")
        data_analysis.plot_fourier(reg_sample.weighted_sum, title="regression")

    print(f"z_ops: {z_ops}")
    print(f"best_sample.rmse_sum {best_sample.rmse}")
    print(f"best_sample.rmse_sum-norm {norm_sample.rmse}")
    print(f"best_sample.rmse_fit {reg_sample.rmse}")
    print(f"best_sample.rmse_fit-norm {norm_reg_sample.rmse}")
    
    plt.show()

def simple_algo_sweep(algo_sweep: AlgoSweep, sampling_rate: int, target: np.ndarray, raw_dtype: np.dtype, visual: bool = False) -> None:
    """algo sweep without averaging or collecting results"""
    for Algo, algo_args in zip(algo_sweep.algo, algo_sweep.algo_args):
        search_alg = Algo(algo_args)
        best_sample, z_ops = search_alg.search()
        if visual: post_main(best_sample, sampling_rate, target, raw_dtype, z_ops)

def produce_all_results(algo_sweep: AlgoSweep, target: np.ndarray) -> None:
    """run all experiments and plot results"""
    show_all = True
    exp = experimenteur.Experimenteur(mp = True)

    results = exp.run_rand_args_sweep(algo_sweep, params.expo_time_sweep, params.py_rand_args_uniform)
    df = expan.conv_results_to_pd(results)
    expan.plot_n_vs_rmse(df, len(target), show=show_all)

    results = exp.run_z_ops_sweep(algo_sweep, params.z_ops_sweep)
    df = expan.conv_results_to_pd(results)
    expan.plot_z_vs_rmse(df, len(target), show=show_all)

    results = exp.run_rand_args_sweep(algo_sweep, params.freq_sweep, params.py_rand_args_uniform)
    df = expan.conv_results_to_pd(results)
    expan.plot_freq_range_vs_rmse(df, len(target), show=show_all)

    results = exp.run_rand_args_sweep(algo_sweep, params.weight_sweep, params.py_rand_args_uniform)
    df = expan.conv_results_to_pd(results)
    expan.plot_weight_range_vs_rmse(df, len(target), show=show_all)

    results = exp.run_rand_args_sweep(algo_sweep, params.phase_sweep, params.py_rand_args_uniform)
    df = expan.conv_results_to_pd(results)
    expan.plot_phase_range_vs_rmse(df, len(target), show=show_all)

    results = exp.run_rand_args_sweep(algo_sweep, params.offset_sweep, params.py_rand_args_uniform)
    df = expan.conv_results_to_pd(results)
    expan.plot_offset_range_vs_rmse(df, len(target), show=show_all)

    results = exp.run_rand_args_sweep(algo_sweep, params.amplitude_sweep, params.py_rand_args_uniform)
    df = expan.conv_results_to_pd(results)
    expan.plot_amplitude_vs_rmse(df, len(target), show=show_all)

@data_analysis.print_time
def main():
    rand_args, meta_target = params.init_target2rand_args()
    target = meta_target[1]
    algo_sweep = params.init_algo_sweep(target)

    # produce_all_results(algo_sweep, target)
    # simple_algo_sweep(algo_sweep, *meta_target)
    
    exp = experimenteur.Experimenteur(mp = True)
    # results = exp.run_algo_sweep(algo_sweep)
    results = exp.run_rand_args_sweep(algo_sweep, params.offset_sweep, params.py_rand_args_uniform)
    # results = exp.run_rand_args_sweep(algo_sweep, params.expo_time_sweep, params.py_rand_args_uniform)
    # results = exp.run_sampling_rate_sweep(params.sampling_rate_sweep)
    # results = exp.run_z_ops_sweep(algo_sweep, params.z_ops_sweep)
    for r in results:
        print(f"{r}\n")
    data_io.pickle_results(results, Path("data/results.pickle"))

    df = expan.conv_results_to_pd(results)
    df.to_csv(Path("data/experiment.csv"))

    print(df.describe())
    print(f"df.columns: {df.columns}")
    
    expan.plot_offset_range_vs_rmse(df, len(target), show=True)

if __name__ == "__main__":
    main()

# IDEA: maybe it would be cool to write an algorithm for self-adjustment of the correct number of oscillators