
# add code in src to path
import os
import sys
import copy
import csv
from pathlib import Path
import re
from typing import Callable, Final, List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
from data_analysis import compute_rmse, hist_rmse, plot_n, plot_pred_target, plot_signal, plot_fourier
from data_preprocessor import norm1d, sample_down, sample_down_int, take_middle_third
from data_io import DATA_PATH, load_data, load_data_numpy, save_signal_to_wav
import gen_signal_python
import test_params as params
import data_io
import data_analysis

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

from param_types import PythonSignalDetArgs, PythonSignalRandArgs
from search_module import Sample, SearchModule

def init_search():
    # loading and manipulating the target signal
    raw_sampling_rate, raw_target, raw_dtype = data_io.load_data()
    scale_factor = 0.01
    target_full_len: Final = sample_down_int(raw_target, scale_factor)
    # shorten the target
    target: Final = take_middle_third(target_full_len)
    # normalize to range 0 1
    target_norm: Final = norm1d(target)
    # save to wav
    sampling_rate = int(scale_factor*raw_sampling_rate)
    data_io.save_signal_to_wav(target, sampling_rate, raw_dtype, Path("data/target_downsampled.wav"))
    # init search params
    rand_args = params.py_rand_args_uniform
    rand_args.samples = len(target) # generated signals match length of target
                                    # NOTE: the sampling rate could also be set lower instead
    return target, rand_args

def run_all_algs(search: SearchModule) -> None:
    best_sample, dradels = search.random_one_shot(params.k_samples, False, False, None)
    # best_sample = search.random_one_shot_weight()
    #   zero, ones, uniform, dist
    best_sample, dradels = search.las_vegas(params.k_samples, None, False, False, None)
    for w in params.weight_inits: #   zero, ones, uniform, dist
        best_sample, dradels = search.las_vegas_weight(params.k_samples, w, False, False, None)
    best_sample, dradels = search.random_exploit(params.k_samples, params.j_exploits)
    best_sample, dradels = search.random_exploit(params.k_samples, params.j_exploits, zero_model=True)
    for w in params.weight_inits: #   zero, ones, uniform, dist
        best_sample, dradels = search.random_weight_exploit(params.k_samples, params.j_exploits, w, False, False, None)

def eval_alg(search: SearchModule, best_sample: Sample) -> None:
        # find best sample and save
    print(f"mean: {np.mean(best_sample.signal_sum)}")
    data_io.save_sample(best_sample)
    data_io.save_signal_to_wav(best_sample.signal_sum, sampling_rate, raw_dtype, Path("data/best_sample.wav"))

    norm_sample = Sample.norm_sample(best_sample, target_norm)
    
    # compute regression against target
    reg_sample = Sample.regress_sample(best_sample, target)
    data_io.save_signal_to_wav(reg_sample.signal_sum, sampling_rate, raw_dtype, Path("data/fit.wav"))
    
    # norm regression after fit (good enough)
    norm_reg_sample = Sample.norm_sample(reg_sample, target_norm)

    # plots
    if True: # time-domain
        #hist_rmse(rmse_list, title="sum distribution")
        #hist_rmse(rmse_norm_list, title="norm-sum distribution")
        data_analysis.plot_pred_target(best_sample.signal_sum, target, title="sum")
        data_analysis.plot_pred_target(reg_sample.signal_sum, target, title="regression")
        data_analysis.plot_pred_target(norm_sample.signal_sum, target_norm, title="norm-sum")
        data_analysis.plot_pred_target(norm_reg_sample.signal_sum, target_norm, title="norm after fit")
    if True: # frequency-domain
        data_analysis.plot_fourier(target, title="target")
        data_analysis.plot_fourier(best_sample.signal_sum, title="sum")
        data_analysis.plot_fourier(reg_sample.signal_sum, title="regression")

    print(f"best_sample.rmse_sum {best_sample.rmse}")
    print(f"best_sample.rmse_sum-norm {norm_sample.rmse}")
    print(f"best_sample.rmse_fit {reg_sample.rmse}")
    print(f"best_sample.rmse_fit-norm {norm_reg_sample.rmse}")
    

def test_main():
    target, rand_args = init_search()
    search = SearchModule(
                rand_args=rand_args,
                target=target)
    run_all_algs(search)
    

    
