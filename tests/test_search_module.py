
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
from data_preprocessor import norm, sample_down, sample_down_int, take_middle_third
from data_io import DATA_PATH, load_data, load_data_numpy, save_signal_to_wav
import gen_signal_python
import params

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

from param_types import PythonSignalDetArgs, PythonSignalRandArgs
from search_module import Sample, SearchModule

def test_main() -> None:
    # loading and manipulating the target signal
    raw_sampling_rate, raw_target, raw_dtype = load_data()
    scale_factor = 0.01
    target_full_len: Final = sample_down_int(raw_target, scale_factor)
    # shorten the target
    target: Final = take_middle_third(target_full_len)
    # normalize to range 0 1
    target_norm: Final = norm(target)
    # save to wav
    sampling_rate = int(scale_factor*raw_sampling_rate)
    save_signal_to_wav(target, sampling_rate, raw_dtype, Path("data/target_downsampled.wav"))
    
    # initialize and start search
    rand_args = params.py_rand_args_uniform
    rand_args.samples = len(target) # generated signals match length of target

    kwargs = {
        "target": target,
        "target_norm": target_norm,
        "sampling_rate": sampling_rate,
        "raw_dtype": raw_dtype
    }

    search = SearchModule(
        k_samples=10,
        rand_args=rand_args,
        target=target)
    search.random_one_shot()
    main(search, **kwargs)

    search = SearchModule(
    k_samples=10,
    rand_args=rand_args,
    target=target)
    search.random_one_shot()
    main(search, **kwargs)
    
    search = SearchModule(
    k_samples=10,
    rand_args=rand_args,
    target=target)
    search.random_one_shot()
    main(search, **kwargs)

def main(search: SearchModule, target: np.ndarray, target_norm: np.ndarray, sampling_rate: int, raw_dtype: np.dtype) -> None:
    
    # without random phase shifts between 0 and 2 pi
    # signals will overlap at the first sample
    if True:
        for s in search.samples:
            s.sum_y[0] = 0 # set first point to 0
            s.rmse_sum = compute_rmse(s.sum_y, target)
            s.rmse_norm = compute_rmse(norm(s.sum_y), target_norm)

    # find best sample and save
    best_sample, rmse_list, rmse_norm_list = search.gather_samples()
    best_sample.save()
    save_signal_to_wav(best_sample.sum_y, sampling_rate, raw_dtype, Path("data/best_sample.wav"))

    # normalize best sample
    norm_sum = norm(best_sample.sum_y)
    norm_rmse = compute_rmse(norm_sum, target_norm)
    
    # compute regression against target
    reg = Sample.regress_linear(best_sample.matrix_y, target, verbose=False)
    pred = Sample.predict(best_sample.matrix_y, reg.coef_, reg.intercept_)
    best_sample.fit_y = pred
    save_signal_to_wav(best_sample.fit_y, sampling_rate, raw_dtype, Path("data/fit.wav"))
    best_sample.rmse_fit = compute_rmse(pred, target)
    
    # norm regression after fit (good enough)
    norm_reg = norm(best_sample.fit_y)
    norm_reg_rmse = compute_rmse(norm_reg, target_norm)
    
    # plots
    if True: # time-domain
        #hist_rmse(rmse_list, title="sum distribution")
        #hist_rmse(rmse_norm_list, title="norm-sum distribution")
        plot_pred_target(best_sample.sum_y, target, title="sum")
        plot_pred_target(best_sample.fit_y, target, title="regression")
        plot_pred_target(norm_sum, target_norm, title="norm-sum")
        plot_pred_target(norm_reg, target_norm, title="norm after fit")
    if True: # frequency-domain
        plot_fourier(target, title="target")
        plot_fourier(best_sample.sum_y, title="sum")
        plot_fourier(best_sample.fit_y, title="regression")
