from pathlib import Path
import time
from typing import Final, List, Tuple

import data_analysis
import data_io
import params
from sample import Sample
from data_preprocessor import norm1d, sample_down, sample_down_int, take_middle_third
import algo_las_vegas
import param_types as party
import experiments

import numpy as np
import matplotlib.pyplot as plt

def init_main() -> Tuple[party.PythonSignalRandArgs, Tuple]:
    """load target and define rand_args"""
    # loading and manipulating the target signal
    raw_sampling_rate, raw_target, raw_dtype = data_io.load_data()
    scale_factor = 0.01
    target_full_len: Final = sample_down_int(raw_target, scale_factor)
    # shorten the target
    target: Final = take_middle_third(target_full_len)
    # save to wav
    sampling_rate = int(scale_factor*raw_sampling_rate)
    data_io.save_signal_to_wav(target, sampling_rate, raw_dtype, Path("data/target_downsampled.wav"))
    # init search params
    rand_args = params.py_rand_args_uniform
    rand_args.samples = len(target) # generated signals match length of target
                                    # NOTE: the sampling rate could also be set lower instead
    return rand_args, (sampling_rate, target, raw_dtype)

def post_main(best_sample: Sample, sampling_rate: int, target: np.ndarray, raw_dtype: np.dtype,
    time: bool = True, freq: bool = False) -> None:
    # normalize target to range 0 1
    target_norm = norm1d(target)

    # find best sample and save
    print(f"signal_sum mean: {np.mean(best_sample.signal_sum)}")
    best_sample.save_sample()
    data_io.save_signal_to_wav(best_sample.signal_sum, sampling_rate, raw_dtype, Path("data/best_sample.wav"))

    norm_sample = Sample.norm_sample(best_sample, target_norm)
    
    # compute regression against target
    reg_sample = Sample.regress_sample(best_sample, target)
    data_io.save_signal_to_wav(reg_sample.signal_sum, sampling_rate, raw_dtype, Path("data/fit.wav"))
    
    # norm regression after fit (good enough)
    norm_reg_sample = Sample.norm_sample(reg_sample, target_norm)

    # plots
    if time: # time-domain
        #hist_rmse(rmse_list, title="sum distribution")
        #hist_rmse(rmse_norm_list, title="norm-sum distribution")
        data_analysis.plot_pred_target(best_sample.signal_sum, target, title="sum")
        data_analysis.plot_pred_target(reg_sample.signal_sum, target, title="regression")
        data_analysis.plot_pred_target(norm_sample.signal_sum, target_norm, title="norm-sum")
        data_analysis.plot_pred_target(norm_reg_sample.signal_sum, target_norm, title="norm after fit")
    if freq: # frequency-domain
        data_analysis.plot_fourier(target, title="target")
        data_analysis.plot_fourier(best_sample.signal_sum, title="sum")
        data_analysis.plot_fourier(reg_sample.signal_sum, title="regression")

    print(f"best_sample.rmse_sum {best_sample.rmse}")
    print(f"best_sample.rmse_sum-norm {norm_sample.rmse}")
    print(f"best_sample.rmse_fit {reg_sample.rmse}")
    print(f"best_sample.rmse_fit-norm {norm_reg_sample.rmse}")
    
    plt.show()

def main():
    rand_args, meta_target = init_main()
    #search = algo_las_vegas.LasVegas(rand_args=rand_args, target=meta_target[1])
    
    const_time_args = params.sweep_py_const_time_args
    algos = params.algo_list
    
    experiments.sweep_const_time_args(rand_args, const_time_args, algos)

    exit()
    @data_analysis.print_timee
    def decorated(): return search.las_vegas(1, None)
    best_sample, j = decorated()
    
    post_main(best_sample, *meta_target)

    #k_dependency_one_shot([1, 10, 100, 500, 1000, 2000], 20, rand_args, target, visual=True)
    #n_dependency(rand_args, target, visual=True)
    #time_elapsed=time.time()-t0
    #print(f"time elapsed: {time_elapsed}")

    # search = SearchModule(
    #             rand_args=rand_args,
    #             target=target)

    #search.random_one_shot()
    #search.random_exploit(search.rand_args.n_osc*5, zero_model=True)
    #best_sample, j = search.las_vegas(5, None)
    #print(f"j: {j}")
    #search.random_weight_exploit(search.rand_args.n_osc*30)
    #search.random_weight_hybrid()
    # best_sample = search.random_one_shot(10, store_det_args=True,
    #     history=True, args_path=Path("data/test_args.pickle"))
    # best_sample, j = search.random_stateless_hybrid(5, store_det_args=True,
    #     history=True, args_path=Path("data/test_args.pickle"))

    # best_sample: Sample = search.las_vegas_weight(weight_init="dist", store_det_args=False)
    #best_sample: Sample = search.random_weight_exploit(k_samples = 3, j_exploits = 100, weight_init="dist")
    


if __name__ == "__main__":
    main()

# IDEA: maybe it would be cool to write an algorithm for self-adjustment of the correct number of oscillators