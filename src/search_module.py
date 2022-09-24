import copy
import csv
from operator import mod
from pathlib import Path
import re
import time
from typing import Callable, Final, List, Tuple

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
from run_experiments import k_dependency_hybrid, k_dependency_one_shot, n_dependency

class Sample():
    """a sample corresponds to a randomly drawn signal approximating the target"""
    def __init__(self, sum_y: np.ndarray,
        signal_matrix: np.ndarray, det_param_li: List[PythonSignalDetArgs]):

                                                    # store single oscillator signals for linear regression
        self.matrix_y: np.ndarray = signal_matrix   #   matrix of single-oscillator signals
        self.sum_y: Final = sum_y   # generated signal y coords
        self.fit_y = None           # summed signal fit to target with regression
        
        # list of determined parameters
        #   one set of parameters corresponds to a single oscillator
        self.det_param_li = det_param_li
        
        self.rmse_sum = None        # root mean square error of summed signal
        self.rmse_fit = None        # root mean square error of regression fit signal
        self.rmse_norm = None       # rmse after normalization of target and sum_y

    def __str__(self) -> str:
        x = f"x: {self.sum_x}\n"
        y = f"y: {self.sum_y}\n"
        rmse_sum = f"rmse sum: {self.rmse_sum}\n"
        rmse_fit = f"rmse fit: {self.rmse_fit}"
        return x + y + rmse_sum + rmse_fit

    def save(self, path: Path = "data/best_sample.csv") -> None:
        """save the determined parameters of a sample to a CSV"""
        with open(path, "w") as f:
            writer = csv.writer(f)
            # write header
            writer.writerow(self.det_param_li[0].__dict__)
            # write data
            for osc in self.det_param_li:
                writer.writerow(osc.__dict__.values())

    @staticmethod
    def regress_linear(p: np.ndarray, t: np.ndarray, verbose: bool = False):
        """apply linear regression"""
        # matrix_y refers to the y-values of a generated signal
        # the individual signals are used as regressors 

        r = p.T

        if verbose:
            print("Computing regression")
            print(f"r {r}")
            print(f"r.shape {r.shape}")
            print(f"t {t}")
            print(f"t.shape {t.shape}")
        
        reg = LinearRegression().fit(r, t)

        if verbose:
            print(f"Coefficient: {reg.coef_}")
            print(f"Coefficient.shape: {reg.coef_.shape}")
            print(f"Intercept: {reg.intercept_}")

        return reg

    @staticmethod
    def predict(X: list, coef: np.ndarray, intercept: float) -> np.ndarray:
        """generate approximation of target, y"""
        fit = np.sum(X.T * coef, axis=1) + intercept
        return fit

class SearchModule():

    def __init__(self, rand_args: PythonSignalRandArgs, k_samples: int, target: np.ndarray, start: np.ndarray = None):
        self.rand_args = rand_args # search parameters
        self.k_samples = k_samples # number of samples
        self.target = target # target function to approximate
        self.start = start # a matrix to start search
        
        self.samples: List[Sample] = list() # list of samples and results
    
    def __str__(self) -> str:
        rand_args = f"rand_args: {self.rand_args}\n"
        k_samples = f"k_samples: {self.k_samples}\n"
        sig_gen_func = f"sig_gen_func: {self.sig_gen_func.__name__}\n"
        samples = ""
        for s in self.samples:
            samples += str(s) + "\n"
        return rand_args + k_samples + sig_gen_func + samples

    def random_one_shot(self) -> None:
        """generate k-signals which are a sum of n-oscillators"""
        for _ in tqdm(range(self.k_samples)):
            # TODO: idem, sig_gen_func
            # compose a signal of single oscillators
            signal_matrix, det_param_li = gen_signal_python.sum_atomic_signals(self.rand_args)
            signal_sum = sum(signal_matrix)
            # store summed signal, single signals, and parameters for each single signal
            sample = Sample(signal_sum, signal_matrix, det_param_li)
            # calculate rmse
            sample.rmse_sum = compute_rmse(signal_sum, self.target)
            # store sample
            self.samples.append(sample)

    def random_stateless_one_shot(self) -> tuple:
        """generate k-signals which are a sum of n-oscillators
        only return the best signal"""

        best_rmse = np.inf
        best_matrix = None
        for _ in tqdm(range(self.k_samples)):
            # compose a signal of single oscillators
            temp_matrix, _ = gen_signal_python.sum_atomic_signals(self.rand_args)
            temp_sum = sum(temp_matrix)
            temp_rmse = compute_rmse(temp_sum, self.target)
            if temp_rmse < best_rmse:
                best_matrix = temp_matrix
                best_rmse = temp_rmse

        self.samples.append(best_matrix)
        return best_matrix, best_rmse
    
    def random_hybrid(self) -> None:
        """generate k-signals which are a sum of n-oscillators
        only accept an oscillator into a sum if it decreases RMSE
        """
        mod_args = copy.deepcopy(self.rand_args)
        mod_args.n_osc = 1
        # TODO: hacky solution to draw 1 oscillator from sum_atomic signals
        for _ in tqdm(range(self.k_samples)):
            # init empty sample
            det_param_li = list()
            signal_matrix = np.zeros((self.rand_args.n_osc, self.rand_args.samples))
            zeros = np.zeros(self.rand_args.samples)
            sample = Sample(zeros, signal_matrix, det_param_li)
            sample.rmse_sum = np.inf

            i = 0 # number of added oscillators
            j = 0 # number of loops
            while i < self.rand_args.n_osc:
                # draw an oscillator
                signal, det_param = gen_signal_python.sum_atomic_signals(mod_args)
                signal = signal.flatten()
                # compute rmse with new oscillator
                temp_sum = sample.sum_y + signal
                temp_rmse = compute_rmse(temp_sum, self.target)

                # accept oscillators when they lower the RMSE
                if temp_rmse < sample.rmse_sum:
                    sample.det_param_li.append(det_param[0])
                    sample.sum_y = temp_sum
                    sample.rmse_sum = temp_rmse
                    sample.matrix_y[i,:] = signal
                    i += 1
                j += 1
            
            self.samples.append(sample)
            print(f"\nj={j}\n")

    def random_stateless_hybrid(self) -> Tuple[np.ndarray, float]:
        """generate k-signals which are a sum of n-oscillators
        only accept an oscillator into a sum if it decreases RMSE
        for k samples, return only that with lowest RMSE
        """
        # TODO: hacky solution to draw 1 oscillator from sum_atomic signals
        mod_args = copy.deepcopy(self.rand_args)
        mod_args.n_osc = 1

        best_model = None
        best_rmse = np.inf
        best_model_j = np.inf
        for _ in tqdm(range(self.k_samples)):
            # init empty sample            
            model = np.zeros((self.rand_args.n_osc, self.rand_args.samples))
            rmse = np.inf

            i = 0 # number of added oscillators
            j = 0 # number of iterations
            while i < self.rand_args.n_osc: # TODO: non-deterministic runtime
                # draw an oscillator
                signal, _ = gen_signal_python.sum_atomic_signals(mod_args)
                signal = signal.flatten()
                # compute rmse with new oscillator
                temp_sum = model.sum(axis=0) + signal
                temp_rmse = compute_rmse(temp_sum, self.target)

                # accept oscillators when they lower the RMSE
                if temp_rmse < rmse:
                    model[i,:] = signal
                    rmse = temp_rmse
                    i += 1
                j += 1
            
            # evaluate model against k models
            if rmse < best_rmse:
                best_model = model
                best_rmse = rmse
                best_model_j = j
        
        return best_model, best_rmse, j  
    
    def random_exploit(self, j_exploits: int, zero_model: bool = False) -> None:
        """generate an ensemble of n-oscillators
        reduce loss by iteratively replacing oscillators in the ensemble
        
        param:
            j_exploits: number of re-drawn oscillators per model
            zero_model: if True, start out with a model of zero-oscillators
        """
        # TODO: hacky solution to draw l-oscillators from sum_atomic signals
        mod_args = copy.deepcopy(self.rand_args)
        mod_args.n_osc = 1

        rng = np.random.default_rng(params.GLOBAL_SEED)

        best_model = None
        best_rmse = np.inf
        for _ in tqdm(range(self.k_samples)):
            # generate initial model
            # TODO: det_param_li is not updated
            model, det_param_li = gen_signal_python.sum_atomic_signals(self.rand_args)
            signal_sum = sum(model)
            if zero_model:
                model = np.zeros((self.rand_args.n_osc, self.rand_args.samples))
                signal_sum = sum(model)
            sample = Sample(signal_sum, model, det_param_li)
            # calculate initial rmse
            sample.rmse_sum = compute_rmse(sample.sum_y, self.target)
            for _ in tqdm(range(j_exploits)):
                # draw 1 oscillator
                signal, _ = gen_signal_python.sum_atomic_signals(mod_args)
                signal = signal.flatten()

                # replace random oscillator in the model
                i = rng.integers(0, self.rand_args.n_osc)
                temp_sum = sum(sample.matrix_y[0:i,:]) + signal + sum(sample.matrix_y[1+i:,:])
                temp_rmse = compute_rmse(temp_sum, self.target)

                # evaluate replacement
                if temp_rmse < sample.rmse_sum:
                    sample.matrix_y[i,:] = signal
                    sample.rmse_sum = temp_rmse
                    sample.sum_y = temp_sum

                self.samples.append(sample)
            
            if self.samples[-1].rmse_sum < best_rmse:
                best_rmse = self.samples[-1].rmse_sum
                best_model = self.samples[-1].matrix_y
        

        

    def gather_samples(self) -> tuple[Sample, list]:
        """find the sample with the lowest root mean square error
        and return a list of all rmse"""
        best_sample = self.samples[0]
        rmse_li, rmse_norm_li = list(), list()

        for s in self.samples:
            rmse_li.append(s.rmse_sum)
            rmse_norm_li.append(s.rmse_norm)
        
            if s.rmse_sum < best_sample.rmse_sum:
                best_sample = s
        
        return best_sample, rmse_li, rmse_norm_li


def main():
    # track elapsed time
    t0 = time.time()
    # loading and manipulating the target signal
    raw_sampling_rate, raw_target, raw_dtype = load_data(Path("resources/yes.wav"))
    scale_factor = 0.2
    target_full_len: Final = sample_down_int(raw_target, scale_factor)
    # shorten the target
    target: Final = target_full_len #take_middle_third(target_full_len)
    # normalize to range 0 1
    target_norm: Final = norm(target)
    # save to wav
    sampling_rate = int(scale_factor*raw_sampling_rate)
    save_signal_to_wav(target, sampling_rate, raw_dtype, Path("data/target_downsampled.wav"))

    # initialize and start search
    rand_args = params.py_rand_args_uniform
    rand_args.samples = len(target) # generated signals match length of target
                                    # NOTE: the sampling rate could also be set lower instead
    
    #k_dependency_one_shot([1, 10, 100, 500, 1000, 2000], 20, rand_args, target, visual=True)
    #n_dependency(rand_args, target, visual=True)
    #time_elapsed=time.time()-t0
    #print(f"time elapsed: {time_elapsed}")

    search = SearchModule(
                k_samples=1, # number of generated sum-signals
                rand_args=rand_args,
                target=target)
    search.random_exploit(20000, zero_model=True)
    #search.random_hybrid()
    
    # without random phase shifts between 0 and 2 pi
    # signals will overlap at the first sample
    if False:
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
    if False: # frequency-domain
        plot_fourier(target, title="target")
        plot_fourier(best_sample.sum_y, title="sum")
        plot_fourier(best_sample.fit_y, title="regression")

    print(f"best_sample.rmse_sum {best_sample.rmse_sum}")
    print(f"best_sample.rmse_sum-norm {norm_rmse}")
    print(f"best_sample.rmse_fit {best_sample.rmse_fit}")
    print(f"best_sample.rmse_fit-norm {norm_reg_rmse}")
    
    time_elapsed=time.time()-t0
    print(f"time elapsed: {time_elapsed:.2f} s")
    
    plt.show()
    

if __name__ == "__main__":
    main()

# IDEA: maybe it would be cool to write an algorithm for self-adjustment of the correct number of oscillators