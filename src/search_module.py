import csv
from pathlib import Path
from random import sample
from typing import Callable, Final, List

from data_analysis import compute_rmse, hist_rmse, plot_n, plot_pred_target, plot_signal, plot_fourier
from data_preprocessor import scale_down, scale_down_int
from data_io import DATA_PATH, load_data, load_data_numpy, save_signal_to_wav
import gen_signal_python
import params

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

from param_types import PythonSignalDetArgs, PythonSignalRandArgs

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

    def __init__(self, rand_args: PythonSignalRandArgs, n_samples: int, target: np.ndarray):
        self.rand_args = rand_args # search parameters
        self.n_samples = n_samples # number of samples
        self.target = target # target function to approximate
        
        self.samples: List[Sample] = list() # list of samples and results
    
    def __str__(self) -> str:
        rand_args = f"rand_args: {self.rand_args}\n"
        n_samples = f"n_samples: {self.n_samples}\n"
        sig_gen_func = f"sig_gen_func: {self.sig_gen_func.__name__}\n"
        samples = ""
        for s in self.samples:
            samples += str(s) + "\n"
        return rand_args + n_samples + sig_gen_func + samples

    def random_search(self) -> None:
        """generate n-signals which are a sum of k-oscillators"""
        for _ in tqdm(range(self.n_samples)):
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

    def gather_samples(self) -> tuple[Sample, list]:
        """find the sample with the lowest root mean square error
        and return a list of all rmse"""
        best_sample = self.samples[0]
        rmse_li = list()

        for s in self.samples:
            rmse_li.append(s.rmse_sum)
        
            if s.rmse_sum < best_sample.rmse_sum:
                best_sample = s
        
        return best_sample, rmse_li
        

def main():
    raw_sampling_rate, raw_target, raw_dtype = load_data()
    save_signal_to_wav(raw_target, raw_sampling_rate, raw_dtype, Path("data/target_control.wav"))
    print(raw_target)

    scale_factor = 0.5
    target: Final = scale_down_int(raw_target, scale_factor)
    sampling_rate = int(scale_factor*raw_sampling_rate)
    print(target)
    save_signal_to_wav(target, sampling_rate, target.dtype, Path("data/target_scipy.wav"))
    
    # this works
    manual_down_sample = raw_target[0:-1:2]
    print(manual_down_sample)
    save_signal_to_wav(manual_down_sample, raw_sampling_rate//2, raw_dtype, Path("data/target_manual.wav"))
    exit()

    rand_args = params.py_rand_args_uniform
    rand_args.n_osc = 4000
    rand_args.sampling_rate = sampling_rate
    rand_args.samples = len(target) # generated signals match length of target

    search = SearchModule(
        n_samples=10, # number of generated sum-signals
        rand_args=rand_args,
        target=target)
    search.random_search()
    
    # without random phase shifts between 0 and 2 pi
    # signals will overlap at the first sample
    for s in search.samples:
        s.sum_y[0] = 0 # set first point to 0
        s.rmse_sum = compute_rmse(s.sum_y, target)

    # find best sample
    best_sample, rmse_list = search.gather_samples()
    best_sample.save()
    save_signal_to_wav(best_sample.sum_y, sampling_rate, Path("data/best_sample.wav"))

    hist_rmse(rmse_list, show=False)

    # apply regression
    reg = Sample.regress_linear(best_sample.matrix_y, target, verbose=False)
    pred = Sample.predict(best_sample.matrix_y, reg.coef_, reg.intercept_)
    best_sample.fit_y = pred
    save_signal_to_wav(best_sample.fit_y, sampling_rate, Path("data/fit.wav"))
    best_sample.rmse_fit = compute_rmse(pred, target)

    plot_pred_target(best_sample.sum_y, target, show=False, title="sum")
    plot_pred_target(best_sample.fit_y, target, show=False, title="regression")
    print(f"best_sample.rmse_sum {best_sample.rmse_sum}")
    print(f"best_sample.rmse_fit {best_sample.rmse_fit}")
    plt.show()
    

if __name__ == "__main__":
    main()
