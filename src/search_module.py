import csv
from pathlib import Path
from typing import Any, Callable, Final, List

from data_analysis import compute_rmse, hist_rmse, plot_n, plot_pred_target, plot_signal, plot_fourier
from data_preprocessor import align_signals, scale_down
from data_io import load_data
from gen_signal_python import gen_inv_sawtooth

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

from gen_signal_spice import gen_random_spice_signal

class Sample():
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.sum_x: Final = x # generated signal x coords
        self.pad_sum_x = None # x coords padded to target
        self.sum_y: Final = y # generated signal y coords
        self.fit_y = None # summed signal fit to target with regression
        
        # list of determined parameters
        # one set of parameters corresponds to a single oscillator
        self.det_param_li = list()
        
        self.rmse_sum = None # root mean square error of summed signal
        self.rmse_fit = None # root mean square error of regression fit signal

        # store single oscillator signals for linear regression
        self.y_li = list() # array of single-oscillator signals

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
            writer.writerow(self.det_param_li[0])
            # write data
            for osc in self.det_param_li:
                writer.writerow(osc.values())

    def regress_linear(self, t: np.ndarray, verbose: bool = False):
        """apply linear regression"""
        # y_li refers to the y-values of a generated signal
        # the individual signals are used as regressors 

        r = np.array(self.y_li).T

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
        X = np.array(X)
        X = X.T
        fit = np.sum(X * coef, axis=1) + intercept
        return fit

class SearchModule():

    def __init__(self, params: dict, n_samples: int, sig_gen_func: Callable, target: np.ndarray):
        self.params = params # search parameters
        self.n_samples = n_samples # number of samples
        self.samples: List[Sample] = list() # list of samples and results
        self.sig_gen_func = sig_gen_func # function to generate a signal
        self.target = target # target function to approximate
    
    def __str__(self) -> str:
        params = f"params: {self.params}\n"
        n_samples = f"n_samples: {self.n_samples}\n"
        sig_gen_func = f"sig_gen_func: {self.sig_gen_func.__name__}\n"
        samples = ""
        for s in self.samples:
            samples += str(s) + "\n"
        return params + n_samples + sig_gen_func + samples

    def draw_params_random(self) -> dict:
        """draw randomly from parameter pool"""
        # NOTE: arg names in det_param correspond to arg names of
        #    the args of sig_gen_func
        det_param = {}
        # frequency
        f_lo = self.params["f_lo"]
        f_hi = self.params["f_hi"]
        f = np.random.uniform(f_lo, f_hi)
        det_param["freq"] = f
        
        # duration
        # det_param["duration"] = self.params["duration"]
        
        # samples
        det_param["samples"] = self.params["samples"]

        # weight
        det_param["weight"] = np.random.uniform(0, 1)
        # random phase
        det_param["random_phase"] = self.params["random_phase"]
        return det_param

    def random_search(self) -> None:
        """generate n-signals which are a sum of k-oscillators"""
        for _ in tqdm(range(self.n_samples)):

            # draw parameters a first time to initialize x and y
            det_params = self.draw_params_random()
            # x doesn't change as oscillators are summed
            sum_x, _ = self.sig_gen_func(**det_params)
            pad_sum_x, _ = align_signals(sum_x, self.target)
            
            # y is initialized as array of zeros
            # y has target length from x
            sum_y = np.zeros(len(pad_sum_x))
            sample = Sample(pad_sum_x, sum_y)

            # compose a signal of single oscillators
            for _ in range(self.params["n_osc"]):
                # determine a set of parameters for a single oscillator
                det_params = self.draw_params_random()
                # store the parameter set
                sample.det_param_li.append(det_params)
                # add single oscillator to sum
                single_signal = self.sig_gen_func(**det_params)
                sample.sum_y = sample.sum_y + single_signal[1]
                # track individual signals for linear regression
                sample.y_li.append(single_signal[1])
            
            # calculate rmse
            sample.rmse_sum = compute_rmse(sample.sum_y, self.target)
        
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
    sampling_rate, raw_target = load_data()
    target: Final = scale_down(raw_target, 0.01)

    params = {
        "n_osc": 1000, # number of oscillators
        "f_lo": 1e5, # frequency bounds
        "f_hi": 1e6,
        #"duration": 0.3, # signal duration in seconds
        "samples": len(target), # number of samples in the signal
        "weight": "random",
        "random_phase": True
    }

    search = SearchModule(params,
        n_samples=100, # number of generated sum-signals
        sig_gen_func=gen_random_spice_signal,
        target=target)
    search.random_search()
    
    # without random phase shifts all signals will be the same at x=0
    if params["random_phase"] == False:
        for s in search.samples:
            s.sum_y[0] = 0
            s.rmse_sum = compute_rmse(s.sum_y, target)

    # find best sample
    best_sample, rmse_list = search.gather_samples()
    best_sample.save()

    hist_rmse(rmse_list, show=False)

    # apply regression
    reg = best_sample.regress_linear(target, verbose=False)
    pred = Sample.predict(best_sample.y_li, reg.coef_, reg.intercept_)
    best_sample.fit_y = pred
    best_sample.rmse_fit = compute_rmse(pred, target)

    plot_pred_target(best_sample.sum_y, target, show=False, title="sum")
    plot_pred_target(best_sample.fit_y, target, show=False, title="regression")
    print(f"best_sample.rmse_sum {best_sample.rmse_sum}")
    print(f"best_sample.rmse_fit {best_sample.rmse_fit}")
    plt.show()
    

if __name__ == "__main__":
    main()
