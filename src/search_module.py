import csv
from pathlib import Path
from tabnanny import verbose
from typing import Any, Callable, Final

from data_analysis import plot_pred_target, plot_signal, plot_fourier
from data_preprocessor import scale_down
from data_io import load_data
from python_signal_generator import gen_inv_sawtooth_api

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Sample():
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x: Final = x # generated signal x coords
        self.y: Final = y # generated signal y coords
        self.padded_y = y # potentially padded y
        
        # list of determined parameters
        # one set of parameters corresponds to a single oscillator
        self.det_param_li = list()
        
        self.rmse = None # root mean square error

        # store single oscillator signals for linear regression
        self.y_li = list() # array of single-oscillator signals
        self.padded_y_li = list() # array of padded single-oscillator signals

    def __str__(self) -> str:
        x = f"x: {self.x}\n"
        y = f"y: {self.y}\n"
        rmse = f"rmse: {self.rmse}"
        return x + y + rmse
        
    def pad_zero(self, short: np.ndarray, len_long: int) -> np.ndarray:
        """evenly zero-pad a short signal up to the desired length"""
        # evenly pad with zeros
        to_pad = len_long - len(short)

        # deal with an odd number of padding so that dimensions are exact
        to_pad_odd = 0
        if to_pad % 2 == 1:
            to_pad_odd = 1

        padded = np.pad(short, (to_pad//2, to_pad//2 + to_pad_odd), mode="constant")
        return padded

    def compute_rmse(self,
    p: np.ndarray, t: np.ndarray,
    visual: bool = False) -> None:
        """
        Compute root mean square error (RMSE) between prediction and target signal.
        """
        get_shortest = lambda p, t: np.argmin([len(p), len(t)])
        get_longest = lambda p, t: np.argmax([len(p), len(t)])

        # if prediction and target have different lengths
        #   pad the shorter shorter signal with zeros
        tup = (p, t)
        short_sig = tup[get_shortest(p, t)]
        long_sig = tup[get_longest(p, t)]
        short_sig = self.pad_zero(short_sig, len(long_sig))
        
        self.rmse = np.sqrt(((short_sig-long_sig)**2).mean())
        
        # store padded signal
        if len(p) < len(t): self.padded_y = short_sig
        
        if visual:
            self.rmse = 99
            print(f"RMSE: {self.rmse}")
            plot_pred_target(p, t)      

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
        # pad individual oscillations
        self.padded_y_li = [self.pad_zero(y, len(t)) for y in self.y_li]
        X = np.array(self.padded_y_li)
        X = X.T

        # t refers to the target signal
        y = t

        if verbose:
            print("Computing regression")
            print(f"X {X}")
            print(f"X.shape {X.shape}")
            print(f"y {y}")
            print(f"y.shape {y.shape}")
        
        reg = LinearRegression().fit(X, y)

        if verbose:
            print(f"Coefficient: {reg.coef_}")
            print(f"Coefficient.shape: {reg.coef_.shape}")
            print(f"Intercept: {reg.intercept_}")

        return reg

    def predict(self, X: list, coef: np.ndarray) -> np.ndarray:
        """generate approximation of target, y"""
        X = np.array(X)
        X = X.T
        return np.sum(X * coef, axis=1)

class SearchModule():

    def __init__(self, params: dict, n_samples: int, sig_gen_func: Callable, target: np.ndarray):
        self.params = params # search parameters
        self.n_samples = n_samples # number of samples
        self.samples = list() # list of samples and results
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
        det_param = {}
        # frequency
        f_lo = self.params["f_lo"]
        f_hi = self.params["f_hi"]
        f = np.random.uniform(f_lo, f_hi)
        det_param["f"] = f
        # duration
        det_param["duration"] = self.params["duration"]
        # weight
        det_param["weight"] = np.random.uniform(0, 1)
        # random phase
        det_param["random_phase"] = self.params["random_phase"]
        return det_param

    def random_search(self) -> None:
        for _ in range(self.n_samples):
            # x doesn't change as oscillators are summed
            det_params = self.draw_params_random()
            x = self.sig_gen_func(det_params)[0]
            # y is initialized as array of zeros
            y = np.zeros(len(x))
            sample = Sample(x, y)

            # compose a signal of single oscillators
            for _ in range(self.params["n_osc"]):
                # determine a set of parameters for a single oscillator
                det_params = self.draw_params_random()
                # store the parameter set
                sample.det_param_li.append(det_params)
                # add single oscillator to sum
                single_signal = self.sig_gen_func(det_params)
                sample.y = sample.y + single_signal[1]
                # track individual signals for linear regression
                sample.y_li.append(single_signal[1])
            
            # calculate rmse
            sample.compute_rmse(sample.y, self.target)
        
            # store sample
            self.samples.append(sample)

    def find_best_sample(self) -> Sample:
        """find the sample with the lowest root mean square error"""
        best = self.samples[0]
        for s in self.samples:
            if s.rmse < best.rmse:
                best = s
        return best

def main():
    params = {
        "n_osc": 10000, # number of oscillators
        "f_lo": 1, # frequency bounds
        "f_hi": 1e2,
        "duration": 1, # signal duration in seconds
        "weight": "random",
        "random_phase": True
    }
    sampling_rate, target = load_data()
    target = scale_down(target, 0.4)
    
    search = SearchModule(params,
        n_samples=1,
        sig_gen_func=gen_inv_sawtooth_api,
        target=target)
    search.random_search()
    
    best_sample = search.find_best_sample()
    plot_pred_target(best_sample.padded_y, target)    
    best_sample.save()
    
    # apply regression
    reg = best_sample.regress_linear(target, verbose=True)
    pred = best_sample.predict(best_sample.padded_y_li, reg.coef_)
    best_sample.compute_rmse(pred, target, visual = True)#
    

if __name__ == "__main__":
    main()
