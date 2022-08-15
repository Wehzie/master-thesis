from xmlrpc.client import Boolean
from data_analysis import plot_signal

from typing import Callable, Final
import numpy as np
from data_io import load_data
from python_signal_generator import gen_inv_sawtooth_api

class Sample():
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x: Final = x # generated signal x coords
        self.y: Final = y # generated signal y coords
        self.det_param_li = list() # list of determined parameters
        self.rmse = None # root mean square error

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
    visual: Boolean = True) -> None:
        """
        Compute root mean square error (RMSE) between prediction and target signal.
        """

        diff_len = lambda p, t: len(p) != len(t)
        get_shortest = lambda p, t: np.argmin([len(p), len(t)])
        get_longest = lambda p, t: np.argmax([len(p), len(t)])

        # if prediction and target have different lengths
        #   pad the shorter shorter signal with zeros
        if diff_len(p, t):
            tup = (p, t)
            short_sig = tup[get_shortest(p, t)]
            long_sig = tup[get_longest(p, t)]
            short_sig = self.pad_zero(short_sig, len(long_sig))
            self.rmse = np.sqrt(((short_sig-long_sig)**2).mean())
            if visual: 
                plot_signal(short_sig)
                plot_signal(long_sig)
            return

        if visual:
            plot_signal(p)
            plot_signal(t)

        self.rmse = np.sqrt(((p-t)**2).mean())

class SearchModule():

    def __init__(self, params: dict, n_samples: int, sig_gen_func: Callable):
        self.params = params # search parameters
        self.n_samples = n_samples # number of samples
        self.samples = list() # list of samples and results
        self.sig_gen_func = sig_gen_func # function to generate a signal
    
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
        # TODO: delete
        #if f_lo < 1 and f_hi < 1:
        #    f = np.random.uniform(f_lo, f_hi)
        #if f_lo >= 1 and f_hi >= 1:
        #    r = np.random.randint(f_lo, f_hi)
        #else:
        #    raise ValueError("unsupported frequency range")
        return det_param

    def random_search(self):
        sampling_rate, target = load_data()
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
            
            # calculate rmse
            sample.compute_rmse(sample.y, target)
        
            # store sample
            self.samples.append(sample)
            

    def find_best_sample(self):
        """find the sample with the lowest root mean square error"""
        best = self.samples[0]
        for s in self.samples:
            if s.rmse < best.rmse:
                best = s
        return best


def main():
    params = {
        "n_osc": 100, # number of oscillators
        "f_lo": 1, # frequency bounds
        "f_hi": 1e4,
        "duration": 3 # signal duration in seconds
    }
    s = SearchModule(params, n_samples=10, sig_gen_func=gen_inv_sawtooth_api)
    s.random_search()
    print(s.find_best_sample())

if __name__ == "__main__":
    main()
