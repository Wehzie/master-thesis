import copy
from pathlib import Path
import pickle
import time
from typing import Final, List, Tuple

import data_analysis
import data_io
import gen_signal_python
import params
from sample import Sample
from data_preprocessor import norm, sample_down, sample_down_int, take_middle_third

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from param_types import PythonSignalRandArgs
from run_experiments import k_dependency_hybrid, k_dependency_one_shot, n_dependency

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

    def random_one_shot(self,
        k_samples: int,
        store_det_args: bool = False,
        history: bool = False,
        args_path: Path = None) -> Sample:
        """generate k-signals which are a sum of n-oscillators
        on each iteration draw a new full model (matrix of n-oscillators)
        
        params:
            k_samples: number of times to draw a matrix
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model
            history: whether to store all generated samples, may run out of RAM
            args_path: when not none, write history to disk instead of RAM at specified path
        """
        if history and args_path: args_path.unlink(missing_ok=True) # delete file if it exists

        best_rmse = np.inf # the lowest rmse
        best_matrix = None # the oscillators generating the lowest rmse when summed
        best_matrix_args = None # list of parameters for each signal in matrix
        for ki in tqdm(range(k_samples)):
            # compose a matrix
            temp_matrix, det_param_li = gen_signal_python.sum_atomic_signals(self.rand_args, store_det_args)
            temp_sum = sum(temp_matrix)
            temp_rmse = data_analysis.compute_rmse(temp_sum, self.target)
            if history: self.samples.append(Sample(temp_sum, temp_matrix, det_param_li))
            if history and args_path and (ki == k_samples-1 or ki % params.SAMPLE_FLUSH_PERIOD == 0):
                with open(args_path, "ab") as f:
                    pickle.dump(self.samples, f)
                    self.samples = list() # clear list  

            # compare with best
            if temp_rmse < best_rmse:
                best_matrix = temp_matrix
                best_rmse = temp_rmse
                best_matrix_args = det_param_li 

        best_sample = Sample(sum(best_matrix), best_matrix, best_matrix_args)
        self.samples.append(best_sample) # TODO: for backwards compatibility -> purify func 
        return best_sample
    
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
                temp_rmse = data_analysis.compute_rmse(temp_sum, self.target)

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
    
    def random_weight_hybrid(self) -> None:
        for _ in tqdm(range(self.k_samples)):
            # init empty sample
            signal_matrix, det_param_li = gen_signal_python.sum_atomic_signals(self.rand_args)
            weights = np.ones(self.rand_args.n_osc)
            signal_sum = Sample.predict(signal_matrix, weights, 0) # weighted sum
            sample = Sample(signal_sum, signal_matrix, det_param_li)
            sample.rmse_sum = np.inf

            i = 0 # number of added oscillators
            j = 0 # number of loops
            while i < self.rand_args.n_osc:
                temp_weights = copy.deepcopy(weights)
                # draw an oscillator
                w = self.rand_args.weight_dist.draw()
                temp_weights[i] = w

                # compute rmse with new oscillator
                temp_sum = Sample.predict(signal_matrix, temp_weights, 0) # weighted sum
                temp_rmse = data_analysis.data_analysis.compute_rmse(temp_sum, self.target)
                # TODO: also compute the weighting of the model

                # accept oscillators when they lower the RMSE
                if temp_rmse < sample.rmse_sum:
                    print(temp_rmse)
                    sample.sum_y = temp_sum
                    sample.rmse_sum = temp_rmse
                    weights = temp_weights
                    i += 1
                j += 1
            
            sample.matrix_y = (signal_matrix.T * weights).T
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
                temp_rmse = data_analysis.compute_rmse(temp_sum, self.target)

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
            sample.rmse_sum = data_analysis.compute_rmse(sample.sum_y, self.target)
            for _ in tqdm(range(j_exploits)):
                # draw 1 oscillator
                signal, _ = gen_signal_python.sum_atomic_signals(mod_args)
                signal = signal.flatten()

                # replace random oscillator in the model
                i = rng.integers(0, self.rand_args.n_osc)
                # TODO: debug whether this works as expected or whether deepcopy is needed
                temp_sum = sum(sample.matrix_y[0:i,:]) + signal + sum(sample.matrix_y[1+i:,:])
                temp_rmse = data_analysis.compute_rmse(temp_sum, self.target)

                # evaluate replacement
                if temp_rmse < sample.rmse_sum:
                    sample.matrix_y[i,:] = signal
                    sample.rmse_sum = temp_rmse
                    sample.sum_y = temp_sum

            self.samples.append(sample)
            
            if self.samples[-1].rmse_sum < best_rmse:
                best_rmse = self.samples[-1].rmse_sum
                best_model = self.samples[-1].matrix_y
        
    def random_weight_exploit(self, j_exploits: int) -> None:
        """generate an ensemble of n-oscillators and a vector of weights
        reduce loss by iteratively replacing weights
        
        param:
            j_exploits: number of re-drawn oscillators per model
        """
        # TODO: hacky solution to draw l-oscillators from sum_atomic signals
        rng = np.random.default_rng(params.GLOBAL_SEED)

        best_model = None
        best_rmse = np.inf
        for _ in range(self.k_samples):
            # generate initial model
            # TODO: det_param_li is not updated
            model, det_param_li = gen_signal_python.sum_atomic_signals(self.rand_args)
            weights = np.ones(self.rand_args.n_osc)
            #weights = rng.uniform(0.1, 100, (self.rand_args.n_osc))
            w_sum = Sample.predict(model, weights, 0) # weighted sum

            sample = Sample(w_sum, model, det_param_li) # TODO: new obj fields
            # calculate initial rmse
            sample.rmse_sum = data_analysis.compute_rmse(sample.sum_y, self.target)
            
            for _ in tqdm(range(j_exploits)):
                # TODO: efficiency improvements
                temp_weights = copy.deepcopy(weights)
                # replace random oscillator in the model
                i = rng.integers(0, self.rand_args.n_osc)
                # draw 1 weight
                w = self.rand_args.weight_dist.draw()
                temp_weights[i] = w
                temp_sum = Sample.predict(model, temp_weights, 0)
                temp_rmse = data_analysis.compute_rmse(temp_sum, self.target)

                # evaluate replacement
                if temp_rmse < sample.rmse_sum:
                    sample.rmse_sum = temp_rmse
                    sample.sum_y = temp_sum
                    weights = temp_weights

            # for compatibility, use best weights
            sample.matrix_y = (model.T * weights).T
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
    raw_sampling_rate, raw_target, raw_dtype = data_io.load_data()
    scale_factor = 0.01
    target_full_len: Final = sample_down_int(raw_target, scale_factor)
    # shorten the target
    target: Final = take_middle_third(target_full_len)
    # normalize to range 0 1
    target_norm: Final = norm(target)
    # save to wav
    sampling_rate = int(scale_factor*raw_sampling_rate)
    data_io.save_signal_to_wav(target, sampling_rate, raw_dtype, Path("data/target_downsampled.wav"))

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
    #search.random_one_shot()
    #search.random_exploit(search.rand_args.n_osc*5, zero_model=True)
    #search.random_hybrid()
    #search.random_weight_exploit(search.rand_args.n_osc*30)
    #search.random_weight_hybrid()
    best_sample = search.random_stateless_one_shot(10, store_det_args=True,
        history=True, args_path=Path("data/test_args.pickle"))
    
    print(data_io.load_pickled_samples(Path("data/test_args.pickle")))

    exit()
    # without random phase shifts between 0 and 2 pi
    # signals will overlap at the first sample
    if False:
        for s in search.samples:
            s.sum_y[0] = 0 # set first point to 0
            s.rmse_sum = data_analysis.compute_rmse(s.sum_y, target)
            s.rmse_norm = data_analysis.compute_rmse(norm(s.sum_y), target_norm)

    # find best sample and save
    #best_sample, rmse_list, rmse_norm_list = search.gather_samples()
    print(f"mean: {np.mean(best_sample.sum_y)}")
    best_sample.save()
    data_io.save_signal_to_wav(best_sample.sum_y, sampling_rate, raw_dtype, Path("data/best_sample.wav"))

    # normalize best sample
    norm_sum = norm(best_sample.sum_y)
    norm_rmse = data_analysis.compute_rmse(norm_sum, target_norm)
    
    # compute regression against target
    reg = Sample.regress_linear(best_sample.matrix_y, target, verbose=False)
    pred = Sample.predict(best_sample.matrix_y, reg.coef_, reg.intercept_)
    best_sample.fit_y = pred
    data_io.save_signal_to_wav(best_sample.fit_y, sampling_rate, raw_dtype, Path("data/fit.wav"))
    best_sample.rmse_fit = data_analysis.compute_rmse(pred, target)
    
    # norm regression after fit (good enough)
    norm_reg = norm(best_sample.fit_y)
    norm_reg_rmse = data_analysis.compute_rmse(norm_reg, target_norm)
    
    # plots
    if True: # time-domain
        #hist_rmse(rmse_list, title="sum distribution")
        #hist_rmse(rmse_norm_list, title="norm-sum distribution")
        data_analysis.plot_pred_target(best_sample.sum_y, target, title="sum")
        data_analysis.plot_pred_target(best_sample.fit_y, target, title="regression")
        data_analysis.plot_pred_target(norm_sum, target_norm, title="norm-sum")
        data_analysis.plot_pred_target(norm_reg, target_norm, title="norm after fit")
    if False: # frequency-domain
        data_analysis.plot_fourier(target, title="target")
        data_analysis.plot_fourier(best_sample.sum_y, title="sum")
        data_analysis.plot_fourier(best_sample.fit_y, title="regression")

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