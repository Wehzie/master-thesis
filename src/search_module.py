import copy
from pathlib import Path
import pickle
import time
from typing import Final, List, Tuple, Union

import data_analysis
import data_io
import gen_signal_python
import params
from sample import Sample
from data_preprocessor import norm1d, sample_down, sample_down_int, take_middle_third

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from param_types import Dist, PythonSignalRandArgs
from run_experiments import k_dependency_hybrid, k_dependency_one_shot, n_dependency

class SearchModule():

    def __init__(self, rand_args: PythonSignalRandArgs, target: np.ndarray, start: np.ndarray = None):
        self.rand_args = rand_args # search parameters
        self.target = target # target function to approximate
        self.start = start # a matrix to start search
        
        self.samples: List[Sample] = list() # list of samples and results
    
    def __str__(self) -> str:
        rand_args = f"rand_args: {self.rand_args}\n"
        sig_gen_func = f"sig_gen_func: {self.sig_gen_func.__name__}\n"
        samples = ""
        for s in self.samples:
            samples += str(s) + "\n"
        return rand_args + sig_gen_func + samples

    def pickle_samples(self, args_path: Path, k_samples: int, ki: int):
        """pickle samples in RAM to disk in order to free RAM
        
        param:
            args_path:  where to store pickle
            k_samples:  total number of samples to draw
            ki:         current k index
        """
        if (ki == k_samples-1 or ki % params.SAMPLE_FLUSH_PERIOD == 0):
            with open(args_path, "ab") as f:
                pickle.dump(self.samples, f)
                self.samples = list() # clear list

    def manage_state(self, base_sample: Sample, k_samples: int, ki: int, history: bool, args_path: Path) -> None:
        """save samples to RAM or file if desired"""
        if history: self.samples.append(base_sample)
        if history and args_path: self.pickle_samples(args_path, k_samples, ki)

    def set_first_point_to_zero(self) -> None:
        """set the first point in each sample to 0 and recompute rmse
        this may be desireable because signals will overlap at the first point
        this can also be prevented by using fully random phase-shifts"""
        for s in self.samples:
            s.signal_sum[0] = 0 # set first point to 0
            s.rmse = data_analysis.compute_rmse(s.signal_sum, self.target)

    def random_one_shot(self,
        k_samples: int,
        store_det_args: bool = False,
        history: bool = False,
        args_path: Path = None) -> Tuple[Sample, int]:
        """generate k-signals which are a sum of n-oscillators
        on each iteration draw a new full model (matrix of n-oscillators)
        
        params:
            k_samples: number of times to draw a matrix
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model
            history: whether to store all generated samples, may run out of RAM
            args_path: when not none, write history to disk instead of RAM at specified path
        """
        if history and args_path: args_path.unlink(missing_ok=True) # delete file if it exists

        best_sample = Sample(None, None, None, 0, np.inf, list())
        for ki in tqdm(range(k_samples)):
            # compose a matrix
            temp_signal_matrix, temp_signal_args = gen_signal_python.sum_atomic_signals(self.rand_args, store_det_args)
            temp_sample = Sample(temp_signal_matrix, None, None, 0, np.inf, temp_signal_args)
            temp_sample.update(self.target)
            if history: self.samples.append(temp_sample)
            if history and args_path: self.pickle_samples(args_path, k_samples, ki)

            # compare with best
            if temp_sample.rmse < best_sample.rmse:
                best_sample = temp_sample

        z_ops = k_samples*self.rand_args.n_osc
        self.samples.append(best_sample) # TODO: for backwards compatibility -> purify func 
        return best_sample, z_ops

    def init_las_vegas(self, weight_init: Union[None, str], store_det_args: bool) -> Sample:
        if weight_init is None: # in this mode, weights are not adapted
            signal_matrix = np.zeros((self.rand_args.n_osc, self.rand_args.samples))
            signal_args = list()
        else:
            signal_matrix, signal_args = gen_signal_python.sum_atomic_signals(self.rand_args, store_det_args)

        if weight_init == "ones":
            weights = np.ones(self.rand_args.n_osc)
        elif weight_init == "uniform":
            rng = np.random.default_rng(params.GLOBAL_SEED)
            weights = rng.uniform(-1, 1, self.rand_args.n_osc)
        elif weight_init == "zeros":
            weights = np.zeros(self.rand_args.n_osc)
        elif weight_init == "dist":
            dist = copy.deepcopy(self.rand_args.weight_dist)
            dist.kwargs["size"] = self.rand_args.n_osc
            weights = dist.draw()
        else:
            weights = None # in this mode, weights are not adapted

        return Sample(signal_matrix, weights, np.sum(signal_matrix, axis=0), 0, np.inf, signal_args)

    def draw_las_vegas_candidate(self, base_sample: Sample, i: int, mod_args: PythonSignalRandArgs, store_det_args: bool, history: bool):
        # TODO: this can be done more efficiently without deepcopy and recomputing all
        temp_sample = copy.deepcopy(base_sample)
        signal, signal_args = gen_signal_python.sum_atomic_signals(mod_args, store_det_args)
        signal = signal.flatten()
        temp_sample.signal_matrix[i,:] = signal
        if store_det_args or history:
            signal_args = signal_args[0]
            temp_sample.signal_args.append(signal_args)
        temp_sample.signal_sum = np.sum(temp_sample.signal_matrix, axis=0)
        temp_sample.rmse = data_analysis.compute_rmse(temp_sample.signal_sum, self.target)
        return temp_sample
    
    def eval_las_vegas(self, temp_sample: Sample, base_sample: Sample, i: int, j: int, best_sample_j: int) -> Tuple[Sample, int, int]:
        if temp_sample.rmse < base_sample.rmse:
            base_sample = temp_sample
            return temp_sample, i+1, j
        return base_sample, i, best_sample_j

    def las_vegas(self,
        k_samples: int,
        weight_init: Union[None, str],
        store_det_args: bool = False,
        history: bool = False,
        args_path: Path = None) -> Tuple[Sample, int]:
        """generate k-signals which are a sum of n-oscillators
        model is constructed by aggregation
        oscillator candidates are accepted into the model when they lower RMSE

        param:
            k_samples: number of times to draw a matrix
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model
            history: whether to store all generated samples, may run out of RAM
            args_path: when not none, write history to disk instead of RAM at specified path

        return:
            best_sample, best_model_j
        """
        if history and args_path: args_path.unlink(missing_ok=True) # delete file if it exists

        # TODO: hacky solution to draw 1 oscillator from sum_atomic signals
        mod_args = copy.deepcopy(self.rand_args)
        mod_args.n_osc = 1

        best_sample = self.init_las_vegas(weight_init, store_det_args)
        best_sample_j = None
        z_ops = 0
        for ki in tqdm(range(k_samples)):
            base_sample = self.init_las_vegas(weight_init, store_det_args) # build up a signal_matrix in here
            i, j = 0, 0 # number of accepted and drawn weights respectively
            while i < self.rand_args.n_osc: # construct a model
                temp_sample = self.draw_las_vegas_candidate(base_sample, i, mod_args, store_det_args, history)
                base_sample, i, _ = self.eval_las_vegas(temp_sample, base_sample, i, None, None)
                j += 1
            
            z_ops += j

            self.manage_state(base_sample, k_samples, ki, history, args_path)
            best_sample, _, best_sample_j = self.eval_las_vegas(base_sample, best_sample, 0, j, best_sample_j)
            
        
        self.samples.append(best_sample)

        return best_sample, z_ops

    def draw_weight(self, weights: np.ndarray, i: int):
        w = self.rand_args.weight_dist.draw()
        weights[i] = w
        return weights

    def draw_las_vegas_weight_candidate(self, base_sample: Sample, i):
        temp_sample = copy.deepcopy(base_sample)
        temp_sample.weights = self.draw_weight(temp_sample.weights, i)
        temp_sample.signal_sum = Sample.predict(temp_sample.signal_matrix, temp_sample.weights, 0)
        temp_sample.rmse = data_analysis.compute_rmse(temp_sample.signal_sum, self.target)
        return temp_sample

    def las_vegas_weight(self,
        k_samples: int,
        weight_init: Union[None, str],
        store_det_args: bool = False,
        history: bool = False,
        args_path: Path = None) -> Tuple[Sample, int]:

        best_sample = self.init_las_vegas(weight_init, store_det_args)
        best_sample_j = None
        z_ops = 0
        for _ in tqdm(range(k_samples)):
            base_sample = self.init_las_vegas(weight_init, store_det_args)
            i, j = 0, 0 # number of accepted and drawn weights respectively
            while i < self.rand_args.n_osc:
                temp_sample = self.draw_las_vegas_weight_candidate(base_sample, i)
                base_sample, i, _ = self.eval_las_vegas(base_sample, temp_sample, i, None, None)
                j += 1

            z_ops += j
            best_sample, _, best_sample_j = self.eval_las_vegas(base_sample, best_sample, 0, j, best_sample_j)
            
        best_sample.signal_matrix = (best_sample.signal_matrix.T * best_sample.weights).T # TODO: change sample type
        self.samples.append(best_sample)
        return best_sample, z_ops
    
    def init_random_exploit(self, zero_model: bool) -> Sample:
        if zero_model:
            signal_matrix, signal_args = np.zeros((self.rand_args.n_osc, self.rand_args.samples)), list()
        else:
            signal_matrix, signal_args = gen_signal_python.sum_atomic_signals(self.rand_args)
        base_sample = Sample(signal_matrix, None, None, 0, None, signal_args)
        base_sample.update(self.target)
        return base_sample

    def draw_exploit_candidate(self, base_sample: Sample, mod_args: PythonSignalRandArgs, store_det_args: bool, history: bool):
        temp_sample = copy.deepcopy(base_sample)
        signal, signal_args = gen_signal_python.sum_atomic_signals(mod_args, store_det_args)
        signal = signal.flatten()
        temp_sample.signal_matrix[i,:] = signal
        if store_det_args or history:
            signal_args = signal_args[0]
            temp_sample.signal_args.append(signal_args)
        temp_sample.signal_sum = np.sum(temp_sample.signal_matrix, axis=0)
        temp_sample.rmse = data_analysis.compute_rmse(temp_sample.signal_sum, self.target)
        return temp_sample

    def compute_z_ops(self, k_samples: int, zero_init: bool, j_exploits: int) -> int:
        if zero_init:
            z_ops = k_samples * j_exploits
        else:
            z_ops = k_samples * (self.rand_args.n_osc + j_exploits)
        return z_ops

    def random_exploit(self, k_samples: int, j_exploits: int, zero_model: bool = False) -> None:
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

        best_sample = Sample(None, None, None, 0, np.inf, list())
        for _ in tqdm(range(k_samples)):
            # generate initial model
            
            base_sample = self.init_random_exploit(zero_model)
            for _ in range(j_exploits):
                # draw 1 oscillator
                signal, _ = gen_signal_python.sum_atomic_signals(mod_args)
                signal = signal.flatten()

                # replace random oscillator in the model
                i = rng.integers(0, self.rand_args.n_osc)
                # TODO: debug whether this works as expected or whether deepcopy is needed
                temp_sum = np.sum(base_sample.signal_matrix[0:i,:], axis=0) + signal + np.sum(base_sample.signal_matrix[1+i:,:], axis=0)
                temp_rmse = data_analysis.compute_rmse(temp_sum, self.target)

                # evaluate replacement
                if temp_rmse < base_sample.rmse:
                    base_sample.signal_matrix[i,:] = signal
                    base_sample.rmse = temp_rmse
                    base_sample.signal_sum = temp_sum

            if base_sample.rmse < best_sample.rmse:
                best_sample = base_sample

        self.samples.append(best_sample)
        
        return best_sample, self.compute_z_ops(k_samples, zero_model, j_exploits)
        
    def random_weight_exploit(self,
        k_samples: int,
        j_exploits: int,
        weight_init: str,
        store_det_args: bool = False,
        history: bool = False,
        args_path: Path = None) -> Tuple[Sample, int]:
        """generate an ensemble of n-oscillators and a vector of weights
        reduce loss by iteratively replacing weights
        
        param:
            j_exploits: number of re-drawn oscillators per model
        """
        # TODO: hacky solution to draw l-oscillators from sum_atomic signals
        rng = np.random.default_rng(params.GLOBAL_SEED)

        best_sample = Sample(None, None, None, 0, np.inf, list())
        for _ in range(k_samples):

            base_sample = self.init_las_vegas(weight_init, store_det_args)
            for _ in range(j_exploits):
                i = rng.integers(0, self.rand_args.n_osc)
                temp_sample = self.draw_las_vegas_weight_candidate(base_sample, i)
                base_sample, _, _ = self.eval_las_vegas(temp_sample, base_sample, 0, 0, 0)

            if base_sample.rmse < best_sample.rmse:
                best_sample = base_sample

        best_sample.signal_matrix = (best_sample.signal_matrix.T * best_sample.weights).T # TODO: change sample type
        self.samples.append(best_sample)
        zero_init = True if weight_init == "zeros" else False
        return best_sample, self.compute_z_ops(k_samples, zero_init, j_exploits)
        
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
    target_norm: Final = norm1d(target)
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
                rand_args=rand_args,
                target=target)
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
    best_sample: Sample = search.random_weight_exploit(k_samples = 3, j_exploits = 100, weight_init="dist")
    


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
    if False: # frequency-domain
        data_analysis.plot_fourier(target, title="target")
        data_analysis.plot_fourier(best_sample.signal_sum, title="sum")
        data_analysis.plot_fourier(reg_sample.signal_sum, title="regression")

    print(f"best_sample.rmse_sum {best_sample.rmse}")
    print(f"best_sample.rmse_sum-norm {norm_sample.rmse}")
    print(f"best_sample.rmse_fit {reg_sample.rmse}")
    print(f"best_sample.rmse_fit-norm {norm_reg_sample.rmse}")
    
    time_elapsed=time.time()-t0
    print(f"time elapsed: {time_elapsed:.2f} s")
    
    plt.show()
    

if __name__ == "__main__":
    main()

# IDEA: maybe it would be cool to write an algorithm for self-adjustment of the correct number of oscillators