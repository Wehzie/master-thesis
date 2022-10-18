from abc import ABC, abstractmethod
import pickle
from typing import Callable, List, Tuple

import sample
import data_analysis
import const
import param_types as party
import gen_signal_python

import numpy as np

class SearchAlgo(ABC):

    def __init__(self, algo_args: party.AlgoArgs):
        self.algo_args = algo_args                          # for storage with algorithm results
        self.rand_args = algo_args.rand_args                # signal generation parameters
        self.target = algo_args.target                       # target function to approximate
        self.weight_mode = algo_args.weight_mode
        self.max_z_ops = algo_args.max_z_ops
        self.k_samples = algo_args.k_samples
        self.j_exploits = algo_args.j_exploits

        self.z_ops_callbacks = algo_args.z_ops_callbacks
        self.store_det_args = algo_args.store_det_args
        self.history = algo_args.history
        self.args_path = algo_args.args_path

        # state
        self.all_samples: List[sample.Sample] = list()              # list of samples and results
        self.best_samples: List[Tuple[sample.Sample, int]] = list() # list of samples and z_ops for intermediate results
        self.z_ops: int = 0                                         # current number of operations
    
    def __str__(self) -> str:
        rand_args = f"rand_args: {self.rand_args}\n"
        sig_gen_func = f"sig_gen_func: {self.sig_gen_func.__name__}\n"
        all_samples = ""
        for s in self.all_samples:
            all_samples += str(s) + "\n"
        best_samples = ""
        for s, z in self.best_samples:
            best_samples += str(s) + f" z_ops: {z}" + "\n"
        z_ops = f"z_ops: {z_ops}"
        return rand_args + sig_gen_func + "all_samples: " + all_samples + "best_samples: " + best_samples + z_ops

    @staticmethod
    def gen_empty_sample() -> sample.Sample:
        return sample.Sample(None, None, None, 0, np.inf, list())

    def pickle_samples(self, k: int) -> None:
        """pickle samples in RAM to disk in order to free RAM
        
        param:
            args_path:  where to store pickle
            k_samples:  total number of samples to draw
            k:          current k out of k_samples
        """
        if (k == self.k_samples-1 or k % const.SAMPLE_FLUSH_PERIOD == 0):
            with open(self.args_path, "ab") as f:
                pickle.dump(self.samples, f)
                self.samples = list() # clear list

    def clear_state(self) -> None:
        """reset state
        1. delete file to which samples are written if it exists
        2. reset z_ops
        3. reset samples list"""
        self.z_ops = 0
        self.samples = list()
        if self.history and self.args_path:
            self.args_path.unlink(missing_ok=True) 

    def eval_z_ops_callback():
        """store a sample and current z_ops when the callback schedule says so"""
        NotImplemented
        # TODO: this is a bit difficult/ugly because we don't know which z_ops we will get exactly
        # therefore we must interpret the list of z_ops_callbacks as ranges
        # so we check whether the current z_ops is in a z_ops range
        # and then, if no sample has been added within this range, add the sample to the 

    def manage_state(self, base_sample: sample.Sample, k: int) -> None:
        """save samples to RAM or file if desired"""
        # if self.z_ops_callbacks: self.eval_z_ops_callback()
        if self.history: self.samples.append(base_sample)
        if self.history and self.args_path: self.pickle_samples(k)

    def set_first_point_to_zero(self) -> None:
        """set the first point in each sample to 0 and recompute rmse
        this may be desireable because signals will overlap at the first point
        this can also be prevented by using fully random phase-shifts"""
        for s in self.samples:
            s.signal_sum[0] = 0 # set first point to 0
            s.rmse = data_analysis.compute_rmse(s.signal_sum, self.target)
    
    def gather_samples(self) -> tuple[sample.Sample, list]:
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

    def eval_max_z_ops(self, verbose: bool = True) -> bool:
        """return true when an algorithm exceeds the maximum number of allowed operations"""
        if self.max_z_ops is None: return False
        if self.z_ops >= self.max_z_ops:
            if verbose:
                print("z_ops: {z_ops} > max_z_ops: {max_z_ops}")
            return True
        return False

    def comp_samples(self, base_sample: sample.Sample, temp_sample: sample.Sample) -> sample.Sample:
        """compare two samples and return the one with lower rmse"""
        if temp_sample.rmse < base_sample.rmse:
            return temp_sample
        return base_sample

    def draw_sample(self) -> sample.Sample:
        """draw a sample and update z_ops"""
        self.z_ops += self.rand_args.n_osc * 2 # weights and oscillators are counted separately
        return gen_signal_python.draw_sample(self.rand_args, self.target, self.store_det_args)

    def draw_sample_weights(self, base_sample: sample.Sample):
        """update z_ops, draw new weights for the sample and recompute metrics"""
        self.z_ops += self.rand_args.n_osc
        return gen_signal_python.draw_sample_weights(base_sample, self.rand_args, self.target)

    def init_best_sample(self) -> sample.Sample:
        """initialize best sample in matrix+weight or weight-only optimization mode"""
        if self.weight_mode: # apply algorithm over weights only
            return self.draw_sample()
        return self.gen_empty_sample()
    
    def draw_temp_sample(self, base_sample: sample.Sample) -> sample.Sample:
        """draw temp sample in matrix+weight or weight-only optimization mode"""
        if self.weight_mode: # apply algorithm over weights only
            return self.draw_sample_weights(base_sample)
        return self.draw_sample()

    def handle_mp(self, sup_func_kwargs: dict) -> None:
        """handle multi processing by modifying numpy the random number generator
        args:
            sup_func_kwargs: the kwargs of the calling function
        """
        # each process needs a unique seed
        if "mp" in sup_func_kwargs and sup_func_kwargs["mp"] == True:
            rng = np.random.default_rng(None)
            #self.rand_args.weight_dist = party.WeightDist(rng.uniform, low=0, high=10, n=100)
            dist = self.rand_args.weight_dist.dist
            # __name__ is used to identify the function
            # this is incredibly ugly
            if isinstance(dist, Callable):
                # if dist is a uniform function, initialise it anew
                if dist.__name__ == rng.uniform.__name__:
                    # can't do
                    # dist = rng.uniform
                    self.rand_args.weight_dist.dist = rng.uniform
                elif dist.__name__ == rng.normal.__name__:
                    self.rand_args.weight_dist.dist = rng.normal

    @abstractmethod
    def search(self, *args, **kwargs): # *args needed to use with map(), not sure why
        NotImplementedError
