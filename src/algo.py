from abc import ABC, abstractmethod
import copy
import pickle
from typing import List
from pathlib import Path

import sample
import data_analysis
import const
import param_types as party
import gen_signal_python

import numpy as np

class SearchAlgo(ABC):

    def __init__(self, algo_args: party.AlgoArgs):
        self.rand_args = algo_args.rand_args                # signal generation parameters
        self.target = algo_args.target                       # target function to approximate
        self.weight_init = algo_args.weight_init
        self.max_z_ops = algo_args.max_z_ops
        self.k_samples = algo_args.k_samples
        self.j_exploits = algo_args.j_exploits

        self.store_det_args = algo_args.store_det_args
        self.history = algo_args.history
        self.args_path = algo_args.args_path

        # state
        self.samples: List[sample.Sample] = list() # list of samples and results
        self.z_ops: int = 0                        # number of operations
    
    def __str__(self) -> str:
        rand_args = f"rand_args: {self.rand_args}\n"
        sig_gen_func = f"sig_gen_func: {self.sig_gen_func.__name__}\n"
        samples = ""
        for s in self.samples:
            samples += str(s) + "\n"
        return rand_args + sig_gen_func + samples

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

    def manage_state(self, base_sample: sample.Sample, k: int) -> None:
        """save samples to RAM or file if desired"""
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

    def eval_z_ops(self, verbose: bool = True) -> bool:
        """return true when an algorithm exceeds the maximum number of allowed operations"""
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

    @abstractmethod
    def search(self):
        NotImplementedError
