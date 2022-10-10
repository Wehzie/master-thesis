from abc import ABC
import copy
import pickle
from typing import List
from pathlib import Path

import sample
import data_analysis
import const
import param_types as party

class SearchAlgo(ABC):

    def __init__(self, algo_args):
        self.rand_args = algo_args.rand_args                # signal generation parameters
        self.target = algo_args.target                       # target function to approximate
        self.weight_init = algo_args.weight_init
        self.max_z_ops = algo_args.max_z_ops
        self.k_samples = algo_args.k_samples
        self.j_exploit = algo_args.j_exploit

        self.store_det_args = algo_args.store_det_args
        self.history = algo_args.history
        self.args_path = algo_args.args_path

        self.samples: List[sample.Sample] = list() # list of samples and results
  
        # TODO: hacky solution to draw 1 oscillator from sum_atomic signals
        single_osc_args = copy.deepcopy(self.rand_args)
        single_osc_args.n_osc = 1
    
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
        if (ki == k_samples-1 or ki % const.SAMPLE_FLUSH_PERIOD == 0):
            with open(args_path, "ab") as f:
                pickle.dump(self.samples, f)
                self.samples = list() # clear list

    def manage_state(self, base_sample: sample.Sample, k_samples: int, ki: int, history: bool, args_path: Path) -> None:
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

    def eval_z_ops(self, z_ops: int, max_z_ops: int, verbose: bool = True) -> bool:
        """return true when an algorithm exceeds the maximum number of allowed operations"""
        if z_ops >= max_z_ops:
            if verbose:
                print("z_ops: {z_ops} > max_z_ops: {max_z_ops}")
            return True
        return False
