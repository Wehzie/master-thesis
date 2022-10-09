import pickle
import sample
import data_analysis
import const

from typing import List
from pathlib import Path

import numpy as np

class SearchAlgo():

    def __init__(self, rand_args, target: np.ndarray):
        self.rand_args = rand_args # search parameters
        self.target = target # target function to approximate
        self.samples: List[sample.Sample] = list() # list of samples and results
    
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
