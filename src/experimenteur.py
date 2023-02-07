import copy
from dataclasses import fields
from typing import Final, Iterable, List, Union
from multiprocessing import Pool, cpu_count
from functools import partial

import param_types as party
import sweep_types as sweety
import result_types as resty
import algo
import meta_target
import const
import param_util
import data_io
import const
import params_target

import numpy as np

class Experimenteur:
    def __init__(self, mp: bool = const.MULTIPROCESSING, clean_work_dir: bool = True, show_plots: bool = False) -> None:
        """
        args:
            mp: if true use multiple CPUs for processing
            clean_work_dir: if true delete all files in the work directory
        """
        self.mp = mp
        self.cpu_count = cpu_count()
        if clean_work_dir:
            data_io.clean_dir(const.WRITE_DIR)
        self.work_dir = data_io.find_dir_name(const.WRITE_DIR, "quantitative_experiment") # directory in which to write all results
        self.sweep_dir = None # directory in which to write results of a sweep for an independent variable
        self.sweep_name = None # name of the sweep for an independent variable
        

    def set_sweep_name_and_dir(self, sweep_name: str) -> None:
        """set the name and directory of the sweep for the next experiment"""
        self.sweep_name = sweep_name
        self.sweep_dir = data_io.find_dir_name(self.work_dir, sweep_name)

    @staticmethod
    def mean_std():
        return np.mean, np.std

    def invoke_search(self, search_alg: algo.SearchAlgo, algo_sweep: sweety.AlgoSweep) -> Iterable:
        """call an algorithm's search function for a given number of times"""
        if self.mp:
            with Pool(self.cpu_count) as p:
                mapfunc = partial(search_alg.search, mp=self.mp) # pass mp in kwargs to search
                raw_result = p.map(mapfunc, range(algo_sweep.m_averages))
                best_samples, z_ops = zip(*raw_result)
        else:
            raw_result = map(search_alg.search, range(algo_sweep.m_averages))
            best_samples, z_ops = zip(*raw_result)
        return best_samples, z_ops

    def average_result(self, samples: Iterable, m_z_ops: Iterable, search_alg: algo.SearchAlgo, algo_sweep: sweety.AlgoSweep) -> resty.ResultSweep:
        """produce mean, stddev for rmse and z-ops over a list of samples (fit signal ensembles)"""
        m_rmse = [s.rmse for s in samples]
        m_rmse = np.array(m_rmse)
        m_z_ops = np.array(m_z_ops)
        mean_rmse, mean_z_ops = np.mean(m_rmse), np.mean(m_z_ops) # map has effects and two functions per map are too ugly
        std_rmse, std_z_ops = np.std(m_rmse), np.std(m_z_ops)
        return resty.ResultSweep(search_alg.__class__.__name__, search_alg.get_algo_args(), mean_rmse, std_rmse, mean_z_ops, std_z_ops, algo_sweep.m_averages)

    def run_algo_sweep(self, algo_sweep: sweety.AlgoSweep) -> List[resty.ResultSweep]:
        """run an experiment comparing multiple algorithms on their rmse and operations"""
        results = list()
        for awa in algo_sweep.algo_with_args:
            awa: sweety.AlgoWithArgs
            search_alg = awa.Algo(awa.algo_args)
            best_samples, z_ops = self.invoke_search(search_alg, algo_sweep)
            result = self.average_result(best_samples, z_ops, search_alg, algo_sweep)
            results.append(result)
        return results
    
    def run_rand_args_sweep(self,
    algo_sweep: sweety.AlgoSweep,
    sweep_args: Union[sweety.ConstTimeSweep, sweety.ExpoTimeSweep],
    base_args: party.PythonSignalRandArgs) -> resty.ResultSweep:
        """
        args:
            algo_sweep: a list of algorithms and algorithm arguments, the algorithm arguments will be modified
            sweep_args: an attribute within a rand_args type, for each attribute a list of values is tested"""
        print("sweeping with", sweep_args.__class__.__name__)
        results = []
        for val_schedule in fields(sweep_args): # for example frequency distribution
            for awa in algo_sweep.algo_with_args: # for example monte carlo search
                awa: sweety.AlgoWithArgs
                for val in getattr(sweep_args, val_schedule.name):    # for example normal vs uniform frequency distribution
                    temp_args = copy.deepcopy(base_args)              # init/reset temporary rand_args
                    setattr(temp_args, val_schedule.name, val)        # for example, for field frequency in base_args set value 10 Hz
                    if val_schedule.name == "n_osc":                  # when n_osc changes
                        temp_args.weight_dist.n = val                 #    update n also in weight_dist
                    awa.algo_args.rand_args = temp_args
                    f_algo_args: Final = copy.deepcopy(awa.algo_args)
                    
                    search_alg: algo.SearchAlgo = awa.Algo(f_algo_args)
                    samples, z_ops = self.invoke_search(search_alg, algo_sweep)
                    result = self.average_result(samples, z_ops, search_alg, algo_sweep)
                    results.append(result)
        # TODO: flush and pickle results
        return results
    
    def run_sampling_rate_sweep(self, sweep_args: sweety.NumSamplesSweep, base_args: party.PythonSignalRandArgs) -> resty.ResultSweep:
        """run all algorithms at different sampling rates of a target"""
        print("sweeping with", sweep_args.__class__.__name__)
        results = list()
        for s in sweep_args.samples:
            temp_args = copy.deepcopy(base_args)
            temp_args.samples = s # inject samples into rand_args
            m_target = meta_target.MetaTargetSample(temp_args, "magpie", params_target.DevSet.MAGPIE.value)
            algo_sweep = param_util.init_algo_sweep(m_target, temp_args)
            results += self.run_algo_sweep(algo_sweep)
        return results

    def run_z_ops_sweep(self, algo_sweep: sweety.AlgoSweep, z_ops_sweep: sweety.ZOpsSweep) -> resty.ResultSweep:
        """run all algorithms with different numbers of z-operations, corresponding to more extensive search"""
        print("sweeping with", z_ops_sweep.__class__.__name__)
        local_algo_sweep = copy.deepcopy(algo_sweep)
        results = list()
        for z_ops in z_ops_sweep.max_z_ops:
            # inject max_z_ops into each algorithm's algo_args
            for awa in local_algo_sweep.algo_with_args:
                awa: sweety.AlgoWithArgs
                awa.algo_args.max_z_ops = z_ops
            results += self.run_algo_sweep(algo_sweep)
        return results

    def run_target_sweep(self, target_sweep: sweety.TargetSweep,
    ) -> List[resty.ResultSweep]:
        """run an experiment comparing multiple targets on their rmse and operations"""
        print("sweeping with", target_sweep.__class__.__name__)
        results = list()
        for target in target_sweep.targets:
            algo_sweep = param_util.init_algo_sweep(target, target_sweep.rand_args, target_sweep.signal_generator, target_sweep.max_z_ops, target_sweep.m_averages)
            results += self.run_algo_sweep(algo_sweep)
        return results