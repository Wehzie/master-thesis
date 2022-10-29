import copy
from dataclasses import fields
from typing import Final, Iterable, List, Union
from multiprocessing import Pool, cpu_count
from functools import partial

import param_types as party
import sweep_types as sweety
import result_types as resty
from algo import SearchAlgo
import params
import meta_target

import numpy as np

class Experimenteur:
    def __init__(self, mp: bool) -> None:
        """
        args:
            mp: if true use multiple CPUs for processing
        """
        self.mp = mp
        self.cpu_count = cpu_count()

    @staticmethod
    def mean_std():
        return np.mean, np.std

    def invoke_search(self, search_alg: SearchAlgo, algo_sweep: sweety.AlgoSweep) -> Iterable:
        """call an algorithm's search function for a given number of times"""
        if self.mp:
            with Pool(self.cpu_count) as p:
                mapfunc = partial(search_alg.search, mp=self.mp) # pass mp in kwargs to search
                samples_z_ops = p.map(mapfunc, range(algo_sweep.m_averages))
        else:
            samples_z_ops = map(search_alg.search, range(algo_sweep.m_averages))
        return samples_z_ops

    def produce_result(self, samples_z_ops: Iterable, search_alg: SearchAlgo, algo_sweep: sweety.AlgoSweep) -> resty.ResultSweep:
        m_rmse_z_ops = [(s.rmse, z_ops) for s, z_ops in samples_z_ops] # List[Tuples[rmse, z_ops]]
        unzipped1 = zip(*m_rmse_z_ops) # unzip to List[rmse], List[z_ops]
        unzipped2 = copy.deepcopy(unzipped1)
        mean_rmse, mean_z_ops = map(np.mean, unzipped1) # map has effects and two functions per map are too ugly
        std_rmse, std_z_ops = map(np.std, unzipped2)
        return resty.ResultSweep(search_alg.__class__.__name__, search_alg.get_algo_args(), mean_rmse, std_rmse, mean_z_ops, std_z_ops, algo_sweep.m_averages)

    def run_algo_sweep(self, algo_sweep: sweety.AlgoSweep) -> List[resty.ResultSweep]:
        """run an experiment comparing multiple algorithms on their rmse and operations"""
        results = list()
        for Algo, algo_args in zip(algo_sweep.algo, algo_sweep.algo_args):
            search_alg = Algo(algo_args)
            samples_z_ops = self.invoke_search(search_alg, algo_sweep)
            result = self.produce_result(samples_z_ops, search_alg, algo_sweep)
            results.append(result)
        return results
    
    def run_rand_args_sweep(self,
    algo_sweep: sweety.AlgoSweep,
    sweep_args: Union[sweety.ConstTimeSweep, sweety.ExpoTimeSweep],
    base_args: party.PythonSignalRandArgs) -> resty.ResultSweep:
        """
        args:
            algo_sweep: a list of algorithms and algorithm arguments, the algorithm arguments will be modified
            sweep_args: a attributes within a rand_args type, for each attribute a list of values is tested"""
        results = []
        for val_schedule in fields(sweep_args): # for example frequency distribution
            for Algo, algo_args in zip(algo_sweep.algo, algo_sweep.algo_args): # for example monte carlo search
                for val in getattr(sweep_args, val_schedule.name):    # for example normal vs uniform frequency distribution
                    temp_args = copy.deepcopy(base_args)              # init/reset temporary rand_args
                    setattr(temp_args, val_schedule.name, val)        # for example, for field frequency in base_args set value 10 Hz
                    if val_schedule.name == "n_osc":                  # when n_osc changes
                        temp_args.weight_dist.n = val                 #    update n also in weight_dist
                    algo_args.rand_args = temp_args
                    f_algo_args: Final = copy.deepcopy(algo_args)
                    
                    search_alg: SearchAlgo = Algo(f_algo_args)
                    samples_z_ops = self.invoke_search(search_alg, algo_sweep)
                    result = self.produce_result(samples_z_ops, search_alg, algo_sweep)
                    results.append(result)
        # TODO: flush and pickle results
        return results
    
    def run_sampling_rate_sweep(self, sweep_args: sweety.NumSamplesSweep, base_args: party.PythonSignalRandArgs) -> resty.ResultSweep:
        """run all algorithms at different sampling rates of a target"""
        results = list()
        for s in sweep_args.samples:
            temp_args = copy.deepcopy(base_args)
            temp_args.samples = s # inject samples into rand_args
            m_target = meta_target.MetaTarget(temp_args)            
            algo_sweep = params.init_algo_sweep(m_target.signal, temp_args)
            results += self.run_algo_sweep(algo_sweep)
        return results

    def run_z_ops_sweep(self, algo_sweep: sweety.AlgoSweep, z_ops_sweep: sweety.ZOpsSweep) -> resty.ResultSweep:
        """run all algorithms with different numbers of z-operations, corresponding to more extensive search"""
        results = list()
        for z_ops in z_ops_sweep.max_z_ops:
            # inject max_z_ops into each algorithm's algo_args
            for algo_args in algo_sweep.algo_args:
                algo_args.max_z_ops = z_ops
            results += self.run_algo_sweep(algo_sweep)
        return results
