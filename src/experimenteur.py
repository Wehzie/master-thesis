
# this implementation is extremely wasteful!
# better: take k_params
# sum(k_params)
# set k = sum(k_params)
# the best value until k reflects the best RMSE for said k_param
# time-complexity is probably one order of of magnitude less
# there would be within-series dependency, but I don't think it matters
from dataclasses import fields
from pathlib import Path
import pickle
from typing import List, Union

import param_types as party
import sweep_types as sweety
import result_types as resty

import numpy as np
import matplotlib.pyplot as plt
from algo import SearchAlgo

"""
experiments:
        # z_ops measures z initialized, drawn, discarded - oscillators or weights
        # zero initialized oscillators or weights are not measured
        # maybe it would have been smarter to only optimize weights
        # not rmse comparisons, because one shot doesn't do that

    algorithms[
        random_one_shot,
        random_one_shot_weights(weight_init=zeros, ones, uniform, dist),
        las_vegas_aggregate,
        las_vegas_weight(weight_init=zeros, ones, uniform, dist),
        # version: pass over a single matrix/weights multiple times
        iterative,
        iterative_weight(weight_init=zeros, ones, uniform, dist),
        annealing (lower num-osc replaced),
        annealing weight(weight_init=zeros, ones, uniform, dist),
        las_vegas_purge:
            draw n oscillators
            loop over each oscillator and remove it if removing lowers the rmse
            remaining n-oscillators is non-deterministic
        genetic,
        genetic weight(weight_init=zeros, ones, uniform, dist)],


"""

def run_algo_sweep(algo_sweep: sweety.AlgoSweep) -> List[resty.ResultAlgoSweep]:
    results = list()
    for Algo, algo_args in zip(algo_sweep.algo, algo_sweep.algo_args):
        search_alg = Algo(algo_args)
        samples_z_ops = map(search_alg.search, range(algo_sweep.m_averages))        
        rmses_z_ops = [(s.rmse, z_ops) for s, z_ops in samples_z_ops]           # List[Tuples[rmse, z_ops]]
        unzipped = zip(*rmses_z_ops)                                            # unzip to List[rmse], List[z_ops]
        mean_rmse, mean_z_ops = map(np.mean, unzipped)
        std_rmse, std_z_ops = map(np.std, unzipped)
        result = resty.ResultAlgoSweep(search_alg.__class__.__name__, search_alg.algo_args, mean_rmse, std_rmse, mean_z_ops, std_z_ops)
        results.append(result)
    return results

def sweep_const_time_args(base_args: party.PythonSignalRandArgs,
arg_schedule: sweety.ConstTimeSweep,
algos: Union[SearchAlgo, List[SearchAlgo]]):

    for val_schedule in fields(arg_schedule):                       # for example frequency distribution
        for algo in algos:                                          # for example monte carlo search
            for val in getattr(arg_schedule, val_schedule.name):    # for example normal vs uniform frequency distribution
                print(val)
                setattr(base_args, val_schedule.name, val)
                algo.search(base_args)

def n_dependency(rand_args: party.PythonSignalRandArgs,
    target: np.ndarray,
    visual: bool = True) -> None:
    """given one algorithm visualize k-dependency for different values of n-oscillators"""
    k_params = [5, 20,]# 50, 100, 200, 300, 500, 1000, 2000]
    m_averages = 3
    n_osc = [25, 50,]# 100, 200, 500, 1000]
    fig = plt.figure()

    for n in n_osc:
        print(f"\nn={n}\n")
        rand_args.n_osc = n
        k_rmse, k_std = k_dependency_hybrid(k_params, m_averages, rand_args, target)
        print(k_rmse, k_std)
    
        plt.errorbar(k_params, k_rmse, k_std, elinewidth=1, capsize=5, capthick=1, markeredgewidth=1, label=n)
        # pickle intermediate steps
        with open(Path("data/n_dependency_plot.pickle"), "wb") as file:
            pickle.dump(fig, file)

    if visual:
        plt.title(f"mean over {m_averages}")
        plt.gca().set_xlabel("k")
        plt.gca().set_ylabel("RMSE(generated, target)")
        plt.legend(title="n oscillators")
        plt.savefig(Path("data/n_dependency"), dpi=300)
        plt.show()

def k_dependency_one_shot(
    k_params: list[int],
    m_averages: int,
    rand_args: party.PythonSignalRandArgs,
    target: np.ndarray, visual: bool = False) -> tuple:
    """visualize the k-dependency of random one-shot search
    
    args:
        k_params: list of ks (number of generated signals) to attempt
        m_averages: amount of times to repeat each k in k_params
        rand_args: arguments to generate an individual signal
        target: the target audio signal to approximate
    """
    k_rmse = np.empty(len(k_params))
    k_std = np.empty(len(k_params))
    for ki, k in enumerate(k_params):
        print(f"\nk={k}\n")
        m_rmse = np.empty(m_averages)
        for m in range(m_averages): # averager
            search = search_module.SearchModule(
                k_samples=k, # number of generated sum-signals
                rand_args=rand_args,
                target=target)
            best_model, best_rmse = search.random_stateless_one_shot()
            m_rmse[m] = best_rmse
        k_rmse[ki] = np.mean(m_rmse)
        k_std[ki] = np.std(m_rmse) # standard deviation over the m repetitions


        import result_types as resty
        res_li = list()
        for alg in algos:
            rmse_arr = np.array(map(algo.search(), range(m_averages)))
            mean_rmse = np.mean(rmse_arr)
            std_rmse = np.std(rmse_arr)
            res = resty.ResultAlgoSweep(algo.__name__, algo.algo_args, mean_rmse, std_rmse)
            res_li.append(res)


    if visual:
        plt.errorbar(k_params, k_rmse, k_std, elinewidth=1, capsize=5, capthick=1, markeredgewidth=1)
        plt.title(f"mean over {m_averages}")
        plt.gca().set_xlabel("k")
        plt.gca().set_ylabel("RMSE(generated, target)")
        plt.savefig(Path("data/random_one_shot-k_dependency"), dpi=300)
        plt.show()

    return k_rmse, k_std

def k_dependency_hybrid(
    k_params: list[int],
    m_averages: int,
    rand_args: party.PythonSignalRandArgs,
    target: np.ndarray,
    visual: bool = False) -> tuple:
    """visualize the k-dependency of random hybrid search"""
    k_rmse = np.empty(len(k_params))
    k_std = np.empty(len(k_params))
    for ki, k in enumerate(k_params):
        m_rmse = np.empty(m_averages)
        for m in range(m_averages): # averager
            search = search_module.SearchModule(
                k_samples=k, # number of generated sum-signals
                rand_args=rand_args,
                target=target)
            best_model, best_rmse, best_model_j = search.random_stateless_hybrid()
            #print(f"best_model_j={best_model_j}")
        k_rmse[ki] = np.mean(m_rmse)
        k_std[ki] = np.std(m_rmse)

    if visual:
        plt.errorbar(k_params, k_rmse, k_std, elinewidth=1, capsize=5, capthick=1, markeredgewidth=1)
        plt.title(f"mean over {m_averages}")
        plt.gca().set_xlabel("k")
        plt.gca().set_ylabel("RMSE(generated, target)")
        plt.savefig(Path("data/random_hybrid-k_dependency"), dpi=300)
        plt.show()
    
    return k_rmse, k_std

