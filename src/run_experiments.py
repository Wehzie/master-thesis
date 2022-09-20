
# this implementation is extremely wasteful!
# better: take k_params
# sum(k_params)
# set k = sum(k_params)
# the best value until k reflects the best RMSE for said k_param
# time-complexity is probably one order of of magnitude less
# there would be within-series dependency, but I don't think it matters
from cProfile import label
from pathlib import Path

from param_types import PythonSignalRandArgs
import search_module

import numpy as np
import matplotlib.pyplot as plt

def n_dependency(rand_args: PythonSignalRandArgs,
    target: np.ndarray,
    visual: bool = True) -> None:
    """visualize k-dependency for different values of n-oscillators"""
    k_params = [1, 10, ]# 50, 100, 200, 300, 500, 1000, 2000]
    m_averages = 3
    n_osc = [25, 50,]#100, 200, 500, 1000]
    for n in n_osc:
        rand_args.n_osc = n
        k_rmse, k_std = k_dependency_one_shot(k_params, m_averages, rand_args, target)
    
        plt.errorbar(k_params, k_rmse, k_std, elinewidth=1, capsize=5, capthick=1, markeredgewidth=1, label=n)
    
    if visual:
        plt.title(f"mean over {m_averages}")
        plt.gca().set_xlabel("k")
        plt.gca().set_ylabel("RMSE(generated, target)")
        plt.savefig(Path("data/n_dependency"), dpi=300)
        plt.show()
        

def k_dependency_one_shot(
    k_params: list[int],
    m_averages: int,
    rand_args: PythonSignalRandArgs,
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
        m_rmse = np.empty(m_averages)
        for m in range(m_averages): # averager
            search = search_module.SearchModule(
                k_samples=k, # number of generated sum-signals
                rand_args=rand_args,
                target=target)
            best_model, best_rmse = search.random_stateless_one_shot()
            m_rmse[m] = best_rmse
        k_rmse[ki] = np.mean(m_rmse)
        k_std[ki] = np.std(m_rmse)

    if visual:
        plt.errorbar(k_params, k_rmse, k_std, elinewidth=1, capsize=5, capthick=1, markeredgewidth=1)
        plt.title(f"mean over {m_averages}")
        plt.gca().set_xlabel("k")
        plt.gca().set_ylabel("RMSE(generated, target)")
        plt.savefig(Path("data/random_one_shot-k_dependency"), dpi=300)
        plt.show()

    return k_rmse, k_std

def k_dependency_hybrid(rand_args: PythonSignalRandArgs,
    target: np.ndarray):
    """visualize the k-dependency of random hybrid search"""
    k_params = [1, 10]
    AVERAGES = 3
    k_rmse = np.empty(len(k_params))
    k_std = np.empty(len(k_params))
    for ki, k in enumerate(k_params):
        m_rmse = np.empty(AVERAGES)
        for m in range(AVERAGES): # averager
            search = search_module.SearchModule(
                k_samples=k, # number of generated sum-signals
                rand_args=rand_args,
                target=target)
            best_model, best_rmse = search.random_stateless_hybrid()
            m_rmse[m] = best_rmse
        k_rmse[ki] = np.mean(search.samples.rmse_sum)
        k_std[ki] = np.std(m_rmse)

    plt.errorbar(k_params, k_rmse, k_std, elinewidth=1, capsize=5, capthick=1, markeredgewidth=1)
    plt.title(f"mean over {AVERAGES}")
    plt.gca().set_xlabel("k")
    plt.gca().set_ylabel("RMSE(generated, target)")
    plt.savefig(Path("data/random_hybrid-k_dependency"), dpi=300)
    plt.show()