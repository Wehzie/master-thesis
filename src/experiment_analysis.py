import result_types as resty
from util import add_str2keys

from typing import List
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def conv_results_to_pd(results: List[resty.ResultSweep]) -> pd.DataFrame:
    """convert ResultSweep to a pandas dataframe for further processing"""
    # the chosen approach here is not great
    # there are more loops here over the data than should be necessary
    res_df = pd.DataFrame(results).drop("algo_args", axis=1)
    algo_args_df = pd.DataFrame([r.algo_args for r in results]).drop(["rand_args", "target"], axis=1)
    rand_args = [r.algo_args.rand_args for r in results]
    rand_args_df = pd.DataFrame(rand_args).drop(["f_dist", "weight_dist", "phase_dist", "offset_dist"], axis=1)
    # attributes
    f_dist_df = pd.DataFrame([add_str2keys("f_dist", ra.f_dist.kwargs) for ra in rand_args])
    weight_dist_df = pd.DataFrame([add_str2keys("weight_dist", ra.weight_dist.kwargs) for ra in rand_args])
    phase_dist_df = pd.DataFrame([add_str2keys("phase_dist", ra.phase_dist.kwargs) for ra in rand_args])
    offset_dist_df = pd.DataFrame([add_str2keys("offset_dist", ra.offset_dist.kwargs) for ra in rand_args])
    return pd.concat([res_df, algo_args_df, rand_args_df,
        f_dist_df, weight_dist_df, phase_dist_df, offset_dist_df], axis=1)

def plot_results_expo_time(df: pd.DataFrame) -> None:
    """exp1: plot number of oscillators against rmse for multiple algorithms with z_ops fixed"""
    m_averages = int(df["m_averages"].iloc[[0]])
    max_z_ops = df["max_z_ops"].values[0]
    df = df.filter(items=["algo_name", "n_osc", "mean_rmse", "std_rmse"])# "mean_z_ops"])

    algo_names = set(df["algo_name"])
    dfs_by_algo = [df.loc[df["algo_name"] == name] for name in algo_names]        
    plt.figure()
    for data, name in zip(dfs_by_algo, algo_names):
        plt.errorbar(data["n_osc"], data["mean_rmse"], data["std_rmse"], elinewidth=1, capsize=5, capthick=1, markeredgewidth=1, label=name)

    plt.title(f"mean over {m_averages}, max-ops={max_z_ops}")
    plt.gca().set_xlabel("number of oscillators")
    plt.gca().set_ylabel("RMSE(generated, target)")
    plt.legend(title="algorithm")
    plt.savefig(Path("data/n_on_rmse.png"), dpi=300)

    plt.show()

def plot_results_const_time(df: pd.DataFrame) -> None:
    NotImplemented
    df = df.filter(items=["algo_name",
        "f_dist_low", "f_dist_high", # TODO: only works with uniform distribution
        "weight_high", "weight_dist_high",
        "phase_high", "phase_dist_high",
        "offset_high", "offset_dist_high",
        "mean_rmse", "mean_z_ops"])
    print(df)
    df.plot()
    plt.show()