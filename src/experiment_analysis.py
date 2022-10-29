import result_types as resty
from util import add_str2keys
import const
import param_types as party

from typing import List, Union
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def compute_dist_ranges(results: List[resty.ResultSweep]) -> List[resty.ResultSweep]:
    """compute the ranges for each distribution in a list of results"""
    for result in results:
        result.algo_args.rand_args.freq_dist.compute_range()
        result.algo_args.rand_args.weight_dist.compute_range()
        result.algo_args.rand_args.phase_dist.compute_range()
        result.algo_args.rand_args.offset_dist.compute_range()
    return results

def conv_results_to_pd(results: List[resty.ResultSweep]) -> pd.DataFrame:
    """convert ResultSweep to a pandas dataframe for further processing"""
    results = compute_dist_ranges(results)

    # initial definitions for convenience
    rand_args: party.PythonSignalRandArgs = [r.algo_args.rand_args for r in results]
    rand_args_dist_names = ["freq_dist", "weight_dist", "phase_dist", "offset_dist"]

    # unpack the nested args such that each entry has its own column
    res_df = pd.DataFrame(results).drop("algo_args", axis=1)
    algo_args_df = pd.DataFrame([r.algo_args for r in results]).drop(["rand_args", "target"], axis=1)
    rand_args_df = pd.DataFrame(rand_args).drop(rand_args_dist_names, axis=1)

    # dist ranges
    freq_dist_range = pd.DataFrame([ra.freq_dist.range for ra in rand_args], columns=["freq_range"])
    weight_dist_range = pd.DataFrame([ra.weight_dist.range for ra in rand_args], columns=["weight_range"])
    phase_dist_range = pd.DataFrame([ra.phase_dist.range for ra in rand_args], columns=["phase_range"])
    offset_dist_range = pd.DataFrame([ra.offset_dist.range for ra in rand_args], columns=["offset_range"])

    # dist bounds
    freq_dist_df = pd.DataFrame([add_str2keys("freq_dist", ra.freq_dist.kwargs) for ra in rand_args])
    weight_dist_df = pd.DataFrame([add_str2keys("weight_dist", ra.weight_dist.kwargs) for ra in rand_args])
    phase_dist_df = pd.DataFrame([add_str2keys("phase_dist", ra.phase_dist.kwargs) for ra in rand_args])
    offset_dist_df = pd.DataFrame([add_str2keys("offset_dist", ra.offset_dist.kwargs) for ra in rand_args])
    
    return pd.concat([res_df, algo_args_df, rand_args_df,
        freq_dist_range, weight_dist_range, phase_dist_range, offset_dist_range,
        freq_dist_df, weight_dist_df, phase_dist_df, offset_dist_df], axis=1)

def get_plot_title(df: pd.DataFrame, target_samples: int) -> str:
    """assemble plot title, execute before filtering out columns from dataframe"""
    m_averages = int(df["m_averages"].iloc[[0]])
    n_osc = df["n_osc"].values[0]
    max_z_ops = int(df["max_z_ops"].values[0])
    return f"m={m_averages}, n={n_osc}, samples={target_samples}, max z-ops={max_z_ops}"

def filter_df_by_dist_name(df: pd.DataFrame, attr_name: str, dist_name: str) -> pd.DataFrame:
    """
    filter a dataframe by a type of distribution name, e.g. "uniform" or "normal"

    args:
        attr_name: in freq, weight, phase, offset
        dist_name: for example uniform
    """
    if dist_name == "uniform":
        return df[df[f"{attr_name}_dist_low"].notna()] # for example freq_dist_low
    if dist_name == "normal":
        return df[df[f"{attr_name}_dist_loc"].notna()]

def plot_rmse_by_algo(df: pd.DataFrame, column_name: str) -> plt.Figure:
    """plot one line for each algorithm given an attribute column in a dataframe

    args:
        column_name: for example amplitude or freq_range
    """
    fig = plt.figure()
    algo_names = set(df["algo_name"])
    dfs_by_algo = [df.loc[df["algo_name"] == name] for name in algo_names]        
    for data, name in zip(dfs_by_algo, algo_names):
        plt.errorbar(data[column_name], data["mean_rmse"], data["std_rmse"], elinewidth=1, capsize=5, capthick=1, markeredgewidth=1, label=name)
    plt.legend(title="algorithm")
    plt.gca().set_ylabel("RMSE(generated, target)")
    return fig

def plot_n_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp1: plot number of oscillators, n, against rmse for multiple algorithms with z_ops and rand_args fixed"""
    title = get_plot_title(df, target_samples)
    df = df.filter(items=["algo_name", "n_osc", "mean_rmse", "std_rmse"])
    _ = plot_rmse_by_algo(df, "n_osc")
    plt.title(title)
    plt.gca().set_xlabel("number of oscillators")
    plt.savefig(Path("data/n_on_rmse.png"), dpi=300)
    if show: plt.show()

def plot_z_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp2: plot number of operations, z_ops, against rmse for multiple algorithms with rand_args fixed"""
    title = get_plot_title(df, target_samples)
    df = df.filter(items=["algo_name", "max_z_ops", "mean_rmse", "std_rmse"])
    _ = plot_rmse_by_algo(df, "max_z_ops")
    plt.title(title)
    plt.gca().set_xlabel(" z-operations")
    plt.savefig(Path("data/z_vs_rmse.png"), dpi=300)
    if show: plt.show()

def plot_samples_vs_rmse(df: pd.DataFrame, show: bool = False) -> None:
    """exp3: plot number of samples against rmse for multiple algorithms with rand_args and target fixed"""
    # before filtering df
    m_averages = int(df["m_averages"].iloc[[0]])
    n_osc = df["n_osc"].values[0]
    max_z_ops = int(df["max_z_ops"].values[0])
    title = f"m={m_averages}, n={n_osc}, max z-ops={max_z_ops}"

    df = df.filter(items=["algo_name", "samples", "mean_rmse", "std_rmse"])
    _ = plot_rmse_by_algo(df, "samples")
    plt.title(title)
    plt.gca().set_xlabel("number of samples")
    plt.savefig(Path(f"data/samples_vs_rmse.png"), dpi=300)
    if show: plt.show()

def plot_range_vs_rmse(df: pd.DataFrame, target_samples: int, attr_name: str, dist_name: str) -> List[plt.Figure]:
    """
    exp4-8: plot range of distribution against rmse for multiple algorithms with rand_args fixed
    
    args:
        attr_name: for example freq
        dist_name: uniform or normal

    returns:
        a list of figures, one for each distribution type
    """
    # assign strings for subsetting the dataframe and for injection to plot
    assert attr_name in ["freq", "weight", "phase", "offset"]
    range_name = attr_name + "_range" # for example freq_range
    title = get_plot_title(df, target_samples)

    # filter df
    df = filter_df_by_dist_name(df, attr_name, dist_name)
    df = df.filter(items=["algo_name", "mean_rmse", "std_rmse", "mean_z_ops", range_name])

    fig = plot_rmse_by_algo(df, range_name)
    plt.title(f"{title}, dist={dist_name}")

    return fig

def plot_freq_range_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp4+5: plot frequency range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in const.LEGAL_DISTS:
        fig = plot_range_vs_rmse(df, target_samples, "freq", dist_name)
        fig.gca().set_xlabel("width of frequency band")
        fig.gca().set_xscale("log")
        fig.savefig(Path(f"data/freq_range_{dist_name}_vs_rmse.png"), dpi=300)
    if show: plt.show()

def plot_weight_range_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp6: plot weight range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in const.LEGAL_DISTS:
        fig = plot_range_vs_rmse(df, target_samples, "weight", dist_name)
        fig.gca().set_xlabel("width of weight band")
        fig.gca().set_xscale("log")
        fig.savefig(Path(f"data/weight_range_{dist_name}_vs_rmse.png"), dpi=300)
    if show: plt.show()

def plot_phase_range_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp7: plot phase range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in const.LEGAL_DISTS:
        fig = plot_range_vs_rmse(df, target_samples, "phase", dist_name)
        fig.gca().set_xlabel("width of phase band")
        fig.savefig(Path(f"data/phase_range_{dist_name}_vs_rmse_.png"), dpi=300)
    if show: plt.show()

def plot_offset_range_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp8: plot offset range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in const.LEGAL_DISTS:
        fig = plot_range_vs_rmse(df, target_samples, "offset", dist_name)
        fig.gca().set_xlabel("width of offset distribution")
        fig.savefig(Path(f"data/offset_range_{dist_name}_vs_rmse.png"))
    if show: plt.show()

def plot_amplitude_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    "exp9: plot amplitude against rmse for multiple algorithms with rand_args and target fixed"
    title = get_plot_title(df, target_samples) # before filtering df
    df = df.filter(items=["algo_name", "mean_rmse", "std_rmse", "amplitude"])
    fig = plot_rmse_by_algo(df, "amplitude")
    plt.title(title)
    fig.gca().set_ylabel("RMSE(generated, target)")
    fig.gca().set_xlabel("amplitude [V]")
    plt.savefig(Path(f"data/amplitude_vs_rmse_.png"), dpi=300)
    if show: plt.show()