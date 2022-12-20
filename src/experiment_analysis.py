import result_types as resty
import const
import param_types as party

from typing import List, Union, Tuple
from pathlib import Path
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

# rainbow colors
cmap = mpl.colormaps['Spectral']
rb_colors = [cmap(i) for i in np.linspace(0, 0.9, 10)]
# tableau colors
colors = plt.cm.tab10.colors
# set up plotting
linestyles = ["solid", "dashed"]
color_list = colors * len(linestyles)
linestyle_list = [[style] * len(colors) for style in linestyles]
# flatten from [[style1, ...], [style2, ...]] to [style1, style1, ..., style2, ...]
linestyle_list = itertools.chain.from_iterable(linestyle_list)
default_cycler = (
    cycler(color=color_list) +
    cycler(linestyle=linestyle_list)
)
plt.rc('axes', prop_cycle=default_cycler)

# TODO: style multiple algo lines with a cycler to increase number of colors
# remove whitespace where possible form plots (right hand side)
# make legends background more transparent
# optionally save legend to the side or as a separate figure

def add_str2keys(s: str, d: dict) -> dict:
    """append the given string to each key of a dictionary"""
    return {f"{s}_{k}": v for k, v in d.items()}

def compute_dist_ranges(results: List[resty.ResultSweep]) -> List[resty.ResultSweep]:
    """compute the ranges for each distribution in a list of results"""
    for result in results:
        result.algo_args.rand_args.freq_dist.compute_range()
        result.algo_args.rand_args.weight_dist.compute_range()
        result.algo_args.rand_args.phase_dist.compute_range()
        result.algo_args.rand_args.offset_dist.compute_range()
    return results

def rename_algos_by_args(name_df: pd.DataFrame, algo_args_df: pd.DataFrame) -> pd.DataFrame:
    """rename identical algorithms in the dataframe by the distinguishing arguments they were called with.
    for example, MCExploitJ1, when j_replacements=1 and MCExploitJ5, when j_replacements=5"""

    # filter for j_replace, keep index intact to merge back into name_df
    temp_df = pd.concat([name_df["algo_name"], algo_args_df["j_replace"]], axis=1)
    temp_df.dropna(inplace=True)
    temp_df["j_replace"] = temp_df["j_replace"].astype(int) # ints are prettier than floats
    temp_df["algo_name"] = temp_df["algo_name"] + ", j=" + temp_df["j_replace"].astype(str)
    temp_df = name_df.merge(temp_df, left_index=True, right_index=True, how="outer")

    # rename the algorithms
    name_df["algo_name"] = temp_df["algo_name_y"].fillna(temp_df["algo_name_x"])
    return name_df

def conv_results_to_pd(results: List[resty.ResultSweep]) -> pd.DataFrame:
    """convert ResultSweep to a pandas dataframe for further processing"""
    results = compute_dist_ranges(results)

    # initial definitions for convenience
    rand_args: party.PythonSignalRandArgs = [r.algo_args.rand_args for r in results]
    rand_args_dist_names = ["freq_dist", "weight_dist", "phase_dist", "offset_dist"]

    # unpack the nested args such that each entry has its own column
    res_df = pd.DataFrame(results).drop("algo_args", axis=1)
    algo_args_df = pd.DataFrame([r.algo_args for r in results]).drop(["rand_args", "target"], axis=1)
    res_df = rename_algos_by_args(res_df, algo_args_df)
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

def get_plot_title(df: pd.DataFrame, target_samples: int, z_ops: bool = True) -> str:
    """assemble plot title, execute before filtering out columns from dataframe"""
    m_averages = int(df["m_averages"].iloc[[0]])
    n_osc = df["n_osc"].values[0]
    max_z_ops = int(df["max_z_ops"].values[0])
    str_max_z_ops = f", max z-ops={max_z_ops}" if z_ops else ""
    return f"m={m_averages}, n={n_osc}, samples={target_samples}{str_max_z_ops}" 

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

def plot_rmse_by_algo(df: pd.DataFrame, column_name: str, separate_legend: bool = const.SEPARATE_LEGEND) -> Tuple[plt.Figure, plt.Figure]:
    """plot one line for each algorithm given an attribute column in a dataframe

    args:
        column_name: for example amplitude or freq_range
    """
    fig = plt.figure()
    algo_names = set(df["algo_name"])
    algo_names = sorted(algo_names)
    dfs_by_algo = [df.loc[df["algo_name"] == name] for name in algo_names]        
    for data, name in zip(dfs_by_algo, algo_names):
        plt.errorbar(data[column_name], data["mean_rmse"], data["std_rmse"], elinewidth=1, capsize=5, capthick=1, markeredgewidth=1, label=name)
    plt.legend(title="Algorithm", framealpha=0.5)
    plt.gca().set_ylabel("RMSE(generated, target)")

    if separate_legend == False: return fig, None
    
    fig.gca().legend().set_visible(False) # hide legend on main plot
    legend_as_fig = plt.figure()
    plt.figlegend(*fig.gca().get_legend_handles_labels(), loc="upper left", title="algorithm")
    
    return fig, legend_as_fig

def save_fig_n_legend(fig: plt.Figure, legend_as_fig: plt.Figure, name: str, show: bool = False) -> None:
    fig.savefig(Path("data") / (name + ".png"), dpi=300)
    if legend_as_fig is not None:
        legend_as_fig.savefig(Path("data") / (name + "_legend.png"), dpi=300)
    if show: plt.show()

def plot_n_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp1: plot number of oscillators, n, against rmse for multiple algorithms with z_ops and rand_args fixed"""
    title = get_plot_title(df, target_samples)
    df = df.filter(items=["algo_name", "n_osc", "mean_rmse", "std_rmse"])
    fig, legend_as_fig = plot_rmse_by_algo(df, "n_osc")
    fig.gca().set_title(title)
    fig.gca().set_xlabel("number of oscillators")
    save_fig_n_legend(fig, legend_as_fig, "n_on_rmse", show)

def plot_z_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp2: plot number of operations, z_ops, against rmse for multiple algorithms with rand_args fixed"""
    title = get_plot_title(df, target_samples, z_ops=False)
    df = df.filter(items=["algo_name", "max_z_ops", "mean_rmse", "std_rmse"])
    fig, legend_as_fig = plot_rmse_by_algo(df, "max_z_ops")
    fig.gca().set_title(title)
    fig.gca().set_xlabel("z-operations")
    save_fig_n_legend(fig, legend_as_fig, "z_vs_rmse", show)

def plot_samples_vs_rmse(df: pd.DataFrame, show: bool = False) -> None:
    """exp3: plot number of samples against rmse for multiple algorithms with rand_args and target fixed"""
    # before filtering df
    m_averages = int(df["m_averages"].iloc[[0]])
    n_osc = df["n_osc"].values[0]
    max_z_ops = int(df["max_z_ops"].values[0])
    title = f"m={m_averages}, n={n_osc}, max z-ops={max_z_ops}"

    df = df.filter(items=["algo_name", "samples", "mean_rmse", "std_rmse"])
    fig, legend_as_fig = plot_rmse_by_algo(df, "samples")
    fig.gca().set_title(title)
    fig.gca().set_xlabel("number of samples")
    save_fig_n_legend(fig, legend_as_fig, "samples_vs_rmse", show)

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

    fig, legend_as_fig = plot_rmse_by_algo(df, range_name)
    fig.gca().set_title(f"{title}, dist={dist_name}")

    return fig, legend_as_fig

def plot_freq_range_vs_rmse(df: pd.DataFrame, target_samples: int, sweep_name: str, show: bool = False) -> None:
    """exp4+5: plot frequency range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in const.LEGAL_DISTS:
        fig, legend_as_fig = plot_range_vs_rmse(df, target_samples, "freq", dist_name)
        x_label = "frequency diversity [Hz]"
        if dist_name == "uniform" and sweep_name == "freq_range_from_zero":
            x_label += r", lower bound $\rightarrow$ 0, upper bound = $x$"
        elif dist_name == "uniform" and sweep_name == "freq_range_around_vo2":
            x_label += r", lower + upper bound = $x$"
        elif dist_name == "normal" and sweep_name == "freq_range_from_zero":
            x_label += r", $\mu + \sigma = x$ , $\mu \approx \sigma$"
        elif dist_name == "normal" and sweep_name == "freq_range_around_vo2":
            mu = df["freq_dist_loc"].dropna().iloc[0]
            x_label += r", $\mu$=" + f"{mu:.0f}, " + r"$\sigma = x/2$"
        fig.gca().set_xlabel(x_label) # width of frequency band
        fig.gca().set_xscale("log")
        save_fig_n_legend(fig, legend_as_fig, f"{sweep_name}_{dist_name}_vs_rmse", show=False)
    if show: plt.show()

def plot_weight_range_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp6: plot weight range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in const.LEGAL_DISTS:
        fig, legend_as_fig = plot_range_vs_rmse(df, target_samples, "weight", dist_name)
        inv_amplitude = 1/df["amplitude"].iloc[0]
        fig.gca().set_xlabel(f"dynamic range (scaled by inverse-of-amplitude={inv_amplitude:.0f})") # width of weight band
        # dynamic range would be given with amplitude=1
        fig.gca().set_xscale("log")
        save_fig_n_legend(fig, legend_as_fig, f"weight_range_{dist_name}_vs_rmse", show=False)
    if show: plt.show()

def plot_phase_range_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp7: plot phase range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in const.LEGAL_DISTS:
        fig, legend_as_fig = plot_range_vs_rmse(df, target_samples, "phase", dist_name)
        fig.gca().set_xlabel("phase diversity") # width of phase band
        fig.gca().xaxis.set_major_formatter("{x:.2f}"+r"$\pi$")
        save_fig_n_legend(fig, legend_as_fig, f"phase_range_{dist_name}_vs_rmse", show=False)
    if show: plt.show()

def plot_offset_range_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    """exp8: plot offset range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in const.LEGAL_DISTS:
        fig, legend_as_fig = plot_range_vs_rmse(df, target_samples, "offset", dist_name)
        fig.gca().set_xlabel("offset diversity") # width of offset distribution
        save_fig_n_legend(fig, legend_as_fig, f"offset_range_{dist_name}_vs_rmse", show=False)
    if show: plt.show()

def plot_amplitude_vs_rmse(df: pd.DataFrame, target_samples: int, show: bool = False) -> None:
    "exp9: plot amplitude against rmse for multiple algorithms with rand_args and target fixed"
    title = get_plot_title(df, target_samples) # before filtering df
    weight_range = df["weight_range"].iloc[0]
    title += f", weight_range={weight_range:.1f}"
    df = df.filter(items=["algo_name", "mean_rmse", "std_rmse", "amplitude"])
    fig, legend_as_fig = plot_rmse_by_algo(df, "amplitude")
    fig.gca().set_title(title)
    fig.gca().set_ylabel("RMSE(generated, target)")
    fig.gca().set_xlabel("unweighted amplitude")
    save_fig_n_legend(fig, legend_as_fig, "amplitude_vs_rmse", show)