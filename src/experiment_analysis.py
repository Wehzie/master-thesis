"""
This module implements functions for plotting and analyzing the results of experiments.

The focus is on comparing multiple algorithms.
"""

import result_types as resty
import const
import gen_signal_args_types as party
import data_io

from typing import Callable, List, Set, Union, Tuple
from pathlib import Path
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import matplotlib.colors as mcolors
import mask_type as param_mask

TABLEAU_COLORS: List[str] = [color for color in list(mcolors.TABLEAU_COLORS.values())]  # HEX colors

# rainbow colors
cmap = mpl.cm.get_cmap("Spectral")
rb_colors = [cmap(i) for i in np.linspace(0, 0.9, 10)]
# tableau colors
colors = plt.cm.tab10.colors
# set up plotting
linestyles = ["solid", "dashed", "dotted"]
color_list = colors * len(linestyles)
linestyle_list = [[style] * len(colors) for style in linestyles]
# flatten from [[style1, ...], [style2, ...]] to [style1, style1, ..., style2, ...]
linestyle_list = itertools.chain.from_iterable(linestyle_list)
default_cycler = cycler(color=color_list) + cycler(linestyle=linestyle_list)
plt.rc("axes", prop_cycle=default_cycler)
plt.rc("legend", fontsize=8)  # using a size in points


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
        # check if rand_args has freq_dist attribute
        if hasattr(result.algo_args.rand_args, "freq_dist"):
            result.algo_args.rand_args.freq_dist.compute_range()
        elif hasattr(result.algo_args.rand_args, "r_dist"):
            result.algo_args.rand_args.r_dist.compute_range()
        else:
            raise ValueError("rand_args has no freq_dist or r_dist attribute")
        result.algo_args.rand_args.weight_dist.compute_range()
        result.algo_args.rand_args.phase_dist.compute_range()
        result.algo_args.rand_args.offset_dist.compute_range()
    return results


def rename_algos_by_args(name_df: pd.DataFrame, algo_args_df: pd.DataFrame) -> pd.DataFrame:
    """
    rename identical algorithms in the dataframe by the distinguishing arguments they were called with.

    for example, MCExploitJ1, when j_replacements=1 and MCExploitJ5, when j_replacements=5
    """
    # filter for j_replace, keep index intact to merge back into name_df
    temp_df = pd.concat([name_df["algo_name"], algo_args_df["j_replace"]], axis=1)
    temp_df.dropna(inplace=True)
    temp_df["j_replace"] = temp_df["j_replace"].astype(int)  # ints are prettier than floats
    temp_df["algo_name"] = temp_df["algo_name"] + ", j=" + temp_df["j_replace"].astype(str)
    temp_df = name_df.merge(temp_df, left_index=True, right_index=True, how="outer")

    # rename the algorithms
    name_df["algo_name"] = temp_df["algo_name_y"].fillna(temp_df["algo_name_x"])
    return name_df


def get_rand_args_dist_names(rand_args: party.UnionRandArgs) -> List[str]:
    """get the names of the distribution attributes of the rand_args class"""
    rand_args_dist_names = ["weight_dist", "phase_dist", "offset_dist", "description"]
    if isinstance(rand_args, party.PythonSignalRandArgs):
        rand_args_dist_names.append("freq_dist")
    elif isinstance(rand_args, party.SpiceSumRandArgs):
        rand_args_dist_names += ["r_last", "r_control", "dependent_component"]
    else:
        raise ValueError("rand_args must be PythonSignalRandArgs or SpiceSumRandArgs")
    return rand_args_dist_names


def select_frequency_controller_dist_range(rand_args: List[party.UnionRandArgs]) -> pd.DataFrame:
    """
    Select the frequency controller based on the type of rand_args.

    Finds the range of a uniform distribution, meaning upper - lower; also known as attribute diversity.
    The implementation is hacky in the sense that r_dist corresponds the resistor values,
    not the frequency of the signal. This is because the frequency is not a random variable.
    However the solution could be improved by mapping R to a frequency distribution.
    """
    if isinstance(rand_args[0], party.PythonSignalRandArgs):
        freq_dist_range = pd.DataFrame(
            [ra.freq_dist.range for ra in rand_args], columns=["freq_range"]
        )
    elif isinstance(rand_args[0], party.SpiceSumRandArgs):
        freq_dist_range = pd.DataFrame(
            [ra.r_dist.range for ra in rand_args], columns=["freq_range"]
        )
    else:
        raise ValueError("rand_args must be PythonSignalRandArgs or SpiceSumRandArgs")
    return freq_dist_range


def select_frequency_controller_bounds(rand_args: List[party.UnionRandArgs]) -> pd.DataFrame:
    """
    Select the frequency controller based on the type of rand_args.

    Finds the upper and lower bound of a distribution.
    The implementation is hacky in the sense that r_dist corresponds the resistor values,
    not the frequency of the signal. This is because the frequency is not a random variable.
    However the solution could be improved by mapping R to a frequency distribution.
    """
    if isinstance(rand_args[0], party.PythonSignalRandArgs):
        freq_dist_df = pd.DataFrame(
            [add_str2keys("freq_dist", ra.freq_dist.kwargs) for ra in rand_args]
        )
    elif isinstance(rand_args[0], party.SpiceSumRandArgs):
        freq_dist_df = pd.DataFrame(
            [add_str2keys("freq_dist", ra.r_dist.kwargs) for ra in rand_args]
        )
    else:
        raise ValueError("rand_args must be PythonSignalRandArgs or SpiceSumRandArgs")
    return freq_dist_df


def infer_duration(results: List[resty.ResultSweep]) -> pd.DataFrame:
    """if duration is not set in the results, infer it from the results"""
    if isinstance(results[0].algo_args.rand_args, party.SpiceSumRandArgs):
        return pd.DataFrame(
            [r.algo_args.rand_args.get_duration() for r in results], columns=["duration"]
        )
    return None


def infer_sampling_rate(results: List[resty.ResultSweep]) -> pd.DataFrame:
    """if sampling_rate is not set in the results, infer it from the results"""
    if isinstance(results[0].algo_args.rand_args, party.SpiceSumRandArgs):
        return pd.DataFrame(
            [r.algo_args.rand_args.get_sampling_rate() for r in results], columns=["sampling_rate"]
        )
    return None


# TODO: write function to recursively unpack objects inside a dataframe and each field as column
def conv_results_to_pd(results: List[resty.ResultSweep]) -> pd.DataFrame:
    """convert ResultSweep to a pandas dataframe for further processing"""
    results = compute_dist_ranges(results)

    # initial definitions for convenience
    rand_args: party.UnionRandArgs = [r.algo_args.rand_args for r in results]
    rand_args_to_drop = get_rand_args_dist_names(rand_args[0])

    # unpack the nested args such that each entry has its own column
    res_df = pd.DataFrame(results).drop("algo_args", axis=1)
    algo_args_df = pd.DataFrame([r.algo_args for r in results]).drop(
        ["rand_args", "meta_target"], axis=1
    )
    res_df = rename_algos_by_args(res_df, algo_args_df)
    rand_args_df = pd.DataFrame(rand_args).drop(rand_args_to_drop, axis=1)
    target_df = pd.DataFrame(
        [r.algo_args.meta_target.name for r in results], columns=["target_name"]
    )
    target_max_freq_df = pd.DataFrame(
        [r.algo_args.meta_target.get_max_freq() for r in results], columns=["target_max_freq"]
    )

    duration_df = infer_duration(results)
    sampling_rate_df = infer_sampling_rate(results)

    # dist ranges
    freq_dist_range = select_frequency_controller_dist_range(rand_args)
    weight_dist_range = pd.DataFrame(
        [ra.weight_dist.range for ra in rand_args], columns=["weight_range"]
    )
    phase_dist_range = pd.DataFrame(
        [ra.phase_dist.range for ra in rand_args], columns=["phase_range"]
    )
    offset_dist_range = pd.DataFrame(
        [ra.offset_dist.range for ra in rand_args], columns=["offset_range"]
    )

    # dist bounds
    freq_dist_df = select_frequency_controller_bounds(rand_args)
    weight_dist_df = pd.DataFrame(
        [add_str2keys("weight_dist", ra.weight_dist.kwargs) for ra in rand_args]
    )
    phase_dist_df = pd.DataFrame(
        [add_str2keys("phase_dist", ra.phase_dist.kwargs) for ra in rand_args]
    )
    offset_dist_df = pd.DataFrame(
        [add_str2keys("offset_dist", ra.offset_dist.kwargs) for ra in rand_args]
    )

    return pd.concat(
        [
            res_df,
            target_df,
            target_max_freq_df,
            algo_args_df,
            rand_args_df,
            duration_df,
            sampling_rate_df,
            freq_dist_range,
            weight_dist_range,
            phase_dist_range,
            offset_dist_range,
            freq_dist_df,
            weight_dist_df,
            phase_dist_df,
            offset_dist_df,
        ],
        axis=1,
    )


def get_plot_title(df: pd.DataFrame, target_samples: int, z_ops: bool = True) -> str:
    """assemble plot title, execute before filtering out columns from dataframe"""
    m_averages = int(df["m_averages"].iloc[[0]])
    n_osc = df["n_osc"].values[0]
    max_z_ops = int(df["max_z_ops"].values[0])
    str_max_z_ops = f", z={max_z_ops}" if z_ops else ""

    duration = df["duration"].values[0] if "duration" in df else None
    sampling_rate = df["sampling_rate"].values[0] if "sampling_rate" in df else None

    duration_txt = f", t={duration} s" if duration is not None else ""
    sampling_rate_txt = f", fs={sampling_rate} Hz" if sampling_rate is not None else ""
    time_vars = (
        duration_txt + sampling_rate_txt
        if duration_txt != "" or sampling_rate_txt != ""
        else f"#s={target_samples}"
    )

    target_name = df["target_name"].values[0]
    title = f"{target_name}, m={m_averages}, n={n_osc}{time_vars}{str_max_z_ops}"
    return title


def filter_df_by_dist_name(df: pd.DataFrame, attr_name: str, dist_name: str) -> pd.DataFrame:
    """
    filter a dataframe by a type of distribution name, e.g. "uniform" or "normal"

    args:
        attr_name: in freq, weight, phase, offset
        dist_name: for example uniform
    """
    if dist_name == "uniform" and f"{attr_name}_dist_low" in df.columns:
        # select the rows where the low bounds of an attribute are not empty; the high bounds won't be empty either
        return df[df[f"{attr_name}_dist_low"].notna()]  # for example freq_dist_low
    if dist_name == "normal" and f"{attr_name}_dist_loc" in df.columns:
        # select the rows where the loc of an attribute are not empty; the scale won't be empty either
        return df[df[f"{attr_name}_dist_loc"].notna()]
    raise ValueError(f"unknown distribution name {dist_name}")


# TODO: the better approach here is to pass the unmodified algorithm name in the df
def clean_algo_name_from_args(algo_name: str) -> str:
    """
    remove the arguments that are part of an algorithm's name.

    for example 'MCExploit, j=1' -> 'MCExploit'.
    """
    return algo_name.split(",", 1)[0]


def get_color_map(algo_names: Set[str], with_pattern: bool = False) -> dict:
    """get matplotlib compatible color map for each algorithm in the list"""
    color_map = {}
    if with_pattern:
        visual_identifiers = itertools.product(TABLEAU_COLORS, [" ", "O", "x", "+"])
    else:
        visual_identifiers = itertools.product(TABLEAU_COLORS, [" "])
    for algo, (color, pattern) in zip(algo_names, visual_identifiers):
        color_map[algo] = (color, pattern)
    return color_map


def pick_color_linestyle(
    mask: param_mask.ExperimentMask, algo_name: str
) -> Tuple[Union[None, str]]:
    """map an algorithm to a color according to the mask"""
    if mask is None:
        return None, None
    algo_name = clean_algo_name_from_args(algo_name)
    return mask.get_color_map()[algo_name], "solid"


def plot_rmse_by_algo(
    df: pd.DataFrame,
    column_name: str,
    separate_legend: bool = const.HOARD_DATA,
    mask: param_mask.ExperimentMask = None,
) -> Tuple[plt.Figure, plt.Figure]:
    """plot one line for each algorithm given an attribute column in a dataframe

    args:
        column_name: for example amplitude or freq_range
    """
    fig = plt.figure()
    algo_names = set(df["algo_name"])
    algo_names = sorted(algo_names)
    dfs_by_algo = [df.loc[df["algo_name"] == name] for name in algo_names]
    for data, name in zip(dfs_by_algo, algo_names):
        color, linestyle = pick_color_linestyle(mask, name)
        plt.errorbar(
            data[column_name],
            data["mean_rmse"],
            data["std_rmse"],
            elinewidth=1,
            capsize=5,
            capthick=1,
            markeredgewidth=1,
            label=name,
            color=color,
            linestyle=linestyle,
        )
    plt.legend(title="Algorithm", framealpha=0.5)
    plt.gca().set_ylabel("RMSE")

    if separate_legend == False:
        return fig, None

    fig.gca().legend().set_visible(False)  # hide legend on main plot
    legend_as_fig = plt.figure()  # create new figure for legend
    plt.figlegend(*fig.gca().get_legend_handles_labels(), loc="upper left", title="algorithm")

    return fig, legend_as_fig


def save_mask_plot(
    fig: plt.Figure,
    legend_as_fig: plt.Figure,
    name: str,
    save_dir: Path,
    mask: param_mask.ExperimentMask,
) -> None:
    if legend_as_fig is not None:
        plt.close(legend_as_fig)  # legend is already on main plot
    name = name + "_MASK_" + mask.filename
    fig.gca().legend().set_visible(True)  # save figure with legend
    fig.savefig(save_dir / (name + ".png"), dpi=300)


def save_unmasked_plot(
    fig: plt.Figure, legend_as_fig: plt.Figure, name: str, save_dir: Path
) -> None:
    if legend_as_fig is None:
        fig.savefig(save_dir / (name + ".png"), dpi=300)
        data_io.pickle_object(fig, save_dir / (name + "_figure.pickle"))
    else:  # const.HOARD_DATA == True
        legend_as_fig.savefig(save_dir / (name + "_legend.png"), dpi=300)
        fig.gca().legend().set_visible(True)  # save figure with legend
        fig.savefig(save_dir / (name + "_with_legend.png"), dpi=300)
        data_io.pickle_object(fig, save_dir / (name + "_figure.pickle"))
        fig.gca().legend().set_visible(False)  # hide legend on main plot for viewing
        fig.savefig(save_dir / (name + "_sans_legend.png"), dpi=300)


def save_fig_n_legend(
    fig: plt.Figure,
    legend_as_fig: plt.Figure,
    name: str,
    save_dir: Path = const.WRITE_DIR,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """save a figure and its legend to disk"""
    if mask is None:
        save_unmasked_plot(fig, legend_as_fig, name, save_dir)
    else:  # there is a mask
        save_mask_plot(fig, legend_as_fig, name, save_dir, mask)
    if show:
        plt.show()
    if legend_as_fig is not None:
        plt.close(legend_as_fig)  # close figures to save memory
    plt.close(fig)


def clean_algo_name_column(df: pd.DataFrame) -> pd.DataFrame:
    """remove the arguments that are part of an algorithm's name in the dataframe"""
    return df["algo_name"].apply(clean_algo_name_from_args)


def apply_mask(df: pd.DataFrame, mask: param_mask.ExperimentMask):
    """filter out algorithms that are not in the list"""
    if mask is not None:
        mask_algo_names = mask.get_algo_names()
        df_algo_names = clean_algo_name_column(df)
        return df.loc[df_algo_names.isin(mask_algo_names)]
    return df


def subset_df_by_n_algos(df: pd.DataFrame, n_algos: int) -> List[pd.DataFrame]:
    """return a dataframe with only n_algos per group"""
    unique_algo_names = df["algo_name"].unique()
    # divide into groups of size n
    algo_groups = [
        unique_algo_names[x : x + n_algos] for x in range(0, len(unique_algo_names), n_algos)
    ]
    dfs_by_algo = []
    for algo_group in algo_groups:
        dfs_by_algo.append(df.loc[df["algo_name"].isin(algo_group)])
    return dfs_by_algo


def plot_masks(masks: List[param_mask.ExperimentMask], plot_func: Callable, *args, **kwargs):
    """call a plot function with all masks"""
    for mask in masks:
        plot_func(*args, mask=mask, **kwargs)


def plot_n_vs_rmse(
    df: pd.DataFrame,
    target_samples: int,
    sweep_name: str,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp1: plot number of oscillators, n, against rmse for multiple algorithms with z_ops and rand_args fixed"""
    title = get_plot_title(df, target_samples)
    df = df.filter(items=["algo_name", "n_osc", "mean_rmse", "std_rmse"])
    df = apply_mask(df, mask)
    fig, legend_as_fig = plot_rmse_by_algo(df, "n_osc", mask=mask)
    fig.gca().set_title(title)
    fig.gca().set_xlabel("number of oscillators")
    save_fig_n_legend(fig, legend_as_fig, sweep_name, save_dir, mask, show)


def plot_z_vs_rmse(
    df: pd.DataFrame,
    target_samples: int,
    sweep_name: str,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp2: plot number of operations, z_ops, against rmse for multiple algorithms with rand_args fixed"""
    title = get_plot_title(df, target_samples, z_ops=False)
    df = df.filter(items=["algo_name", "max_z_ops", "mean_rmse", "std_rmse"])
    df = apply_mask(df, mask)
    fig, legend_as_fig = plot_rmse_by_algo(df, "max_z_ops", mask=mask)
    fig.gca().set_title(title)
    fig.gca().set_xlabel("z-perturbations")
    save_fig_n_legend(fig, legend_as_fig, sweep_name, save_dir, mask, show)


def plot_samples_vs_rmse(
    df: pd.DataFrame,
    sweep_name: str,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp3: plot number of samples against rmse for multiple algorithms with rand_args and target fixed"""
    title = get_plot_title(df, target_samples=None)
    df = df.filter(items=["algo_name", "samples", "mean_rmse", "std_rmse"])
    df = apply_mask(df, mask)
    fig, legend_as_fig = plot_rmse_by_algo(df, "samples", mask=mask)
    fig.gca().set_title(title)
    fig.gca().set_xlabel("time [a.u.]")
    save_fig_n_legend(fig, legend_as_fig, sweep_name, save_dir, mask, show)


def plot_targets_w_all_algos(df: pd.DataFrame, title: str) -> plt.Figure:
    """exp4: plot where each target forms a group of bars, each bar is an algorithm"""
    # compute the label locations
    algo_names = set(df["algo_name"])
    unique_target_names = set(df["target_name"])
    n_targets = len(unique_target_names)
    y_target = np.arange(n_targets)  # bar start locations for each target
    space_between_groups = n_targets / 2  # in relation to the size of the bars
    width = (n_targets - 1) / (n_targets * space_between_groups)  # width of each bar

    def get_positions(n_algos: int, width: float):
        positions = []
        for y_t in y_target:
            position = y_t - width
            for _ in range(n_algos):
                positions.append(position)
                position += width
        return positions

    positions = get_positions(len(algo_names), width)

    color_map = get_color_map(algo_names)

    fig, ax = plt.subplots()
    error_kw = dict(elinewidth=1, capsize=1, markeredgewidth=1)

    first_target = True
    target_names_in_order = []
    for (index, row), position in zip(df.iterrows(), positions):
        if row.target_name not in target_names_in_order:
            target_names_in_order.append(row.target_name)
        if index == len(algo_names):  # avoid duplicate labels
            first_target = False
        ax.barh(
            position,
            row.mean_rmse,
            width,
            xerr=row.std_rmse,
            error_kw=error_kw,
            label=row.algo_name if first_target else "",
            color=color_map[row.algo_name][0],
            hatch=color_map[row.algo_name][1],
        )

    ax.set_yticks(y_target - 0.5 * width, target_names_in_order)
    plt.xlabel("RMSE")
    ax.legend()
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_algos_w_all_target(df: pd.DataFrame, title: str) -> plt.Figure:
    """exp4: plot of subplots where each subplot shows one target and each bar is an algorithm"""
    unique_algo_names = set(df["algo_name"])
    n_algos = len(unique_algo_names)
    y_algo_pos = np.arange(n_algos)

    unique_target_names = set(df["target_name"])
    dfs_by_target = [df[df["target_name"] == target_name] for target_name in unique_target_names]

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)  # constrained_layout=True
    error_kw = dict(elinewidth=1, capsize=1, markeredgewidth=1)
    color_map = get_color_map(set(df["algo_name"]))

    for ax, df in zip(axs.flat, dfs_by_target):
        title_str = df["target_name"].iloc[[0]].values[0]
        ax.set_title(f"target={title_str}")
        for (index, row), position in zip(df.iterrows(), y_algo_pos):
            ax.barh(
                position,
                row.mean_rmse,
                1,
                xerr=row.std_rmse,
                error_kw=error_kw,
                label=row.algo_name,
                color=color_map[row.algo_name][0],
                hatch=color_map[row.algo_name][1],
            )

    ax.set_yticks(y_algo_pos, set(df["algo_name"]))
    plt.xlabel("RMSE")
    ax.legend()
    plt.title(title)
    return fig


def plot_targets_vs_rmse(
    df: pd.DataFrame,
    sweep_name: str,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp4: show the rmse of algorithms with different target samples in a bar plot"""
    title = f"m={int(df['m_averages'].iloc[[0]])}, n={int(df['n_osc'].iloc[[0]])}, z={int(df['max_z_ops'].iloc[[0]])}"
    df = df.filter(items=["algo_name", "target_name", "mean_rmse", "std_rmse"])
    df = apply_mask(df, mask)

    dfs = subset_df_by_n_algos(df, 8)
    for i, df in enumerate(dfs):
        fig = plot_algos_w_all_target(df, title)
        save_fig_n_legend(fig, None, sweep_name + f"_focus_algos_plot{i}", save_dir, mask, show)
    dfs = subset_df_by_n_algos(df, 4)
    for i, df in enumerate(dfs):
        fig = plot_targets_w_all_algos(df, title)
        save_fig_n_legend(fig, None, sweep_name + f"_focus_targets_plot{i}", save_dir, mask, show)


def tab_targets_vs_rmse(
    df: pd.DataFrame, sweep_name: str, save_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """exp4: save a csv file with the mean and std RMSE for each target (across algorithm) and each algorithm (across targets)"""
    target_means = df.groupby(["target_name"])["mean_rmse"].mean().reset_index()
    target_stds = df.groupby(["target_name"])["std_rmse"].mean().reset_index()
    algo_means = df.groupby(["algo_name"])["mean_rmse"].mean().reset_index()
    algo_stds = df.groupby(["algo_name"])["std_rmse"].mean().reset_index()
    df_target = pd.DataFrame(
        {
            "target_name": target_means["target_name"],
            "mean_rmse": target_means["mean_rmse"],
            "std_rmse": target_stds["std_rmse"],
        }
    )
    df_algo = pd.DataFrame(
        {
            "algo_name": algo_means["algo_name"],
            "mean_rmse": algo_means["mean_rmse"],
            "std_rmse": algo_stds["std_rmse"],
        }
    )
    df_target.to_csv(save_dir / f"{sweep_name}_target_means.csv", index=False)
    df_algo.to_csv(save_dir / f"{sweep_name}_algo_means.csv", index=False)
    return df_target, df_algo


def plot_average_target_vs_rmse(
    target_df: pd.DataFrame,
    num_algos: int,
    sweep_name: str,
    save_dir: Path,
    title: str,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
):
    """exp4: show rmse for targets averaged over algorithms"""
    title += f", #a={num_algos}"
    target_df = target_df.sort_values(by="mean_rmse", ascending=False)
    cmap = mpl.cm.get_cmap("Spectral")
    colors = target_df["mean_rmse"].apply(lambda x: cmap(x / target_df["mean_rmse"].max())).tolist()
    ax = target_df.plot.barh(
        x="target_name", y="mean_rmse", xerr="std_rmse", title=title, color=colors, legend=False
    )
    ax.set_xlabel("RMSE")
    ax.set_ylabel("target")
    plt.yticks(rotation=30, ha="right")
    plt.tight_layout()
    save_fig_n_legend(ax.get_figure(), None, sweep_name + "_averaged_targets", save_dir, mask, show)


def plot_average_algo_vs_rmse(
    target_df: pd.DataFrame,
    num_targets: int,
    sweep_name: str,
    save_dir: Path,
    title: str,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
):
    """exp4: show rmse for algorithms averaged over different targets"""
    title += f", #t={num_targets}"
    target_df = target_df.sort_values(by="mean_rmse", ascending=False)
    cmap = mpl.cm.get_cmap("Spectral")
    colors = target_df["mean_rmse"].apply(lambda x: cmap(x / target_df["mean_rmse"].max())).tolist()
    ax = target_df.plot.barh(
        x="algo_name", y="mean_rmse", xerr="std_rmse", title=title, color=colors, legend=True
    )
    ax.set_xlabel("RMSE")
    ax.set_ylabel("algorithm")
    ax.legend().set_visible(False)
    plt.tight_layout()
    save_fig_n_legend(
        ax.get_figure(), None, sweep_name + "_averaged_algorithms", save_dir, mask, show
    )


def analyze_targets_vs_rmse(
    df: pd.DataFrame,
    sweep_name: str,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp4: show the rmse of algorithms with different targets in a bar plot and tab the results"""
    target_df, algo_df = tab_targets_vs_rmse(df, sweep_name, save_dir)
    num_algos = len(algo_df)
    num_targets = len(target_df)
    title = get_plot_title(df, None)

    plot_average_target_vs_rmse(target_df, num_algos, sweep_name, save_dir, title, mask, show)
    plot_average_algo_vs_rmse(algo_df, num_targets, sweep_name, save_dir, title, mask, show)
    # TODO: fix this some other time, the plots are quite messy anyways
    # plot_targets_vs_rmse(df, sweep_name, save_dir, mask, show)


def plot_range_vs_rmse(
    df: pd.DataFrame,
    target_samples: int,
    attr_name: str,
    dist_name: str,
    mask: param_mask.ExperimentMask,
) -> Tuple[plt.Figure]:
    """
    exp5-9: plot range of distribution against rmse for multiple algorithms with rand_args fixed

    args:
        attr_name: for example freq
        dist_name: uniform or normal

    returns:
        the main figure and the legend as a figure
    """
    # assign strings for subsetting the dataframe and for injection to plot
    assert attr_name in ["freq", "weight", "phase", "offset"]
    range_name = attr_name + "_range"  # for example freq_range
    title = get_plot_title(df, target_samples)

    # filter df
    df = filter_df_by_dist_name(df, attr_name, dist_name)
    df = df.filter(items=["algo_name", "mean_rmse", "std_rmse", "mean_z_ops", range_name])
    df = apply_mask(df, mask)

    fig, legend_as_fig = plot_rmse_by_algo(df, range_name, mask=mask)
    fig.gca().set_title(f"{title}, dist={dist_name}")

    return fig, legend_as_fig


def find_dists_in_df(df: pd.DataFrame) -> List[str]:
    """find the distributions that are present in the dataframe"""
    dists = []
    for attr_name in ["freq", "weight", "phase", "offset"]:
        if f"{attr_name}_dist_low" in df.columns:
            dists.append("uniform")
        if f"{attr_name}_dist_loc" in df.columns:
            dists.append("normal")
    return list(set(dists))


def get_freq_plot_label(dist_name: str, sweep_name: str, df: pd.DataFrame) -> str:
    """get the x label for the frequency plots"""
    x_label = "frequency diversity [Hz]"
    # if dist_name == "uniform" and sweep_name == "freq_range_from_zero":
    #     x_label += r", lower bound $\rightarrow$ 0, upper bound = $x$"
    # elif dist_name == "uniform" and sweep_name == "freq_range_around_vo2":
    #     x_label += r", lower + upper bound = $x$"
    # elif dist_name == "normal" and sweep_name == "freq_range_from_zero":
    #     x_label += r", $\mu + \sigma = x$ , $\mu \approx \sigma$"
    # elif dist_name == "normal" and sweep_name == "freq_range_around_vo2":
    #     mu = df["freq_dist_loc"].dropna().iloc[0]
    #     x_label += r", $\mu$=" + f"{mu:.0f}, " + r"$\sigma = x/2$"
    return x_label


def get_freq_sup_title(dist_name: str, sweep_name: str, df: pd.DataFrame) -> str:
    """get the super title for the frequency plots"""
    if sweep_name == "freq_range_around_vo2":
        if dist_name == "uniform":
            bounds = df[["freq_dist_high", "freq_dist_low"]].dropna().iloc[0]
            high = bounds["freq_dist_high"]
            low = bounds["freq_dist_low"]
            mean_freq = (high + low) / 2
        if dist_name == "normal":
            mean_freq = df["freq_dist_loc"].dropna().iloc[0]
        sup_title = r"$\mu(f)=$" + f"{mean_freq:.0f} Hz"
    else:
        sup_title = "min(f) = 0 Hz"
    return sup_title


def plot_freq_range_vs_rmse(
    df: pd.DataFrame,
    target_samples: int,
    sweep_name: str,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp5+6: plot frequency range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in find_dists_in_df(df):
        fig, legend_as_fig = plot_range_vs_rmse(df, target_samples, "freq", dist_name, mask)
        x_label = get_freq_plot_label(dist_name, sweep_name, df)
        suptitle = get_freq_sup_title(dist_name, sweep_name, df)
        fig.suptitle(suptitle)
        fig.gca().set_xlabel(x_label)  # width of frequency band
        fig.gca().set_xscale("log")
        sweep_name_with_dist = f"{sweep_name}_{dist_name}_vs_rmse"
        save_fig_n_legend(fig, legend_as_fig, sweep_name_with_dist, save_dir, mask, show)
    if show:
        plt.show()


def select_generator_inverse_amplitude(df: pd.DataFrame) -> float:
    """select the inverse amplitude of the generator used in the experiment"""
    if "amplitude" in df.columns:
        return 1 / df["amplitude"].iloc[0]
    else:
        return 1 / 0.5
        # TODO: 0.5 corresponds to the voltage of the VO2-RC circuit
        # however this should be read from the data


def plot_weight_range_vs_rmse(
    df: pd.DataFrame,
    target_samples: int,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp7: plot weight range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in find_dists_in_df(df):
        fig, legend_as_fig = plot_range_vs_rmse(df, target_samples, "weight", dist_name, mask)
        fig.gca().set_xlabel("dynamic range")  # width of weight band
        # dynamic range would be given with amplitude=1
        fig.gca().set_xscale("log")
        save_fig_n_legend(
            fig, legend_as_fig, f"weight_range_{dist_name}_vs_rmse", save_dir, mask, show
        )
    if show:
        plt.show()


def plot_phase_range_vs_rmse(
    df: pd.DataFrame,
    target_samples: int,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp8: plot phase range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in find_dists_in_df(df):
        fig, legend_as_fig = plot_range_vs_rmse(df, target_samples, "phase", dist_name, mask)
        fig.gca().set_xlabel("phase diversity")  # width of phase band
        fig.gca().xaxis.set_major_formatter("{x:.2f}" + r"$\pi$")
        save_fig_n_legend(
            fig, legend_as_fig, f"phase_range_{dist_name}_vs_rmse", save_dir, mask, show
        )
    if show:
        plt.show()


def plot_offset_range_vs_rmse(
    df: pd.DataFrame,
    target_samples: int,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp9: plot offset range against rmse for multiple algorithms with rand_args and target fixed"""
    for dist_name in find_dists_in_df(df):
        fig, legend_as_fig = plot_range_vs_rmse(df, target_samples, "offset", dist_name, mask)
        fig.gca().set_xlabel("offset diversity")  # width of offset distribution
        save_fig_n_legend(
            fig, legend_as_fig, f"offset_range_{dist_name}_vs_rmse", save_dir, mask, show
        )
    if show:
        plt.show()


def plot_amplitude_vs_rmse(
    df: pd.DataFrame,
    target_samples: int,
    sweep_name: str,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp10: plot amplitude against rmse for multiple algorithms with rand_args and target fixed"""
    title = get_plot_title(df, target_samples)  # before filtering df
    weight_range = df["weight_range"].iloc[0]
    title += f", dynamic range={weight_range:.0f}"
    df = df.filter(items=["algo_name", "mean_rmse", "std_rmse", "amplitude"])
    df = apply_mask(df, mask)
    fig, legend_as_fig = plot_rmse_by_algo(df, "amplitude", mask=mask)
    fig.gca().set_title(title)
    fig.gca().set_ylabel("RMSE")
    fig.gca().set_xlabel("unweighted amplitude")
    save_fig_n_legend(fig, legend_as_fig, sweep_name, save_dir, mask, show)


def plot_resistor_range_vs_rmse(
    df: pd.DataFrame,
    target_samples: int,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp11: plot offset range against rmse for multiple algorithms with rand_args and target fixed"""

    def define_plot(dist_name: str) -> None:
        fig, legend_as_fig = plot_range_vs_rmse(df, target_samples, "freq", dist_name, mask)
        fig.gca().set_xlabel(r"resistor diversity $\Omega$")  # width of offset distribution
        save_fig_n_legend(
            fig, legend_as_fig, f"resistor_range_{dist_name}_vs_rmse", save_dir, mask, show
        )

    dist_names = find_dists_in_df(df)
    [define_plot(dist_name) for dist_name in dist_names]

    if show:
        plt.show()


def plot_duration_vs_rmse(
    df: pd.DataFrame,
    sweep_name: str,
    save_dir: Path,
    mask: param_mask.ExperimentMask = None,
    show: bool = False,
) -> None:
    """exp12: plot duration against rmse for multiple algorithms with rand_args (signal generator args) and target fixed"""
    # before filtering df
    m_averages = int(df["m_averages"].iloc[[0]])
    n_osc = df["n_osc"].values[0]
    max_z_ops = int(df["max_z_ops"].values[0])
    sampling_rate = int(df["sampling_rate"].values[0])
    target_name = df["target_name"].values[0]
    title = f"m={m_averages}, n={n_osc}, z={max_z_ops}, fs={sampling_rate} Hz, {target_name}"

    df = df.filter(items=["algo_name", "duration", "mean_rmse", "std_rmse"])
    df = apply_mask(df, mask)
    fig, legend_as_fig = plot_rmse_by_algo(df, "duration", mask=mask)
    fig.gca().set_title(title)
    fig.gca().set_xlabel("duration [s]")
    fig.gca().set_xscale("log")
    save_fig_n_legend(fig, legend_as_fig, sweep_name, save_dir, mask, show)


def plot_multi_weight_hist(results: dict, save_path: Path, show: bool = False) -> None:
    """exp13: plot a histogram to visualize the distribution of weights in an oscillator ensemble over multiple runs for multiple algorithms"""
    for key, val in results.items():  # for each algorithm
        plt.figure()
        log = False
        if key == "LinearRegression":
            log = True
        plt.hist(val["data"]["weights"], bins="auto", density=True, log=log)
        plt.gca().set_xlabel("gain")
        plt.gca().set_ylabel("probability density")

        n_osc = val["meta"]["n_osc"]
        z_ops = val["meta"]["max_z_ops"]
        sampling_rate = val["meta"]["sampling_rate"]
        target_name = val["meta"]["target_name"]
        m_averages = val["meta"]["m_averages"]
        plt.title(
            f"m={m_averages}, n={n_osc}, z={z_ops}, t={target_name}, fs={sampling_rate:.2e}, {key}"
        )

        temp_path = save_path / f"multi_weight_hist_{key}.png"
        plt.savefig(temp_path, dpi=300)

        if show:
            plt.show()


def plot_multi_weight_hist_2x2(results: dict, save_path: Path, show: bool = False) -> None:
    """exp13: combine histograms into one figure for readability"""
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    algos_of_interest = [
        "LasVegas",
        "LinearRegression",
        "MCExploitNeighborWeight",
        "MCExploitWeight",
    ]
    algo_it = 0
    for row in ax:
        for col in row:
            key = algos_of_interest[algo_it]
            val = results[key]
            col.hist(val["data"]["weights"], density=True, log=True, bins=1000)
            col.set_title(key)
            col.label_outer()
            algo_it += 1

    # Set common labels for the shared axes
    fig.text(0.5, 0.04, "gain", ha="center")
    fig.text(0.04, 0.5, "probability density", va="center", rotation="vertical")

    temp_path = save_path / "multi_weight_hist_2x2.png"
    plt.savefig(temp_path, dpi=300)

    if show:
        plt.show()


def plot_multi_freq_hist(results: dict, save_path: Path, show: bool = False) -> None:
    """exp14: plot a histogram to visualize the distribution of frequencies in an oscillator ensemble over multiple runs for multiple algorithms"""
    for key, val in results.items():  # for each algorithm
        plt.figure()
        plt.hist(val["data"]["freq"], bins="auto", density=True, log=False)
        plt.gca().set_xlabel("fundamental frequency [Hz]")
        plt.gca().set_ylabel("probability density")

        n_osc = val["meta"]["n_osc"]
        z_ops = val["meta"]["max_z_ops"]
        sampling_rate = val["meta"]["sampling_rate"]
        target_name = val["meta"]["target_name"]
        m_averages = val["meta"]["m_averages"]
        plt.title(
            f"m={m_averages}, n={n_osc}, z={z_ops}, t={target_name}, fs={sampling_rate:.2e}, {key}"
        )

        temp_path = save_path / f"multi_frequency_hist_{key}.png"
        plt.savefig(temp_path, dpi=300)

        if show:
            plt.show()


def plot_multi_freq_hist_2x2(results: dict, save_path: Path, show: bool = False) -> None:
    """exp14: combine histograms into one figure for readability of axis labels"""
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    algos_of_interest = ["LasVegas", "LinearRegression", "DifferentialEvolution", "MCExploitWeight"]
    algo_it = 0
    for row in ax:
        for col in row:
            key = algos_of_interest[algo_it]
            val = results[key]
            col.hist(val["data"]["freq"], density=True, bins="auto")
            col.set_title(key)
            col.label_outer()
            algo_it += 1

    # Set common labels for the shared axes
    fig.text(0.5, 0.01, "fundamental frequency [Hz]", ha="center")
    fig.text(0.04, 0.5, "probability density", va="center", rotation="vertical")

    temp_path = save_path / "multi_frequency_hist_2x2.png"
    plt.savefig(temp_path, dpi=300)

    if show:
        plt.show()
