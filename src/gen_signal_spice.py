from data_analysis import compute_rmse, plot_fourier, plot_signal
from data_io import find_dir_name, json_to_df, load_data, load_sim_data
from data_preprocessor import clean_signal
from param_types import SpiceSingleDetArgs, SpiceSumDetArgs, SpiceSumRandArgs
from params import bird_params
from netlist_generator import build_single_netlist, build_sum_netlist, run_ngspice, select_netlist_generator

from typing import Callable, List, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAM = bird_params["magpie_single_oscillator"]
    
# ability to generate single oscillator in ngspice
# ability to generate k oscillators and sum them in ngspice
def gen_random_spice_signal(netlist_generator: Callable, param: SpiceSingleDetArgs) -> np.ndarray:
    """generate a single signal which is a composition
    of oscillators in ngspice.
    
    the signal may be
    
    a single oscillator
    or a sum of k oscillators
    or a tree of k oscillators
    or a graph of k oscillators
    
    NOT however l single oscillators,
    l sums of k oscillators,
    l trees or graphs
    
    the random search module is responsible for said abstraction"""
    sampling_rate, target = load_data()
    # generate path to save signal and circuit
    experiment_path = find_dir_name(Path("data"))
    tmp_path = experiment_path / "netlist.cir"
    # build netlist and store parameters
    build_single_netlist(tmp_path, param)
    # run simulation on ngspice
    run_ngspice(tmp_path)
    # load the simulation data written by ngspice into Python
    df = load_sim_data(Path(str(tmp_path) + ".dat"))
    # convert from dataframe to numpy
    pred = df.iloc[:,1].to_numpy()
    # compute rmse of generated signal with the target
    
    # TODO: migrate
    # with open(experiment_path / f"param.json", "w") as f:
    #     json.dump(det_params, f)
    
    # df = json_to_df(experiment_path)
    return pred


def draw_params_random(args: SpiceSumRandArgs) -> SpiceSingleDetArgs:
    """draw randomly from parameter pool"""
    assert args.c_max <= 1, "Randomly generating capacitors with >1 Farad is not implemented!"

    # TODO: generalize for >0 and <0 values
    #   so over randint vs uniform
    r = np.random.randint(args.r_min, 1 + args.r_max)
    c = np.random.uniform(args.c_min, args.c_max)

    return SpiceSingleDetArgs(args.n_osc, args.v_in,
    r, args.r_last, args.r_control, c,
    args.time_step, args.time_stop, args.time_start)


def sum_atomic_signals(args: SpiceSumRandArgs) -> Tuple[np.ndarray, List[SpiceSingleDetArgs]]:
    """compose a signal of single oscillators"""
    # TODO: is this how samples actually works? check this
    samples = int((args.time_stop - args.time_start) // args.time_step)
    signal_matrix = np.empty((args.n_osc, samples))
    det_arg_li = list()

    for i in range(args.n_osc):
        # determine a set of parameters for a single oscillator
        det_params = draw_params_random(args)
        # store the parameter set
        det_arg_li.append(det_params)
        # generate single oscillator signal and add to matrix
        single_signal = gen_random_spice_signal(build_sum_netlist, det_params)
        signal_matrix[i,:] = single_signal[0:samples] # exact number of samples is non-deterministic
                                                        # around 2% cutoff
    return signal_matrix, det_arg_li

def main():
    args = SpiceSumRandArgs()
    atomic_signals, det_arg_li = sum_atomic_signals(args)
    sig_sum = sum(atomic_signals)
    plot_signal(sig_sum, show=True)

    exit()
    netlist_generator = select_netlist_generator("sum")
    pred, df_meta = gen_random_spice_signal(netlist_generator, param=PARAM)

    plot_signal(pred)
    plot_signal(pred, x=np.linspace(0, PARAM["time_stop"], len(pred)), ylabel=PARAM["dependent_component"])

    clean_pred = clean_signal(pred)
    plot_fourier(clean_pred)

    plt.show()


if __name__ == "__main__":
    main()