from data_analysis import compute_rmse, plot_fourier, plot_signal
from data_io import find_dir_name, json_to_df, load_data, load_sim_data
from data_preprocessor import clean_signal
from params import bird_params
from netlist_generator import run_ngspice, select_netlist_generator

from typing import Callable
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAM = bird_params["magpie_single_oscillator"]
    
# ability to generate single oscillator in ngspice
# ability to generate k oscillators and sum them in ngspice
def gen_random_spice_signal(netlist_generator: Callable, param: dict) -> tuple[list, pd.DataFrame]:
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
    det_params = netlist_generator(tmp_path, param)
    # run simulation on ngspice
    run_ngspice(tmp_path)
    # load the simulation data written by ngspice into Python
    df = load_sim_data(Path(str(tmp_path) + ".dat"))
    # convert from dataframe to numpy
    pred = df.iloc[:,1].to_numpy()
    # compute rmse of generated signal with the target
    det_params["rmse"] = compute_rmse(pred, target, pad=True)
    
    with open(experiment_path / f"param.json", "w") as f:
        json.dump(det_params, f)
    
    # aggregate the results of n trials into a single dataframe
    df = json_to_df(experiment_path)
    return pred, df

def main():
    netlist_generator = select_netlist_generator("sum")
    pred, df_meta = gen_random_spice_signal(netlist_generator, param=PARAM)

    plot_signal(pred)
    plot_signal(pred, x=np.linspace(0, PARAM["time_stop"], len(pred)), ylabel=PARAM["dependent_component"])

    clean_pred = clean_signal(pred)
    plot_fourier(clean_pred)

    plt.show()


if __name__ == "__main__":
    main()
