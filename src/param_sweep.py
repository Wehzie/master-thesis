
import shutil
from params import param_sweep_schedule

import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


CIRCUIT_PATH = Path("circuit_lib/single_oscillator_RC.cir")
DATA_PATH = Path("data")
CONTROL_PARAMETERS = param_sweep_schedule["vo2_r1"]


CONTROL_TEMPLATE = """* Control commands
.control
set wr_vecnames * to print header
set wr_singlescale * to not print scale vectors

let start = {start}
let stop = {stop}
let step = {step}
let temp = start
*let counter = 1
* loop
while temp le stop
    set title = {changed_component}=\{{$&temp}}
    set path = data/{{$title}}.txt
    *set path = data/{{$&counter}}.txt

    alter {changed_component} temp
    tran {time_step} {time_stop} {time_start} uic
    *set curplottitle = $title
    *plot {dependent_component}
    let temp = temp + step

    wrdata $path {dependent_component}

    *set gnuplot_terminal = png/quit
    *gnuplot $path {dependent_component}
    *+ title $title
    *+ xlabel "time [s]" ylabel "{dependent_component} [V]"
end

quit
.endc
"""

def add_controls(netlist: Path, controls: str, temp_name: str = "temp.cir") -> Path:
    """create temporary netlist, add controls, return path to temp file"""
    temp_path = netlist.parent/temp_name
    shutil.copyfile(netlist, temp_path)
    with open(temp_path, "a") as f:
        f.write(controls)
    return temp_path

def run_ngspice(netlist: Path) -> None:
    os.system(f"ngspice {netlist}")

def clean_signal(df, points_dropped=200):
    """remove startup and y-offset"""
    arr = df.to_numpy()
    no_startup = arr[points_dropped:]
    no_offset = no_startup - min(no_startup)
    return no_offset

def sweep(visual=True):
    # execute ngspice simulation
    controls = CONTROL_TEMPLATE.format(**CONTROL_PARAMETERS)
    tmp_path = add_controls(CIRCUIT_PATH, controls)
    run_ngspice(tmp_path)

    df = pd.DataFrame()
    # iterate results files
    for f in sorted(DATA_PATH.glob("*.txt")):
        # load single data file and add to df
        df_tmp = pd.read_csv(f, sep="[ ]+", engine="python") # match any number of spaces
        if "time" not in df.columns: df["time"] = df_tmp["time"]
        col_name = f.name[:-4]
        df[col_name] = df_tmp["v(/A)"]

        # apply fourier transform to signal
        signal = clean_signal(df[col_name])  
        spectrum = np.fft.fft(signal)
        abs_spec = abs(spectrum)
        freq = np.fft.fftfreq(len(abs_spec), d=5e-9)

         # compute fundamental frequency
        nlargest = pd.Series(abs_spec).nlargest(2)
        nlargest_arg = nlargest.index.values.tolist()
        f0 = abs(freq[nlargest_arg[1]])
        print(f"fundamental frequency: {f0} Hz")

        if visual:
            # time-domain plot
            sns.lineplot(data=df_tmp, x="time", y="v(/A)")
            
            fig, ax = plt.subplots()
            plt.plot(freq, abs_spec)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel("log-frequency [Hz]")
            ax.set_ylabel('log-amplitude')

            plt.show()
    
    return df

def fundamental_freq(s: pd.Series) -> float:
    """compute fundamental frequency of a vector"""
    # apply fourier transform to signal
    signal = clean_signal(s)
    spectrum = np.fft.fft(signal)
    abs_spec = abs(spectrum)
    freq = np.fft.fftfreq(len(abs_spec), d=5e-9)

    # compute fundamental frequency
    nlargest = pd.Series(abs_spec).nlargest(2)
    nlargest_arg = nlargest.index.values.tolist()
    return abs(freq[nlargest_arg[1]])

def plot_frequency_dependency(df):
    """dependency between independent variable and oscillator frequency"""
    df = df.drop(["time"], axis=1) # time not needed
    
    # define x
    # drop last character containing quantifier
    x_start = int(CONTROL_PARAMETERS["start"][:-1])
    x_stop  = int(CONTROL_PARAMETERS["stop"][:-1])
    x_step  = int(CONTROL_PARAMETERS["step"][:-1])
    iv_li = range(x_start, x_stop+x_step, x_step)

    # define y
    f0_li = []
    for col in df:
        f0 = fundamental_freq(df[col])
        f0_li.append(f0)
    
    # plot
    _, ax = plt.subplots()
    plt.plot(iv_li, f0_li)
    ax.set_xlabel("R1 [k Ohms]")
    ax.set_ylabel("frequency [Hz]")
    plt.show()

def main():
    df = sweep()
    plot_frequency_dependency(df)
    
main()

# TODO:
# control netlist parameters from within python
# out.txt should be renameable from within python
# implement aggregate, tendency and dependency plots
# random parameter search or hyperband search