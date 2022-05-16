
import shutil
from params import param_sweep_schedule

import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CIRCUIT_PATH = Path("circuit_lib/single_oscillator_RC_version_davide.cir")
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
let counter = 1
* loop
while temp le stop
    set title = {changed_component}=\{{$&temp}}
    *set path = data/{{$title}}
    set path = data/{{$&counter}}.txt

    alter {changed_component} temp
    tran {time_step} {time_stop} {time_start} uic
    *set curplottitle = $title
    *plot {dependent_component}
    let r_act = r_act + delta_r

    wrdata $path {dependent_component} {changed_component}

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

controls = CONTROL_TEMPLATE.format(**CONTROL_PARAMETERS)
tmp_path = add_controls(CIRCUIT_PATH, controls)
run_ngspice(tmp_path)
for f in DATA_PATH.glob("*.txt"):
    df = pd.read_csv(f, sep="[ ]+", engine="python") # match any number of spaces
    sns.lineplot(data=df, x="time", y="v(/A)")
    plt.show()

# TODO:
# control netlist parameters from within python
# out.txt should be renameable from within python
# implement aggregate, tendency and dependency plots
# random parameter search or hyperband search