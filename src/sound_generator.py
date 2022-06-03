import os
from params import bird_params

from pathlib import Path

import numpy as np

DATA_PATH = Path("circuit_lib")
PARAM = bird_params["magpie"]

POWER = """
.include ../circuit_lib/VO2_Sto_rand.cir
V1 /A 0 dc {v}
"""

# vo2 only
OSC_TEMPLATE = """
XU{i} /A /osc{i} control{i} VO2_Sto
C{i} /osc{i} 0 {c}
R{i} /osc{i} /sum {r}
"""

POST_SUM = """
R_last /sum 0 {r_last}
"""

CONTROL_TEMPLATE = """* Control commands
.control
tran {time_step} {time_stop} {time_start} uic

set wr_vecnames * to print header
set wr_singlescale * to not print scale vectors

set curplottitle = test_title
plot {dependent_component}
*******temp-open
plot v("/osc1")
*******temp-close

wrdata data/test_data.txt {dependent_component}

*set gnuplot_terminal = png/quit
*gnuplot $path {dependent_component}
*+ title $title
*+ xlabel "time [s]" ylabel "{dependent_component} [V]"

*quit
.endc
"""

def build_netlist(path: Path, temp_name: str = "temp.cir") -> Path:
    """create temporary netlist, return path to temp file"""
    temp_path = path / temp_name

    netlist = POWER.format(v=PARAM["v_in"])
    for i in range(1, 1+PARAM["num_osc"]):
        # TODO: generalize for >0 and <0 values
        # so over randint vs uniform
        r = np.random.randint(PARAM["r_min"], 1+PARAM["r_max"])
        c = np.random.uniform(PARAM["c_min"], PARAM["c_max"])        
        r_out = np.random.randint(PARAM["r_out_min"], 1+PARAM["r_out_max"])

        netlist += OSC_TEMPLATE.format(i=i, r=r, c=c, r_out=r_out)

    netlist += POST_SUM.format(r_last=PARAM["r_last"])
    netlist += CONTROL_TEMPLATE.format(
        time_step=PARAM["time_step"],
        time_stop=PARAM["time_stop"],
        time_start=PARAM["time_start"],
        dependent_component=PARAM["dependent_component"])

    with open(temp_path, "w") as f:
        f.write(netlist)
    return temp_path

def run_ngspice(netlist: Path) -> None:
    os.system(f"ngspice {netlist}")

tmp_path = build_netlist(DATA_PATH)
run_ngspice(tmp_path)
