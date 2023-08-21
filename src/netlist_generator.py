"""This module applies meta-programming to generate SPICE netlists."""

import subprocess
from pathlib import Path
from typing import Callable
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import gen_signal_args_types as party
import const

# see https://github.com/PySpice-org/PySpice

INCLUDE = """
.include ./circuit_lib/VO2_Sto_rand.cir
"""
POWER = """
V1 bridge 0 dc {v}
"""
RESISTOR_0 = """
R{i} 0 wire{i} {r}
"""
RESISTOR = """
R{i} wire{i} wire{j} {r}
"""
POST_SUM = """
R_last sum 0 {r_last}
"""
TREE_POWER = """
V{i} bridge{i} 0 dc {v}
"""
SUM_OSC_TEMPLATE = """
XU{i} bridge osc{i} control{i} VO2_Sto
C{i} osc{i} 0 {c}
R{i} osc{i} sum {r}
R{i}control control{i} 0 {r_control}
"""
LEAF_OSC_TEMPLATE = """
XU{i} bridge{i} osc{i} control{i} VO2_Sto
C{i} osc{i} 0 {c}
R{i} osc{i} wire{j} {r}
R{i}control control{i} 0 {r_control}
"""
TREE_OSC_TEMPLATE = """
XU{i} out{j} osc{i} control{i} VO2_Sto
C{i} osc{i} 0 {c}
R{i} osc{i} out{i} {r}
R{i}control control{i} 0 {r_control}
"""
SINGLE_OSCILLATOR_CIRCUIT = """
V1 power 0 dc 14
XU1 power osc control VO2_Sto
C1 osc 0 {c}
R1 osc 0 {r}
R1control control 0 {r_control}
"""
CONTROL_TEMPLATE = """* Control commands
.control
tran {time_step} {time_stop} {time_start} uic
set wr_vecnames * to print header
set wr_singlescale * to not print scale vectors
wrdata {file_path} {dependent_component}
quit
.endc
"""


def build_control(path: Path, PARAM: dict) -> str:
    return CONTROL_TEMPLATE.format(
        time_step=PARAM["time_step"],
        time_stop=PARAM["time_stop"],
        time_start=PARAM["time_start"],
        dependent_component=PARAM["dependent_component"],
        file_path=Path(str(path) + ".dat"),
    )


def build_tree_netlist(path: Path, PARAM: dict, visual: bool = False) -> dict:
    """
    Write a netlist to file where oscillators signals are summed into one node.

    Return dictionary of deterministic parameters.
    """
    assert PARAM["c_max"] <= 1, "Randomly generating capacitors with >1 Farad is not implemented!"

    # build graph
    G = nx.balanced_tree(PARAM["branching"], PARAM["height"], create_using=nx.DiGraph())
    if visual:
        nx.draw(G, with_labels=True)
        plt.show()

    det_param = PARAM  # probabilistic to deterministic parameters

    netlist = []  # netlist as list of lines
    # add includes and power supply to netlist
    netlist.append(INCLUDE)

    # add root to netlist
    root = list(nx.topological_sort(G))[0]
    netlist.append(RESISTOR_0.format(i=root, r=PARAM["r_tree"]))

    # add root's children to netlist
    def add_children(root, parent):
        print(root, parent, "root, parent")
        edges = nx.edges(G, [root])
        children = [edge[1] for edge in edges]

        # base case: no children
        if len(edges) == 0:
            r = np.random.randint(PARAM["r_min"], 1 + PARAM["r_max"])
            c = np.random.uniform(PARAM["c_min"], PARAM["c_max"])
            r_control = PARAM["r_control"]
            det_param[f"r{root}"] = r
            det_param[f"c{root}"] = c
            netlist.append(
                LEAF_OSC_TEMPLATE.format(i=root, j=parent, r=r, c=c, r_control=r_control)
            )
            netlist.append(TREE_POWER.format(i=root, v=PARAM["v_in"]))
            return

        # recursive case: node has children

        det_param[f"r{root}"] = PARAM["r_tree"]
        netlist.append(RESISTOR.format(i=root, j=parent, r=PARAM["r_tree"]))
        for child in children:
            add_children(root=child, parent=root)

    # we have already added the root node as a special case
    # so build tree starting with its children
    for edge in nx.edges(G, [root]):
        child = edge[1]
        add_children(child, root)

    # add control statements
    netlist.append(build_control(path, PARAM))

    with open(path, "w") as f:
        f.write("\n".join(netlist))

    return det_param


def build_sum_netlist(path: Path, PARAM: dict) -> dict:
    """
    Write netlist to file where oscillators are summed into one node.

    Return dictionary of deterministic parameters.
    """
    assert PARAM["c_max"] <= 1, "Randomly generating capacitors with >1 Farad is not implemented!"

    # from probabilistic to deterministic parameters
    det_param = PARAM

    netlist = INCLUDE
    netlist += POWER.format(v=PARAM["v_in"])
    for i in range(1, 1 + PARAM["n_osc"]):
        # TODO: generalize for >0 and <0 values
        # so over randint vs uniform
        r = np.random.randint(PARAM["r_min"], 1 + PARAM["r_max"])
        c = np.random.uniform(PARAM["c_min"], PARAM["c_max"])
        r_control = PARAM["r_control"]
        netlist += SUM_OSC_TEMPLATE.format(i=i, r=r, c=c, r_control=r_control)
        det_param[f"r{i}"] = r
        det_param[f"c{i}"] = c

    netlist += POST_SUM.format(r_last=PARAM["r_last"])
    netlist += build_control(path, PARAM)

    with open(path, "w") as f:
        f.write(netlist)

    return det_param


def build_single_netlist(
    path: Path, det_args: party.SpiceSingleDetArgs, debug: bool = False
) -> None:
    """Write netlist to file with a single oscillator."""
    netlist = INCLUDE
    netlist += POWER.format(v=det_args.v_in)
    netlist += SUM_OSC_TEMPLATE.format(
        i=1, r=det_args.r, c=det_args.c, r_control=det_args.r_control
    )
    netlist += POST_SUM.format(r_last=det_args.r_last)
    netlist += build_control(path, det_args.__dict__)

    if debug:
        print(f"writing netlist to path {path}")
    with open(path, "w") as f:
        f.write(netlist)


def select_netlist_generator(builder: str) -> Callable:
    """
    select a netlist generator and return appropriate function

    selection: "tree", "sum"
    """
    if builder == "tree":
        return build_tree_netlist
    if builder == "sum":
        return build_sum_netlist
    raise ValueError()


def run_ngspice(netlist: Path) -> None:
    """start an ngspice simulation from python"""
    try:
        subprocess.run(["ngspice", netlist], timeout=const.SPICE_TIMEOUT, stdout=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print(f"ngspice timed out after {const.SPICE_TIMEOUT} seconds")
