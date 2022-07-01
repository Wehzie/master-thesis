from pathlib import Path
import networkx as nx
from params import bird_params
import numpy as np

PARAM = bird_params["magpie"]

INCLUDE = """
.include ./circuit_lib/VO2_Sto_rand.cir
"""
POWER = """
V1 /bridge 0 dc {v}
"""
ROOT = """
XU{i} /bridge /osc{i} control{i} VO2_Sto
C{i} /osc{i} 0 {c}
R{i} /osc{i} /out{i} {r}
R{i}control control{i} 0 {r_control}
"""
OSC_TEMPLATE = """
XU{i} /out{j} /osc{i} control{i} VO2_Sto
C{i} /osc{i} 0 {c}
R{i} /osc{i} /out{i} {r}
R{i}control control{i} 0 {r_control}
"""

def generate_tree_netlist(G: nx.DiGraph, path: Path):
    det_param = PARAM
    # TODO: saving params

    nl_lines = [] # netlist as lines
    nl_lines.append(INCLUDE)
    nl_lines.append(POWER.format(v=PARAM["v_in"]))

    root = list(nx.topological_sort(G))[0]
    
    nl_lines.append(ROOT.format(i=root, r=99, c=99, r_control=99))

    def add_children(root):
        for edge in nx.edges(G, [root]): # add all children of root to netlist
            child = edge[1]
            nl_lines.append(OSC_TEMPLATE.format(i=child, j=root, r=99, c=99, r_control=99))
            # recursive case: children means appending
            add_children(child)
        # base case: no children means do nothing

    add_children(root)

    with open(path, "w") as f:
        f.write("\n".join(nl_lines))