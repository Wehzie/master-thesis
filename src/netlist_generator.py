from pathlib import Path
import networkx as nx
import numpy as np

INCLUDE = """
.include ./circuit_lib/VO2_Sto_rand.cir
"""
POWER = """
V1 /bridge 0 dc {v}
"""
RESISTOR_0 = """
R{i} 0 right{i} {r}
"""
RESISTOR = """
R{i} left{i} right{j} {r}
"""
POST_SUM = """
R_last /sum 0 {r_last}
"""
TREE_POWER = """
V{i} /bridge{i} 0 dc {v}
"""
SUM_OSC_TEMPLATE = """
XU{i} /bridge /osc{i} control{i} VO2_Sto
C{i} /osc{i} 0 {c}
R{i} /osc{i} /sum {r}
R{i}control control{i} 0 {r_control}
"""
LEAF_OSC_TEMPLATE = """
XU{i} /bridge{i} /osc{i} control{i} VO2_Sto
C{i} /osc{i} 0 {c}
R{i} /osc{i} /right{j} {r}
R{i}control control{i} 0 {r_control}
"""
TREE_OSC_TEMPLATE = """
XU{i} /out{j} /osc{i} control{i} VO2_Sto
C{i} /osc{i} 0 {c}
R{i} /osc{i} /out{i} {r}
R{i}control control{i} 0 {r_control}
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

# TODO: reverse tree such that leafs are powered and root is integrator
def build_tree_netlist(G: nx.DiGraph, path: Path, PARAM: dict) -> dict:
    """
    Write a netlist to file where oscillators signals are summed in a.
    
    Return dictionary of deterministic parameters.
    """
    assert PARAM["c_max"] <= 1, "Randomly generating capacitors with >1 Farad is not implemented!"
    
    det_param = PARAM # probabilistic to deterministic parameters

    netlist = [] # netlist as list of lines
    # add includes and power supply to netlist
    netlist.append(INCLUDE)

    # add root to netlist
    root = list(nx.topological_sort(G))[0]
    netlist.append(RESISTOR_0.format(i=root, r=PARAM["r_tree"]))

    # add root's children to netlist
    def add_children(root):
        edges = nx.edges(G, [root])
        # base case: no children
        if len(edges) == 0:
            parent = list(G.in_edges(root))[0][0]
            r = np.random.randint(PARAM["r_min"], 1+PARAM["r_max"])
            c = np.random.uniform(PARAM["c_min"], PARAM["c_max"])
            r_control = PARAM["r_control"]
            det_param[f"r{root}"] = r
            det_param[f"c{root}"] = c
            netlist.append(LEAF_OSC_TEMPLATE.format(i=root, j=parent, r=r, c=c, r_control=r_control))
            netlist.append(TREE_POWER.format(i=root, v=PARAM["v_in"]))
            return
        
        # recursive case: node has children
        def add_resistor_node(child):
            det_param[f"r{child}"] = PARAM["r_tree"]
            netlist.append(RESISTOR.format(i=child, j=root, r=PARAM["r_tree"]))
            add_children(child)
        
        children = [edge[1] for edge in edges] # get children
        map(add_resistor_node, children)
      
    # we have already added the root node as a special case
    # so build tree starting with its children
    for edge in nx.edges(G, [root]):
        child = edge[1]
        add_children(child)

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
    for i in range(1, 1+PARAM["num_osc"]):
        # TODO: generalize for >0 and <0 values
        # so over randint vs uniform
        r = np.random.randint(PARAM["r_min"], 1+PARAM["r_max"])
        c = np.random.uniform(PARAM["c_min"], PARAM["c_max"])
        r_control = PARAM["r_control"]
        netlist += SUM_OSC_TEMPLATE.format(i=i, r=r, c=c, r_control=r_control)
        det_param[f"r{i}"] = r
        det_param[f"c{i}"] = c

    netlist += POST_SUM.format(r_last=PARAM["r_last"])
    netlist += CONTROL_TEMPLATE.format(
        time_step=PARAM["time_step"],
        time_stop=PARAM["time_stop"],
        time_start=PARAM["time_start"],
        dependent_component=PARAM["dependent_component"],
        file_path=Path(str(path) + ".dat"))

    with open(path, "w") as f:
        f.write(netlist)

    return det_param
