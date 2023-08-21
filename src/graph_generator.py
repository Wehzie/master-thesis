"""
This module creates trees and graphs.

The structures are mapped to SPICE netlists to simulate circuits.
"""

from netlist_generator import build_tree_netlist
from params_spice import bird_params

from pathlib import Path
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

PARAM = bird_params["magpie"]


def connected_components(G: nx.Graph) -> nx.Graph:
    """extract subgraphs ordered from largest to smallest."""
    subgraphs = [G.subgraph(c) for c in nx.connected_components(G) if len(c) > 1]
    sorted_subgraphs = sorted(subgraphs, key=len)
    return sorted_subgraphs[::-1]


def watts_strogatz_graph() -> nx.Graph:
    """build a simple Watts Strogatz graph."""
    G = nx.Graph()
    G = nx.connected_watts_strogatz_graph(n=100, k=5, p=0.4, tries=100, seed=2)
    return G


def simple_graph() -> nx.Graph:
    """define a simple weighted graph."""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 1 / 6), (2, 0, 1 / 3), (2, 3, 1 / 2)])
    # TODO: a dict can be used to set individual attributes
    oscillator_model = "hartley"
    nx.set_edge_attributes(G, oscillator_model, "model")
    return G


def main():
    G = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True)
    plt.show()
    build_tree_netlist(G, Path("tree_netlist.cir"), PARAM)


main()
