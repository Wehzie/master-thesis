from netlist_generation import SpiceNetlist
import networkx as nx
import matplotlib.pyplot as plt

def connected_components(G: nx.Graph) -> nx.Graph:
    """extract subgraphs ordered from largest to smallest."""
    subgraphs = [G.subgraph(c) for c in nx.connected_components(G) if len(c) > 1]
    sorted_subgraphs = sorted(subgraphs, key=len)
    return sorted_subgraphs[::-1]

def watts_strogatz_graph() -> nx.Graph:
    """build a simple Watts Strogatz graph."""
    G = nx.Graph()
    G = nx.connected_watts_strogatz_graph(n=100, k=5, p=.4, tries=100, seed=2)
    return G

def simple_graph() -> nx.Graph:
    """define a simple weighted graph."""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 1/6), (2, 0, 1/3), (2, 3, 1/2)])
    # TODO: a dict can be used to set individual attributes
    oscillator_model = "hartley"
    nx.set_edge_attributes(G, oscillator_model, "model")
    return G

def main():
    G = simple_graph()
    nx.draw(G, with_labels=True)
    plt.show()
    exit()
    for edge in G.edges:
        print(G.get_edge_data(edge[0], edge[1]))


    netlist = SpiceNetlist()
    netlist.generate_netlist(G)
    netlist.run_ngspice()

main()