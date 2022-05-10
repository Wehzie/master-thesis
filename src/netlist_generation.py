import params

import os
from pathlib import Path

FILEPATH = Path(".")
DEBUG = True

class SpiceNetlist():

    def __init__(self):
        self.connections = []
        self.state_nodes = []
    
    def generate_netlist(self, G, netlist_name="netlist"):
        for edge in G.edges:
            node1 = edge[0]
            node2 = edge[1]
            gnd_node = 1 # TODO
            voltage_input_node = list(G.nodes)[0] 
            
            if DEBUG: print(node1, node2)
            # possibly this only makes sense in a grid
            n1 = f"n{node1[0]}{node1[1]}"
            n2 = f"n{node2[0]}{node2[1]}"
            if DEBUG: print(n1, n2)

            if node1 == voltage_input_node:
                n1 = "vin"
                
            if node1 == gnd_node:
                n1 = "gnd"
                
            if node2 == gnd_node:
                n2 = "gnd"
            
            self.state_nodes.append(f"L({node1[0]};{node1[1]})({node2[0]};{node2[1]})")
            self.connections.append((n1,n2))
        
        self._write_netlist_file(self.connections, netlist_name)

    def run_ngspice(self, netlist_name="netlist"):
        os.system(f"ngspice {FILEPATH}/{netlist_name}.cir")

    def _get_gnd_node(self, graph, N):
        NotImplemented
        # the old function seems to be designed around square matrices

    def _write_netlist_file(self, connections, circ_name="name", model="hartley", netlist_name="netlist"):
        states = ""
        f = open(f"{FILEPATH}/{netlist_name}.cir", "w+")   
        f.write("* {circ_name}\n")
        f.write(f".include ../oscillators/{model}.sub\n")
        vin = "VIN-not-implemented"
        f.write(f"V1 vin gnd {vin}\n")
        
        for idx, c in enumerate(connections):
            model_params = f""
            
            f.write(f"Xosc{idx} {c[0]} {c[1]} l{idx} {model} {model_params}\n")
            states += f"l{idx} "

        f.write(f".tran {params.tstep} {params.tstop} {params.tstart} uic\n")
        f.write(".control\n")
        f.write("run\n")
        # f.write("option numdgt=7\n")
        f.write("set wr_vecnames\n")
        f.write("set wr_singlescale\n")
        f.write(f"wrdata {FILEPATH}/{netlist_name}_states.csv " + states + " \n")
        f.write(f"wrdata {FILEPATH}/{netlist_name}_iv.csv i(v1) vin\n")
        # f.write("plot -i(v1) vs vin\n")
        f.write("quit\n")
        f.write(".endc\n")
        f.write(".end\n")
        f.close()

    def _get_vin_type(self):
        NotImplemented
        # dc transient power
