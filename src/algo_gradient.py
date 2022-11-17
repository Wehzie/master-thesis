"""module implements gradient based optimization algorithms

in a neuromorphic context gradient based algorithms are difficult because obtaining gradient information is difficult.
this is in particular due to the attribution problem.
This is to ask by what amount does oscillator_i contribute to the error?
"""

import algo

class LinearRegression(algo.SearchAlgo):
    NotImplemented