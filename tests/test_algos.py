
# add code in src to path
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from main import *

def test_lazy():
    rand_args, meta_target = init_main()
    #algo = algo_las_vegas.LasVegas(rand_args=rand_args, target=meta_target[1])
    for algo in params.algo_list:
        alg_instance = algo.__init__(rand_args=rand_args, target=meta_target[1])
        alg_instance.search()
    
