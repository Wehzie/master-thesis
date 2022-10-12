"""
This is an example of the application of DeepESN model for multivariate time-series prediction task 
on Piano-midi.de (see http://www-etud.iro.umontreal.ca/~boulanni/icml2012) dataset.
The dataset is a polyphonic music task characterized by 88-dimensional sequences representing musical compositions.
Starting from played notes at time t, the aim is to predict the played notes at time t+1.

Reference paper for DeepESN model:
C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical Experimental Analysis", 
Neurocomputing, 2017, vol. 268, pp. 87-99

In this Example we consider the hyper-parameters designed in the following paper:
C. Gallicchio, A. Micheli, L. Pedrelli, "Design of deep echo state networks",
Neural Networks, 2018, vol. 108, pp. 33-47
"""

from pathlib import Path
import time

import numpy as np
from DeepESN import DeepESN
from utils import computeMusicAccuracy, config_pianomidi, load_pianomidi, select_indexes
class Struct(object): pass

# sistemare indici per IP in config_pianomidi, mettere da un'altra parte
# translation: set indexes by IP in config_plans, put elsewhere
# this probably means the confusion of controlling how intrinsic plasticity is used

# sistema selezione indici con transiente messi all'interno della rete
# index selection system with transient placed inside the network
# this probably means that the transient component in main is now redundant?

def main():
    # measure time for this code section
    t0 = time.perf_counter()
    
    # fix a seed for the reproducibility of results
    np.random.seed(7)
   
    # dataset path 
    path = Path("data")

    (dataset,
    Nu, # dimension of a single data point
        # for example 88 for piano-midi.de
        # where 88 corresponds the number of keys on a piano
    error_function,
    optimization_problem,
    TR_indexes, # train set indices
    VL_indexes, # validation set indices
    TS_indexes # test set indices
    ) = load_pianomidi(path, computeMusicAccuracy)

    # load configuration for pianomidi task
    configs = config_pianomidi(list(TR_indexes) + list(VL_indexes))   
    
    # Be careful with memory usage
    # TODO: What does careful with memory usage mean?
    # What are the limits?
    Nr = 50 # number of recurrent units
    Nl = 1 # number of recurrent layers
    reg = 10.0**-2 # probably refers to lambda_r, readout regularization
                    # BUG: however we also set regularization in the config file
    transient = 5
    
    # initialize the ESN
    deepESN = DeepESN(Nu, Nr, Nl, configs, verbose=1)
    # 
    states = deepESN.computeState(dataset.inputs, deepESN.IPconf.DeepIP, verbose=1)
    
    train_states = select_indexes(states, list(TR_indexes) + list(VL_indexes), transient)
    train_targets = select_indexes(dataset.targets, list(TR_indexes) + list(VL_indexes), transient)
    test_states = select_indexes(states, TS_indexes)
    test_targets = select_indexes(dataset.targets, TS_indexes)
    
    deepESN.trainReadout(train_states, train_targets, reg)
    
    train_outputs = deepESN.computeOutput(train_states)
    train_error = error_function(train_outputs, train_targets)
    print(f"Training ACC: {np.mean(train_error):0.5f} \n")
    
    test_outputs = deepESN.computeOutput(test_states)
    test_error = error_function(test_outputs, test_targets)
    print(f"Test ACC: {np.mean(test_error):0.5f} \n")

    # duration is difference between end time and start time
    t1 = time.perf_counter()
    print(f"Time elapsed: {t1-t0:0.5f} s")
 
 
if __name__ == "__main__":
    main()
    
