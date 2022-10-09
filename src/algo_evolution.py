from scipy.optimize import differential_evolution, basinhopping
import numpy as np

from data_analysis import compute_rmse, plot_pred_target
from search_module import Sample
rng = np.random.default_rng()


def func1(weights, *args):
    """adapt weights only"""
    model, x, y = args
    weighted_sum = np.sum(model.T * weights, axis=1)
    rmse = compute_rmse(weighted_sum, y)
    return rmse

def func2(model, *args):
    """adapt entire matrix""" # this doesn't even make sense because a time-series can't be cut in half
    x, y, n_osc, samples = args
    model = model.reshape(n_osc, samples)
    unweighted_sum = np.sum(model, axis=0)
    rmse = compute_rmse(unweighted_sum, y)
    return rmse

# uses globals
def func3(x):
    """adapt weights for simulated annealing"""
    weighted_sum = np.sum(model.T * x, axis=1)
    rmse = compute_rmse(weighted_sum, y)
    return rmse

# uses globals
def func4(x):
    """adapt maxtrix for simulated annealing""" # this doesn't even make sense because a time-series can't be cut in half
    x = x.reshape(n_osc, samples) 
    unweighted_sum = np.sum(x, axis=0)
    rmse = compute_rmse(unweighted_sum, y)
    return rmse

if __name__ == '__main__':
    samples = 6
    n_osc = 10

    # target data
    x = [i for i in range(samples)]
    y = [np.sin(x) for x in x]

    # unweighted model
    model = rng.normal(-1, 1, (n_osc, samples))
    # weights will be learned
    weights = [(0, 1) for _ in range(n_osc)]

    def prog(*args, **kwargs):
        print(args)

    if False: # evolution on weights
        # pack model and data into args
        args = (model, x, y)
        result = differential_evolution(func1, weights, args=args)

        print(result.x, result.fun, result.nit, result.nfev)
        pred = Sample.predict(model, result.x, 0)
        rmse = compute_rmse(pred, y)
        print("RMSE: {rmse}")
        plot_pred_target(pred, y, show=True)

    if False: # evolution on full model
        # differential evolution can only handle 1 dimensional bounds
        args = x, y, n_osc, samples
        bound_1d = [(-10, 10) for _ in range(samples)] # single oscillator
        bounds = list()
        for _ in range(n_osc): # flatten n oscillators
            bounds += bound_1d        
        result = differential_evolution(func2, bounds, args=args, workers=4)

        print(result.x, result.fun, result.nit, result.nfev)
        model = result.x.reshape(n_osc, samples)
        pred = np.sum(model, axis=0)
        rmse = compute_rmse(pred, y)
        print("RMSE: {rmse}")
        plot_pred_target(pred, y, show=True)

    if False: # anneal on weights
        # pack model and data into args
        weights = np.ones(n_osc)
        minimizer_kwargs = {"method": "BFGS"}
        GLOB_MODEL = model
        result = basinhopping(func3, weights, minimizer_kwargs=minimizer_kwargs,
                   niter=200)

        print(result.x, result.fun, result.nit, result.nfev)
        pred = Sample.predict(model, result.x, 0)
        rmse = compute_rmse(pred, y)
        print("RMSE: {rmse}")
        plot_pred_target(pred, y, show=True)

    if False:
        model = model.flatten()
        minimizer_kwargs = {"method": "BFGS"}
        result = basinhopping(func4, model, minimizer_kwargs=minimizer_kwargs,
                   niter=200)

        print(result.x, result.fun, result.nit, result.nfev)
        model = result.x.reshape(n_osc, samples)
        pred = np.sum(model, axis=0)
        rmse = compute_rmse(pred, y)
        print("RMSE: {rmse}")
        plot_pred_target(pred, y, show=True)
    

# random search
    # monte carlo one-shot
    # las vegas
    # monte carlo exploit
    # simulated annealing
        # advantage: most easy to implement in hardware
    
# population search
    # genetic algo
        # problem: neuromorphic memory

# hebbian learning
    # boltzman learning, boltzman machine
    # hopfield 
    # kohonen SOM - not on sklearn or scipy
    # particle swarmp opt
    # stdp

# bayesian
    # not feasible at all - random variables and distributions are not intuitive to implement in materio
    # contra arg is that brain is argued to be bayesian
    # advantage is also that gradients aren't needed
    
    # of course a normal distribution can be esitmated by mean and variance
    # and uniform dist can be represented by lower and upper bound
    # however drawing from such a distribution in-materio that is truly random
    # and then changing the dist on-fly on such a device would be a necessary requirement
    # Schomaker: work is being done on this at cognigron, ask for citation

# gradient based
    # not feasible
    # attribution problem: by what amount does oscillator_i contribute to the error?
    # see this paper which manages to do it a bit

# not discussed
    # decision tree and random forest
