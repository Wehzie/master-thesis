"""an oscillator classifier is a set of independently trained oscillator ensembles

given a forest of ensembles trained on various target functions,
the ensembles are compared to an unknown target function by RMSE,
this generates a probability distribution over the ensembles
"""
import const

if const.TEST_PARAMS:
    print("Import test parameters.")
    import params_hybrid_test as hybrid_parameters
else:
    print("Import production parameters.")
    import params_hybrid as hybrid_parameters

import meta_target
import sample
import data_analysis
import algo_args_bundle
import gen_signal_spipy
import shared_params_target
import algo_las_vegas
import algo_args_type

import numpy as np

from typing import List


class Classifier:
    """A classifier from a forest of ensembles that classifies an unknown target function"""

    def __init__(
        self,
        train_targets: List[meta_target.MetaTarget],
        awa_list: List[algo_args_bundle.AlgoWithArgs],
        mp: bool = const.MULTIPROCESSING,
    ) -> None:
        self.mp = mp
        assert len(awa_list) == len(
            train_targets
        ), "number of trained ensembles doesn't match the number of training targets"
        self.awa_list = awa_list
        self.train_targets = train_targets
        self.ensembles: List[sample.Sample] = []

    def train(self):
        """train the classifier by training each ensemble in the forest on a different function"""
        if self.mp:
            NotImplemented
        else:
            for awa in self.awa_list:
                search_alg = awa.Algo(awa.algo_args)
                best_sample, _ = search_alg.search()
                self.ensembles.append(best_sample)

    def predict(self, target: np.ndarray, verbose: bool = False) -> int:
        """predict the target function by comparing it to the ensembles in the forest"""
        # note these RMSE's are not from training but for inference
        comp_rmse = data_analysis.compute_rmse
        rmse_list = [comp_rmse(ens.weighted_sum, target) for ens in self.ensembles]

        def norm(rmse_list):  # norm_list sums to 1
            s = sum(rmse_list)
            norm_list = [val / s for val in rmse_list]
            return norm_list

        rmse_distribution = norm(rmse_list)
        if verbose:
            print(f"rmse_distribution: {rmse_distribution}")
        return np.argmin(rmse_distribution)

    def map_label_to_string(self, argmin: int) -> str:
        """convert numerical label of a prediction to string output"""
        return self.train_targets[argmin].name


if __name__ == "__main__":
    """example usage of the classifier"""

    sig_gen = gen_signal_spipy.SpipySignalGenerator()
    generator_args = hybrid_parameters.spice_rand_args_uniform
    synth_freq = hybrid_parameters.SYNTH_FREQ

    sine = shared_params_target.select_target_by_string("sine", generator_args, synth_freq)
    triangle = shared_params_target.select_target_by_string("triangle", generator_args, synth_freq)
    train_targets = [sine, triangle]

    max_z_ops = 5000
    awa_list = []
    for t in train_targets:
        search_algo = algo_las_vegas.LasVegas
        algo_args = algo_args_type.AlgoArgs(sig_gen, generator_args, t, max_z_ops)
        awa = algo_args_bundle.AlgoWithArgs(search_algo, algo_args)
        awa_list.append(awa)

    cls = Classifier(train_targets, awa_list)
    cls.train()

    easy_sine = shared_params_target.select_target_by_string("sine", generator_args, synth_freq)
    pred = cls.predict(easy_sine.signal, verbose=True)
    print(cls.map_label_to_string(pred))

    # change the frequency of the sine wave to make approximation more difficult
    difficult_sine = shared_params_target.select_target_by_string(
        "sine", generator_args, synth_freq + 1000
    )
    pred = cls.predict(difficult_sine.signal, verbose=True)
    print(cls.map_label_to_string(pred))
