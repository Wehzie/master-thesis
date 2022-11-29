from typing import Callable, List, Tuple, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import gen_signal
import data_analysis
import data_io
import data_preprocessor
import param_types as party
import params_test_spice as params
import netlist_generator


class SpiceSignalGenerator(gen_signal.SignalGenerator):
    """spice netlists generate single signals and python sums them up
    this approach scales better than pure spice signal generation
    but worse than pure python signal generation
    
    when a sample is drawn SPICE is called to generate the signals and Pytho is used for post-processing    
    the underlying time series are SPICE generated
    weights are added in Python, instead of using an operational amplifier for gain in SPICE
    phase is added in Python, instead of using an operational amplifier integrator in SPICE
    offset is added in Python, instead of using an operational amplifier in SPICE
    """

    # ability to generate single oscillator in ngspice
    # ability to generate k oscillators and sum them in ngspice
    @staticmethod
    def draw_single_oscillator(det_args: party.SpiceSingleDetArgs, work_dir: Path = Path("data")) -> np.ndarray:
        """generate a time series from a single spice oscillator"""
        # generate path to save signal and circuit
        experiment_path = data_io.find_dir_name(work_dir)
        tmp_path = experiment_path / "netlist.cir"
        # build netlist with a single oscillator
        netlist_generator.build_single_netlist(tmp_path, det_args)
        # run simulation on ngspice
        netlist_generator.run_ngspice(tmp_path)
        # load the simulation data written by ngspice into Python
        df = data_io.load_sim_data(Path(str(tmp_path) + ".dat"))
        # convert from dataframe to numpy
        arr = df.iloc[:,1].to_numpy()
        # add phase shift to the netlist
        freq = data_analysis.get_freq_from_fft(arr, det_args.time_step)
        pred = data_preprocessor.add_phase_to_oscillator(arr, det_args.phase, 1/freq, det_args.time_step)

        return pred

    @staticmethod
    def draw_params_random(args: party.SpiceSumRandArgs) -> party.SpiceSingleDetArgs:
        """draw randomly from parameter pool"""
        n_osc = args.n_osc
        v_in = args.v_in
        r = args.r_dist.draw()
        r_last = args.r_last
        r_control = args.r_control
        c = args.c_dist.draw()
        time_step = args.time_step
        time_stop = args.time_stop
        time_start = args.time_start
        dependent_component = args.dependent_component
        phase = args.phase_dist.draw()

        return party.SpiceSingleDetArgs(n_osc, v_in, r, r_last, r_control, c, time_step, time_stop, time_start, dependent_component, phase)

    @staticmethod
    def simulation_successful(single_signal: np.ndarray, samples: int) -> bool:
        """check if SPICE run was successful
        
        returns True if the simulation was successful, False otherwise"""
        return False if len(single_signal) < samples else True

    @staticmethod
    def draw_n_oscillators(rand_args: party.SpiceSumRandArgs, store_det_args: bool = False) -> Tuple[np.ndarray, List[party.SpiceSingleDetArgs]]:
        det_arg_li = list()
        # allocate more memory than necessary
        # exact number of samples is non-deterministic
        # the cost of np.zeros is higher than np.empty, but this seems safer
        num_samples_float = (rand_args.time_stop - rand_args.time_start) / rand_args.time_step
        num_samples = np.ceil(num_samples_float).astype(int)
        signal_matrix = np.zeros((rand_args.n_osc, num_samples))

        i = 0
        ifinity_breaker = 0 # prevent infinite loop
        # non-deterministic runtime due to possible simulation failure
        while i < rand_args.n_osc:
            det_params = SpiceSignalGenerator.draw_params_random(rand_args)
            single_signal = SpiceSignalGenerator.draw_single_oscillator(det_params)
            success = SpiceSignalGenerator.simulation_successful(single_signal, num_samples)
            if success:
                # exact number of samples is non-deterministic
                # around 2% cutoff
                single_signal_cut = single_signal[0:num_samples] 
                signal_matrix[i] = single_signal_cut
                i += 1
                if store_det_args: det_arg_li.append(det_params)
            ifinity_breaker += 1
            if ifinity_breaker > 10:
                raise Exception("Infinite loop detected")
        return signal_matrix, det_arg_li

    
def main():
    sig_gen = SpiceSignalGenerator()
    
    if False:
        # test single oscillator generation
        det_args = params.spice_single_det_args
        single_oscillator = sig_gen.draw_single_oscillator(det_args)
        data_analysis.plot_signal(single_oscillator, show=True)

    if False:
        # test drawing single oscillators from SPICE and summing them up in Python
        rand_args = params.spice_rand_args_uniform
        signal_matrix, det_args = sig_gen.draw_n_oscillators(rand_args)
        sig_sum = sum(signal_matrix)
        data_analysis.plot_signal(sig_sum, show=True)
    
    if True:
        # test drawing a sample
        rand_args = params.spice_rand_args_uniform
        sample = sig_gen.draw_sample(rand_args)
        data_analysis.plot_signal(sample.weighted_sum, show=True)

if __name__ == "__main__":
    main()