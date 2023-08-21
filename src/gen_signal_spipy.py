"""
This module implements the Hybrid or SpiPy (SPICE+Python) signal generator class.

The oscillation of a single oscillator circuit is generated within SPICE.
The resulting signal is loaded into Python.
The processes is repeated to obtain an ensemble of signals.
Meta-programming is used to modify the SPICE netlist to generate different signals.
"""

import copy
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import gen_signal
import data_analysis
import data_io
import data_preprocessor
import gen_signal_args_types as party
import netlist_generator
import const

DF_CACHE = None


class SpipySignalGenerator(gen_signal.SignalGenerator):
    """
    spice netlists generate single signals and Python sums them up

    this approach scales better than pure spice signal generation
    but worse than pure Python signal generation

    when a sample is drawn SPICE is called to generate the signals and Python is used for post-processing
    the underlying time series are SPICE generated
    weights are added in Python, instead of using an operational amplifier for gain in SPICE
    phase is added in Python, instead of using an operational amplifier integrator in SPICE
    offset is added in Python, instead of using an operational amplifier in SPICE
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: make this a class variable
        global DF_CACHE
        if DF_CACHE is None:
            DF_CACHE = data_io.load_signal_cache()

    @staticmethod
    def get_tmp_path(work_dir: Path = const.WRITE_DIR) -> Path:
        """get the path to save the netlist and simulation data"""
        experiment_path = work_dir / "spice_single_oscillator"
        experiment_path.mkdir(parents=True, exist_ok=True)
        return experiment_path / "netlist.cir"

    @staticmethod
    def load_from_cache_and_extrapolate_signal(
        det_args: party.SpiceSingleDetArgs, df_cache: pd.DataFrame
    ) -> np.ndarray:
        """load a signal from cache and extrapolate it to a longer duration"""
        dist_between_rs = df_cache["r"].iloc[1] - df_cache["r"].iloc[0]
        decimal_places = (np.log10(dist_between_rs)).astype(int)
        rounded_r = int(round(det_args.r, -decimal_places))
        row = df_cache[df_cache["r"] == rounded_r]
        single_period_signal = row["signal"].values[0]
        sampling_rate = row["sampling_rate"].values[0]
        goal_duration = det_args.time_stop
        phase = det_args.phase
        full_signal = data_preprocessor.extrapolate_from_period(
            single_period_signal, sampling_rate, goal_duration, phase
        )
        offset_realigned = data_preprocessor.remove_offset_two_sided(full_signal)
        return offset_realigned

    @staticmethod
    def simulate_and_extrapolate_signal(
        det_args: party.SpiceSingleDetArgs, work_dir: Path, period_multiplier: float = 1
    ) -> np.ndarray:
        """simulate a signal for a short time and extrapolate it to a longer time"""
        original_duration = copy.deepcopy(det_args.time_stop)
        det_args.time_stop = (
            const.LONGEST_VO2_PERIOD * period_multiplier
        )  # inject duration needed to observe the first period for frequencies up to f(R=69k)
        netlist_generator.build_single_netlist(work_dir, det_args)
        netlist_generator.run_ngspice(work_dir)
        df = data_io.load_sim_data(Path(str(work_dir) + ".dat"))
        arr = df.iloc[:, 1].to_numpy()
        # FIXME: sampling_rate = np.round((1 / det_args.time_step) * det_args.down_sample_factor).astype(int)
        # maybe reason for down_sample_factor**2
        sampling_rate = 1 / det_args.time_step
        det_args.time_stop = original_duration
        full_signal = data_preprocessor.extrapolate_oscillation(
            arr, sampling_rate, original_duration, det_args.phase
        )
        return full_signal

    @staticmethod
    def simulate_single_period(
        det_args: party.SpiceSingleDetArgs, work_dir: Path, period_multiplier: float = 1
    ) -> np.ndarray:
        """simulate a single oscillator for a short time preparing the signal to be cached to non volatile memory"""
        det_args.time_stop = (
            const.LONGEST_VO2_PERIOD * 3
        )  # inject duration needed to observe the first period for frequencies up to f(R=69k)
        netlist_generator.build_single_netlist(work_dir, det_args)
        netlist_generator.run_ngspice(work_dir)
        df = data_io.load_sim_data(Path(str(work_dir) + ".dat"))
        arr = df.iloc[:, 1].to_numpy()
        sampling_rate = np.round(1 / det_args.time_step).astype(int)
        period_duration, period_signal = data_preprocessor.extract_single_period(arr, sampling_rate)
        return period_duration, sampling_rate, period_signal

    @staticmethod
    def fully_simulate_signal(det_args: party.SpiceSingleDetArgs, work_dir: Path) -> np.ndarray:
        """simulate a signal for the full extent of its duration in ngspice"""
        # build netlist with a single oscillator
        netlist_generator.build_single_netlist(work_dir, det_args)
        # run simulation on ngspice
        netlist_generator.run_ngspice(work_dir)
        # load the simulation data written by ngspice into Python
        df = data_io.load_sim_data(Path(str(work_dir) + ".dat"))
        # convert from dataframe to numpy
        arr = df.iloc[:, 1].to_numpy()

        # remove startup and offset
        clean = data_preprocessor.clean_spice_signal(arr, len(arr))
        # add phase shift to the netlist
        freq = data_analysis.get_freq_from_fft_v2(clean, det_args.time_step)
        pred = data_preprocessor.add_phase_to_oscillator(
            clean, det_args.phase, 1 / freq, det_args.time_step
        )
        return pred

    @staticmethod
    def draw_single_oscillator(
        det_args: party.SpiceSingleDetArgs, work_dir: Path = const.WRITE_DIR
    ) -> np.ndarray:
        """generate a time series from a single spice oscillator"""
        tmp_path = SpipySignalGenerator.get_tmp_path(work_dir)

        if det_args.generator_mode == party.SpipyGeneratorMode.CACHE:
            pred = SpipySignalGenerator.load_from_cache_and_extrapolate_signal(det_args, DF_CACHE)
        elif det_args.generator_mode == party.SpipyGeneratorMode.EXTRAPOLATE:
            patience_counter = 0
            while patience_counter < const.SPICE_PATIENCE:
                pred = SpipySignalGenerator.simulate_and_extrapolate_signal(
                    det_args, tmp_path, patience_counter + 1
                )
                if pred is not None:
                    break
                patience_counter += 1
            if patience_counter == const.SPICE_PATIENCE:
                raise Exception(f"SPICE simulation failed {const.SPICE_PATIENCE} times in a row")
        elif det_args.generator_mode == party.SpipyGeneratorMode.SPICE:
            pred = SpipySignalGenerator.fully_simulate_signal(det_args, tmp_path)
        else:
            raise ValueError(f"generator mode {det_args.generator_mode} not supported")

        if (
            det_args.generator_mode
            in [party.SpipyGeneratorMode.EXTRAPOLATE, party.SpipyGeneratorMode.SPICE]
            and det_args.down_sample_factor is not None
        ):
            return data_preprocessor.downsample_by_factor_typesafe(
                pred, det_args.down_sample_factor
            )

        return pred

    @staticmethod
    def draw_params_random(args: party.SpiceSumRandArgs) -> party.SpiceSingleDetArgs:
        """draw randomly from a set of random variables to define the parameters of a single oscillator"""
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
        generator_mode = args.generator_mode
        down_sample_factor = args.down_sample_factor

        return party.SpiceSingleDetArgs(
            n_osc,
            v_in,
            r,
            r_last,
            r_control,
            c,
            time_step,
            time_stop,
            time_start,
            dependent_component,
            phase,
            generator_mode,
            down_sample_factor,
        )

    @staticmethod
    def simulation_successful(single_signal: np.ndarray, samples: int) -> bool:
        """
        check if a SPICE run was successful

        returns True if the simulation was successful, False otherwise
        """
        return False if single_signal is None or len(single_signal) < samples else True

    @staticmethod
    def draw_n_oscillators(
        rand_args: party.SpiceSumRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, List[party.SpiceSingleDetArgs]]:
        """
        draw a matrix of n oscillators.

        handles SPICE simulation failures.
        """
        det_arg_li = list()
        # allocate more memory than necessary
        # exact number of samples is non-deterministic
        # the cost of np.zeros is higher than np.empty, but this seems safer
        num_samples = rand_args.estimate_number_of_samples()
        signal_matrix = np.zeros((rand_args.n_osc, num_samples))

        i = 0
        infinity_breaker = 0  # prevent infinite loop
        # non-deterministic runtime due to possible simulation failure
        while i < rand_args.n_osc:
            if rand_args.n_osc > 1 and rand_args.generator_mode != party.SpipyGeneratorMode.CACHE:
                print(f"added {i} out of {rand_args.n_osc} oscillators")
            det_params = SpipySignalGenerator.draw_params_random(rand_args)
            single_signal = SpipySignalGenerator.draw_single_oscillator(det_params)
            success = SpipySignalGenerator.simulation_successful(single_signal, num_samples)
            if success:
                # exact number of samples is non-deterministic
                # around 2% cutoff
                infinity_breaker = 0
                single_signal_cut = single_signal[0:num_samples]
                signal_matrix[i] = single_signal_cut
                i += 1
                if store_det_args:
                    det_arg_li.append(det_params)
            else:
                infinity_breaker += 1
            if infinity_breaker > const.SPICE_PATIENCE:
                raise Exception("Infinite loop detected")
        return signal_matrix, det_arg_li


def main():
    import params_hybrid as params

    sig_gen = SpipySignalGenerator()

    if True:
        # test single oscillator generation
        det_args = params.spice_single_det_args
        det_args.time_stop = 2e-5
        single_oscillator = sig_gen.draw_single_oscillator(det_args)
        x_time = np.linspace(0, det_args.time_stop, len(single_oscillator), endpoint=False)
        sampling_rate = np.around(1 / det_args.time_step).astype(int)
        data_analysis.plot_signal(single_oscillator, x_time, sampling_rate, show=True)

    # test drawing single oscillators from SPICE and summing them up in Python
    if True:
        print("test drawing single oscillators from SPICE and summing them up in Python")
        rand_args = params.spice_rand_args_uniform
        rand_args.n_osc = 25
        rand_args.time_stop = 1e-5
        signal_matrix, det_args = sig_gen.draw_n_oscillators(rand_args)
        x_time = np.linspace(0, rand_args.time_stop, signal_matrix.shape[1], endpoint=False)
        sig_sum = sum(signal_matrix)
        data_analysis.plot_signal(sig_sum, x_time)
        data_analysis.plot_individual_oscillators(
            signal_matrix, x_time, show=True, save_path=Path("data/hybrid_oscillators.png")
        )

    # test drawing a sample
    if True:
        print("test drawing a sample")
        rand_args = params.spice_rand_args_uniform
        n_osc = 10
        rand_args.n_osc = n_osc
        rand_args.weight_dist.n = n_osc
        rand_args.time_stop = 1e-5
        sample = sig_gen.draw_sample(rand_args)
        x_time = np.linspace(0, rand_args.time_stop, sample.signal_matrix.shape[1], endpoint=False)
        data_analysis.plot_signal(sample.weighted_sum, x_time)
        data_analysis.plot_individual_oscillators(sample.signal_matrix, x_time, show=True)


if __name__ == "__main__":
    main()
