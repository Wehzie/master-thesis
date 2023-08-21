"""
This module implements an experiment that evaluates oscillation frequency of a single RC oscillator as a function of R.

Multiple netlists are constructed with different values of R.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gen_signal_spipy
import data_analysis
import gen_signal_args_types as party
import const
import data_preprocessor

base_args = party.SpiceSingleDetArgs(
    n_osc=1,
    v_in=14,
    r=47e3,
    r_last=1,
    r_control=1,
    c=300e-12,
    time_step=2e-9,
    time_stop=1e-5,
    time_start=0,
    dependent_component="v(osc1)",
    phase=0,
    generator_mode=party.SpipyGeneratorMode.CACHE,
    down_sample_factor=1 / 200,  # must be same as in rand_args of params_hybrid.py
)


def freq_to_r_sweep(debug=True, show=True):
    # r values to sweep
    r_sweep = range(int(20e3), int(140e3), int(1e3))

    sig_gen = gen_signal_spipy.SpipySignalGenerator()

    freqs = []
    rs = list(r_sweep)
    for r in r_sweep:
        base_args.r = r
        single_oscillator = sig_gen.draw_single_oscillator(base_args)
        x_time = np.linspace(0, base_args.time_stop, len(single_oscillator), endpoint=False)
        freq = data_analysis.get_freq_from_fft(single_oscillator, base_args.time_step)
        freqs.append(freq)
        if debug:
            print(f"r={r}, freq={freq}")
            data_analysis.plot_signal(single_oscillator, x_time, show=True)

    plt.plot(rs, freqs, "o")
    plt.xlabel(r"R [$\Omega$]")
    plt.ylabel("Frequency [Hz]")
    plt.savefig(const.WRITE_DIR / "r_to_freq_sweep.png", dpi=300)
    if show:
        plt.show()


def build_signal_cache(
    r_min: int = 19e3, r_max: int = 141e3, r_step: int = 100, debug: bool = False
):
    """
    Build a cache of signals with different values of R.

    This is useful for debugging and testing.
    """
    r_min, r_max, r_step = int(r_min), int(r_max), int(r_step)

    def simulate_failure_tolerant():
        patience_counter = 0
        while patience_counter < const.SPICE_PATIENCE:
            (
                period_duration,
                sampling_rate,
                period_signal,
            ) = gen_signal_spipy.SpipySignalGenerator.simulate_single_period(
                base_args, tmp_path, patience_counter + 1
            )
            if period_duration is not None:
                break
            else:
                patience_counter += 1

        if patience_counter >= const.SPICE_PATIENCE:
            raise Exception(f"SPICE simulation failed {const.SPICE_PATIENCE} times in a row")

        return period_duration, sampling_rate, period_signal

    r_sweep = range(r_min, r_max, r_step)
    df = pd.DataFrame(
        columns=["r", "freq", "duration", "sampling_rate", "signal"], index=range(len(r_sweep))
    )
    tmp_path = gen_signal_spipy.SpipySignalGenerator.get_tmp_path()
    for i, r in enumerate(r_sweep):
        base_args.r = r
        period_duration, sampling_rate, period_signal = simulate_failure_tolerant()
        freq = 1 / period_duration

        if base_args.down_sample_factor < 1:
            period_signal = data_preprocessor.downsample_by_factor_typesafe(
                period_signal, base_args.down_sample_factor
            )
            sampling_rate = np.round(sampling_rate * base_args.down_sample_factor).astype(int)
        assert sampling_rate > 2 * freq, "Sampling rate is below Nyquist rate."

        df.loc[i] = [r, freq, period_duration, sampling_rate, period_signal]
        if debug:
            x_time = np.linspace(0, period_duration, len(period_signal), endpoint=False)
            data_analysis.plot_signal(period_signal, x_time, show=True)

    const.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_dir = const.CACHE_DIR
    df.to_pickle(save_dir / "signal_cache.pickle")
    df.to_csv(save_dir / "signal_cache.csv")


if __name__ == "__main__":
    # freq_to_r_sweep()
    build_signal_cache()
