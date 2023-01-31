import gen_signal_spipy

import numpy as np
import data_analysis
import param_types as party
import data_analysis
import matplotlib.pyplot as plt
import const

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
    down_sample_factor=1,
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

if __name__ == "__main__":
    freq_to_r_sweep()