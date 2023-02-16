"""
The main module serves as the entry point to load parameters and start a simulation.
"""

import argparse

parser = argparse.ArgumentParser(description="Run a simulation.")
parser.add_argument("--production", action="store_true", help="Run a simulation with production parameters.", default=False, required=False)
parser.add_argument("--clean", action="store_true", help="Clean the write directory before running the simulation.", default=False, required=False)
parser.add_argument("--signal_generator", type=str, help="Select the signal generator to use.", choices=["python", "spipy", "hybrid", "all"], default="all", required=False)
parser.add_argument("--qualitative", action="store_true", help="Run optimization without statistical analysis.", default=False, required=False)
parser.add_argument("--experiment", type=str, help="Select the quantitative experiments to run; not all experiments are compatible with each signal generator.",
default="all", required=False, choices=[
    "none", "all",
    "target",
    "n_osc", "z_ops", "samples", "duration",
    "frequency", "resistor", "weight", "offset", "phase", "amplitude"])
parser.add_argument("--target", type=str, help="Select the default target to approximate.", default="sine", required=False, choices=[
    "sine", "triangle", "square", "sawtooth", "inverse_sawtooth",
    "chirp", "beat", "damp_chirp",
    "smooth_gauss", "smooth_uniform", "gauss_noise", "uniform_noise",
    "magpie", "human_yes", "bellbird", "human_okay",
    ])

args = parser.parse_args()

print(f"Running simulation with {args}")

import const

if args.production:
    print("Configuration set to run simulation with production parameters.")
    const.TEST_PARAMS = False

import data_io
import data_analysis
import experimenteur
if const.TEST_PARAMS:
    print("Import test parameters.")
    import params_python_test as python_parameters
    import params_hybrid_test as hybrid_parameters
else:
    print("Import production parameters.")
    import params_python as python_parameters
    import params_hybrid as hybrid_parameters
import shared_params_target
import gen_signal_python
import gen_signal_spipy
import sweep_builder
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

legal_python_experiments = [
    "target",
    "n_osc",
    "z_ops",
    "samples",
    "frequency",
    "weight",
    "offset",
    "phase",
    "amplitude",
    "all",
    "none"
]

legal_hybrid_experiments = [
    "target",
    "n_osc",
    "z_ops",
    "duration",
    "resistor",
    "weight",
    "offset",
    "phase",
    "all",
    "none"
]

if "python" in args.signal_generator:
    assert args.experiment in legal_python_experiments, f"Experiment {args.experiment} is not legal for Python signal generator."

if "spipy" in args.signal_generator:
    assert args.experiment in legal_hybrid_experiments, f"Experiment {args.experiment} is not legal for SpiPy signal generator."

@data_analysis.print_time
def main():
    if args.clean:
        print(f"Cleaning {const.WRITE_DIR} directory.")
        data_io.clean_dir(const.WRITE_DIR)

    if args.signal_generator in ["python", "all"]:
        
        exp = experimenteur.Experimenteur("python_sweep")
        sig_gen = gen_signal_python.PythonSigGen()
        generator_args = python_parameters.py_rand_args_uniform
        m_target = shared_params_target.select_target_by_string(args.target, generator_args, python_parameters.SYNTH_FREQ, python_parameters.DURATION)
        sweep_bundle = sweep_builder.bundle_python_sweep(sig_gen, generator_args, m_target, algo_selector="all")
        
        if args.qualitative:
            exp.run_qualitative_algo_sweep(sweep_bundle.algo_sweep, m_target)
        
        if args.experiment != "none":
            exp.run_all_experiments(sweep_bundle, m_target.samples, generator_args, args.experiment)

    if args.signal_generator in ["spipy", "hybrid", "all"]:
        
        exp = experimenteur.Experimenteur("hybrid_sweep")
        sig_gen = gen_signal_spipy.SpipySignalGenerator()
        generator_args = hybrid_parameters.spice_rand_args_uniform
        m_target = shared_params_target.select_target_by_string(args.target, generator_args, hybrid_parameters.SYNTH_FREQ)
        sweep_bundle = sweep_builder.bundle_hybrid_sweep(sig_gen, generator_args, m_target, algo_selector="all")

        if args.qualitative:
            exp.run_qualitative_algo_sweep(sweep_bundle.algo_sweep, m_target)

        if args.experiment != "none":
            exp.run_all_experiments(sweep_bundle, m_target.samples, generator_args, args.experiment, args.target)

if __name__ == "__main__":
    main()