from typing import Callable
from params import bird_params
from netlist_generator import select_netlist_generator

import glob
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write

TARGET_AUDIO_PATH = Path("resources/magpie. 35k, mono, 8-bit, 11025 Hz, 3.3 seconds.wav")
DATA_PATH = Path("data")
PARAM = bird_params["magpie_single_oscillator"]

def run_ngspice(netlist: Path) -> None:
    """start an ngspice simulation from python"""
    os.system(f"ngspice {netlist}")

def load_sim_data(data_path: Path) -> pd.DataFrame:
    """load the simulation data written to file by ngspice into python"""
    df = pd.DataFrame()
    df = pd.read_csv(data_path, sep="[ ]+", engine="python") # match any number of spaces
    return df

def clean_signal(s: pd.Series, points_dropped: int = 200) -> pd.Series:
    """remove startup and y-offset"""
    arr = s.to_numpy()
    no_startup = arr[points_dropped:]
    no_offset = no_startup - min(no_startup)
    return no_offset

def analyze_data(data: pd.Series,
sample_spacing : float = PARAM["time_step"]
) -> tuple:
    # apply fourier transform to signal
    signal = None
    try:
        signal = clean_signal(data)
    except:
        print("EXCEPT")
        signal = data.to_numpy()
    spectrum = np.fft.fft(signal)
    abs_spec = abs(spectrum)
    freq = np.fft.fftfreq(len(abs_spec), d=sample_spacing)
    return freq, abs_spec

def visualize_analysis(df: pd.DataFrame,
dv: str, freq: np.ndarray, abs_spec: np.ndarray
) -> None:
    # time-domain plot
    dv = dv.replace('"', '') # ngspice removes "
    sns.lineplot(data=df, x="time", y=dv)
    
    # frequency-domain plot
    fig, ax = plt.subplots()
    plt.plot(freq, abs_spec)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("log-frequency [Hz]")
    ax.set_ylabel('log-amplitude')

    plt.show()

def signal_to_wav(s: pd.Series,
path: Path = Path("data/out.wav"),
) -> None:
    """Convert a pandas series into a .wav file."""
    samp_rate: int = int(0.00001/PARAM["time_step"])
    print(f"samplerate: {samp_rate}")
    # TODO: auto-set sampling rate so that duration of clip is 1s
    
    s = s.repeat(3) # stretch signal duration
    s = s.to_numpy()
    amplitude = np.iinfo(np.uint8).max # 255
    s *= amplitude # scale
    write(path, samp_rate, s.astype(np.uint8))

def scale_up(short: np.ndarray, len_long: int) -> np.ndarray:
    """scale a short signal up to the desired length by repeating each symbol in-place"""
    short = pd.Series(short)
    short = short.repeat(len_long//len(short))
    # pad with zeros at the end
    to_pad = len_long - len(short)
    padded = np.pad(short, (0, to_pad))
    return padded

def rmse(p: pd.Series, t: np.ndarray) -> float:
    """
    Compute root mean square error (RMSE) between prediction and target signal.
    Scale up smaller signal to enable metric.
    """
    diff_len = lambda p, t: len(p) != len(t)
    get_shortest = lambda p, t: np.argmin([len(p), len(t)])
    get_longest = lambda p, t: np.argmax([len(p), len(t)])

    p = p.to_numpy()
    if diff_len(p, t):
        tup = (p, t)
        short_sig = tup[get_shortest(p, t)]
        long_sig = tup[get_longest(p, t)]
        short_sig = scale_up(short_sig, len(long_sig))
        return np.sqrt(((short_sig-long_sig)**2).mean())

    return np.sqrt(((p-t)**2).mean())

def find_dir_name() -> Path:
    """find unused directory name for a search experiment and make directory"""
    path = DATA_PATH
    for i in range(100):
        path = Path(DATA_PATH / f"experiment{i}")
        if not path.exists():
            path.mkdir()
            break
    return path

def json_to_df(path: Path) -> pd.DataFrame:
    """aggregate parameters and results of multiple json files into a dataframe"""
    data_rows = []
    for json_file in glob.glob(str(path) + "/*.json"):
        with open(json_file) as f:
            data_rows.append(json.load(f))
    df = pd.DataFrame(data_rows)
    return df

def random_search(netlist_generator: Callable, param: dict, visual: bool = True) -> None:
    """random search over oscillator space"""
    experiment_path = find_dir_name()
    target = read(TARGET_AUDIO_PATH)[1]
    for i in range(param["trials"]):
        tmp_path = experiment_path / f"netlist{i}.cir"
        # build netlist and store parameters
        det_params = netlist_generator(tmp_path, param)
        # run simulation on netlist
        run_ngspice(tmp_path)
        # load the simulation data written by ngspice into Python
        df = load_sim_data(Path(str(tmp_path) + ".dat"))
        s = df.iloc[:,1] # column as series
        if visual:
            freq, abs_spec = analyze_data(s)
            visualize_analysis(df, param["dependent_component"], freq, abs_spec)
        det_params["rmse"] = rmse(s, target) 
        with open(experiment_path / f"param{i}.json", "w") as f:
            json.dump(det_params, f)
    
    # aggregate the results of n trials into a single dataframe
    df = json_to_df(experiment_path)
    print(df)

def main():
    netlist_generator = select_netlist_generator("sum")
    random_search(netlist_generator, param=PARAM)

if __name__ == "__main__":
    main()
