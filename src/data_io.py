"""This module is designated to handling data input and output from and to the filesystem."""

from typing import Tuple
import functools
import glob
import json
import operator
from pathlib import Path
from typing import List, Union
import wave
import copy
import pickle
import shutil
import sys

import const

from scipy.io.wavfile import write
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(data_path: Path, verbose: bool = True) -> Tuple:
    """load a wav file with scipy"""
    sampling_rate, data = read(data_path)

    msg_convert_to_mono = False
    if len(data.shape) == 2:
        data = (np.sum(data, axis=1) // 2).T
        msg_convert_to_mono = True

    # remove y-offset of unsigned encoding
    msg_convert_uint8 = False
    if data.dtype == np.uint8:
        msg_convert_uint8 = True
        data = data.astype("int16") - np.iinfo(np.uint8).max // 2

    if verbose:
        print(f"loading file: {data_path}")
        if msg_convert_to_mono:
            print("converting signal from stereo to mono")
        if msg_convert_uint8:
            print("convert uint8 to int16 to remove y-offset")
        print(f"sampling rate: {sampling_rate}")
        print(f"data type: {data.dtype}")
        print(f"data shape: {data.shape}\n")

    return sampling_rate, data, data.dtype


def load_data_numpy(data_path: Path) -> Tuple:
    """load a wav file with stdlib's wave module and numpy"""
    # Read file to get buffer
    ifile = wave.open(str(data_path))  # BUG with pathlib
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)
    sampling_rate = ifile.getframerate()

    # Convert buffer to float32 using NumPy
    audio_as_np_uint8 = np.frombuffer(audio, dtype=np.uint8)
    audio_as_np_float32 = audio_as_np_uint8.astype(np.float32)

    # Normalise float32 array so that values are between 0 and +1.0
    max_uint8 = 255
    audio_normalised = audio_as_np_float32 / max_uint8

    return sampling_rate, audio_normalised, audio_normalised.dtype


def save_signal_to_wav(
    data: np.ndarray, samplerate: int, dtype: np.dtype, path: Path = Path("data/test.wav")
) -> None:
    """save a singal to a .wav file"""
    data_copy = copy.deepcopy(data)
    # normalize such that max value in signal has max amplitude
    # the resulting file will sound louder
    if dtype in [np.float32, np.float64]:
        norm_factor = np.finfo(dtype).max // max(data)
    else:
        norm_factor = np.iinfo(dtype).max // max(data)
    data_copy *= norm_factor
    write(path, samplerate, data_copy.astype(dtype))


def find_dir_name(parent_path: Path, experiment_description: Union[str, None]) -> Path:
    """find unused directory name for a search experiment and make directory"""
    if experiment_description is None:
        experiment_description = "experiment"
    for i in range(100):
        path = Path(parent_path / f"{experiment_description}{i}")
        if not path.exists():
            path.mkdir(
                exist_ok=True, parents=True
            )  # TODO: hack for launching multiple experiments in parallel via external slurm script
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


def load_sim_data(data_path: Path) -> pd.DataFrame:
    """load the simulation data written to file by ngspice into python"""
    df = pd.DataFrame()
    df = pd.read_csv(data_path, sep="[ ]+", engine="python")  # match any number of spaces
    return df


def load_pickled_fig(data_path: Path) -> None:
    """load a pickled matplotlib figure and display it"""
    _ = plt.figure()
    plt.close()
    with open(data_path, "rb") as file:
        pickle.load(file)
    plt.show()


def load_pickled_samples(data_path: Path) -> List:  # List[Sample]
    """load a list of pickled samples"""
    data_path = Path("data/test_args.pickle")
    obj_li = list()
    with open(data_path, "rb") as f:
        while True:
            try:
                obj_li.append(pickle.load(f))
            except EOFError:
                break

    flat = functools.reduce(operator.iconcat, obj_li, [])
    return flat


def pickle_object(obj: object, data_path: Path) -> None:
    """pickle an object"""
    with open(data_path, "wb") as f:
        pickle.dump(obj, f)


def hoard_experiment_results(
    experiment_description: str, results: List, df: pd.DataFrame, directory: Path = const.WRITE_DIR
) -> None:
    """save experiment results to file and close all figures if still open"""
    plt.close("all")
    if not const.HOARD_DATA:
        return
    pickle_object(results, directory / (experiment_description + "_results.pickle"))
    if df is not None:
        df.to_csv(directory / (experiment_description + "_dataframe.csv"), index=False)
    print(f"saved {experiment_description} results to {directory}")


def clean_dir(path: Path) -> None:
    """delete all files in a directory"""
    for file in path.glob("*"):
        if file.is_dir():
            shutil.rmtree(file)
        elif file.is_file():
            file.unlink()
        else:
            print(f"unknown file type: {file}")


def load_signal_cache(path: Path = const.CACHE_DIR / "signal_cache.pickle") -> pd.DataFrame:
    """load a dataframe from file that maps signal's resistance to frequency and waveform

    args:
        path: path to dataframe saved as pickle
    returns:
        df: dataframe with columns ["r", "freq", "duration", "sampling_rate", "signal"]
    """
    if not path.exists():
        print(f"signal cache not found at {path}")
        print("would you like to create a new cache? (y/n)")
        if input() == "y":
            print("creating new cache...")
            import spice_sweep

            spice_sweep.build_signal_cache()
            print("done")
        else:
            print("exiting...")
            sys.exit()
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


def save_object_to_string(obj: object, path: Path) -> None:
    """save meta information to a file as a string"""
    with open(path, "w") as f:
        f.write(str(obj))


def load_experiment_results(file_name: str) -> pd.DataFrame:
    """load an experiments dataframe from file"""
    if ".pickle" in file_name:
        path = const.WRITE_DIR / file_name
        with open(path, "rb") as f:
            results = pickle.load(f)
        return results
    else:
        path = const.WRITE_DIR / (file_name + ".csv")
        df = pd.read_csv(path)
    return df


def main():
    sampling_rate, data, dtype = load_data()
    save_signal_to_wav(data, sampling_rate, dtype)


if __name__ == "__main__":
    main()
