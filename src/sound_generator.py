from params import bird_params

import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

DATA_PATH = Path("circuit_lib")
PARAM = bird_params["magpie"]

POWER = """
.include ../circuit_lib/VO2_Sto_rand.cir
V1 /bridge 0 dc {v}
"""

# vo2 only
OSC_TEMPLATE = """
XU{i} /bridge /osc{i} control{i} VO2_Sto
C{i} /osc{i} 0 {c}
R{i} /osc{i} /sum {r}
"""

POST_SUM = """
R_last /sum 0 {r_last}
"""

CONTROL_TEMPLATE = """* Control commands
.control
tran {time_step} {time_stop} {time_start} uic
set wr_vecnames * to print header
set wr_singlescale * to not print scale vectors
wrdata data/{file_name}.txt {dependent_component}
quit
.endc
"""

def build_netlist(path: Path, temp_name: str = "temp.cir") -> Path:
    """create temporary netlist, return path to temp file"""
    temp_path = path / temp_name

    netlist = POWER.format(v=PARAM["v_in"])
    for i in range(1, 1+PARAM["num_osc"]):
        # TODO: generalize for >0 and <0 values
        # so over randint vs uniform
        r = np.random.randint(PARAM["r_min"], 1+PARAM["r_max"])
        c = np.random.uniform(PARAM["c_min"], PARAM["c_max"])        
        r_out = np.random.randint(PARAM["r_out_min"], 1+PARAM["r_out_max"])

        netlist += OSC_TEMPLATE.format(i=i, r=r, c=c, r_out=r_out)

    netlist += POST_SUM.format(r_last=PARAM["r_last"])
    netlist += CONTROL_TEMPLATE.format(
        time_step=PARAM["time_step"],
        time_stop=PARAM["time_stop"],
        time_start=PARAM["time_start"],
        dependent_component=PARAM["dependent_component"],
        file_name=PARAM["file_name"])

    with open(temp_path, "w") as f:
        f.write(netlist)
    return temp_path

def run_ngspice(netlist: Path) -> None:
    os.system(f"ngspice {netlist}")

def load_data(data_path: Path = f"data/{PARAM['file_name']}.txt") -> pd.DataFrame:
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
    signal = clean_signal(data)
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
    
    fig, ax = plt.subplots()
    plt.plot(freq, abs_spec)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("log-frequency [Hz]")
    ax.set_ylabel('log-amplitude')

    plt.show()

def signal_to_wav(s: pd.Series,
path: Path = Path("data/out.wav"),
samp_rate: int = int(1/PARAM["time_step"])
) -> None:
    """Convert a pandas series into a .wav file."""
    amplitude = np.iinfo(np.int16).max
    s = amplitude * s.to_numpy()
    write(path, samp_rate, s.astype(np.int64))

# tmp_path = build_netlist(DATA_PATH)
# run_ngspice(tmp_path)
df = load_data()
s = df.iloc[:,1] # column as series
# freq, abs_spec = analyze_data(s)
# visualize_analysis(df, PARAM["dependent_component"], freq, abs_spec)
signal_to_wav(s)