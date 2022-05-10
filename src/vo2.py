

import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CIRCUIT_PATH = Path("circuit_lib/single_oscillator_RC_version_davide.cir")
DATA_PATH = Path("out.txt")

def run_ngspice(netlist_name="netlist"):
    os.system(f"ngspice {netlist_name}")

run_ngspice(CIRCUIT_PATH)
df = pd.read_csv(DATA_PATH, sep="[ ]+") # match any number of spaces
print(df.head)


sns.lineplot(data=df, x="time", y="v(/A)")
plt.show()

# TODO: out.txt should be renameable from within python
# implement aggregate, tendency and dependency plots
# search