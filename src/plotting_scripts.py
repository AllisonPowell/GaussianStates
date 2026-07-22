import numpy as np
import matplotlib.pyplot as plt


import os
from pathlib import Path
import csv
import pandas as pd

PROJ_DIR = Path(__file__).parent.parent

"""
df = pd.read_csv(f'{PROJ_DIR}/data/harmonic_chain_sizes.csv', usecols=[0, 1],header=None)

size = df[0]
time = df[1]

log = np.log(size)

plt.plot(size,time,'.',label="data")
plt.plot(size,log,label="log(size)")
plt.xlabel("system size")
plt.ylabel("scrambling time")
plt.legend()
plt.show()



spin_sizes = [3,5,7,9,11]
spin_times = [3,5,7,9,11]


plt.plot(size[0:7],time[0:7],'.',label="Gaussian")
plt.plot(spin_sizes,spin_times,'.',label="BKP")
plt.xlabel("system size")
plt.ylabel("scrambling time")
plt.legend()
plt.show()
"""

hole_size = [0,10,24,33,40,46,52]
mi = [1.26,1.46,1.98,3.76,6.69,7.987,8.1126]

plt.plot(hole_size,mi,'ko')
plt.xlabel("hole size")
plt.ylabel("mutual information")
plt.show()

print("stop")
