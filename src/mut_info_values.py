from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PROJ_DIR = Path(__file__).parent.parent

"""
n_tube = [0,2,5,8,13]
t0 = [4,.6,3.5,4,4]
mut_info_right = [.3036,.3083,.3322,.3304,.3322]


n_tube = [0,5,8,20]
t0 = [8,12,12,7.78]
mut_info_right = [.3077,.3182,.3170,.2908,.3199]

n_tube =[15]
t0 = [12]
mut_info_right = [.1961]
"""


n_tube = [0,1,3,5,6,7,10,15,20,25,30,35,40,45,48,50,55,60,62,65,70,75,80]
t0 = [12,6,2.3,3.4,3.8,12,6,6,7.1,9.3,6,12,6.4,8,9.2,12,12]
t_couple = [42.16,70.9,81.3,75.6,76.8,78.5,73.8,77.5,60.9,75.2,96.1,76,112,123.9,115.8,109.1,131,125.5,136.5,110.9,152.8,151.9]
mut_info_right = [.1548,.1800,.2031,.2157,.2274,.2061,.1918,.1996,.2197,.2425,.2326,.2256,.2413,.2489,.2396,.2256,.2291,.2577,.2256,.2099,.2528,.2419,.2343]

meas_data = np.loadtxt("outputs/fin.txt")
meas_lengths = np.linspace(0,400,120)

df = pd.read_csv(f'{PROJ_DIR}/data/coupling_data.csv', usecols=[0, 2, 3],header=None)
n_tube = df[0]
mut_info_right = df[3]
coupling_times = df[2]

plt.plot(n_tube,mut_info_right,'.',color = "black")
plt.xlabel("tube length")
plt.ylabel("mutual information")
plt.show()


plt.plot(n_tube,coupling_times,'.',color = "black")
plt.xlabel("tube length")
plt.ylabel("coupling")
plt.show()


plt.plot(meas_lengths,meas_data,'.',color = "black")
plt.xlabel("tube length")
plt.ylabel("mutual information")
plt.show()