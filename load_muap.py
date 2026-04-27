import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from load_h5 import load_h5_to_dict

def load_muap(plotOn=0):
    # Determine the directory of the current script
    root = os.getcwd() 

    # Define paths
    h5_path = os.path.join(root, "data_MUAP", "muap.h5")

    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            tmp = load_h5_to_dict(f)

        muaps = tmp["muaps"] #
        t = tmp["t"].T #
        # ------------------------------------

    if plotOn:
        fig = plt.figure()
        for i in range(1, 21): # 1:20
            ax = fig.add_subplot(4, 5, i)
            # muaps[:, i*5-4 : i*5] in MATLAB (1-based) 
            # is muaps[:, (i-1)*5 : i*5] in Python (0-based)
            start_col = (i - 1) * 5
            end_col = i * 5
            
            ax.plot(t, 1e6 * muaps[:, start_col:end_col], linewidth=1.5)
            ax.grid(True)
            
            # Legend labels
            labels = [str(val) for val in range(start_col + 1, end_col + 1)]
            ax.legend(labels, loc="upper right")
            
        plt.xlabel("ms")
        plt.ylabel("μV")
        plt.show() #

    return muaps, t