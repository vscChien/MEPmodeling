import os
import numpy as np
import h5py
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def load_muap(plotOn=0):
    # Determine the directory of the current script
    root = os.getcwd() 

    # Define paths
    mat_path = os.path.join(root, "data_MUAP", "muap.mat")
    h5_path = os.path.join(root, "data_MUAP", "muscle_model", "Dist1_Monopolar_Rest_NormalCV_New.hdf5")

    if os.path.exists(mat_path):
        # Using scipy.io.loadmat to handle the .mat file
        from scipy.io import loadmat
        tmp = loadmat(mat_path) #
        muaps = tmp["muaps"] #
        t = tmp["t"] #
    else:
        # Read from HDF5 file
        with h5py.File(h5_path, "r") as f:
            # h5read(..., "/MUAPShapes")
            muaps = np.array(f["MUAPShapes"]) #
        
        muaps = muaps.T # muaps'
        tmuap = np.linspace(0, 20, 20001) # 0~20 ms, dt = 0.001 msec
        
        # ---cut zeros (around first 2000 points)----
        # Find index where the sum of absolute values across columns is non-zero
        row_sums = np.sum(np.abs(muaps), axis=1)
        idx = np.where(row_sums != 0)[0][0] # find(..., 1) - 1
        
        muaps = -muaps[idx:, :] # flipped
        tmuap = tmuap[idx:]
        tmuap = tmuap - np.min(tmuap) #
        
        # ------------------------------------
        # downsample to dt = 0.1 msec
        dt = 0.1 # ms
        t = np.arange(0, 20, dt) # 0:dt:(20-dt)
        
        # Interpolation: interp1(tmuap, muaps, t, "linear", 0)
        f_interp = interp1d(tmuap, muaps, axis=0, kind="linear", bounds_error=False, fill_value=0)
        muaps = f_interp(t) #
        # ------------------------------------

    if plotOn:
        fig = plt.figure()
        for i in range(1, 21): # 1:20
            ax = fig.add_subplot(4, 5, i)
            # muaps[:, i*5-4 : i*5] in MATLAB (1-based) 
            # is muaps[:, (i-1)*5 : i*5] in Python (0-based)
            start_col = (i - 1) * 5
            end_col = i * 5
            
            ax.plot(t.T, 1e6 * muaps[:, start_col:end_col], linewidth=1.5)
            ax.grid(True)
            
            # Legend labels
            labels = [str(val) for val in range(start_col + 1, end_col + 1)]
            ax.legend(labels, loc="upper right")
            
        plt.xlabel("ms")
        plt.ylabel("μV")
        plt.show() #

    return muaps, t