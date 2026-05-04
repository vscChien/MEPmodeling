import os
import h5py
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from load_h5 import load_h5_to_dict


def load_MEP(subj, iidx=None, tcrop=[20, 50], plotOn=1):
    TSTIM = 100  # TMS stimulus was at t=100 ms

    # ----- load data -----
    root = os.getcwd()
    mat_path = os.path.join(
        root,
        "data_Oxford_MEP",
        f"S{subj}_Magstim_data",
        f"S{subj}.h5"
    )
    
    with h5py.File(mat_path, 'r') as f:
        tmp = load_h5_to_dict(f)

    mep = tmp["mep"]                 # shape: [N x time x trials]
    intensities = tmp["intensities"].flatten()
    times = tmp["t"].flatten() - TSTIM

    # ----- select intensities -----
    if iidx is None:
        iidx = np.arange(len(intensities))

    mep = mep[iidx, :, :]
    intensities = intensities[iidx]

    # ----- crop -----
    tidx = np.where((times >= tcrop[0]) & (times < tcrop[1]))[0]
    t = times[tidx]

    yall = mep[:, tidx, :]
    y = np.mean(yall, axis=2)

    # ----- remove baseline [-20, -10] ms -----
    baseline_idx = np.where((times >= -20) & (times < -10))[0]

    baseline = np.mean(
        np.mean(mep[:, baseline_idx, :], axis=2),
        axis=1
    )  # shape: [n]

    # subtract baseline
    y = y - baseline[:, np.newaxis]
    mep = mep - baseline[:, np.newaxis, np.newaxis]

    # ----- plotting -----
    if plotOn:

        # figure 1: all trials
        plt.figure()
        for i in range(len(intensities)):
            plt.subplot(2, 5, i + 1)

            plt.plot(times, mep[i, :, :], 'c', linewidth=0.8)
            plt.grid(True)

            mean_trace = np.mean(mep[i, :, :], axis=1)
            plt.plot(times, mean_trace, 'k', linewidth=1.5)

            plt.xlim([20, 50])
            plt.ylim([-2, 5])

            plt.title(f"{intensities[i]}% MSO")
            plt.xlabel("msec")

        # figure 2: averaged signals
        plt.figure()
        for i in range(len(intensities)):
            plt.plot(t, y[i, :], linewidth=1.5)

        plt.grid(True)
        plt.title(f"Subject {subj}")
        plt.xlabel("Time (msec)")
        plt.ylabel("Amplitude (mV)")
        plt.xlim([20, 50])

        legend_labels = [f"{val}%MSO" for val in intensities]
        plt.legend(legend_labels)

        plt.show()

    return y, t, mep, intensities, times, yall