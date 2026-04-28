import os
import numpy as np
from load_MEP import load_MEP
from load_muap import load_muap
from gen_DIwave import gen_DIwave
from plot_DIwave import plot_DIwave
from gen_kernels import gen_kernels
from deconv_DIwave import deconv_DIwave

def config_model_bio(subj, withRC, AMPAweight=[]):

    model = {}
    model["AMPAweight"] = AMPAweight
    model["withRC"] = withRC
    model["maxES"] = 5  # take first 5 spikes

    # -----subject MEP (target) -----
    if subj == 1 or subj == 8:
        intensity_idx = np.arange(0, 10)
    elif subj == 5 or subj == 7:
        intensity_idx = np.arange(0, 8)
    elif subj == 6 or subj == 10:
        intensity_idx = np.arange(0, 6)
    else:
        intensity_idx = np.arange(0, 7)

    ref = {}
    ref["subj"] = subj
    ref["intensity_idx"] = intensity_idx
    ref["tcrop"] = [20, 50]
    y0, t0, _, intensities, _, _ = load_MEP(subj, intensity_idx, ref["tcrop"], 0)
    ref["intensities"] = intensities
    ref["t0"] = t0
    ref["y0"] = y0.T
    model["muaps"], model["tmuap"] = load_muap()

    # -----subject RMT ---------
    # (by eyeballing the IO curve)
    if ref["subj"] == 1 or ref["subj"] == 8:
        ref["RMT"] = 32
    elif ref["subj"] == 2 or ref["subj"] == 10:
        ref["RMT"] = 35
    elif ref["subj"] == 3 or ref["subj"] == 9:
        ref["RMT"] = 38
    elif ref["subj"] == 4:
        ref["RMT"] = 41
    elif ref["subj"] == 5 or ref["subj"] == 6:
        ref["RMT"] = 47
    else:
        ref["RMT"] = 44

    # ----- simulated DI wave -----
    tlength = 50 # ms
    dt = 0.1 # ms
    t = np.arange(0, tlength, dt)
    model["DIwave0"] = np.zeros((len(ref["intensities"]), len(t)))
    for i in range(len(ref["intensities"])):
        model["DIwave0"][i,:] = gen_DIwave(t, ref["intensities"][i] / ref["RMT"])
    #model["DIwave"] = deconv_DIwave(t, model["DIwave0"], ref)
    from scipy.io import loadmat
    model["DIwave"] = loadmat("DIwave.mat")
    model["DIwave"] = model["DIwave"]["DIwave"]

    # ----- AMPA, NMDA kernels -----
    model["AMPA"], model["NMDA"] = gen_kernels(dt, tlength)

    # ----- AchR, GlyR kernels -----
    model[("kernel", "tau")] = [[0.5, 3.6], [1.8, 20.2], [1, 6]]
    model[("kernel", "h")] = [[1.5977], [1.3908], [1.7175]]

    # ----- Motor neuron -----
    model["tauLIF"] = 10 # ms
    model["Eexc"] = 0 # mV
    model["Einh"] = -75 # mV
    model["V_rest"] = -65 # mV
    model["V_thr"] = -55 # mV
    model["T_ref"] = 2 # ms, refractory period
    model ["fastAChRweight"] = 0.5 # AchR (MN->RC)

    # ----- tDIwave (input) -----
    model["DIwaveConv"] = model["DIwave"]
    model["DIwaveConv_AMPA"] = np.zeros((len(ref["intensities"]), len(t)))
    model["DIwaveConv_NMDA"] = np.zeros((len(ref["intensities"]), len(t)))
    fr = 1
    for i in range (len(ref["intensities"])):
        g = np.convolve(fr * model["DIwave"][i,:], model["AMPA"])
        model["DIwaveConv_AMPA"][i,:] = g[:len(t)]
        g = np.convolve(fr * model["DIwave"][i,:], model["NMDA"])
        model["DIwaveConv_NMDA"][i,:] = g[:len(t)]
    model["t"] = t
    model["dt"] = dt

    plot_DIwave(model, ref)

    # ----- search boundary-----
    if withRC == 0:
        nParams = 7
        boundary = np.zeros((nParams, 2)) # [lower, upper]
        boundary[0, :] = np.array([0, 10]) # R1
        boundary[1, :] = np.array([0, 10]) # R2
        boundary[2, :] = np.array([0, 10]) # R3
        boundary[3, :] = np.array([0, 10]) # R4
        boundary[4, :] = np.array([0, 10]) # R5
        boundary[5, :] = np.array([5, 10]) # Tmuap
        boundary[6, :] = np.array([0, 1]) # AMPAweight
        if AMPAweight:
            boundary[6, :] = np.array([1, 1]) * AMPAweight # fixed value
        model["boundarytext"] = ['R1','R2','R3','R4','R5','MU.T','AMPAw']
    else:
        nParams = 12
        boundary = np.zeros((nParams, 2)) # [lower, upper]
        boundary[0, :] = np.array([0, 10]) # R1
        boundary[1, :] = np.array([0, 10]) # R2
        boundary[2, :] = np.array([0, 10]) # R3
        boundary[3, :] = np.array([0, 10]) # R4
        boundary[4, :] = np.array([0, 10]) # R5
        boundary[5, :] = np.array([0, 20]) # E1 (MN1 -> RC)
        boundary[6, :] = np.array([0, 20]) # E2 (MN100 -> RC)
        boundary[7, :] = np.array([0, 10]) # I1 (RC -> MN1)
        boundary[8, :] = np.array([0, 10]) # I2 (RC -> MN100)
        boundary[9, :] = np.array([1, 10]) # RC.th
        boundary[10, :] = np.array([5, 10]) # Tmuap
        boundary[11, :] = np.array([0, 1]) # AMPAweight
        if AMPAweight:
            boundary[11, :] = np.array([1, 1]) * AMPAweight # fixed value
        model["boundarytext"] = ['R1','R2','R3','R4','R5','E1','E2','I1','I2','RC.th','MU.T','AMPAw']

    model["boundary"] = boundary
    ref["model"] = model

    if withRC == 1:
        if not AMPAweight == None:
            ref["resultname"] = os.path.join("fitted_results", "bio", f"result_bio_s{subj}.h5")
        else:
            ref["resultname"] = os.path.join("fitted_results", "bio", "fixed_AMPAweight", f"result_bio_s{subj}.h5")
    else:
            ref["resultname"] = os.path.join("fitted_results", "bioNoRC", f"result_bioNoRC_s{subj}.h5")
    ref["figname"] = ref["resultname"][:-4]+".svg"

    return ref