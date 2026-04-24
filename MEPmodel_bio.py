import numpy as np
from scipy.interpolate import interp1d
from plot_summary import plot_summary
from plot_param_panel import plot_param_panel
from plot_TMScondition import plot_TMScondition
from MEPmodel_bio_core import MEPmodel_bio_core
from gen_resistance_mono import gen_resistance_mono

# --------- Main function ---------
def MEPmodel_bio(p, ref, plotOn=0):

    ref = update_model_bio(ref, p)
    sim = MEPmodel_bio_core(ref["model"])  # simulation 
    ref = cal_error(ref, sim)
    simMEP = ref["sim"]["simMEP2"]

    if plotOn:
        plot_param_panel(p, ref)
        plot_summary(p, ref)

        idx = len(ref["intensities"])  # same as MATLAB
        plot_TMScondition(ref, idx)

    return simMEP, ref


# ==========================================================================
def update_model_bio(ref, p):

    if ref["model"]["withRC"] == 0:
        R     = p[0:5]
        E1    = 0
        E2    = 0
        I1    = 0
        I2    = 0
        RCth  = 10
        Tmu   = p[5]
        AMPAw = p[6]

    elif ref["model"]["withRC"] == 1:
        R     = p[0:5]
        E1    = p[5]
        E2    = p[6]
        I1    = p[7]
        I2    = p[8]
        RCth  = p[9]
        Tmu   = p[10]
        AMPAw = p[11]

    # ----- membrane resistances of MNs -----
    ref["model"]["R"] = gen_resistance_mono(R)

    # ----- Wexc (MNs -> RC) -----
    ref["model"]["Wexc"] = np.linspace(E1, E2, 100)

    # ----- R*Winh (RC->MNs) -----
    ref["model"]["RWinh"] = np.linspace(I1, I2, 100)

    # ----- RC sigmoid -----
    ref["model"]["rc"] = RC_setting(RCth)

    # ----- MUAP -----
    ref["model"]["Tmu"] = Tmu

    # ----- DI-waves -----
    ref["model"]["AMPAweight"] = AMPAw
    ref["model"]["DIwaveConv"] = (
        ref["model"]["DIwaveConv_AMPA"] * ref["model"]["AMPAweight"]
        + ref["model"]["DIwaveConv_NMDA"] * (1 - ref["model"]["AMPAweight"])
    )

    # ----- Motor neuron -----
    ref["model"]["tauLIF"] = 10
    ref["model"]["Eexc"] = 0
    ref["model"]["Einh"] = -75
    ref["model"]["V_rest"] = -65
    ref["model"]["V_thr"] = -55
    ref["model"]["T_ref"] = 2
    ref["model"]["fastAChRweight"] = 0.5

    return ref


# ==========================================================================
# Renshaw cell population params
def RC_setting(p):

    rc = {}

    # RC sigmoid function
    rc["r"] = 10
    rc["v_thr"] = p
    rc["fmax"] = 1

    # synaptic kernel (RC->MN, Glycine receptor)
    taur = 1
    tauf = 6
    rc["tau1"] = taur * tauf / (taur + tauf)
    rc["tau2"] = tauf
    rc["h"] = (rc["tau2"] - rc["tau1"]) / rc["tau1"] / rc["tau2"]

    return rc


# ==========================================================================
def cal_error(ref, sim):

    y0 = ref["y0"]
    t0 = ref["t0"]

    simMEP = sim["simMEP"].T  # transpose like MATLAB '
    t = sim["t"]

    # align peaks
    i1 = np.argmax(y0[:, -1])
    i2 = np.argmax(simMEP[:, -1])

    axonalDelay = max(0, t0[i1] - t[i2])

    # interpolation (MATLAB interp1 equivalent)
    f_interp = interp1d(
        t + axonalDelay,
        simMEP,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=0
    )
    simMEP = f_interp(t0)

    # normalization
    if np.max(simMEP) > 0:
        simMEP2 = simMEP / np.max(simMEP) * np.max(y0)
    else:
        simMEP2 = simMEP

    error = simMEP2 - y0

    NRMSD = (
        np.linalg.norm(y0 - simMEP2)
        / np.sqrt(len(t0))
        / (np.max(y0) - np.min(y0))
    )

    R2 = 1 - np.sum((y0 - simMEP2) ** 2) / np.sum((y0 - np.mean(y0)) ** 2)

    ref["NRMSD"] = NRMSD
    sim["simMEP2"] = simMEP2
    ref["model"]["axonalDelay"] = axonalDelay
    ref["R2"] = R2
    ref["error"] = error
    ref["sim"] = sim

    return ref