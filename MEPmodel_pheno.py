import numpy as np
from MEPmodel_pheno_core import MEPmodel_pheno_core
from plot_param_panel_pheno import plot_param_panel_pheno
from plot_summary_pheno import plot_summary_pheno


def MEPmodel_pheno(p, ref, plotOn=0):
    """
    3 + i free parameters
      p[0]   : alpha  – shape of gamma distribution
      p[1]   : lambda – rate of gamma distribution
      p[2]   : d      – time lag between TMS conditions
      p[3+i] : active MNs (range [4-100]) by TMS intensity i
    """
    # ----- simulate MEP -----
    sim = MEPmodel_pheno_core(ref, p)
    ref = cal_error(ref, sim)
    simMEP = ref['sim']['simMEP2']

    if plotOn:
        plot_param_panel_pheno(p, ref)
        plot_summary_pheno(p, ref)

    return simMEP, ref


# ==========================================================================
def cal_error(ref, sim):
    y0     = ref['y0']
    t0     = ref['t0']
    simMEP = sim['simMEP'].T          # transpose to match [t x N] layout of y0

    simMEP2 = simMEP / np.max(simMEP) * np.max(y0)
    error   = np.ravel(simMEP2) - np.ravel(y0)

    denom_nrmsd = np.sqrt(len(t0)) * (np.max(y0) - np.min(y0))
    NRMSD = np.linalg.norm(np.ravel(y0) - np.ravel(simMEP2)) / denom_nrmsd

    ss_res = np.sum((np.ravel(y0) - np.ravel(simMEP2)) ** 2)
    ss_tot = np.sum((np.ravel(y0) - np.mean(y0))        ** 2)
    R2     = 1.0 - ss_res / ss_tot

    ref['NRMSD'] = NRMSD
    ref['error'] = error
    ref['R2']    = R2

    sim['simMEP2'] = simMEP2
    ref['sim']     = sim

    return ref