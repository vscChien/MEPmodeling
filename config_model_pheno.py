import numpy as np
import os
from load_MEP import load_MEP
from load_muap import load_muap


def config_model_pheno(subj):
    ref = {}

    # ----- MEP (target) -----
    if subj == 1 or subj == 8:
        intensity_idx = np.arange(0, 10)
    elif subj == 5 or subj == 7:
        intensity_idx = np.arange(0, 8)
    elif subj == 6 or subj == 10:
        intensity_idx = np.arange(0, 6)
    else:
        intensity_idx = np.arange(0, 7)

    ref['tcrop'] = [20, 50]   # ms, time window of interest

    y0, t0, _, intensities, _, _ = load_MEP(subj, intensity_idx, ref["tcrop"], 0)

    ref['y0']          = y0.T          # [t x N]
    ref['t0']          = t0
    ref['intensities'] = intensities
    ref['muaps'], ref['tmuap'] = load_muap()
    ref['subj']          = subj
    ref['intensity_idx'] = intensity_idx

    # ======================================================================
    # ----- search boundary -----
    # curveType: Gamma + tshift
    nParams = 3 + len(intensity_idx)
    boundary = np.zeros((nParams, 2))   # [lower, upper]
    boundary[0, :] = [0.1, 5]          # alpha
    boundary[1, :] = [0.1, 5]          # lambda
    boundary[2, :] = [0, 2]            # tshift (msec)

    boundary_text = ['shape', 'rate', 'tLag']
    for i in range(len(intensity_idx)):
        boundary[3 + i, :] = [4, 100]  # N_i
        boundary_text.append(f'N{i + 1}')

    ref['boundary']     = boundary
    ref['boundarytext'] = boundary_text

    ref['resultname'] = os.path.join(
        'fitted_results', 'pheno',
        f'result_pheno_s{subj}.h5'
    )
    ref['figname'] = ref['resultname'][:-3] + '.svg'

    return ref