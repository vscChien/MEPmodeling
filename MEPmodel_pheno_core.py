import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.interpolate import interp1d


def MEPmodel_pheno_core(ref, p):
    y0    = ref['y0'].T          # transpose back to [N x t]
    t0    = ref['t0']
    muaps = ref['muaps']
    tmuap = ref['tmuap']

    nIntensities = y0.shape[0]
    simMEP       = np.zeros((nIntensities, len(t0)))
    spike_times  = {}

    for i in range(nIntensities):
        shape  = p[0]
        rate   = p[1]
        tshift = p[2]
        N      = int(round(p[3 + i]))

        spike_times[i] = (
            gen_spike_times_gamma(shape, rate, N)
            + (nIntensities - 1 - i) * tshift   # 0-indexed: highest intensity = i=0
        )

        MEPcomps = np.zeros((N, len(t0)))
        for n in range(N):
            xp  = tmuap + spike_times[i][n] + t0[0]
            fp  = muaps[:, n]
            f   = interp1d(xp, fp, kind='linear', bounds_error=False, fill_value=0.0)
            MEPcomps[n, :] = f(t0)

        simMEP[i, :] = MEPcomps.sum(axis=0)

    # align peaks of simMEP and MEP
    i1 = int(np.argmax(y0[-1, :]))          # largest TMS intensity (last row)
    i2 = int(np.argmax(simMEP[-1, :]))
    axonalDelay = max(0.0, t0[i1] - t0[i2])  # d >= 0 ms

    # shift simMEP by axonalDelay
    simMEP_shifted = np.zeros_like(simMEP)
    for i in range(nIntensities):
        f = interp1d(
            t0 + axonalDelay, simMEP[i, :],
            kind='linear', bounds_error=False, fill_value=0.0
        )
        simMEP_shifted[i, :] = f(t0)
    simMEP = simMEP_shifted

    for i in range(nIntensities):
        spike_times[i] = spike_times[i] + axonalDelay

    # ----------------------
    sim = {
        'simMEP':       simMEP,
        'spike_times':  spike_times,
        'MEPcomps':     MEPcomps,
        'axonalDelday': axonalDelay + t0[0],
        'shape':        shape,
        'rate':         rate,
        'tshift':       tshift,
    }
    return sim


# ==========================================================================
def gen_spike_times_gamma(shape, rate, N):
    """Generate spike times from quantiles of a Gamma distribution."""
    scale       = 1.0 / rate
    quantiles   = np.linspace(0, 0.99, N)
    spike_times = gamma_dist.ppf(quantiles, a=shape, scale=scale)
    return spike_times