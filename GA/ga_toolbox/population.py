import numpy as np


def population(N, nParams, LR, UR):
    """
    Generate N random solutions uniformly within the boundary.

    Parameters
    ----------
    N       : int        number of random solutions to generate
    nParams : int        number of parameters
    LR      : array-like [nParams,]  lower boundary
    UR      : array-like [nParams,]  upper boundary

    Returns
    -------
    P : np.ndarray  [N x nParams]  random solutions
    """
    LR = np.atleast_1d(LR)
    UR = np.atleast_1d(UR)

    P = np.zeros((N, nParams))
    for i in range(nParams):
        P[:, i] = (UR[i] - LR[i]) * np.random.rand(N) + LR[i]

    return P