import numpy as np


def super_mutation(P, n, t, LR, UR):
    """
    Generate n mutated copies of P by adding zero-mean Gaussian noise
    with standard deviation t, then clamp to [LR, UR].

    Parameters
    ----------
    P  : np.ndarray  [1 x dim] or [dim,]  parent solution
    n  : int         number of mutated offspring to return
    t  : float       standard deviation for the zero-mean normal noise
    LR : float or array-like  lower boundary
    UR : float or array-like  upper boundary

    Returns
    -------
    P1 : np.ndarray  [n x dim]  mutated population
    """
    P = np.atleast_2d(P)
    _, dim = P.shape

    P1 = np.ones((n, dim)) * P  # broadcast P to n rows

    if np.isscalar(LR) or len(np.atleast_1d(LR)) == 1:
        # All parameters share the same boundary
        LR = float(np.atleast_1d(LR)[0])
        UR = float(np.atleast_1d(UR)[0])

        P1 = P1 + np.random.normal(0, t, size=(n, dim))  # zero-mean, std = t

        # Clamp to boundary
        P1[P1 < LR] = LR
        P1[P1 > UR] = UR

    else:
        LR = np.atleast_1d(LR)
        UR = np.atleast_1d(UR)

        P1 = P1 + np.random.normal(0, t, size=(n, dim))  # zero-mean, std = t

        for i in range(dim):
            P_ = P1[:, i]
            P_[P_ < LR[i]] = LR[i]
            P_[P_ > UR[i]] = UR[i]
            P1[:, i] = P_

    return P1