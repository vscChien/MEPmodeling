import numpy as np


def super_mutation_wild(P, n, t, LR, UR):
    """
    Generate n mutated copies of P by adding zero-mean Gaussian noise whose
    standard deviation is scaled by the parameter range (t * (UR - LR)),
    then clamp to [LR, UR].

    Parameters
    ----------
    P  : np.ndarray  [1 x dim] or [dim,]  parent solution
    n  : int         number of mutated offspring to return
    t  : float       fraction of the parameter range used as standard deviation
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

        # Noise std scaled by the full range
        P1 = P1 + np.random.normal(0, t * (UR - LR), size=(n, dim))

        # Clamp to boundary
        P1[P1 < LR] = LR
        P1[P1 > UR] = UR

    else:
        LR = np.atleast_1d(LR)
        UR = np.atleast_1d(UR)

        for i in range(dim):
            # Per-parameter noise std scaled by that parameter's range
            P1[:, i] = P1[:, i] + np.random.normal(0, t * (UR[i] - LR[i]), size=n)

            P_ = P1[:, i]
            P_[P_ < LR[i]] = LR[i]
            P_[P_ > UR[i]] = UR[i]
            P1[:, i] = P_

    return P1