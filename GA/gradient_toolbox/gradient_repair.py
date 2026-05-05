import numpy as np
from GA.ga_toolbox.population import population

def gradient_repair(Para_E, LR, UR):
    """
    Clamp / repair a parameter vector so that every value stays within [LR, UR].

    Parameters
    ----------
    Para_E : np.ndarray  [nParams,] or [nParams x 1]  parameter vector to repair
    LR     : float or array-like  lower boundary
    UR     : float or array-like  upper boundary

    Returns
    -------
    Para_E_new : np.ndarray  repaired parameter vector (same shape as input)
    """
    Para_E = Para_E.copy()
    LR_arr = np.atleast_1d(LR)
    UR_arr = np.atleast_1d(UR)

    if len(LR_arr) == 1:
        # All parameters share the same constraint
        LR_val = float(LR_arr[0])
        UR_val = float(UR_arr[0])

        mask_hi = Para_E > UR_val
        if np.any(mask_hi):
            Para_E[mask_hi] = population(1, int(np.sum(mask_hi)), LR_val, UR_val).ravel()

        mask_lo = Para_E < LR_val
        if np.any(mask_lo):
            Para_E[mask_lo] = population(1, int(np.sum(mask_lo)), LR_val, UR_val).ravel()

    else:
        for i in range(len(Para_E)):
            if Para_E[i] > UR_arr[i]:
                Para_E[i] = UR_arr[i]
            elif Para_E[i] < LR_arr[i]:
                Para_E[i] = LR_arr[i]

    Para_E_new = Para_E
    return Para_E_new