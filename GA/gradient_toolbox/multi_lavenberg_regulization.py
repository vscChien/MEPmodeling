import numpy as np
from GA.gradient_toolbox.gradient_repair import gradient_repair


def multi_lavenberg_regulization(n, reg0, reg1, Para_E, J, h_output, LR, UR):
    """
    Generate n candidate parameter updates via Levenberg-Marquardt regularisation,
    each with a different regularisation strength on a log scale [reg0, reg1].

    Parameters
    ----------
    n       : int        number of candidate solutions to return
    reg0    : float      log10 of minimum regularisation value
    reg1    : float      log10 of maximum regularisation value
    Para_E  : np.ndarray [nParams x 1] or [nParams,]  current parameter vector
    J       : np.ndarray [nData x nParams]  Jacobian matrix
    h_output: np.ndarray [nData,] or [nData x 1]  current residual (f(x) - y_goal)
    LR      : float or array-like  lower boundary
    UR      : float or array-like  upper boundary

    Returns
    -------
    Y : np.ndarray  [n x nParams]  updated candidate solutions
    """
    Para_E = Para_E.ravel(order='F')   # ensure 1-D, Fortran ordering to match MATLAB (:)
    nParams = len(Para_E)

    Y = np.zeros((n, nParams))

    reg = 10.0 ** np.linspace(reg0, reg1, n)

    h_output_flat = h_output.ravel(order='F')   # flatten as MATLAB column-major

    for i in range(n):
        # Levenberg-Marquardt regularisation step
        try:
            D = np.linalg.pinv(J.T @ J + reg[i] * np.eye(nParams))
        except Exception:
            return Y

        d = -D @ J.T @ h_output_flat   # parameter update (f(x) - y_goal convention)

        if np.any(np.isnan(d)):
            # equivalent to MATLAB's keyboard — raise to allow inspection
            raise RuntimeError(
                f'NaN detected in gradient update at regularisation step {i}. '
                f'reg={reg[i]:.4g}, check J and h_output.'
            )

        Para_E_new = Para_E + d

        # Clamp to boundary
        Y[i, :] = gradient_repair(Para_E_new, LR, UR)

    return Y