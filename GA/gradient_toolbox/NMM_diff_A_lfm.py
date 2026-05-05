import numpy as np


def NMM_diff_A_lfm(para, houtput, myfunc, y_goal):
    """
    Estimate the Jacobian of a black-box function using forward finite differences.

    Approximation:
        df/dp_i ≈ ( f(p + h·eᵢ) − f(p) ) / h

    Parameters
    ----------
    para    : np.ndarray  [nParams,] or [nParams x 1]  current parameter vector
    houtput : np.ndarray  [T,] or [T x 1]  function output at `para` (reference)
    myfunc  : callable    black-box function: error = myfunc(para, y_goal)
    y_goal  : target output passed through to myfunc

    Returns
    -------
    j : np.ndarray  [T x nParams]  Jacobian matrix
    """
    h = 1e-6

    para = para.ravel(order='F')   # ensure 1-D, Fortran ordering to match MATLAB (:)
    houtput_flat = houtput.ravel(order='F')

    para1 = para + h   # perturbed parameter vector (all elements shifted; only one used at a time)

    para_save = para.copy()
    T = len(houtput_flat)
    nParams = len(para)

    f = np.zeros((T, nParams))

    for i in range(nParams):
        para_1 = para_save.copy()
        para_1[i] = para1[i]                         # perturb only parameter i

        houtput_new = myfunc(para_1, y_goal)
        houtput_new_flat = np.asarray(houtput_new).ravel(order='F')

        f[:, i] = (houtput_new_flat - houtput_flat) / h

    j = f   # [T x nParams]

    j[np.isnan(j)] = 0.0
    j[np.isinf(j)] = 0.0

    return j