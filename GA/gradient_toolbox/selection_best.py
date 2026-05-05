import numpy as np


def selection_best(P, E, R, p, op):
    """
    Select the top-p solutions by fitness.

    Parameters
    ----------
    P  : np.ndarray  [n_pop x n_parameter]  population
    E  : np.ndarray  [n_pop,]               fitness values
    R  : np.ndarray  [n_data_sample x n_pop] residuals (y - h_output)
    p  : int         number of solutions to return
    op : int         -1 → select minimum fitness, +1 → select maximum fitness

    Returns
    -------
    YY1 : np.ndarray  [p x n_parameter]      selected population
    YY2 : np.ndarray  [p,]                   fitness of YY1
    YY3 : np.ndarray  [n_data_sample x p]    residuals of YY1
    """
    E = np.atleast_1d(E).ravel(order='F').copy()
    R = np.atleast_2d(R)

    # Turn minimisation into maximisation if necessary
    E = op * E

    # Sort from high to low — best first
    index = np.argsort(E)[::-1]
    print(np.shape(E))
    E = E[index]
    P = P[index, :]
    R = R[:, index]

    YY1 = P[:p, :]
    YY2 = op * E[:p]   # turn back to original sign
    YY3 = R[:, :p]

    return YY1, YY2, YY3