import numpy as np
from GA.ga_toolbox.population import population


def selection_uniq(P1, B, p, r, op, LR, UR):
    """
    Select p unique solutions, keeping the top-r best and randomly sampling
    the rest from the remaining pool.

    Parameters
    ----------
    P1 : np.ndarray  [n_pop x n_parameter]  population
    B  : np.ndarray  [n_pop,]               fitness values
    p  : int         desired output population size
    r  : int         number of top solutions guaranteed to be kept
    op : int         -1 → select minimum, +1 → select maximum
    LR : array-like  lower boundary
    UR : array-like  upper boundary

    Returns
    -------
    YY1 : np.ndarray  [p x n_parameter]  selected population
    YY2 : np.ndarray  [p,]               fitness of YY1
    """
    # Turn minimisation into maximisation if necessary
    B = op * B.copy()
    dim = P1.shape[1]

    # Remove inf entries
    index = np.isinf(B)
    B  = B[~index]
    P1 = P1[~index, :]

    # Remove nan entries
    index = np.isnan(B)
    B  = B[~index]
    P1 = P1[~index, :]

    # Keep only unique rows
    P1, ip = np.unique(P1, axis=0, return_index=True)
    B = B[ip]

    # Sort from high to low — best first
    index = np.argsort(B)[::-1]
    E  = B[index]
    P1 = P1[index, :]

    # Recheck length; pad with random solutions if fewer than p remain
    len_counter = len(E)
    if len_counter < p:
        new_len = p - len_counter
        P1 = np.vstack([P1, population(new_len, dim, LR, UR)])
        E  = np.concatenate([E, np.full(new_len, np.nan)])

    # Select best r, then randomly sample the rest
    if len_counter > p:
        P1_best = P1[:r, :]
        E_best  = E[:r]

        P2 = P1[r:, :]
        E2 = E[r:]

        index = np.random.permutation(len(E2))
        P2 = P2[index, :]
        E2 = E2[index]

        P2 = P2[:p - r, :]
        E2 = E2[:p - r]

        P1 = np.vstack([P1_best, P2])
        E  = np.concatenate([E_best, E2])

    YY1 = P1[:p, :]
    YY2 = op * E[:p]   # turn back to original sign

    return YY1, YY2