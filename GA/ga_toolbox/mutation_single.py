import numpy as np
from GA.ga_toolbox.population import population


def mutation_single(solution, LR, UR):
    """
    Single-parameter mutation on the best solution.

    Parameters
    ----------
    solution : np.ndarray  [1 x nParams] or [nParams,]  best solution
    LR       : array-like  lower boundary
    UR       : array-like  upper boundary

    Returns
    -------
    P : np.ndarray  [nParams x nParams]  nParams mutated solutions,
        each differing from `solution` in exactly one parameter position.
    """
    solution = np.atleast_1d(solution).ravel(order='F')
    nParams  = len(solution)

    P = np.tile(solution, (nParams, 1))           # replicate solution nParams times

    # Replace the diagonal entries with fresh random values (one per row)
    diag_mask = np.eye(nParams, dtype=bool)
    P[diag_mask] = population(1, nParams, LR, UR).ravel(order='F')

    return P