import numpy as np
from GA.ga_toolbox.population import population


def mutationV(P, lowchance, highchance, LR, UR):
    """
    Mutate all positions by chance, where mutation probability increases
    linearly from best (first row) to worst (last row) of the ranked population.

    Parameters
    ----------
    P          : np.ndarray  [n_pop x n_parameter]  ranked population,
                             best (lowest cost) individual at the first row
    lowchance  : float  mutation probability for the best individual
    highchance : float  mutation probability for the worst individual
    LR         : array-like  lower boundary
    UR         : array-like  upper boundary

    Returns
    -------
    mutateP : np.ndarray  [n_pop x n_parameter]  mutated population
    """
    mutateP = P.copy()                                          # [n_pop x n_parameter]

    mutateChance = np.linspace(lowchance, highchance, mutateP.shape[0])  # per-row probability

    mask = np.random.rand(*mutateP.shape) < mutateChance[:, np.newaxis]  # broadcast

    tmp = population(mutateP.shape[0], mutateP.shape[1], LR, UR)
    mutateP[mask] = tmp[mask]

    return mutateP