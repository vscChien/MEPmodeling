import numpy as np


def crossover(X, n):
    """
    Choose n pairs from X, to get 2*n new crossover children.

    Parameters
    ----------
    X : np.ndarray  [x x y]  population
    n : int         number of pairs of chromosomes to crossover

    Returns
    -------
    Y : np.ndarray  [2*n x y]  offspring after crossover
    """
    x, y = X.shape  # x: population size, y: parameter size
    E = np.zeros((2 * n, y))

    for i in range(n):  # crossover repeat n times
        # r: select two chromosomes for crossover
        r = np.random.randint(0, x, size=2)  # two indices in [0, x)
        while r[0] == r[1]:                  # make sure they are different
            r = np.random.randint(0, x, size=2)

        A = X[r[0], :].copy()  # chromosome 1 for crossover
        B = X[r[1], :].copy()  # chromosome 2 for crossover

        c = 1 + np.random.randint(0, y - 1)  # select cut point (1-indexed equivalent)

        D = A[c:y].copy()   # reserve
        A[c:y] = B[c:y]
        B[c:y] = D
        # Now A and B are chromosomes after crossover

        E[2 * i, :]     = A
        E[2 * i + 1, :] = B

    Y = E
    return Y