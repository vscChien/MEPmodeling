import numpy as np


def mutation(X, n):
    """
    Single-point swap mutation between n randomly selected chromosome pairs.

    Parameters
    ----------
    X : np.ndarray  [x x y]  population
    n : int         number of pairs of chromosomes to mutate

    Returns
    -------
    Y : np.ndarray  [2*n x y]  mutated offspring
    """
    x, y = X.shape
    E = np.zeros((2 * n, y))

    for i in range(n):
        r = np.random.randint(0, x, size=2)  # select two chromosomes for mutation
        while r[0] == r[1]:
            r = np.random.randint(0, x, size=2)

        A = X[r[0], :].copy()  # chromosome 1 for mutation
        B = X[r[1], :].copy()  # chromosome 2 for mutation

        c = np.random.randint(0, y)  # select cut point (0-indexed)

        D    = A[c]   # reserve
        A[c] = B[c]   # exchange 1 parameter for mutation
        B[c] = D

        E[2 * i, :]     = A
        E[2 * i + 1, :] = B

    Y = E
    return Y