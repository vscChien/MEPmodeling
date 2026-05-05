import numpy as np


def fitness_function(y_goal, h_output):
    """
    Compute the goodness of fit: var(y - h_output) / var(y)  [i.e. 1 - R²]

    Parameters
    ----------
    y_goal   : np.ndarray  target output
    h_output : np.ndarray  model output

    Returns
    -------
    fit : float  fitness value (lower is better)
    """
    tmp = y_goal - h_output
    fit = np.var(tmp.ravel(order='F')) / np.var(y_goal.ravel(order='F'))  # 1 - R2

    # Alternative metrics (commented out, matching original):
    # fit = -1 * np.sum(np.var(y_goal) * np.log(np.var(h_output))
    #         + (1 - np.var(y_goal)) * np.log(1 - np.var(h_output))) / len(y_goal)
    # fit = np.sqrt(np.sum((y_goal - h_output) ** 2)) / len(y_goal)
    # fit = np.corrcoef(y_goal.ravel(order='F'), h_output.ravel(order='F'))[0, 1]

    return fit