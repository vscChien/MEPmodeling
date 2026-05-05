import numpy as np
from GA.gradient_toolbox.evaluation import evaluation
from GA.gradient_toolbox.gauss_newton_slow import gauss_newton_slow


def run_gradient(P, target, boundary, conf=None):
    """
    High-level wrapper: evaluate initial solutions then run Gauss-Newton
    gradient search.

    Parameters
    ----------
    P        : np.ndarray  [N x nParams]  initial population
    target   : target output (y_goal)
    boundary : np.ndarray  [nParams x 2]  columns are [lower, upper] bounds
    conf     : dict or None
               If None, defaults are used (matching the MATLAB nargin<4 branch).
               Expected keys when provided:
                   'op', 'y_goal', 'myfunc', 'LR', 'UR',
                   'gLoop', 'gL', 'gU', 'gT', 'gTol'

    Returns
    -------
    P_post : np.ndarray  [nParams,]  best solution after gradient search
    """
    if conf is None:
        from ga_MEPmodel_pheno import objective_function   # assumed defined elsewhere

        conf = {}
        conf['op']     = -1                           # -1: find global minimum
        conf['y_goal'] = target
        conf['myfunc'] = objective_function
        conf['LR']     = boundary[:, 0]
        conf['UR']     = boundary[:, 1]
        conf['gLoop']  = 64
        conf['gL']     = -12
        conf['gU']     = 12
        conf['gT']     = abs(conf['gU'] - conf['gL']) + 1
        conf['gTol']   = 0.05

    # ---- Evaluate initial population ----
    Para_E = P
    fit, error, _ = evaluation(Para_E, conf['myfunc'], conf['y_goal'])
    print('fit: ' + str(fit))   # fit is sumsquare(error)

    # ---- Gradient search ----
    print('gradient search....')
    fit_new, P_post, error_new = gauss_newton_slow(
        conf['op'],
        Para_E.T,           # [nParams x N] — matches MATLAB Para_E'
        error,              # [nData x N]
        conf['myfunc'],
        conf['y_goal'],
        conf['gL'],
        conf['gU'],
        conf['gT'],
        conf['gLoop'],
        conf['gTol'],
        conf['LR'],
        conf['UR'],
        error,              # fit_crit = error (mirrors MATLAB call)
    )

    return P_post