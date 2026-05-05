import numpy as np
from GA.gradient_toolbox.NMM_diff_A_lfm import NMM_diff_A_lfm
# Alternatives (uncomment to switch):
# from NMM_diff_A_lfm_pade   import NMM_diff_A_lfm_pade   as NMM_diff_A_lfm
# from NMM_diff_A_lfm_pade33 import NMM_diff_A_lfm_pade33 as NMM_diff_A_lfm
from GA.gradient_toolbox.multi_lavenberg_regulization import multi_lavenberg_regulization
from GA.gradient_toolbox.evaluation import evaluation
from GA.gradient_toolbox.selection_best import selection_best


def gauss_newton_slow(op, Para_E_test, r_test, func, y_goal,
                      reg0, reg1, steps, loop, tol, LR, UR, fit_crit):
    """
    Iterative Gauss-Newton optimisation with Levenberg-Marquardt regularisation.

    Parameters
    ----------
    op           : int        -1 → minimise, +1 → maximise
    Para_E_test  : np.ndarray [nParams x 1] or [nParams,]  initial parameters
    r_test       : np.ndarray [nData,] or [nData x 1]      initial residual
    func         : callable   objective: error, houtput = func(P, y_goal)
    y_goal       : target output
    reg0         : float      log10 of minimum regularisation (passed to MLR)
    reg1         : float      log10 of maximum regularisation (passed to MLR)
    steps        : int        number of regularisation candidates per iteration
    loop         : int        maximum number of iterations
    tol          : float      convergence tolerance (stop if improvement < tol)
    LR           : array-like lower boundary
    UR           : array-like upper boundary
    fit_crit     : not currently active (mirrors commented-out MATLAB code)

    Returns
    -------
    fit_after_g   : float      best fitness after gradient search
    Para_E_after_g: np.ndarray [nParams,]  best parameters after gradient search
    error_after_g : np.ndarray [nData,]    residual of best solution
    """
    Para_E_test = np.atleast_1d(Para_E_test).ravel(order='F')   # [nParams,]
    r_test      = np.atleast_1d(r_test).ravel(order='F')        # [nData,]

    fit_     = []           # fitness history
    Para_E_  = []           # parameter history  [iter x nParams]
    error_   = []           # residual history   [nData x iter]

    j = 0
    while j < loop:
        # ---- Jacobian ----
        J = NMM_diff_A_lfm(Para_E_test, r_test, func, y_goal)

        # ---- Candidate updates via Levenberg-Marquardt ----
        Para_E_new_group = multi_lavenberg_regulization(
            steps, reg0, reg1, Para_E_test, J, r_test, LR, UR
        )   # [steps x nParams]

        print(f'[{j + 1}/{loop}] ', end='', flush=True)

        # ---- Evaluate candidates ----
        fit_grp, error_grp, _ = evaluation(Para_E_new_group, func, y_goal)

        # ---- Select best candidate ----
        Para_E_new, fit_new_arr, error_new = selection_best(
            Para_E_new_group, fit_grp.ravel(), error_grp, 1, op
        )
        fit_new   = float(fit_new_arr.ravel()[0])
        r_test    = error_new.ravel(order='F')          # [nData,]
        Para_E_test = Para_E_new.ravel(order='F')       # [nParams,]

        print(fit_new, flush=True)

        fit_.append(fit_new)
        Para_E_.append(Para_E_test.copy())
        error_.append(r_test.copy())

        j += 1

        # ---- Convergence check ----
        if len(fit_) > 1 and op * fit_[-1] - op * fit_[-2] < tol:
            print(f'Quit: improvement < tol({tol:g})')
            break

    # ---- Return best overall or last iterate ----
    fit_arr   = np.array(fit_)            # [iter,]
    Para_E_mat = np.vstack(Para_E_)       # [iter x nParams]
    error_mat  = np.column_stack(error_)  # [nData x iter]

    if j == loop:
        # Loop ran to completion — pick the best across all iterations
        YY1, fit_after_arr, YY3 = selection_best(
            Para_E_mat, fit_arr, error_mat, 1, op
        )
        Para_E_after_g = YY1.ravel(order='F')
        fit_after_g    = float(fit_after_arr.ravel()[0])
        error_after_g  = YY3.ravel(order='F')
    else:
        # Converged early — use the last iterate
        Para_E_after_g = Para_E_test
        fit_after_g    = fit_new
        error_after_g  = r_test

    return fit_after_g, Para_E_after_g, error_after_g