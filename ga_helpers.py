"""
ga_helpers.py
-------------
Standalone GA helper functions (no state dict dependency).
Imported by Optimizer.py and available for direct use.
"""

import numpy as np
import time as _time


def ga_population(N, nParams, LR, UR):
    """
    Generate N random solutions uniformly within [LR, UR].

    Parameters
    ----------
    N       : int        number of solutions
    nParams : int        number of parameters
    LR      : array-like [nParams,]  lower boundary
    UR      : array-like [nParams,]  upper boundary

    Returns
    -------
    P : np.ndarray  [N x nParams]
    """
    P = np.zeros((N, nParams))
    for i in range(nParams):
        P[:, i] = (UR[i] - LR[i]) * np.random.rand(N) + LR[i]
    return P


def ga_fitness_function(y, h_out):
    """
    Compute fitness as var(y - h_out) / var(y).

    Parameters
    ----------
    y     : np.ndarray  goal output
    h_out : np.ndarray  model output

    Returns
    -------
    fit : float
    """
    difference = y - h_out
    fit = np.var(difference) / np.var(y)
    return fit


def ga_crossover(X, n):
    """
    Perform n single-point crossover operations on population X.

    Parameters
    ----------
    X : np.ndarray  [population_size x parameter_size]
    n : int         number of crossover pairs

    Returns
    -------
    E : np.ndarray  [2*n x parameter_size]
    """
    x, y = X.shape
    E = np.zeros((2 * n, y))

    for i in range(n):
        r = np.random.choice(x, size=2, replace=False)
        A = X[r[0]].copy()
        B = X[r[1]].copy()
        c = np.random.randint(1, y)
        A_back_part = A[c:].copy()
        A[c:] = B[c:]
        B[c:] = A_back_part
        E[2 * i]     = A
        E[2 * i + 1] = B

    return E


def ga_mutation(X, n):
    """
    Perform single-gene swap mutation on n randomly selected chromosome pairs.

    Parameters
    ----------
    X : np.ndarray  [x x y]  population
    n : int         number of pairs to mutate

    Returns
    -------
    E : np.ndarray  [2*n x y]
    """
    x, y = X.shape
    E = np.zeros((2 * n, y))

    for i in range(n):
        r = np.random.randint(0, x, size=2)
        while r[0] == r[1]:
            r = np.random.randint(0, x, size=2)
        A = X[r[0]].copy()
        B = X[r[1]].copy()
        c = np.random.randint(0, y)
        A[c], B[c] = B[c], A[c]
        E[2 * i]     = A
        E[2 * i + 1] = B

    return E


def ga_mutationV(P, lowchance, highchance, LR, UR):
    """
    Mutate all positions by chance; probability increases linearly from best
    to worst individual.

    Parameters
    ----------
    P          : np.ndarray  [n_pop x n_param]  ranked population (best first)
    lowchance  : float  mutation probability for the best individual
    highchance : float  mutation probability for the worst individual
    LR, UR     : array-like  lower / upper boundaries

    Returns
    -------
    mutateP : np.ndarray  [n_pop x n_param]
    """
    mutateP = P.copy()
    n_pop, n_param = mutateP.shape
    mutateChance = np.linspace(lowchance, highchance, n_pop)
    mask = np.random.rand(n_pop, n_param) < mutateChance[:, np.newaxis]
    mutation_vals = ga_population(n_pop, n_param, LR, UR)
    mutateP[mask] = mutation_vals[mask]
    return mutateP


def ga_mutation_single(solution, LR, UR):
    """
    Single-parameter mutation on the best solution.

    Parameters
    ----------
    solution : np.ndarray  [nParams,]  best solution
    LR, UR   : array-like  lower / upper boundaries

    Returns
    -------
    P : np.ndarray  [nParams x nParams]  nParams mutated solutions
    """
    solution = np.array(solution)
    nParams = solution.size
    P = np.tile(solution, (nParams, 1))
    mutation_values = ga_population(1, nParams, LR, UR).flatten()
    np.fill_diagonal(P, mutation_values)
    return P


def ga_selection_best(P, E, R, n_out, op=-1):
    """
    Return the n_out best solutions sorted by fitness.

    Parameters
    ----------
    P     : np.ndarray  [n_pop x n_parameter]
    E     : np.ndarray  [n_pop,]   fitness values
    R     : np.ndarray             residuals [nData x n_pop]  (columns = solutions)
    n_out : int         number of solutions to return
    op    : int         -1 → select minimum, +1 → select maximum

    Returns
    -------
    P_out : np.ndarray  [n_out x n_parameter]  (squeezed to [nParams,] when n_out==1)
    E_out : np.ndarray  [n_out,]               (squeezed to scalar float when n_out==1)
    R_out : np.ndarray  [nData x n_out]        (squeezed to [nData,] when n_out==1)
    """
    E = np.atleast_1d(E).ravel()
    P = np.atleast_2d(P)

    E_signed = op * E
    index    = np.argsort(E_signed)[::-1]   # descending: best first

    P_sorted = P[index]
    E_sorted = op * E_signed[index]         # restore original sign

    # Normalise R to [nData x n_pop] regardless of how it was passed in
    R = np.atleast_2d(R)
    if R.shape[1] == len(E):
        # already [nData x n_pop]
        R_sorted = R[:, index]
    elif R.shape[0] == len(E):
        # [n_pop x nData] — transpose first
        R_sorted = R[index].T
    else:
        # 1-D or ambiguous — keep as-is indexed by row
        R_sorted = R[index]

    P_out = P_sorted[:n_out]
    E_out = E_sorted[:n_out]
    R_out = R_sorted[:, :n_out] if R_sorted.ndim == 2 else R_sorted[:n_out]

    # Squeeze when only one solution is requested so callers get plain values
    if n_out == 1:
        P_out = P_out[0]              # [nParams,]
        E_out = float(E_out[0])       # scalar
        R_out = R_out[:, 0] if R_out.ndim == 2 else R_out[0]  # [nData,]

    return P_out, E_out, R_out


def ga_selection_uniq(P1, E, p, r, op, LR, UR):
    """
    Select p unique solutions, keeping the top-r best and randomly sampling
    the rest from the remaining pool.

    Parameters
    ----------
    P1       : np.ndarray  [n_pop x n_parameter]
    E        : np.ndarray  [n_pop,]
    p        : int         desired output size
    r        : int         number of top individuals guaranteed to be kept
    op       : int         -1 → minimise, +1 → maximise
    LR, UR   : array-like  lower / upper boundaries

    Returns
    -------
    P1_new : np.ndarray  [p x n_parameter]
    E_new  : np.ndarray  [p,]
    """
    E   = op * E
    dim = P1.shape[1]

    # Remove inf and nan entries
    valid_mask = ~np.isinf(E) & ~np.isnan(E)
    E  = E[valid_mask]
    P1 = P1[valid_mask]

    # Keep unique rows
    P1_unique, idx = np.unique(P1, axis=0, return_index=True)
    E  = E[idx]
    P1 = P1_unique

    # Sort descending (best first)
    sorted_idx = np.argsort(E)[::-1]
    P1       = P1[sorted_idx]
    E_sorted = E[sorted_idx]

    n_E = len(E_sorted)
    if n_E < p:
        n_new    = p - n_E
        P1       = np.vstack([P1, ga_population(n_new, dim, LR, UR)])
        E_sorted = np.concatenate([E_sorted, np.full(n_new, np.nan)])

    if n_E > p:
        P1_best  = P1[:r]
        E_best   = E_sorted[:r]
        P2       = P1[r:]
        E2       = E_sorted[r:]
        rand_idx = np.random.permutation(len(E2))
        P2       = P2[rand_idx][:p - r]
        E2       = E2[rand_idx][:p - r]
        P1       = np.vstack([P1_best, P2])
        E_sorted = np.concatenate([E_best, E2])

    P1_new = P1[:p]
    E_new  = op * E_sorted[:p]   # restore original sign

    return P1_new, E_new


def ga_gradient_repair(Para_E, LR, UR):
    """
    Clamp each parameter to its [LR, UR] boundary.

    Parameters
    ----------
    Para_E : np.ndarray  [nParams,]
    LR, UR : array-like

    Returns
    -------
    Para_E : np.ndarray  (modified in place and returned)
    """
    LR = np.atleast_1d(LR)
    UR = np.atleast_1d(UR)

    if len(UR) == 1:
        upper_cut = Para_E > UR[0]
        under_cut = Para_E < LR[0]
        Para_E[upper_cut] = ga_population(1, int(np.sum(upper_cut)), LR, UR).ravel()
        Para_E[under_cut] = ga_population(1, int(np.sum(under_cut)), LR, UR).ravel()
    else:
        for i in range(Para_E.shape[0]):
            if Para_E[i] > UR[i]:
                Para_E[i] = UR[i]
            elif Para_E[i] < LR[i]:
                Para_E[i] = LR[i]
    return Para_E


def ga_multi_lavenberg_regularization(n, reg0, reg1, Para_E, J, h_output, LR, UR):
    """
    Generate n candidate parameter updates via Levenberg-Marquardt regularisation.

    Parameters
    ----------
    n        : int    number of candidates
    reg0     : float  log10 of minimum regularisation
    reg1     : float  log10 of maximum regularisation
    Para_E   : np.ndarray  [nParams,]  current parameters
    J        : np.ndarray  [nData x nParams]  Jacobian
    h_output : np.ndarray or scalar  current residual
    LR, UR   : array-like  lower / upper boundaries

    Returns
    -------
    Y : np.ndarray  [n x nParams]
    """
    Y   = np.zeros((n, Para_E.shape[0]))
    reg = 10 ** np.linspace(reg0, reg1, n)

    if isinstance(h_output, (int, float)):
        h_output = np.array([h_output])

    for i in range(reg.shape[0]):
        try:
            D = np.linalg.pinv(J.T @ J + reg[i] * np.eye(Para_E.shape[0]))
        except Exception:
            return Y
        if isinstance(h_output, (int, float)):
            d = -D @ J.T * h_output
        else:
            d = -D @ J.T @ h_output
        if np.isnan(d).any():
            continue

        Para_E_new = Para_E + d
        Y[i, :]    = ga_gradient_repair(Para_E_new, LR, UR)

    return Y


def ga_NMM_diff_A_lfm(parameter, h_output, function_call):
    """
    Compute the Jacobian matrix using forward finite differences.

    Parameters
    ----------
    parameter     : np.ndarray  [nParams,]
    h_output      : np.ndarray  reference output at `parameter` (shape [nData,])
    function_call : callable    function_call(parameters) → np.ndarray [nData,]

    Returns
    -------
    j : np.ndarray  [nData x nParams]
    """
    h = 1e-6
    parameter_pert = parameter + h
    p_shape = parameter.shape[0]

    h_output = np.atleast_1d(h_output).ravel()
    h_shape  = len(h_output)          # always derived from the reference output

    j = np.zeros((h_shape, p_shape))

    for i in range(p_shape):
        parameter_update = parameter.copy()
        parameter_update[i] = parameter_pert[i]
        h_output_new = np.atleast_1d(function_call(parameter_update)).ravel()
        j[:, i] = (h_output_new - h_output) / h

    j[np.isnan(j)] = 0
    j[np.isinf(j)] = 0
    return j



def ga_evaluation(X, objective_function, reference):
    """
    Evaluate the objective function for every solution in X.

    Parameters
    ----------
    X                  : np.ndarray  [n_pop x nParams]
    objective_function : callable    objective_function(p, reference) -> error, output
    reference          : target reference passed to objective_function

    Returns
    -------
    fits   : np.ndarray  [n_pop,]  sum-squared error per solution
    errors : np.ndarray  [nData x n_pop]  raw error vectors
    Houtput : list  model outputs per solution
    """
    n_pop = X.shape[0]
    fits    = np.zeros(n_pop)
    errors  = None   # built on first iteration once error shape is known
    Houtput = [None] * n_pop

    for j in range(n_pop):
        t0 = _time.time()
        error, houtput = objective_function(X[j], reference)
        elapsed = _time.time() - t0
        error_flat = np.atleast_1d(error).ravel()
        fit = np.sum(error_flat ** 2)

        if errors is None:
            errors = np.zeros((len(error_flat), n_pop))
        errors[:, j] = error_flat
        fits[j]      = fit
        Houtput[j]   = houtput
        print(f'    eval [{j+1:3d}/{n_pop}]  fit = {fit:.6g}  ({elapsed:.2f}s)',
              flush=True)

    return fits, errors, Houtput

def ga_gauss_newton_slow(op, Para_E_test, r_test, reg0, reg1, steps, loop, tol,
                         LR, UR, fit_crit, evaluation_fn, function_call, reference):
    """
    Iterative Gauss-Newton optimisation with Levenberg-Marquardt regularisation.

    Parameters
    ----------
    op            : int        -1 → minimise, +1 → maximise
    Para_E_test   : np.ndarray [nParams,]
    r_test        : np.ndarray or scalar  initial residual
    reg0, reg1    : float      regularisation log-range
    steps         : int        number of regularisation candidates per iteration
    loop          : int        maximum iterations
    tol           : float      convergence tolerance
    LR, UR        : array-like lower / upper boundaries
    fit_crit      : stopping criterion (reserved, mirrors original)
    evaluation_fn : callable   evaluation_fn(X, reference) → fits, errors, h_outs
    function_call : callable   function_call(parameters) → h_out
    reference     : target reference passed to evaluation_fn

    Returns
    -------
    fit_after_g    : float
    Para_E_after_g : np.ndarray  [nParams,]
    error_after_g  : np.ndarray
    """
    j       = 1
    fit_    = []
    Para_E_ = []
    error_  = []

    Para_E_test = np.atleast_1d(Para_E_test).ravel()
    r_test      = np.atleast_1d(r_test).ravel()

    print(f'  Gradient descent  (max {loop} steps, tol={tol:g})', flush=True)
    t_gd_start = _time.time()
    while j <= loop:
        print(f'    step [{j:3d}/{loop}]  computing Jacobian...', end=' ', flush=True)
        t_step = _time.time()
        J = ga_NMM_diff_A_lfm(Para_E_test, r_test, function_call)
        Para_E_new_group = ga_multi_lavenberg_regularization(
            steps, reg0, reg1, Para_E_test, J, r_test, LR, UR
        )

        fit_grp, error_grp, hout_group = evaluation_fn(Para_E_new_group, reference)
        # Select best candidate: rank by fit_grp, take matching error column
        Para_E_new, fit_new, r_new = ga_selection_best(
            Para_E_new_group, fit_grp, error_grp, 1, op
        )
        r_new = np.atleast_1d(r_new).ravel()

        r_test      = r_new
        Para_E_test = np.atleast_1d(Para_E_new).ravel()

        improvement = float(op * (fit_new - fit_[-1])) if fit_ else float('nan')
        fit_.append(fit_new)
        Para_E_.append(Para_E_test.copy())
        error_.append(r_test.copy())

        elapsed_step = _time.time() - t_step
        if len(fit_) == 1:
            print(f'fit = {fit_new:.6g}  ({elapsed_step:.2f}s)', flush=True)
        else:
            print(f'fit = {fit_new:.6g}  Δ = {improvement:+.4g}  ({elapsed_step:.2f}s)',
                  flush=True)
        j += 1

        if len(fit_) > 1 and op * (fit_[-1] - fit_[-2]) < tol:
            print(f'    → converged at step {j-1}: improvement {improvement:+.4g} < tol({tol:g})',
                  flush=True)
            break

    elapsed_gd = _time.time() - t_gd_start
    best_gd = float(min(fit_) if op == -1 else max(fit_))
    print(f'  Gradient descent done: {j-1} steps, best fit = {best_gd:.6g}, '
          f'total time = {elapsed_gd:.1f}s', flush=True)

    if j == loop:
        Para_E_after_g, fit_after_g, error_after_g = ga_selection_best(
            np.array(Para_E_), np.array(fit_), np.array(error_), 1, op
        )
        # ga_selection_best with n_out=1 already squeezes to 1-D / scalar
        Para_E_after_g = np.atleast_1d(Para_E_after_g).ravel()
        error_after_g  = np.atleast_1d(error_after_g).ravel()
    else:
        Para_E_after_g = np.atleast_1d(Para_E_test).ravel()
        fit_after_g    = float(fit_new)
        error_after_g  = np.atleast_1d(r_test).ravel()

    return fit_after_g, Para_E_after_g, error_after_g


def ga_gradient_search(P, r, conf, stop_crit, verbose=0):
    """
    Run Gauss-Newton gradient search on one or more solutions.

    Parameters
    ----------
    P         : np.ndarray  [N x nParams] or [nParams,]
    r         : np.ndarray or scalar  residual(s)
    conf      : dict with keys:
                    'op', 'gL', 'gU', 'gT', 'gLoop', 'gTol', 'LR', 'UR',
                    'evaluation_fn'  – callable(X, reference) → fits, errors, h_outs
                    'function_call'  – callable(parameters) → h_out
                    'reference'      – target reference passed to evaluation_fn
    stop_crit : stopping criterion passed through to ga_gauss_newton_slow
    verbose   : int   0 = silent

    Returns
    -------
    P_post   : np.ndarray  [N x nParams]
    fit_post : np.ndarray  [N,]
    r_post   : np.ndarray  [N x ...]
    """
    evaluation_fn = conf['evaluation_fn']
    function_call = conf['function_call']
    reference     = conf['reference']
    if P.ndim < 2:
        P = P[np.newaxis, :]
    if isinstance(r, (int, float)):
        r = np.array([[r]])
    elif r.ndim < 2:
        r = r[np.newaxis, :]

    N, nParams = P.shape
    fit_post = np.zeros(N)
    P_post   = np.zeros((N, nParams))
    r_post   = None   # allocated on first iteration once true error length is known

    for i in range(N):
        print(f'  [gradient search  candidate {i+1}/{N}]', flush=True)
        t_cand = _time.time()
        fit_i, P_i, r_i = ga_gauss_newton_slow(
            conf['op'],
            P[i],
            r[i],
            conf['gL'],
            conf['gU'],
            conf['gT'],
            conf['gLoop'],
            conf['gTol'],
            conf['LR'],
            conf['UR'],
            stop_crit,
            evaluation_fn,
            function_call,
            reference,
        )
        r_i = np.atleast_1d(r_i).ravel()
        if r_post is None:
            r_post = np.zeros((N, len(r_i)))
        fit_post[i]  = fit_i
        P_post[i]    = P_i
        r_post[i]    = r_i
        print(f'  [gradient search  candidate {i+1}/{N}  final fit = {fit_i:.6g}  ' +
              f'time = {_time.time()-t_cand:.1f}s]', flush=True)

    # Transpose so output is [nData x N] — same convention as ga_evaluation
    return P_post, fit_post, r_post.T