"""
Optimizer.py
------------
Contains:
    make_optimizer_state()            – base state defaults
    make_hierarchical_random_state()  – Hierarchical_Random initialisation
    hierarchical_random_run()         – Hierarchical_Random main loop
    ga_run(ref, objective_function)   – GA main loop (called from ga_MEPmodel_pheno)
    ga_plot_fit(errors)               – plot error evolution

All low-level GA helper functions (including evaluation) live in ga_helpers.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from tqdm.contrib import itertools

from ga_helpers import (
    ga_population        as population,
    ga_crossover         as crossover,
    ga_mutation          as mutation,
    ga_mutationV         as mutationV,
    ga_mutation_single   as mutation_single,
    ga_selection_best    as selection_best,
    ga_selection_uniq    as selection_uniq,
    ga_gradient_search   as gradient_search,
    ga_fitness_function  as fitness_function,
    ga_evaluation        as evaluation,
    ga_gauss_newton_slow,
    ga_gradient_repair,
    ga_multi_lavenberg_regularization,
    ga_NMM_diff_A_lfm,
)


# =============================================================================
# Base state
# =============================================================================

def make_optimizer_state():
    """Return the base state dict shared by all optimizers."""
    return {
        'opt_parameters': np.zeros(2),
        'optimum':        None,
        'results_folder': 'optimization_temp',
        'save_results':   False,
    }


# =============================================================================
# Hierarchical_Random
# =============================================================================

def make_hierarchical_random_state(parameters):
    """
    Build and validate the state dict for the Hierarchical_Random optimizer.

    Parameters
    ----------
    parameters : dict
        Must contain at least: 'y' (np.ndarray), 'simulate' (callable),
        'model_parameters' (list), 'bounds' (list of [lo, hi] pairs).
        Optional keys override the defaults below.

    Returns
    -------
    state : dict
    """
    state = make_optimizer_state()
    state.update({
        't':                np.zeros(100),
        'max_iter':         10,
        'noise_term':       0.005,
        'eps':              0.01,
        'min_errors':       [],
        'optimum':          None,
        'opt_idxs':         [],
        'n_grid':           20,
        'x_out':            None,
        'simulate':         None,
        'y':                None,
        'model_parameters': None,
        'simulation_class': None,
        'bounds':           None,
    })
    state.update(parameters)
    state['parameters'] = parameters

    assert type(state['y']) == np.ndarray,        'please provide a validation function!'
    assert state['simulate'] is not None,          'please provide a model!'
    assert state['model_parameters'] is not None, 'please provide parameter names!'
    assert state['bounds'] is not None,            'please provide parameter ranges!'

    n_param = len(state['model_parameters'])
    state['n_param'] = n_param

    lower_bound = np.zeros(n_param)
    upper_bound = np.zeros(n_param)
    for i in range(n_param):
        lower_bound[i] = state['bounds'][i][0]
        upper_bound[i] = state['bounds'][i][1]
    state['lower_bound'] = lower_bound
    state['upper_bound'] = upper_bound

    state['error']          = np.ones((state['max_iter'], state['n_grid']))
    state['min_error_idxs'] = np.zeros(state['max_iter'])

    if hasattr(state['simulation_class'], 't'):
        t_shape = state['simulation_class'].t.shape[0]
        state['x_vals'] = np.zeros((state['max_iter'], state['n_grid'], t_shape))
    else:
        state['x_vals'] = np.zeros((state['max_iter'], state['n_grid']))

    state['opt_parameters'] = np.zeros(
        (state['max_iter'], state['n_grid'], lower_bound.shape[0])
    )
    return state


def hierarchical_random_run(state):
    """
    Run the Hierarchical_Random optimiser.

    Parameters
    ----------
    state : dict   as returned by make_hierarchical_random_state()

    Returns
    -------
    state : dict   updated in-place with results
    """
    from Utils import nrmse   # local import to avoid hard top-level dependency

    previous_min_error = 1
    i_count = -1

    for i, k in itertools.product(range(state['max_iter']), range(state['n_grid'])):
        if i_count < i:
            i_count += 1
            param_values = np.zeros((state['n_param'], state['n_grid']))
            keywords = state['parameters']
            for j in range(state['n_param']):
                param_values[j] = np.random.uniform(
                    state['lower_bound'][j], state['upper_bound'][j], state['n_grid']
                )

        state['opt_parameters'][i, k] = param_values[:, k]

        for l in range(state['n_param']):
            keywords[state['model_parameters'][l]] = param_values[l, k]
        keywords['y']                = state['y']
        keywords['idx']              = f'{i}_{k}'
        keywords['simulation_class'] = None

        if state['simulation_class'] is None:
            x = state['simulate'](**keywords)
        elif state['x_out'] is None:
            state['simulation_class'].__init__(parameters=keywords)
            state['simulate']()
        else:
            state['simulation_class'].__init__(parameters=keywords)
            state['simulate']()
            x = getattr(state['simulation_class'], state['x_out'])

        state['x_vals'][i, k] = x

        if hasattr(state['simulation_class'], 'error'):
            state['error'][i, k] = state['simulation_class'].error
        else:
            state['error'][i, k] = nrmse(state['y'], x)

        orig_name = state['simulation_class'].mass_model.name
        if state['save_results']:
            run_name = os.path.join(state['results_folder'], f'diw_sim_opt_hu_{i}_{k}')
            state['simulation_class'].name = run_name
            state['simulation_class'].plot_validation(save_fig=True)
            state['simulation_class'].mass_model.name = run_name
            state['simulation_class'].mass_model.plot(
                savefig=True, plot_input=True, z_limit=0.001, fname=orig_name
            )

        if k == state['n_grid'] - 1:
            min_error     = np.nanmin(state['error'][i])
            min_error_idx = (i, np.nanargmin(state['error'][i]))
            state['min_error_idxs'][i] = min_error_idx[1]
            state['min_errors'].append(min_error)
            state['opt_idxs'].append(min_error_idx)
            print('\n#########################################################################')
            print(f'error: {min_error:.5f}, at index {min_error_idx}')
            print(f'{param_values[:, min_error_idx[1]]}')
            print('#########################################################################')
            print(f'plotted results for diw_sim_opt_hu_{i}')

            if min_error < state['eps']:
                print(f'error: {min_error:.4f}')
                state['optimum'] = param_values[:, min_error_idx[1]]
                print(f'optimal values: {state["optimum"]}')
                break

            if i > 0:
                previous_min_error = np.min(np.array(state['min_errors'][:i]))
            if min_error < previous_min_error - state['noise_term']:
                print('new min error, updating parameters')
                p_new = param_values[:, min_error_idx[1]]
                state['optimum'] = p_new
                print(f'optimal values: {state["optimum"]}')
                delta = state['upper_bound'] - state['lower_bound']
                for j in range(state['n_param']):
                    state['lower_bound'][j] = max(
                        state['lower_bound'][j], p_new[j] - 0.5 * delta[j]
                    )
                    state['upper_bound'][j] = min(
                        state['upper_bound'][j], p_new[j] + 0.5 * delta[j]
                    )
            else:
                print(f'error not smaller than {previous_min_error:.4f}-{state["noise_term"]}')

    return state


# =============================================================================
# GA main loop — called directly by ga_MEPmodel_pheno
# =============================================================================

def ga_run(ref, objective_function,
           N1=60, N2=100, N3=100, tg=50,
           op=-1, verbose=0,
           solution_ini=None, plot_callback=None):
    """
    Run the genetic algorithm optimiser for the MEP model.

    Parameters
    ----------
    ref                : dict   configuration dict produced by config_model_pheno;
                                must contain keys: 'boundary', 'y0', 'resultname'
    objective_function : callable
                         objective_function(p, ref) → error, updated_ref
    N1                 : int   population size  (default 60)
    N2                 : int   crossover pairs  (default 100)
    N3                 : int   mutation pairs   (default 100)
    tg                 : int   maximum generations (default 50)
    op                 : int   -1 minimise, +1 maximise (default -1)
    verbose            : int   0 = silent per-candidate printout (default 0)
    solution_ini       : np.ndarray or None
                         [M x nParams] previously fitted solutions to seed the
                         initial population before evaluation and selection
    plot_callback      : callable or None
                         Called at the end of each generation as:
                         plot_callback(KP, KS, K, E, GA_counter, R1)
                         Use this for live plotting without coupling to a GUI.

    Returns
    -------
    p_post  : np.ndarray  [nParams,]               best parameter set
    KP_arr  : np.ndarray  [generations x nParams]  per-generation best params
    KS_arr  : np.ndarray  [generations,]           per-generation best cost
    P       : np.ndarray  [N1 x nParams]           final population
    """
    LR      = ref['boundary'][:, 0]
    UR      = ref['boundary'][:, 1]
    nParams = len(LR)

    def _function_call(parameters):
        """Return the raw error vector for a single parameter set."""
        error, _ = objective_function(parameters, ref)
        return np.atleast_1d(error).ravel()

    def _evaluation_fn(X, reference):
        """Wrap ga_evaluation so conf['evaluation_fn'] has the right signature."""
        return evaluation(X, objective_function, reference)

    conf = {
        'UR':            UR,
        'LR':            LR,
        'op':            op,
        'myfunc':        objective_function,
        'y_goal':        ref,
        'gLoop':         10,
        'gL':           -12,
        'gU':            12,
        'gT':            abs(12 - (-12)) + 1,
        'gTol':          0.01,
        'evaluation_fn': _evaluation_fn,
        'function_call': _function_call,
        'reference':     ref,
    }

    K          = []   # [avg_cost, best_cost] per generation
    KP         = []   # best solution per generation
    KS         = []   # best cost per generation
    GA_counter = []
    w          = 0    # generation index

    # ----- initialisation -----
    print('=' * 60)
    print('  INITIALIZATION')
    print(f'  Population size : {N1}   nParams : {nParams}')
    print(f'  Max generations : {tg}   op      : {"minimise" if op == -1 else "maximise"}')
    print('=' * 60)
    P = population(N1, nParams, LR, UR)
    n_ini = 0
    if solution_ini is not None and np.asarray(solution_ini).size > 0:
        ini_arr = np.atleast_2d(solution_ini)
        P = np.vstack([P, ini_arr])
        n_ini = ini_arr.shape[0]
    print(f'  Evaluating {P.shape[0]} solutions ({N1} random + {n_ini} seeded)...')
    E, R, _  = evaluation(P, objective_function, ref)
    P, E, R  = selection_best(P, E, R, N1, op)
    R1       = R[:, 0]
    print(f'  Initialization done  |  best fit = {E[0]:.6g}  |  mean fit = {E.mean():.6g}')
    print('=' * 60)
    E_crit = E[0]

    # ----- main loop -----
    # R convention throughout: [nData x n_pop]  (columns = solutions)
    import time as _time
    t_run_start = _time.time()
    j = 1
    while True:
        t_gen_start = _time.time()
        print()
        print('=' * 60)
        print(f'  GENERATION {j} / {tg}')
        print(f'  Current best fit = {E[0]:.6g}  |  mean fit = {E.mean():.6g}')
        print('=' * 60)

        # ── Step 1: gradient search on best solution ──────────────────────────
        print(f'\n[Step 1/3]  Gradient search on best solution')
        t1 = _time.time()
        Para_E_grd, E_grd, R_grd = gradient_search(
            P[0, :].reshape((1, -1)), R1.reshape((1, -1)), conf, E_crit
        )
        if op * E_grd[0] > op * E[0]:
            print(f'  ✓ improved: {E[0]:.6g} → {E_grd[0]:.6g}  ({_time.time()-t1:.1f}s)',
                  flush=True)
            P[0, :]  = Para_E_grd[0]
            E[0]     = E_grd[0]
            R[:, 0]  = R_grd[:, 0]
        else:
            print(f'  – no improvement  (best = {E[0]:.6g}, grad result = {E_grd[0]:.6g})'
                  f'  ({_time.time()-t1:.1f}s)', flush=True)

        # ── Step 2: single-parameter mutation + gradient search ───────────────
        print(f'\n[Step 2/3]  Single-parameter mutation  ({nParams} candidates)')
        t2 = _time.time()
        P_        = mutation_single(P[0, :], LR, UR)
        E_, R_, _ = evaluation(P_, objective_function, ref)
        print(f'  Mutation evaluation done  |  best candidate fit = {E_.min():.6g}'
              f'  ({_time.time()-t2:.1f}s)', flush=True)

        print(f'  Gradient search on {len(E_)} mutated candidates:')
        nData          = R_.shape[0]
        n_cands        = len(E_)
        Para_E_grd_arr = np.zeros_like(P_)
        E_grd_arr      = np.zeros(n_cands)
        R_grd_arr      = np.zeros((nData, n_cands))

        for i in range(n_cands):
            print(f'\n  [mutation candidate {i+1}/{n_cands}]  '
                  f'pre-gradient fit = {E_[i]:.6g}', flush=True)
            P_grd_i, E_grd_i, R_grd_i = gradient_search(
                P_[i, :].reshape((1, -1)), R_[:, i].reshape((1, -1)), conf, E_crit
            )
            Para_E_grd_arr[i, :] = P_grd_i[0]
            E_grd_arr[i]         = E_grd_i[0]
            R_grd_arr[:, i]      = R_grd_i[:, 0]
            delta = E_grd_arr[i] - E_[i]
            marker = '✓' if op * delta > 0 else '–'
            print(f'  {marker} candidate {i+1}/{n_cands}  ' +
                  f'post-gradient fit = {E_grd_arr[i]:.6g}  (Δ = {delta:+.4g})',
                  flush=True)

        n_improved = int(np.sum(op * E_grd_arr > op * E_))
        idx            = op * E_grd_arr > op * E_
        P_[idx, :]     = Para_E_grd_arr[idx, :]
        E_[idx]        = E_grd_arr[idx]
        R_[:, idx]     = R_grd_arr[:, idx]
        print(f'\n  Mutation+gradient done: {n_improved}/{n_cands} candidates improved'
              f'  ({_time.time()-t2:.1f}s total)', flush=True)

        P = np.vstack([P, P_])
        E = np.concatenate([E, E_])
        R = np.hstack([R, R_])

        _, E_show, _ = selection_best(P, E, R, 1, op)
        print(f'  Best after Steps 1–2: {E_show:.6g}', flush=True)

        # ── Step 3: GA operators ──────────────────────────────────────────────
        n_offspring = N1 + 2*N2 + 2*N3
        print(f'\n[Step 3/3]  GA operators')
        print(f'  {N1} mutV  +  {2*N2} crossover  +  {2*N3} mutation  =  {n_offspring} offspring')
        t3 = _time.time()
        P_mutV  = mutationV(P[:N1, :], 0.1, 0.9, LR, UR)
        P_cross = crossover(P, N2)
        P_mut   = mutation(P, N3)
        P_new   = np.vstack([P_mutV, P_cross, P_mut])

        print(f'  Evaluating {n_offspring} offspring...', flush=True)
        E_new, _, _ = evaluation(P_new, objective_function, ref)
        P = np.vstack([P, P_new])
        E = np.concatenate([E, E_new])

        P, E = selection_uniq(P, E, N1, N1, op, LR, UR)
        _, R1_2d, _ = evaluation(P[[0], :], objective_function, ref)
        R1 = R1_2d[:, 0]
        print(f'  GA operators done  ({_time.time()-t3:.1f}s)', flush=True)

        # ── Generation summary ────────────────────────────────────────────────
        avg_cost  = E.mean()
        best_cost = E[0]
        K.append([avg_cost, best_cost])
        KP.append(P[0, :].copy())
        KS.append(E[0])
        E_crit = E[0]

        ga_worked = bool(E_show > E[0])
        GA_counter.append(1 if ga_worked else 0)
        gof = fitness_function(ref['y0'].ravel(order='F'), R1)
        suc_rate = sum(GA_counter) / len(GA_counter)
        t_gen = _time.time() - t_gen_start

        print()
        print('─' * 60)
        print(f'  GENERATION {j} / {tg}  SUMMARY  ({t_gen:.1f}s)')
        print(f'  Best fit      : {KS[-1]:.6g}')
        print(f'  Mean fit      : {E.mean():.6g}')
        print(f'  R² (best)     : {gof:.4f}')
        print(f'  GA effective  : {"yes ✓" if ga_worked else "no  ✗"}'
              f'  (success rate {suc_rate:.0%}, {sum(GA_counter)}/{len(GA_counter)})')
        print(f'  Total elapsed : {_time.time()-t_run_start:.1f}s')
        print('─' * 60)

        # ----- online plot (if callback provided) -----
        if plot_callback is not None:
            plot_callback(KP, KS, K, E, GA_counter, R1)

        w += 1
        j += 1

        if j > tg:
            print(f'\n  Reached max generations ({tg}).', flush=True)
            break
        if KS[-1] < 0.01:
            print(f'\n  Early stop: fit {KS[-1]:.6g} < 0.01.', flush=True)
            break

    # ----- select best result -----
    KS_arr = np.array(KS)
    KP_arr = np.array(KP)

    if op == -1:
        idx_best = int(np.argmin(KS_arr))
    else:
        idx_best = int(np.argmax(KS_arr))

    print(f'{"minimum" if op == -1 else "maximum"}: {KS_arr[idx_best]}')

    p_post = KP_arr[-1, :]
    return p_post, KP_arr, KS_arr, P


# =============================================================================
# Utility
# =============================================================================

def ga_plot_fit(errors):
    """
    Plot the error evolution of a completed GA run.

    Parameters
    ----------
    errors : list or np.ndarray  best-cost per generation
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(np.array(errors))
    ax.set_xlabel('# iteration')
    ax.set_ylabel('fit error')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(tck.MultipleLocator())
    plt.show()