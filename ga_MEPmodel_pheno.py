"""
Phenomenological model

Example usage:
    from ga_MEPmodel_pheno import ga_MEPmodel_pheno
    ga_MEPmodel_pheno(subj=1, reRun=0)

    subj  : subject 1~10
    reRun : 0 – load fitted result and plot simulated MEP
            1 – re-run model fitting (backs up previous fitted result)
"""

import os
import sys
import h5py
import shutil
import numpy as np
import scipy.io
from datetime import datetime
import matplotlib.pyplot as plt
from load_h5 import load_h5_to_dict
from config_model_pheno import config_model_pheno
from MEPmodel_pheno import MEPmodel_pheno
# from GA.ga_toolbox.population        import population
# from GA.ga_toolbox.evaluation        import evaluation
# from GA.ga_toolbox.selection_best    import selection_best
# from GA.ga_toolbox.selection_uniq    import selection_uniq
# from GA.ga_toolbox.crossover         import crossover
# from GA.ga_toolbox.mutation          import mutation
# from GA.ga_toolbox.mutationV         import mutationV
# from GA.ga_toolbox.mutation_single   import mutation_single
# from GA.ga_toolbox.fitness_function  import fitness_function
# from GA.gradient_toolbox.gradient_search import gradient_search


def ga_MEPmodel_pheno(subj, reRun=0):
    root = os.getcwd()

    # ----- model setting -----
    ref = config_model_pheno(subj)

    # ----- run GA -----
    result_path = os.path.join(root, ref['resultname'])
    if os.path.isfile(result_path) and not reRun:
        print(f'Use fitted result: \n{ref["resultname"]}')
        with h5py.File(result_path, 'r') as f:
            tmp = load_h5_to_dict(f)
        # Flattening to ensure it's a 1D array as expected in Python
        p_post = tmp['p_post'].flatten()
    else:
        p_post = run_ga(ref)

    # ----- show result -----
    plotOn = 1
    MEPmodel_pheno(p_post, ref, plotOn)


# ==========================================================================
def objective_function(p, ref):
    _, ref = MEPmodel_pheno(p, ref)
    error  = ref['error']
    return error, ref


# ==========================================================================
def run_ga(ref):
    root = os.path.dirname(os.path.abspath(__file__))

    myfunc  = objective_function
    op      = -1        # -1: find minimum, +1: find maximum

    LR      = ref['boundary'][:, 0]
    UR      = ref['boundary'][:, 1]
    nParams = len(LR)

    conf = {
        'UR':     UR,
        'LR':     LR,
        'op':     op,
        'myfunc': myfunc,
        'y_goal': ref,
        'gLoop':  10,
        'gL':    -12,
        'gU':     12,
        'gT':     abs(12 - (-12)) + 1,
        'gTol':   0.01,
    }

    # ------------------------------------------------------------------
    N1 = 60    # population size
    N2 = 100   # crossover pairs
    N3 = 100   # mutation pairs
    tg = 5     # total generations

    K          = []   # history: [average cost, best cost]
    KP         = []   # history: best solution per generation
    KS         = []   # history: best cost per generation
    GA_counter = []
    w          = 0    # generation index
    j          = 1    # generation counter

    fig, axes = plt.subplots(5, 1, figsize=(8, 10))
    plt.ion()
    plt.show()

    # ---- collect previous solutions ----
    tmpname      = os.path.join(root, ref['resultname'])
    solution_ini = np.empty((0, nParams))
    if os.path.isfile(tmpname):
        print(f'{tmpname} found.')
        tmp          = scipy.io.loadmat(tmpname, squeeze_me=True)
        solution_ini = np.atleast_2d(tmp['p_post'])

    # rectify min/max
    for i in range(nParams):
        solution_ini[:, i] = np.clip(solution_ini[:, i], LR[i], UR[i])

    # ----- initialization -----
    print('======== Initialization ========')
    P     = population(N1, nParams, LR, UR)
    if solution_ini.size > 0:
        P = np.vstack([P, solution_ini])
    E, R  = evaluation(P, myfunc, ref)
    P, E, R = selection_best(P, E, R, N1, op)
    R1    = R[:, 0]
    print('done')
    print(f'Minimum cost: {E[0]}')
    print('================================')
    E_crit = E[0]

    # ----- main loop -----
    while True:
        print('======= Gradient search ========')
        Para_E_grd, E_grd, R_grd = gradient_search(P[0, :], R1, conf, E_crit)
        if op * E_grd > op * E[0]:
            P[0, :]  = Para_E_grd
            E[0]     = E_grd
            R[:, 0]  = R_grd
        print('done')

        print('======= single-parameter mutation ========')
        P_ = mutation_single(P[0, :], LR, UR)
        E_, R_ = evaluation(P_, myfunc, ref)
        print('done')

        print('======= Gradient search ========')
        Para_E_grd_arr = np.zeros_like(P_)
        E_grd_arr      = np.zeros(len(E_))
        R_grd_arr      = np.zeros((R_.shape[0], len(E_)))
        for i in range(len(E_)):
            print(f'[{i+1}/{len(E_)}] cost: {E_[i]:.6f}')
            Para_E_grd_arr[i, :], E_grd_arr[i], R_grd_arr[:, i] = gradient_search(
                P_[i, :], R_[:, i], conf, E_crit
            )
        idx = op * E_grd_arr > op * E_
        P_[idx, :]      = Para_E_grd_arr[idx, :]
        E_[idx]         = E_grd_arr[idx]
        R_[:, idx]      = R_grd_arr[:, idx]

        P = np.vstack([P, P_])
        E = np.concatenate([E, E_])
        R = np.hstack([R, R_])
        print('done')

        _, E_show, _ = selection_best(P, E, R, 1, op)
        print(f'best after gradient: {E_show}')

        # GA
        print('GA search...')
        P_mutV  = mutationV(P[:N1, :], 0.1, 0.9, LR, UR)
        P_cross = crossover(P, N2)
        P_mut   = mutation(P, N3)
        P_new   = np.vstack([P_mutV, P_cross, P_mut])

        E_new, _, _ = evaluation(P_new, myfunc, ref)
        P = np.vstack([P, P_new])
        E = np.concatenate([E, E_new])

        P, E = selection_uniq(P, E, N1, N1, op, LR, UR)
        _, R1, _ = evaluation(P[[0], :], myfunc, ref)
        R1 = R1[:, 0]
        print('done')

        avg_cost  = E.mean()
        best_cost = E[0]
        K.append([avg_cost, best_cost])
        KP.append(P[0, :].copy())
        KS.append(E[0])
        E_crit = E[0]

        print('========')
        print(f'current best Loss: {KS[-1]}')
        print('========')

        gof = fitness_function(ref['y0'].ravel(), R1)
        print('========')
        print(f'current best R2: {gof}')
        print('========')

        if E_show > E[0]:
            print('GA works')
            GA_counter.append(1)
        else:
            print("GA doesn't work")
            GA_counter.append(0)

        w += 1
        j += 1

        # ----- online plot -----
        _, houtput = myfunc(KP[-1], ref)
        K_arr      = np.array(K)
        GA_arr     = np.array(GA_counter)

        for ax in axes:
            ax.cla()

        axes[0].plot(K_arr[:, 1], 'b.')
        axes[0].plot(K_arr[:, 0], 'r.')
        axes[0].set_title('Blue - Best            Red - Average')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Loss function')
        axes[0].set_yscale('log')
        axes[0].grid(True)

        axes[1].plot(E, 'b.')
        axes[1].set_xlabel('Chromosomes')
        axes[1].set_ylabel('Loss function')
        axes[1].set_yscale('log')
        axes[1].grid(True)

        axes[2].plot(KP[-1], '-ko')
        axes[2].set_title('parameter')

        axes[3].plot(ref['y0'].ravel(), 'k', linewidth=1.5)
        axes[3].plot(houtput['sim']['simMEP2'].ravel(), 'r', linewidth=1.0)
        axes[3].set_title('target & best fit')

        axes[4].plot(GA_arr, 'b.')
        axes[4].set_xlabel('Generations')
        suc_rate = GA_arr.sum() / len(GA_arr) if len(GA_arr) else 0
        axes[4].set_title(f'0--not work, 1--work, total success rate: {suc_rate:.2f}')

        plt.pause(0.01)
        fig.canvas.draw()

        # stop: number of generations
        if j > tg:
            break

        # stop: good fit
        if KS[-1] < 0.01:
            break

    # ----- get final result -----
    KS_arr = np.array(KS)
    KP_arr = np.array(KP)

    if op == -1:
        idx_best       = int(np.argmin(KS_arr))
        find_parameter = KP_arr[idx_best, :]
        print(f'minimum: {KS_arr[idx_best]}')
    else:
        idx_best       = int(np.argmax(KS_arr))
        find_parameter = KP_arr[idx_best, :]
        print(f'maximum: {KS_arr[idx_best]}')

    # make a copy of previous fitted result
    result_path = os.path.join(root, ref['resultname'])
    if os.path.isfile(result_path):
        timestamp       = datetime.now().strftime('%Y-%m%d-%H%M')
        backup_name     = ref['resultname'][:-4] + f'_backup-{timestamp}.mat'
        backup_path     = os.path.join(root, backup_name)
        shutil.copyfile(result_path, backup_path)

    # save fitted result
    p_post = KP_arr[-1, :]
    _, ref = MEPmodel_pheno(p_post, ref, 0)   # update ref

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    scipy.io.savemat(result_path, {
        'p_post': p_post,
        'KP':     KP_arr,
        'ref':    ref,     
        'P':      P,
        'KS':     KS_arr,
    })
    print('fitted result saved:')
    print(ref['resultname'])

    return p_post


# ==========================================================================
if __name__ == '__main__':
    subj  = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    reRun = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ga_MEPmodel_pheno(subj, reRun)