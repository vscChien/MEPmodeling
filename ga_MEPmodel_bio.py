import os
import h5py
import shutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from load_h5 import load_h5_to_dict
from scipy.io import loadmat, savemat
from MEPmodel_bio import MEPmodel_bio
from config_model_bio import config_model_bio

def ga_MEPmodel_bio(subj, withRC=2, AMPAweight=None, reRun=0):
    """
    Main function for biological MEP model fitting using Genetic Algorithm.
    """
    # Get the directory of the current script
    root = os.getcwd()
    
    # model setting
    # Treat 'ref' as a dictionary with tuple keys for nested fields
    ref = config_model_bio(subj, withRC, AMPAweight)
    
    result_file = os.path.join(root, ref['resultname'])
    
    # run GA or load existing results
    if os.path.exists(result_file) and not reRun:
        print(f"Use fitted result: \n{ref['resultname']}")
        with h5py.File(result_file, 'r') as f:
            tmp = load_h5_to_dict(f)
        # Flattening to ensure it's a 1D array as expected in Python
        p_post = tmp['p_post'].flatten()
    else:
        p_post = run_ga(ref)
 
    # show result
    plotOn = 1
    MEPmodel_bio(p_post, ref, plotOn)

def objective_function(p, ref):
    """
    Objective function for optimization mapping to the MATLAB subfunction.
    """
    # MEPmodel_bio returns [sim, ref]
    _, ref_updated = MEPmodel_bio(p, ref)
    error = ref_updated['error']
    return error, ref_updated

def run_ga(ref):
    """
    Genetic Algorithm implementation following the MATLAB run_ga subfunction.
    """
    myfunc = objective_function
    op = -1   # -1: find minimum, 1: find maximum
    
    # Access nested structures using tuples as keys
    LR = ref[('model', 'boundary')][:, 0]
    UR = ref[('model', 'boundary')][:, 1]
    nParams = len(LR)

    # Optimization configuration dictionary
    conf = {
        'UR': UR,
        'LR': LR,
        'op': op,
        'myfunc': myfunc,
        'y_goal': ref,
        'gLoop': 10,
        'gL': -12,
        'gU': 12,
        'gT': abs(12 - (-12)) + 1,
        'gTol': 0.01
    }
    
    # GA parameters
    N1 = 60   # population size
    N2 = 100  # crossover pairs
    N3 = 100  # mutation pairs
    tg = 1    # total generations (matched to current script logic)

    K = np.empty((0, 2))        # history of [average cost, best cost]
    KP = np.empty((0, nParams)) # history of best solutions
    KS = []                     # history of best costs
    GA_counter = []
    
    w = 0 # 0-indexed counter for history
    j = 1 # 1-indexed generation counter
    
    plt.figure(figsize=(8, 12))
    
    # Collect previous solutions from file
    root = os.path.dirname(os.path.abspath(__file__))
    tmpname = os.path.join(root, 'fitted_results', 'bio', f"result_bio_s{ref['subj']}.csv")
    solution_ini = np.empty((0, nParams))
    
    if os.path.exists(tmpname):
        print(f"{tmpname} found.")
        tmp = loadmat(tmpname)
        if 'p_post' in tmp:
            solution_ini = np.vstack([solution_ini, tmp['p_post'].flatten()])
    
    # Check fixed AMPA weight results
    for AMPAw in np.arange(0.2, 0.9, 0.1):
        # Format float to match MATLAB's %g
        tmpname_fixed = os.path.join(root, 'fitted_results', 'bio', 'fixed_AMPAweight', 
                                     f"result_bio_s{ref['subj']}[{AMPAw:g}].csv")
        if os.path.exists(tmpname_fixed):
            print(f"{tmpname_fixed} found.")
            tmp_fixed = loadmat(tmpname_fixed)
            if 'p_post' in tmp_fixed:
                p_tmp = tmp_fixed['p_post'].flatten()
                if ref[('model', 'AMPAweight')] is not None:
                    # Index 12 (MATLAB) -> Index 11 (Python)
                    p_tmp[11] = ref[('model', 'AMPAweight')]
                solution_ini = np.vstack([solution_ini, p_tmp])
    
    # Rectify boundaries for initial solutions
    if solution_ini.size > 0:
        for i in range(nParams):
            solution_ini[:, i] = np.clip(solution_ini[:, i], LR[i], UR[i])

    # --- Initialization ---
    print('======== Initialization ========')
    P = population(N1, nParams, LR, UR)
    if solution_ini.size > 0:
        P = np.vstack([P, solution_ini])
        
    E, R_pop = evaluation(P, myfunc, ref)
    P, E, R_pop = selection_best(P, E, R_pop, N1, op)
    R1 = R_pop[:, 0] # Best residual
    
    print('done')
    print(f"Minimum cost: {E[0]}")
    print('================================')
    E_crit = E[0]
    
    # --- Main GA loop ---
    while True:
        print('======= Gradient search ========')
        Para_E_grd, E_grd, R_grd = gradient_search(P[0, :], R1, conf, E_crit)
        
        if op * E_grd > op * E[0]:
            P[0, :] = Para_E_grd
            E[0] = E_grd
            R_pop[:, 0] = R_grd
        print('done')
              
        print('======= single-parameter mutation ========')
        P_mut_s = mutation_single(P[0, :], LR, UR)
        E_mut_s, R_mut_s = evaluation(P_mut_s, myfunc, ref)
        print('done')
        
        print('======= Gradient search (post-mutation) ========')
        n_mut = len(E_mut_s)
        Para_E_grd_all = np.zeros_like(P_mut_s)
        E_grd_all = np.zeros(n_mut)
        R_grd_all = np.zeros_like(R_mut_s)
        
        for i in range(n_mut):
            print(f"[{i+1}/{n_mut}] cost: {E_mut_s[i]}")
            p_g, e_g, r_g = gradient_search(P_mut_s[i, :], R_mut_s[:, i], conf, E_crit)
            Para_E_grd_all[i, :] = p_g
            E_grd_all[i] = e_g
            R_grd_all[:, i] = r_g
            
        replace_idx = (op * E_grd_all) > (op * E_mut_s)
        P_mut_s[replace_idx, :] = Para_E_grd_all[replace_idx, :]
        E_mut_s[replace_idx] = E_grd_all[replace_idx]
        R_mut_s[:, replace_idx] = R_grd_all[:, replace_idx]
        
        P = np.vstack([P, P_mut_s])
        E = np.concatenate([E, E_mut_s])
        R_pop = np.hstack([R_pop, R_mut_s])
        print('done')
                           
        # Check current best before GA search
        _, E_show_arr, _ = selection_best(P, E, R_pop, 1, op)
        E_show = E_show_arr[0]
        print(f"best after gradient: {E_show}")
        
        # --- GA Search ---
        print('GA search...')
        P_mutV = mutationV(P[:N1, :], 0.1, 0.9, LR, UR)
        P_cross = crossover(P, N2)
        P_mut_ga = mutation(P, N3)
        
        P_offspring = np.vstack([P_mutV, P_cross, P_mut_ga])
        E_offspring, _, _ = evaluation(P_offspring, myfunc, ref)
        
        P = np.vstack([P, P_offspring])
        E = np.concatenate([E, E_offspring])
        P, E = selection_uniq(P, E, N1, N1, op, LR, UR)
        
        _, R1, _ = evaluation(P[0, :], myfunc, ref)
        print('done')
               
        # Statistics
        K = np.vstack([K, [np.mean(E), E[0]]])
        KP = np.vstack([KP, P[0, :]])
        KS.append(E[0])
        E_crit = E[0]
        
        print(f"========\ncurrent best Loss: {KS[-1]}\n========")
        # Match MATLAB flattening for R^2 calculation
        gof = fitness_function(ref['y0'].flatten(order='F'), R1)
        print(f"current best R2: {gof}\n========")
        
        GA_counter.append(1 if E_show > E[0] else 0)
                        
        w += 1 
        j += 1 
        
        # --- Online Visualization ---
        _, houtput = myfunc(KP[-1, :], ref)
        plt.clf()

        plt.subplot(5, 1, 1)
        plt.plot(K[:, 1], 'b.', label='Best')
        plt.plot(K[:, 0], 'r.', label='Average')
        plt.title('Blue - Best            Red - Average')
        plt.ylabel('Loss function')
        plt.grid(True); plt.yscale('log')

        plt.subplot(5, 1, 2)
        plt.plot(E, 'b.')
        plt.xlabel('Chromosomes'); plt.ylabel('Loss function')
        plt.grid(True); plt.yscale('log')

        plt.subplot(5, 1, 3)
        plt.plot(KP[-1, :], '-ko')
        plt.title('parameter')

        plt.subplot(5, 1, 4)
        plt.plot(ref['y0'].flatten(order='F'), 'k', linewidth=1.5, label='Target')
        plt.plot(houtput[('sim', 'simMEP2')].flatten(order='F'), 'r', linewidth=1, label='Best fit')
        plt.title('target & best fit')
   
        plt.subplot(5, 1, 5)
        plt.plot(GA_counter, 'b.')
        plt.xlabel('Generations')
        if GA_counter:
            rate = sum(GA_counter) / len(GA_counter)
            plt.title(f"0--not work, 1--work, total successful rate: {rate:.4f}")
        
        plt.draw()
        plt.pause(0.01)
                
        # --- Termination ---
        if j > tg or KS[-1] < 0.01:
            break
                
    # --- Finalization ---
    if op == -1:
        idx = np.argmin(KS)
    else:
        idx = np.argmax(KS)
    find_parameter = KP[idx, :]
     
    # Backup existing result
    result_path = os.path.join(root, ref['resultname'])
    if os.path.exists(result_path):
        date_tag = datetime.now().strftime('%Y-%m%d-%H%M')
        backup_name = f"{ref['resultname'][:-4]}_backup-{date_tag}.csv"
        shutil.copy2(result_path, os.path.join(root, backup_name))
    
    # Save final results
    p_post = KP[-1, :]
    _, ref_final = MEPmodel_bio(p_post, ref, 0)
    
    data_to_save = {
        'p_post': p_post,
        'KP': KP,
        'ref': ref_final,
        'P': P,
        'KS': np.array(KS)
    }
    savemat(result_path, data_to_save)
    print(f"fitted result saved: \n{ref_final['resultname']}")

    return p_post