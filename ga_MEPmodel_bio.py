"""
Biological MEP model entry point.

Example usage:
    from ga_MEPmodel_bio import ga_MEPmodel_bio
    ga_MEPmodel_bio(subj=1, withRC=1, AMPAweight=None, reRun=0)

    subj        : subject number
    withRC      : include recurrent connections (default 1)
    AMPAweight  : fixed AMPA weight value, or None to fit freely
    reRun       : 0 – load fitted result and plot simulated MEP
                  1 – re-run model fitting (backs up previous fitted result)
"""

import os
import sys
import h5py
import shutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from load_h5            import load_h5_to_dict
from MEPmodel_bio       import MEPmodel_bio
from config_model_bio   import config_model_bio
from Optimizer          import ga_run


# ==========================================================================
def objective_function(p, ref):
    """
    Objective function: runs the biological MEP model and returns the residual.
    """
    _, ref_updated = MEPmodel_bio(p, ref)
    error = ref_updated['error']
    return error, ref_updated


# ==========================================================================
def _to_h5_compatible(value):
    """
    Convert *value* to something h5py can write as a dataset.

    Resolution order
    ----------------
    1. None              → store as empty bytes
    2. dict              → signal caller to recurse (returns None sentinel)
    3. str/bytes         → np.bytes_
    4. bool              → int (must come before int check)
    5. int / float       → as-is
    6. np.ndarray
       a. 0-d object     → unwrap and recurse
       b. object dtype   → convert element-wise to str, store as bytes array
       c. Unicode dtype  → encode each element to bytes
       d. numeric/bool   → as-is
    7. list / tuple      → convert to np.ndarray; fall back to bytes on failure
    8. everything else   → str repr stored as bytes
    """
    if value is None:
        return np.bytes_(b'')
    if isinstance(value, dict):
        return None                          # sentinel: caller must recurse
    if isinstance(value, (bytes, np.bytes_)):
        return np.bytes_(value)
    if isinstance(value, str):
        return np.bytes_(value.encode('utf-8'))
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_h5_compatible(value.item())
        if value.dtype == object:
            flat = [str(v).encode('utf-8') for v in value.ravel()]
            return np.array(flat, dtype='S').reshape(value.shape)
        if value.dtype.kind == 'U':
            flat = [s.encode('utf-8') for s in value.ravel()]
            return np.array(flat, dtype='S').reshape(value.shape)
        return value
    if isinstance(value, (list, tuple)):
        try:
            arr = np.array(value)
            if arr.dtype == object:
                raise ValueError('object array')
            if arr.dtype.kind == 'U':
                flat = [s.encode('utf-8') for s in arr.ravel()]
                return np.array(flat, dtype='S').reshape(arr.shape)
            return arr
        except (ValueError, TypeError):
            return np.bytes_(str(value).encode('utf-8'))
    return np.bytes_(str(value).encode('utf-8'))


def _save_dict_to_h5(h5file, data):
    """
    Recursively write a dict to an open h5py.File or h5py.Group.

    Tuple keys (used for nested bio-model fields) are serialised as
    their string representation so they round-trip safely.
    """
    for key, value in data.items():
        safe_key = str(key)          # h5py requires string keys; tuple → "('a','b')"
        if isinstance(value, dict):
            grp = h5file.require_group(safe_key)
            _save_dict_to_h5(grp, value)
        else:
            converted = _to_h5_compatible(value)
            if converted is None:
                grp = h5file.require_group(safe_key)
                _save_dict_to_h5(grp, value)
            else:
                h5file.create_dataset(safe_key, data=converted)


# ==========================================================================
def ga_MEPmodel_bio(subj, withRC=1, AMPAweight=None, reRun=0):
    """
    Main entry point for biological MEP model fitting using a Genetic Algorithm.
    """
    root = os.getcwd()

    # Normalise AMPAweight=[] (MATLAB empty) to None
    if isinstance(AMPAweight, (list, np.ndarray)) and len(AMPAweight) == 0:
        AMPAweight = None

    # ----- model setting -----
    ref = config_model_bio(subj, withRC, AMPAweight)

    # ----- derive h5 result path (replace any existing extension) -----
    resultname_h5 = os.path.splitext(ref['resultname'])[0] + '.h5'
    result_path   = os.path.join(root, resultname_h5)

    # ----- run GA or load fitted result -----
    if os.path.isfile(result_path) and not reRun:
        print(f'Use fitted result: \n{resultname_h5}')
        with h5py.File(result_path, 'r') as f:
            tmp = load_h5_to_dict(f)
        p_post = tmp['p_post'].flatten()
    else:
        p_post = _run_and_save(ref, root, result_path)

    # ----- show result -----
    plotOn = 1
    MEPmodel_bio(p_post, ref, plotOn)


# ==========================================================================
def _run_and_save(ref, root, result_path):
    """
    Run the GA, save the result as an HDF5 file, and return the best
    parameter set.

    Parameters
    ----------
    ref         : dict   model configuration from config_model_bio
    root        : str    working directory
    result_path : str    full path to the .h5 output file

    Returns
    -------
    p_post : np.ndarray  [nParams,]
    """
    # Promote ref['model']['boundary'] to ref['boundary'] so that ga_run,
    # which expects the pheno convention, can find it at the top level.
    # The full ref['model'] sub-dict is left intact for MEPmodel_bio.
    ref['boundary'] = ref['model']['boundary']

    LR      = ref['boundary'][:, 0]
    UR      = ref['boundary'][:, 1]
    nParams = len(LR)

    # ---- collect any previous solutions to seed the population ----
    solution_ini = np.empty((0, nParams))

    # Primary result file
    if os.path.isfile(result_path):
        print(f'{result_path} found.')
        with h5py.File(result_path, 'r') as f:
            tmp = load_h5_to_dict(f)
        solution_ini = np.vstack([solution_ini, np.atleast_2d(tmp['p_post'])])

    # Fixed-AMPAweight result files (h5 versions)
    fixed_dir = os.path.join(root, 'fitted_results', 'bio', 'fixed_AMPAweight')
    for AMPAw in np.arange(0.2, 0.9, 0.1):
        tmpname_fixed = os.path.join(
            fixed_dir, f"result_bio_s{ref['subj']}[{AMPAw:g}].h5"
        )
        if os.path.isfile(tmpname_fixed):
            print(f'{tmpname_fixed} found.')
            with h5py.File(tmpname_fixed, 'r') as f:
                tmp_fixed = load_h5_to_dict(f)
            if 'p_post' in tmp_fixed:
                p_tmp = np.atleast_1d(tmp_fixed['p_post']).ravel()
                ampa = ref['model'].get('AMPAweight')
                if ampa is not None and ampa != []:
                    p_tmp[11] = ampa
                solution_ini = np.vstack([solution_ini, p_tmp])

    # Rectify boundaries for seeded solutions
    if solution_ini.size > 0:
        for i in range(nParams):
            solution_ini[:, i] = np.clip(solution_ini[:, i], LR[i], UR[i])

    # ---- set up online plot ----
    fig, axes = plt.subplots(5, 1, figsize=(8, 12))
    plt.ion()
    plt.show()

    # ---- GA run ----
    def plot_callback(KP, KS, K, E, GA_counter, R1):
        K_arr  = np.array(K)
        GA_arr = np.array(GA_counter)
        _, houtput = objective_function(KP[-1], ref)

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

        axes[3].plot(ref['y0'].flatten(order='F'), 'k', linewidth=1.5)
        axes[3].plot(
            houtput['sim']['simMEP2'].flatten(order='F'), 'r', linewidth=1.0
        )
        axes[3].set_title('target & best fit')

        axes[4].plot(GA_arr, 'b.')
        axes[4].set_xlabel('Generations')
        suc_rate = GA_arr.sum() / len(GA_arr) if len(GA_arr) else 0
        axes[4].set_title(
            f'0--not work, 1--work, total success rate: {suc_rate:.2f}'
        )

        plt.pause(0.01)
        fig.canvas.draw()

    p_post, KP_arr, KS_arr, P = ga_run(
        ref,
        objective_function,
        N1=60, N2=100, N3=100, tg=1,
        op=-1,
        solution_ini=solution_ini,
        plot_callback=plot_callback,
    )

    # ---- backup previous result if it exists ----
    if os.path.isfile(result_path):
        timestamp   = datetime.now().strftime('%Y-%m%d-%H%M')
        backup_path = os.path.splitext(result_path)[0] + f'_backup-{timestamp}.h5'
        shutil.copyfile(result_path, backup_path)
        print(f'Previous result backed up to: {backup_path}')

    # ---- update ref and save to HDF5 ----
    _, ref_final = MEPmodel_bio(p_post, ref, 0)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with h5py.File(result_path, 'w') as f:
        f.create_dataset('p_post', data=p_post)
        f.create_dataset('KP',     data=KP_arr)
        f.create_dataset('KS',     data=KS_arr)
        f.create_dataset('P',      data=P)
        grp = f.create_group('ref')
        _save_dict_to_h5(grp, ref_final)

    print('fitted result saved:')
    print(result_path)

    return p_post


# ==========================================================================
if __name__ == '__main__':
    subj       = int(sys.argv[1])   if len(sys.argv) > 1 else 1
    withRC     = int(sys.argv[2])   if len(sys.argv) > 2 else 1
    AMPAweight = float(sys.argv[3]) if len(sys.argv) > 3 else None
    reRun      = int(sys.argv[4])   if len(sys.argv) > 4 else 0
    ga_MEPmodel_bio(subj, withRC, AMPAweight, reRun)