import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def generate_EP(d=0.01, plotOn=0, Axontype=1):
    # ----------load AP (by Arancibia-Carcamo model)
    root = os.path.dirname(os.path.abspath(__file__))
    
    # Path logic equivalent to MATLAB's switch Axontype
    if Axontype == 1:
        data_path = os.path.join(root, 'MyelinatedAxonModel', 'EP', 'AP.mat')
    elif Axontype == 2:
        data_path = os.path.join(root, 'MyelinatedAxonModel', 'EP', 'AP2.mat')
    else:
        # Default fallback if needed
        data_path = os.path.join(root, 'MyelinatedAxonModel', 'EP', 'AP.mat')

    # Load the .mat file
    mat_contents = loadmat(data_path)
    TIME_VECTOR = mat_contents['TIME_VECTOR'].flatten()
    MEMBRANE_POTENTIAL = mat_contents['MEMBRANE_POTENTIAL']

    # ----------------------------------------------------
    idx = 20  # MATLAB idx=21 corresponds to index 20 in 0-based Python
    times = TIME_VECTOR.copy()
    padding = 1000
    dt = times[1] - times[0]
    
    # Append padding to times
    padding_times = times[-1] + (np.arange(1, padding + 1) * dt)
    times = np.concatenate([times, padding_times])
    
    # Extract specific AP and add padding
    v_segment = MEMBRANE_POTENTIAL[:, idx]
    v_padding = np.ones(padding) * v_segment[-1]
    v = np.concatenate([v_segment, v_padding]).reshape(-1, 1)
    
    # Calculate derivatives (preserving MATLAB structure)
    dv = np.vstack([np.diff(v, axis=0), np.zeros((1, v.shape[1]))])
    ddv = np.vstack([np.diff(dv, axis=0), np.zeros((1, dv.shape[1]))])

    # ---------calculate extracellular potential------------
    EP = np.zeros(ddv.shape)
    for i in range(v.shape[1]):
        for t_idx in range(len(times)):
            # mask = 1./sqrt((times-times(t)).^2+d^2)'
            mask = 1.0 / np.sqrt((times - times[t_idx])**2 + d**2)
            EP[t_idx, i] = np.sum(ddv[:, i] * mask)

    # ----------------------------------------------------
    if plotOn:
        fig, axs = plt.subplots(4, 1, figsize=(4, 8))
        axs[0].plot(times, v); axs[0].set_ylabel('mV'); axs[0].set_title('action potential by AC model\nV')
        axs[1].plot(times, dv); axs[1].set_title("V'")
        axs[2].plot(times, ddv); axs[2].set_title("V''")
        axs[3].plot(times, EP); axs[3].set_title(f"extracellular potential (d={d})")
        axs[3].set_xlabel('time (ms)')
        plt.tight_layout()
        plt.show()

    # ---------------prepare output data---------------
    dt2 = 1e-3
    times2 = np.arange(-0.5, 0.5 + dt2, dt2)
    
    # Process AP2
    AP2_raw = v.flatten() - np.min(v)
    max_val_ap = np.max(AP2_raw)
    i_max_ap = np.argmax(AP2_raw)
    AP2_norm = AP2_raw / max_val_ap
    
    f_ap = interp1d(times - times[i_max_ap], AP2_norm, kind='linear', fill_value="extrapolate")
    AP2 = f_ap(times2)
    
    # Process EP2
    min_val_ep = np.min(EP)
    i_min_ep = np.argmin(EP)
    EP2_norm = EP.flatten() / np.abs(min_val_ep)
    
    f_ep = interp1d(times - times[i_min_ep], EP2_norm, kind='linear', fill_value="extrapolate")
    EP2 = f_ep(times2)

    return EP2, times2, AP2