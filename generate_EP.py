import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from load_h5 import load_h5_to_dict
from scipy.interpolate import interp1d

def generate_EP(d=0.01, plotOn=0, Axontype=1):
    # ----------load AP (by Arancibia-Carcamo model)
    root = os.path.dirname(os.path.abspath(__file__))
    
    # Path logic equivalent to MATLAB's switch Axontype
    if Axontype == 1:
        data_path = os.path.join(root, 'MyelinatedAxonModel', 'EP', 'AP.h5')
    else:
        data_path = os.path.join(root, 'MyelinatedAxonModel', 'EP', 'AP2.h5')

    # Load the file
    with h5py.File(data_path, 'r') as f:
        mat_contents = load_h5_to_dict(f)
    times = mat_contents['TIME_VECTOR']
    times = times.reshape((len(times), 1))
    MEMBRANE_POTENTIAL = mat_contents['MEMBRANE_POTENTIAL']

    # ----------------------------------------------------
    idx = 20  # MATLAB idx=21 corresponds to index 20 in 0-based Python
    padding = 1000
    dt = times[1] - times[0]
    
    # Append padding to times
    padding_times = times[-1] + (np.arange(1, padding + 1) * dt)
    padding_times = padding_times.reshape((len(padding_times), 1))
    times = np.concatenate((times, padding_times))
    # Extract specific AP and add padding
    v_segment = MEMBRANE_POTENTIAL[:, idx]
    v_segment = v_segment.reshape(-1, 1)
    v_padding = np.ones(padding) * MEMBRANE_POTENTIAL[-1, idx]
    v_padding = v_padding.reshape(-1, 1)
    v = np.concatenate((v_segment, v_padding))
    np.savetxt("v.txt", v)

    # Calculate derivatives (preserving MATLAB structure)
    if isinstance(idx, int):
        s = 1
    else:
        s = len(idx)
    dv = np.vstack((np.diff(v, axis=0), np.zeros((1, s))))
    ddv = np.vstack((np.diff(dv, axis=0), np.zeros((1, s))))

    # ---------calculate extracellular potential------------
    EP = np.zeros(ddv.shape)
    for i in range(v.shape[1]):
        for t_idx in range(len(times)):
            # mask = 1./sqrt((times-times(t)).^2+d^2)'
            mask = 1.0 / np.sqrt((times - times[t_idx])**2 + d**2).T
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
    
    f_ap = interp1d((times - times[i_max_ap]).flatten(), AP2_norm, kind='linear')
    AP2 = f_ap(times2).T
    
    # Process EP2
    min_val_ep = np.min(EP)
    i_min_ep = np.argmin(EP)
    EP2_norm = EP.flatten() / np.abs(min_val_ep)
    
    f_ep = interp1d((times - times[i_min_ep]).flatten(), EP2_norm, kind='linear')
    EP2 = f_ep(times2).T
    #EP2 = EP2.reshape(-1, 1) # rehsape to column vector
    np.savetxt("EP2.txt", EP2)
    return EP2, times2, AP2