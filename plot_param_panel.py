import numpy as np
import matplotlib.pyplot as plt

def plot_param_panel(p, ref):
    # Mapping variables from the ref dictionary using tuple keys for nested structures
    spike_times  = ref['sim']['spike_times']
    NRMSD        = ref['NRMSD']
    R            = ref['model']['R']
    Wexc         = ref['model']['Wexc']
    RWinh        = ref['model']['RWinh']  # R*Winh
    simMEP       = ref['sim']['simMEP2']
    DIwave       = ref['model']['DIwave']
    t            = ref['model']['t']
    y0           = ref['y0']
    intensities  = ref['intensities']
    subj         = ref['subj']
    tcrop        = ref['tcrop']
    
    # Optional/conditional fields
    boundary     = ref['model']['boundary'] if 'boundary' in ref['model'] else None
    boundarytext = ref['model']['boundarytext'] if 'boundarytext' in ref['model'] else []
    axonalDelay  = ref['model']['axonalDelay'] if 'axonalDelay' in ref['model'] else 0
    maxES        = ref['model']['maxES'] if 'maxES' in ref['model'] else 1
    withRC       = ref['model']['withRC'] if 'withRC' in ref['model'] else False

    nParams = len(p)
    plt.figure(figsize=(10, 8)) # Replicating set(gcf,'position',[50 50 800 600]) ratio

    # --- Top Row: Parameter value indicators ---
    for i in range(nParams):
        plt.subplot(3, nParams, i + 1)
        rescale = 0.9
        if boundary is not None:
            # Map p into 0~1 range for visualization
            locP = rescale * (p[i] - boundary[i, 0]) / (boundary[i, 1] - boundary[i, 0])
        else:
            locP = 0.5 * rescale
        
        plt.scatter([1], [locP], color='C0', zorder=5) # "filled" scatter
        plt.text(1.01, locP-0.05, f"{p[i]:.2f}")
        
        if boundary is not None:
            plt.scatter([1, 1], [0, rescale], marker='_', color='black')
            plt.plot([1, 1], [0, rescale], 'k')
            plt.text(1, 0-0.03, str(boundary[i, 0]), ha='center', va='top')
            plt.text(1, rescale+0.03, str(boundary[i, 1]), ha='center', va='bottom')
            
        plt.ylim([-0.1, 1.1])
        plt.axis('off')
        if boundarytext:
            plt.title(boundarytext[min(i, len(boundarytext)-1)], fontsize=9)

    # --- Middle Row: MEP comparison across intensities (Subplot 312) ---
    plt.subplot(3, 1, 2)
    # Using 'F' order (column-major) to flatten arrays to match MATLAB's (:) operator
    y0_flat = y0.flatten(order='F')
    simMEP_flat = simMEP.flatten(order='F')
    
    plt.plot(y0_flat, 'k', linewidth=2, label='MEP')
    plt.plot(simMEP_flat, 'r', linewidth=1, label='simMEP')
    
    # R^2 = 1 - sumsqr(err) / sumsqr(y - mean(y))
    R2 = 1 - (np.sum((y0_flat - simMEP_flat)**2) / np.sum((y0_flat - np.mean(y0_flat))**2))
    print(f'R^2 = {R2}')
    plt.title(f"MEPs of subject {subj} \n(R^2 = {R2:.2g}, NRMSD = {NRMSD*100:.2g}%)")

    # Draw vertical lines for intensity boundaries
    # size(simMEP, 1) in MATLAB refers to the number of points per intensity block
    pts_per_intensity = simMEP.shape[0]
    for i in range(simMEP.shape[1]):
        plt.axvline(x=i * pts_per_intensity, color='black', linewidth=0.5)
    
    plt.xlim([0, len(simMEP_flat)])
    plt.xlabel(f"Time ({tcrop[0]}-{tcrop[1]} ms)")
    plt.xticks([]) # Equivalent to xticklabel, []
    
    ylimit = plt.gca().get_ylim()
    for i in range(len(intensities)):
        plt.text(pts_per_intensity * (i + 0.5), ylimit[1] * 0.5, 
                 f"{int(intensities[i])}%\nMSO", ha='center', fontsize=8)
    
    plt.legend(loc='center left', prop={'size': 8}) # 'west' equivalent
    plt.ylabel('Amplitude (mV)')

    # Select 3 intensities to show in detail [Last, Middle, First]
    sel = [len(intensities) - 1, int(round(len(intensities) / 2)) - 1, 0]
    legend_labels = [f"{int(intensities[s])}%" for s in sel]

    # --- Subplot 9: DI-waves ---
    plt.subplot(3, 4, 9)
    plt.plot(t, DIwave[sel, :].T, linewidth=1.5)
    plt.xlabel('Time (ms)')
    plt.title('DIwaves')
    plt.xlim([0, 20])
    curr_yl = plt.gca().get_ylim()
    plt.ylim([-0.05, curr_yl[1]])
    lgd = plt.legend(legend_labels, loc='upper right', fontsize=7)
    lgd.get_frame().set_linewidth(0.5)

    # --- Subplot 10: Model Parameters across Neurons ---
    ax1 = plt.subplot(3, 4, 10)
    neurons = np.arange(1, 101)
    h1, = plt.plot(neurons, R, 'k.', label='R')
    plt.title('Model param.')
    plt.xlabel('Motor neuron')
    # Highlight specific neurons [1, 10, 20, 60, 100]
    hi_idx = [0, 9, 19, 59, 99]
    plt.plot(neurons[hi_idx], R[hi_idx], 'ko', fillstyle='none')
    
    if withRC:
        h2, = plt.plot(neurons, Wexc, 'b', linewidth=1.5, label='Wexc')
        h3, = plt.plot(neurons, RWinh, 'm', linewidth=1.5, label='R*Winh')
        plt.legend(handles=[h1, h2, h3], fontsize=7)
    else:
        plt.legend(handles=[h1], labels=['R'], fontsize=7)
    ax1.set_xlim([0, 100])
    plt.ylim([-1, plt.gca().get_ylim()[1]])

    # --- Subplot 11: MU Trigger Time (Raster Plot) ---
    plt.subplot(3, 4, 11)
    # spike_times is (100, maxES, nIntensities)
    for s_idx in sel:
        # Replicate MATLAB's reshape/repmat logic for scatter plotting
        st = (spike_times[:, :, s_idx] + axonalDelay).flatten()
        units = np.repeat(np.arange(1, 101), maxES)
        mask = ~np.isnan(st)
        plt.plot(st[mask], units[mask], '.', markersize=2)
        
    plt.ylim([-0.5, 100])
    plt.xlim([15, 50])
    plt.ylabel('Motor unit')
    plt.xlabel('Time (ms)')
    plt.title('MU trigger time')
    plt.legend(legend_labels, loc='upper right', fontsize=7)

    # --- Subplot 12: Trigger Time Histogram ---
    plt.subplot(3, 4, 12)
    bins = np.arange(15, 51, 2)
    for s_idx in sel:
        data = (spike_times[:, :, s_idx] + axonalDelay).flatten()
        data = data[~np.isnan(data)]
        # count density = count / bin_width
        counts, edges = np.histogram(data, bins=bins)
        centers = edges[:-1] + np.diff(edges) / 2
        plt.plot(centers, counts / np.diff(edges), linewidth=1.5)
        
        
    plt.xlim([15, 50])
    plt.ylim([-1, plt.gca().get_ylim()[1]])
    plt.title('Histogram')
    plt.xlabel('Trigger time (ms)')
    plt.ylabel('Count density')
    plt.legend(legend_labels, loc='upper right', fontsize=7)

    plt.subplots_adjust(hspace=0.4,wspace=0.5)
    plt.show()