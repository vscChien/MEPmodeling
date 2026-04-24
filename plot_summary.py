import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from sigmoid import sigmoid
from load_MEP import load_MEP

from scipy.io import loadmat

def plot_summary(p, ref):
    # Extracting variables from the nested ref dictionary using tuple keys
    spike_times = ref['sim']['spike_times']
    R = ref['model']['R']
    Wexc = ref['model']['Wexc']
    RWinh = ref['model']['RWinh']  # R*Winh
    simMEP = ref['sim']['simMEP2']
    
    # Simple keys
    y0 = ref['y0']
    t0 = ref['t0']
    intensities = ref['intensities']

    fig = plt.figure(figsize=(20/2.54, 6/2.54)) # Convert cm to inches
    gs = GridSpec(1, 4, figure=fig, wspace=0.55, left=0.05, right=0.95, top=0.85, bottom=0.2)

    # --- Nexttile(1): MEP comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    space = (np.max(y0) - np.min(y0)) / 2
    
    # MATLAB: repmat((1:size(ref.y0,2))*space,[size(ref.y0,1),1])
    # y0 is assumed [time x intensities]
    offsets = np.arange(1, y0.shape[1] + 1) * space
    
    h1 = ax1.plot(t0, y0 + offsets, 'k', linewidth=1.5)
    h2 = ax1.plot(t0, simMEP + offsets, 'r', linewidth=1)
    
    # R^2 calculation
    R2 = 1 - np.sum((y0.flatten() - simMEP.flatten())**2) / np.sum((y0.flatten() - np.mean(y0.flatten()))**2)
    ax1.set_title(f'MEP (R^2= {R2:.2g})', fontsize=9)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_yticks([])
    ax1.set_xlabel('Time (ms)', fontsize=10)
    ax1.set_xlim([15, 50])
    ax1.set_ylim([0, space * len(intensities) + np.max(y0)])
    
    for i in range(len(intensities)):
        ax1.text(12, (i + 1) * space, f'{intensities[i]}%', fontsize=6)
    
    ylimit = ax1.get_ylim()
    peak = np.max(y0)
    if peak > 1:
        ax1.plot([47, 47], [ylimit[1] - 1, ylimit[1]], 'k', linewidth=1.5)
        ax1.text(46, ylimit[1] - 0.5, '1 mV', fontsize=6, ha='right')
    else:
        v_scale = np.round(peak / 2, 1)
        ax1.plot([47, 47], [ylimit[1] - v_scale, ylimit[1]], 'k', linewidth=1.5)
        ax1.text(46, ylimit[1] - v_scale / 2, f'{v_scale} mV', fontsize=6, ha='right')

    # --- Nexttile(2): IO Curve ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Assume get_iocurve is defined elsewhere as per prompt
    IO, simIO, myfit1, myfit2 = get_iocurve(simMEP, ref)
    
    x1 = np.linspace(IO[0, 0], IO[-1, 0], 100)
    ax2.plot(x1, sigmoid(x1, *myfit1), 'k', linewidth=1)
  
    x2 = np.linspace(simIO[0, 0], simIO[-1, 0], 100)
    ax2.plot(x2, sigmoid(x2, *myfit2), 'r', linewidth=1)
    
    ax2.scatter(IO[:, 0], IO[:, 1], 15, edgecolors='k', facecolors='none')
    ax2.errorbar(IO[:, 0], IO[:, 1], yerr=IO[:, 2], fmt='none', ecolor='k')
    ax2.scatter(simIO[:, 0], simIO[:, 1], 15, edgecolors='r', facecolors='none')
    
    ax2.set_xlim([27, 58])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('TMS intensity (%MSO)', fontsize=10)
    ax2.set_ylabel('Amplitude (mV)', fontsize=8)
    ax2.set_title('IO curve', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    if ref['subj'] in [1, 2, 4, 7, 9]:
        ticks = np.unique(np.append(ax2.get_yticks(), 0.5))
        ax2.set_yticks(ticks)

    # --- Nexttile(3): Model Parameters ---
    ax3 = fig.add_subplot(gs[0, 2])
    mn_indices = np.arange(1, 101)
    
    if 'withRC' in ref['model']:
        ax3.scatter(mn_indices, Wexc, 10, c='b', marker='.', alpha=0.5)
        ax3.scatter(mn_indices, RWinh, 10, c='m', marker='.', alpha=0.5)
    
    ax3.scatter(mn_indices, R, 10, c='k', marker='.')
    hi_idx = np.array([0, 9, 19, 59, 99]) # 1, 10, 20, 60, 100 in 0-based
    ax3.scatter(mn_indices[hi_idx], R[hi_idx], 20, edgecolors='k', facecolors='none')

    ax3.set_title('Model param.', fontsize=9)
    ylimit3 = ax3.get_ylim()
    
    text_params = {'fontsize': 7, 'backgroundcolor': 'none'}
    ax3.text(15, ylimit3[1] * 0.8, f'R: [{R[0]:.1f}, {R[99]:.1f}]', color='k', **text_params)
    
    if 'withRC' in ref['model']:
        ax3.text(15, ylimit3[1] * 0.7, f'Wexc: [{Wexc[0]:.1f}, {Wexc[99]:.1f}]', color='b', **text_params)
        ax3.text(15, ylimit3[1] * 0.6, f'Winh*R: [{RWinh[0]:.1f}, {RWinh[99]:.1f}]', color='m', **text_params)
        ax3.text(30, ylimit3[1] * 0.5, f'RCth= {np.round(p[9], 1)}', color='k', **text_params)

    ax3.text(30, ylimit3[1] * 0.4, f"dAxon= {np.round(ref['model']['axonalDelay'] , 1)}", color='k', **text_params)
    ax3.text(30, ylimit3[1] * 0.3, f"Tmu= {np.round(ref['model']['Tmu'] , 1)}", color='k', **text_params)
    ax3.text(15, ylimit3[1] * 0.9, f"AMPAw= {np.round(ref['model']['AMPAweight'] , 1)}", color='k', **text_params)

    ax3.set_xlabel('Motor neuron', fontsize=10)
    ax3.set_xticks([1, 50, 100])
    ax3.set_xlim([1, 100])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # --- Nexttile(4): MU Trigger Time ---
    ax4 = fig.add_subplot(gs[0, 3])
    axonal_delay = ref['model']['axonalDelay'] 
    maxES = ref['model']['maxES']
    
    # Using 'case 2' logic from MATLAB
    tmp = np.transpose(spike_times + axonal_delay, (0, 2, 1)) # [100 x intensities x maxES]
    tmp_flat = np.reshape(tmp, (100 * len(intensities), maxES), order='F')

    y_coords = np.arange(1, (100 * len(intensities)) + 1)
    for col in range(maxES):
        ax4.scatter(tmp_flat[:, col], y_coords, 0.2, c='k', marker='.')
        
    ax4.set_ylim([0, 100 * len(intensities)])
    ax4.set_xlim([15, 50])
    
    # Intensity boundary lines
    for i in range(1, len(intensities)):
        ax4.plot([17, 50], [i * 100, i * 100], 'k-', linewidth=0.5)
        
    for i in range(len(intensities)):
        ax4.text(12, (i + 0.7) * 100 - 50, f'{intensities[i]}%', fontsize=6)
        
    ax4.spines['left'].set_visible(False)
    ax4.set_yticks([])
    ax4.plot([47, 47], [1, 100], 'k', linewidth=1.5)
    ax4.text(45, 10, '100 MUs', fontsize=8, ha='right')
    ax4.set_xlabel('Time (ms)', fontsize=10)
    ax4.set_title('MU trigger time', fontsize=9)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.show()

def get_iocurve(simMEP, ref):
    # reload single-trial MEP for std of IO curve
    _, _, _, _, _, y0all = load_MEP(ref["subj"], ref["intensity_idx"], [20, 50], 0)

    IO = np.zeros((len(ref["intensity_idx"]), 3))  # [Int, amplitude, std]
    simIO = np.zeros((len(ref["intensity_idx"]), 2))

    for i in range(len(ref["intensity_idx"])):
        tidx1 = np.where((ref["t0"] >= 24) & (ref["t0"] <= 28))[0]
        peakv1 = np.max(ref["y0"][tidx1, i])
        peaki1 = np.argmax(ref["y0"][tidx1, i])
        tidx2 = np.where((ref["t0"] >= ref["t0"][tidx1[peaki1]]) & (ref["t0"] <= ref["t0"][tidx1[peaki1]] + 6))[0]
        peakv2 = np.min(ref["y0"][tidx2, i])
        IO[i, 0] = ref["intensities"][i]
        IO[i, 1] = peakv1 - peakv2

        tmp = np.zeros(y0all.shape[2])  # 15 trials
        for j in range(y0all.shape[2]):
            peakv1 = np.max(y0all[i, tidx1, j])
            peakv2 = np.min(y0all[i, tidx2, j])
            tmp[j] = peakv1 - peakv2
        IO[i, 2] = np.std(tmp, ddof=1) / np.sqrt(len(tmp))  # std error of MEP amplitude

        tidx = np.where((ref["t0"] >= 24) & (ref["t0"] <= 28))[0]
        peakv1 = np.max(simMEP[tidx, i])
        peaki1 = np.argmax(simMEP[tidx, i])
        tidx = np.where((ref["t0"] >= ref["t0"][tidx[peaki1]]) & (ref["t0"] <= ref["t0"][tidx[peaki1]] + 6))[0]
        peakv2 = np.min(simMEP[tidx, i])
        simIO[i, 0] = ref["intensities"][i]
        simIO[i, 1] = peakv1 - peakv2

    if ref["subj"] in {1, 2, 3, 4, 9, 10}:
        p0 = [40, 1.4, 10]
    elif ref["subj"] in {5, 7, 8}:
        p0 = [50, 1.4, 5]
    elif ref["subj"] == 6:
        p0 = [60, 1.4, 2]
    else:
        p0 = [40, 1.4, 10]  # default fallback

    popt1, pcov1 = curve_fit(sigmoid, IO[:, 0], IO[:, 1], p0=p0)
    popt2, pcov2 = curve_fit(sigmoid, simIO[:, 0], simIO[:, 1], p0=p0)

    myfit1 = sigmoid(IO[:, 0], *popt1)
    myfit2 = sigmoid(simIO[:, 0], *popt2)

    return IO, simIO, popt1, popt2