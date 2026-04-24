import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

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

    fig = plt.figure(figsize=(15/2.54, 4.5/2.54)) # Convert cm to inches
    gs = GridSpec(1, 4, figure=fig, wspace=0.3, left=0.05, right=0.95, top=0.85, bottom=0.2)

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
    ax1.set_xlabel('Time (ms)', fontsize=10, fontname='Calibri')
    ax1.set_xlim([15, 50])
    ax1.set_ylim([0, space * len(intensities) + np.max(y0)])
    
    for i in range(len(intensities)):
        ax1.text(12, (i + 1) * space, f'{intensities[i]}%', fontsize=6, fontname='Calibri')
    
    ylimit = ax1.get_ylim()
    peak = np.max(y0)
    if peak > 1:
        ax1.plot([47, 47], [ylimit[1] - 1, ylimit[1]], 'k', linewidth=1.5)
        ax1.text(46, ylimit[1] - 0.5, '1 mV', fontsize=6, fontname='Calibri', ha='right')
    else:
        v_scale = np.round(peak / 2, 1)
        ax1.plot([47, 47], [ylimit[1] - v_scale, ylimit[1]], 'k', linewidth=1.5)
        ax1.text(46, ylimit[1] - v_scale / 2, f'{v_scale} mV', fontsize=6, fontname='Calibri', ha='right')

    # --- Nexttile(2): IO Curve ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Assume get_iocurve is defined elsewhere as per prompt
    IO, simIO, myfit1, myfit2 = get_iocurve(simMEP, ref)
    
    x1 = np.linspace(IO[0, 0], IO[-1, 0], 100)
    ax2.plot(x1, myfit1(x1), 'k', linewidth=1)
    
    x2 = np.linspace(simIO[0, 0], simIO[-1, 0], 100)
    ax2.plot(x2, myfit2(x2), 'r', linewidth=1)
    
    ax2.scatter(IO[:, 0], IO[:, 1], 15, edgecolors='k', facecolors='none')
    ax2.errorbar(IO[:, 0], IO[:, 1], yerr=IO[:, 2], fmt='none', ecolor='k')
    ax2.scatter(simIO[:, 0], simIO[:, 1], 15, edgecolors='r', facecolors='none')
    
    ax2.set_xlim([27, 58])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('TMS intensity (%MSO)', fontsize=10, fontname='Calibri')
    ax2.set_ylabel('Amplitude (mV)', fontsize=8, fontname='Calibri')
    ax2.set_title('IO curve', fontsize=9)
    
    if ref['subj'] in [1, 2, 4, 7, 9]:
        ticks = np.unique(np.append(ax2.get_yticks(), 0.5))
        ax2.set_yticks(ticks)

    # --- Nexttile(3): Model Parameters ---
    ax3 = fig.add_subplot(gs[0, 2])
    mn_indices = np.arange(1, 101)
    
    if ref[('model', 'withRC')]:
        ax3.scatter(mn_indices, Wexc, 10, c='b', marker='.', alpha=0.5)
        ax3.scatter(mn_indices, RWinh, 10, c='m', marker='.', alpha=0.5)
    
    ax3.scatter(mn_indices, R, 10, c='k', marker='.')
    hi_idx = np.array([0, 9, 19, 59, 99]) # 1, 10, 20, 60, 100 in 0-based
    ax3.scatter(mn_indices[hi_idx], R[hi_idx], 20, edgecolors='k', facecolors='none')

    ax3.set_title('Model param.', fontsize=9)
    ylimit3 = ax3.get_ylim()
    
    text_params = {'fontname': 'Calibri', 'fontsize': 7, 'backgroundcolor': 'none'}
    ax3.text(15, ylimit3[1] * 0.85, f'R: [{R[0]:.1f}, {R[99]:.1f}]', color='k', **text_params)
    
    if ref[('model', 'withRC')]:
        ax3.text(15, ylimit3[1] * 0.75, f'Wexc: [{Wexc[0]:.1f}, {Wexc[99]:.1f}]', color='b', **text_params)
        ax3.text(15, ylimit3[1] * 0.65, f'Winh*R: [{RWinh[0]:.1f}, {RWinh[99]:.1f}]', color='m', **text_params)
        ax3.text(30, ylimit3[1] * 0.55, f'RCth= {np.round(p[9], 1)}', color='k', **text_params)

    ax3.text(30, ylimit3[1] * 0.45, f"dAxon= {np.round(ref[('model', 'axonalDelay')], 1)}", color='k', **text_params)
    ax3.text(30, ylimit3[1] * 0.35, f"Tmu= {np.round(ref[('model', 'Tmu')], 1)}", color='k', **text_params)
    ax3.text(15, ylimit3[1] * 0.95, f"AMPAw= {np.round(ref[('model', 'AMPAweight')], 1)}", color='k', **text_params)

    ax3.set_xlabel('Motor neuron', fontsize=10, fontname='Calibri')
    ax3.set_xticks([1, 50, 100])

    # --- Nexttile(4): MU Trigger Time ---
    ax4 = fig.add_subplot(gs[0, 3])
    axonal_delay = ref[('model', 'axonalDelay')]
    maxES = ref[('model', 'maxES')]
    
    # Using 'case 2' logic from MATLAB
    tmp = np.transpose(spike_times + axonal_delay, (0, 2, 1)) # [100 x intensities x maxES]
    tmp_flat = tmp.reshape(100 * len(intensities), maxES)
    
    y_coords = np.arange(1, (100 * len(intensities)) + 1)
    for col in range(maxES):
        ax4.scatter(tmp_flat[:, col], y_coords, 5, c='k', marker='.')
        
    ax4.set_ylim([0, 100 * len(intensities)])
    ax4.set_xlim([15, 50])
    
    # Intensity boundary lines
    for i in range(1, len(intensities)):
        ax4.plot([17, 50], [i * 100, i * 100], 'k-', linewidth=0.5)
        
    for i in range(len(intensities)):
        ax4.text(12, (i + 1) * 100 - 50, f'{intensities[i]}%', fontsize=6, fontname='Calibri')
        
    ax4.spines['left'].set_visible(False)
    ax4.set_yticks([])
    ax4.plot([47, 47], [1, 100], 'k', linewidth=1.5)
    ax4.text(45, 50, '100 MUs', fontsize=8, fontname='Calibri', ha='right')
    ax4.set_xlabel('Time (ms)', fontsize=10, fontname='Calibri')
    ax4.set_title('MU trigger time', fontsize=9)

    plt.show()

# Placeholder for custom function assumed to be defined elsewhere
def get_iocurve(simMEP, ref):
    # This should implement the logic found in the second half of the MATLAB file
    pass