import numpy as np
import matplotlib.pyplot as plt
from cal_R2 import cal_R2
from cal_NRMSD import cal_NRMSD

def plot_TMScondition(ref, idx):
    """
    Translates the MATLAB function plot_TMScondition to Python.
    Maintains the same variable names and structure.
    
    Assumes:
    - ref is a dictionary where nested fields are accessed via tuples: ref[['parent', 'child')]
    - cal_R2 and cal_NRMSD are defined elsewhere.
    """
    
    # Extract variables from ref dictionary
    # MATLAB: spike_times=ref.sim.spike_times;
    spike_times = ref['sim']['spike_times']
    spike_times2 = ref['sim']['spike_times2']
    simMEP = ref['sim']['simMEP2']
    DIwave = ref['model']['DIwave']
    t = ref['model']['t']
    gexc_all = ref['sim']['gexc_all']
    ginh_all = ref['sim']['ginh_all']
    Rr_all = ref['sim']['mRC_all']
    Vr_all = ref['sim']['vRC_all']

    # Setup figure
    # width=10; height=15; centimeters to inches conversion (~2.54 cm/inch)
    fig, axes = plt.subplots(6, 1, figsize=(10/2.54, 15/2.54), constrained_layout=True)
    fig.canvas.manager.set_window_title('plot_result_detail')

    # Nexttile 1: DI-waves
    ax = axes[0]
    ax.plot(t, DIwave[idx, :], 'k', linewidth=1)
    ax.set_ylim([-0.1, 1.1])
    ylimit = ax.get_ylim()
    ax.text(10, ylimit[1] * 0.7, f"{ref['intensities'][idx]}% MSO", 
            fontsize=8, fontname='Calibri')
    ax.set_title(f"DI-waves (subject {ref['subj']})")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Rate (Hz)', fontsize=10, fontname='Calibri')

    # Nexttile 2: Effective conductances
    ax = axes[1]
    # area(t, gexc_all)
    ax.fill_between(t, gexc_all[idx, :], color=[0.5, 0.5, 0.5], alpha=0.5, label='g$_{exc}$/g$_{leak}$')
    # area(t, ginh_all)
    ax.fill_between(t, ginh_all[idx, :], color=[109/255, 158/255, 235/255], alpha=0.5, label='g$_{inh}$/g$_{leak}$')
    
    ylimit = ax.get_ylim()
    ampa_w = ref['model']['AMPAweight']
    ax.text(1.5, ylimit[1] * 0.7, f"AMPA:NMDA \n= {round(ampa_w, 2)} : {round(1-ampa_w, 2)}",
            fontsize=8, fontname='Calibri')
    ax.set_title('Effective conductances of MN1')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Nexttile 3: MN spikes
    ax = axes[2]
    # spike_times2{idx} is a matrix where col 2 is time and col 1 is neuron index
    # Note: Adjusting index for Python if spike_times2 was a cell array
    spikes = spike_times2[idx]
    ax.scatter(spikes[:, 1], spikes[:, 0], s=20, c='k', marker='.')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 100])
    ax.set_yticks([1, 50, 100])
    ax.set_title('MN spikes')
    ax.set_ylabel('MN', fontsize=10, fontname='Calibri')

    # Nexttile 4: RC activity
    ax = axes[3]
    h1, = ax.plot(t[:-1], Rr_all[idx, :-1], 'r', linewidth=1, label='rate')
    ax.set_ylabel('Rate (Hz)', fontsize=10, fontname='Calibri')
    ax.set_ylim([-0.1, 1.1])
    ax.set_title('RC activity')
    
    ax_right = ax.twinx()
    h2 = ax_right.fill_between(t, Vr_all[idx, :], color=[234/255, 153/255, 153/255], alpha=0.5, label='EPSP')
    ax_right.set_ylabel('EPSP(mV)', fontsize=10, fontname='Calibri')
    ax_right.spines['top'].set_visible(False)
    ax.legend([h2, h1], ['EPSP', 'rate'], fontsize=8)

    # Nexttile 5: MU trigger times
    ax = axes[4]
    # scatter(spike_times(:,:,idx)+ref.model.axonalDelay, 1:100)
    # spike_times structure is [Neurons x Spikes x Intensity]
    delay = ref['model']['axonalDelay']
    for n in range(100):
        row_spikes = spike_times[n, :, idx] + delay
        # Filter out NaNs if they represent empty spike slots
        valid_spikes = row_spikes[~np.isnan(row_spikes)]
        ax.scatter(valid_spikes, np.full_like(valid_spikes, n + 1), s=20, c='k', marker='.')
    
    ax.set_ylim([0, 100])
    ax.set_yticks([1, 50, 100])
    ylimit = ax.get_ylim()
    ax.text(1.5, ylimit[1] * 0.7, f"Axonal delay \n= {delay:.2g} ms", fontsize=8, fontname='Calibri')
    ax.set_title('MU trigger times')
    ax.set_xlim([0, 50])
    ax.set_ylabel('MU', fontsize=10, fontname='Calibri')

    # Nexttile 6: MEP Comparison
    ax = axes[5]
    ax.plot(ref['t0'], ref['y0'][:, idx], 'k', linewidth=2, label=f"{ref['intensities'][idx]}%MSO")
    ax.plot(ref['t0'], simMEP[:, idx], 'r', linewidth=1.5, label='simMEP')
    ax.legend(loc='upper left', fontsize=8)
    
    # Assuming cal_R2 and cal_NRMSD are costume functions available in the environment
    R2 = cal_R2(ref['y0'][:, idx], simMEP[:, idx])
    NRMSD = cal_NRMSD(ref['y0'][:, idx], simMEP[:, idx])
    
    ax.set_title(f"MEP (R^2 = {R2:.2g}, NRMSD = {NRMSD*100:.2g}%)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (ms)', fontsize=10, fontname='Calibri')
    ax.set_xlim([0, 50])
    ax.set_ylabel('Amplitude (mV)', fontsize=10, fontname='Calibri')

    plt.show()