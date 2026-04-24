import numpy as np
import matplotlib.pyplot as plt

def plot_DIwave(model, ref):
    # select 3 TMS intensities to show: [last, middle, first]
    # MATLAB: sel=[length(ref.intensities), round(length(ref.intensities)/2), 1]
    # Python 0-based conversion:
    intensities_len = len(ref['intensities'])
    sel = [intensities_len - 1, int(round(intensities_len / 2)) - 1, 0]

    fig = plt.figure(figsize=(4, 8)) # Replicating set(gcf,'position',...) ratio
    
    # --- Subplot 1: DIwave0 (potential) ---
    plt.subplot(5, 1, 1)
    # Selecting rows based on 'sel' and plotting against time 'model.t'
    plt.plot(model['t'], model['DIwave0'][sel, :].T, linewidth=1)
    plt.ylabel('potential')
    plt.title(f"Subject {ref['subj']}\nDI-waves (potential)")
    plt.xlim([0, 50])
    plt.legend([str(int(round(val))) for val in ref['intensities'][sel]])

    # --- Subplot 2: DIwave (firing rate) ---
    plt.subplot(5, 1, 2)
    plt.plot(model['t'], model['DIwave'][sel, :].T, linewidth=1)
    plt.title('DI-waves (firing rate)')
    plt.ylabel('rate')
    plt.xlim([0, 50])

    # --- Subplot 3: Synaptic kernels ---
    plt.subplot(5, 1, 3)
    # model.AMPA' and model.NMDA' combined
    plt.plot(model['t'], np.vstack([model['AMPA'], model['NMDA']]).T, linewidth=1)
    plt.title('Synaptic kernels')
    plt.legend(['AMPA', 'NMDA'])
    plt.ylabel('cond.')
    plt.xlim([0, 50])

    # --- Subplot 4: DIwaveConv_AMPA ---
    plt.subplot(5, 1, 4)
    plt.plot(model['t'], model['DIwaveConv_AMPA'][sel, :].T, linewidth=1)
    plt.title('DI-waves (AMPA conductance)')
    plt.ylabel('cond.')
    plt.xlim([0, 50])

    # --- Subplot 5: DIwaveConv_NMDA ---
    plt.subplot(5, 1, 5)
    plt.plot(model['t'], model['DIwaveConv_NMDA'][sel, :].T, linewidth=1)
    plt.xlabel('Time (ms)')
    plt.title('DI-waves (NMDA conductance)')
    plt.ylabel('cond.')
    plt.xlim([0, 50])

    plt.tight_layout()
    plt.show()