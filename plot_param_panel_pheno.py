import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist


def plot_param_panel_pheno(p, ref):
    simMEP      = ref['sim']['simMEP2']
    spike_times = ref['sim']['spike_times']
    NRMSD       = ref['NRMSD']

    nParams = len(p)
    p       = list(p)
    for i in range(3, nParams):
        p[i] = round(p[i])   # number of recruited MNs

    boundary      = ref.get('boundary', None)
    boundary_text = ref['boundarytext']

    fig = plt.figure(figsize=(10, 8)) 

    # ------------------------------------------------------------------
    # Row 1 – parameter sliders
    # ------------------------------------------------------------------
    rescale = 0.9
    for i in range(nParams):
        ax = fig.add_subplot(3, nParams, i + 1)
        ax.set_xlim([0.8, 2.0])
        ax.set_ylim([-0.1, 1.1])
        ax.axis('off')

        if boundary is not None:
            lo, hi = boundary[i, 0], boundary[i, 1]
            locP   = rescale * (p[i] - lo) / (hi - lo)
            ax.scatter([1], [0],       marker='_', c='k', s=50)
            ax.scatter([1], [rescale], marker='_', c='k', s=50)
            ax.plot([1, 1], [0, rescale], 'k')
            ax.text(1, 0-0.04,       f'{lo:.4g}', ha='center', va='top')
            ax.text(1, rescale+0.02, f'{hi:.4g}', ha='center', va='bottom')
        else:
            locP = 0.5 * rescale

        ax.scatter([1], [locP], zorder=3)
        label = f'{p[i]:.2f}' if i < 3 else f'{int(p[i]):d}'
        ax.text(1.2, locP-0.035, label)
        ax.set_title(boundary_text[min(i, len(boundary_text) - 1)], loc='left')

    # ------------------------------------------------------------------
    # Row 2 (full width) – MEP traces
    # ------------------------------------------------------------------
    ax_mep = fig.add_subplot(3, 1, 2)
    y0 = ref['y0']
    ax_mep.plot(y0.flatten(order='F'),     'k', linewidth=2, label='MEP')
    ax_mep.plot(simMEP.flatten(order='F'), 'r', linewidth=1, label='simMEP')

    ss_res = np.sum((y0.ravel(order='F') - simMEP.ravel(order='F')) ** 2)
    ss_tot = np.sum((y0.ravel(order='F') - y0.mean())       ** 2)
    R2     = 1.0 - ss_res / ss_tot
    print(f'R² = {R2:.4g}')
    ax_mep.set_title(
        f'MEPs of subject {ref["subj"]}\n'
        f'(R² = {R2:.2g}, NRMSD = {NRMSD * 100:.2g}%)'
    )

    nT = simMEP.shape[0]
    nI = simMEP.shape[1]
    for k in range(1, nI):
        ax_mep.axvline(x=k * nT, color='k')
    ax_mep.set_xlim([0, len(simMEP.ravel(order='F'))])
    ax_mep.set_xlabel(f'Time ({ref["tcrop"][0]}-{ref["tcrop"][1]} ms)')
    ax_mep.set_xticks([(k + 0.5) * nT for k in range(nI)])  # invisible ticks for labels
    ax_mep.set_xticklabels([])
    ylimit = ax_mep.get_ylim()
    for i, intens in enumerate(ref['intensities']):
        ax_mep.text(
            nT * (i + 0.5), ylimit[1] * 0.7,
            f'{intens:.4g}%\nMSO',
            ha='center', fontsize=8
        )
    lgd = ax_mep.legend(loc='lower left')
    for handle in lgd.legend_handles:
        handle.set_linewidth(1.0)
    ax_mep.set_ylabel('Amplitude (mV)')

    n_int   = len(ref['intensities'])
    sel_idx = [n_int - 1, round((n_int - 1) / 2), 0]   # 0-based indices
    legend_labels = [f"{ref['intensities'][s]:.4g}%" for s in sel_idx]

    # ------------------------------------------------------------------
    # Row 3, panel 10 – gamma PDF
    # ------------------------------------------------------------------
    ax_pdf = fig.add_subplot(3, 4, 10)
    t      = np.linspace(0, 30, 2000)
    shape  = ref['sim']['shape']
    rate   = ref['sim']['rate']
    tshift = ref['sim']['tshift']
    delay  = ref['sim']['axonalDelday']
    pdf_vals = gamma_dist.pdf(t, a=shape, scale=1.0 / rate)

    for s in sel_idx:
        offset = (n_int - 1 - s) * tshift   # 0-indexed equivalent of (nIntensities - sel(i))
        ax_pdf.plot(t + delay + offset, pdf_vals, linewidth=1.5)

    ax_pdf.set_title('Model param.')
    ax_pdf.set_ylabel('Probability density')
    ax_pdf.set_xlabel('Time (ms)')
    ylimit_pdf = ax_pdf.get_ylim()
    ax_pdf.set_ylim([-ylimit_pdf[1] / 10, ylimit_pdf[1]])
    ax_pdf.set_xlim([15, 50])
    ymax = ax_pdf.get_ylim()[1]
    ax_pdf.text(25, ymax * 0.9, f'shape={round(shape,2):.4g}',  fontsize=7)
    ax_pdf.text(25, ymax * 0.8, f'rate={round(rate,2):.4g}',    fontsize=7)
    ax_pdf.text(25, ymax * 0.7, f'dAxon={round(delay,2):.4g}',  fontsize=7)
    ax_pdf.text(25, ymax * 0.6, f'tLag={round(tshift,2):.4g}',  fontsize=7)

    # ------------------------------------------------------------------
    # Row 3, panel 11 – spike raster for selected intensities
    # ------------------------------------------------------------------
    ax_raster = fig.add_subplot(3, 4, 11)
    t0 = ref['t0']
    for i, s in enumerate(sel_idx):
        st = spike_times[s]
        ax_raster.plot(
            st + t0[0],
            np.arange(1, len(st) + 1),
            '.', markersize=3
        )
    lgd2 = ax_raster.legend(legend_labels, loc='upper left')
    for handle in lgd2.legend_handles:
        handle.set_markersize(5)
    ax_raster.set_ylim([-0.5, 100])
    ax_raster.set_xlim([15, 50])
    ax_raster.set_ylabel('Motor unit')
    ax_raster.set_xlabel('Time (ms)')
    ax_raster.set_title('MU trigger time')

    # ------------------------------------------------------------------
    # Row 3, panel 12 – histogram of spike times
    # ------------------------------------------------------------------
    ax_hist = fig.add_subplot(3, 4, 12)
    bins    = np.arange(15, 51, 2)
    for s in sel_idx:
        st     = spike_times[s] + t0[0]
        counts, edges = np.histogram(st, bins=bins, density=False)
        bin_width = edges[1] - edges[0]
        centers   = edges[:-1] + bin_width / 2
        ax_hist.plot(centers, counts / bin_width, linewidth=1.5)

    ax_hist.set_xlim([15, 50])
    ylimit_hist = ax_hist.get_ylim()
    ax_hist.set_ylim([-1, ylimit_hist[1]])
    lgd3 = ax_hist.legend(legend_labels, loc='upper right')
    for handle in lgd3.legend_handles:
        handle.set_linewidth(1.5)
    ax_hist.set_title('Histogram')
    ax_hist.set_xlabel('Trigger time (ms)')
    ax_hist.set_ylabel('Count density')

    plt.subplots_adjust(hspace=0.4,wspace=0.5)
    plt.show()