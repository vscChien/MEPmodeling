import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from scipy.optimize import curve_fit
from sigmoid import sigmoid 
from load_MEP import load_MEP

def plot_summary_pheno(p, ref):
    simMEP      = ref['sim']['simMEP2']
    spike_times = ref['sim']['spike_times']

    fig = plt.figure(figsize=(20/2.54, 6/2.54))
    t_layout = fig.add_gridspec(1, 4, wspace=0.55, left=0.05, right=0.95, top=0.85, bottom=0.2)

    # ------------------------------------------------------------------
    # Tile 1 – MEP traces
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(t_layout[0])
    y0    = ref['y0']
    t0    = ref['t0']
    space = (y0.max() - y0.min()) / 2.0
    n_int = y0.shape[1]

    offsets = np.arange(1, n_int + 1) * space
    for col in range(n_int):
        ax1.plot(t0, y0[:, col]    + offsets[col], 'k', linewidth=1.5)
        ax1.plot(t0, simMEP[:, col] + offsets[col], 'r', linewidth=1.0)

    ss_res = np.sum((y0.ravel(order='F') - simMEP.ravel(order='F')) ** 2)
    ss_tot = np.sum((y0.ravel(order='F') - y0.mean())       ** 2)
    R2     = 1.0 - ss_res / ss_tot
    ax1.set_title(f'MEP (R²= {R2:.2g})')
    ax1.set_yticks([])
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(direction='out')
    ax1.set_xlabel('Time (ms)', fontsize=10, fontname='DejaVu Sans')
    ax1.set_xlim([15, 50])
    ax1.set_ylim([0, space * len(ref['intensities']) + y0.max()])

    for i, intens in enumerate(ref['intensities']):
        ax1.text(12, (i + 1) * space, f"{intens:.4g}%", fontsize=6)

    ylimit = ax1.get_ylim()
    peak   = y0.max()
    if peak > 1:
        ax1.plot([47, 47], [-1 + ylimit[1], ylimit[1]], 'k', linewidth=1.5)
        ax1.text(46, -0.5 + ylimit[1], '1 mV', fontsize=6,
                 horizontalalignment='right')
    else:
        bar_h = round(peak / 2, 1)
        ax1.plot([47, 47], [-bar_h + ylimit[1], ylimit[1]], 'k', linewidth=1.5)
        ax1.text(46, -bar_h / 2 + ylimit[1],
                 f'{bar_h:.4g} mV', fontsize=6,
                 horizontalalignment='right')

    # ------------------------------------------------------------------
    # Tile 2 – IO curve
    # ------------------------------------------------------------------
    ax2  = fig.add_subplot(t_layout[1])
    IO, simIO, myfit1, myfit2 = get_iocurve(simMEP, ref)

    x1 = np.linspace(IO[0, 0], IO[-1, 0], 100)
    ax2.plot(x1, sigmoid(x1, *myfit1), 'k', linewidth=1)
    x2 = np.linspace(simIO[0, 0], simIO[-1, 0], 100)
    ax2.plot(x2, sigmoid(x2, *myfit2), 'r', linewidth=1)
    ax2.scatter(IO[:, 0],    IO[:, 1],    s=15, c='k')
    ax2.errorbar(IO[:, 0],   IO[:, 1], yerr=IO[:, 2],
                 fmt='none', ecolor='k')
    ax2.scatter(simIO[:, 0], simIO[:, 1], s=15, c='r')
    ax2.set_xlim([27, 58])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('TMS intensity (%MSO)', fontsize=10)
    ax2.set_ylabel('Amplitude (mV)', fontsize=8)
    ax2.set_title('IO curve')

    if ref['subj'] in (1, 2, 4, 7, 9):
        yticks = sorted(set(list(ax2.get_yticks()) + [0.5]))
        ax2.set_yticks(yticks)

    # ------------------------------------------------------------------
    # Tile 3 – Model parameters (gamma PDF)
    # ------------------------------------------------------------------
    ax3   = fig.add_subplot(t_layout[2])
    x     = np.linspace(0, 30, 2000)
    shape = ref['sim']['shape']
    rate  = ref['sim']['rate']
    tshift = ref['sim']['tshift']
    delay  = ref['sim']['axonalDelday']
    pdf_vals = gamma_dist.pdf(x, a=shape, scale=1.0 / rate)

    ax3.plot(x + delay, pdf_vals, 'k', linewidth=1)
    ax3.set_title('Model param.')
    ax3.set_ylabel('Prob. density')
    ax3.set_xlabel('Time (ms)')
    ylimit3 = ax3.get_ylim()
    ax3.set_ylim([-ylimit3[1] / 10, ylimit3[1]])
    ax3.set_xlim([15, 50])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ymax = ax3.get_ylim()[1]
    ax3.text(25, ymax * 0.9, f'shape={round(shape,2):.4g}',  fontsize=7)
    ax3.text(25, ymax * 0.8, f'rate={round(rate,2):.4g}',    fontsize=7)
    ax3.text(25, ymax * 0.7, f'dAxon={round(delay,2):.4g}',  fontsize=7)
    ax3.text(25, ymax * 0.6, f'tLag={round(tshift,2):.4g}',  fontsize=7)

    # ------------------------------------------------------------------
    # Tile 4 – MU trigger times
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(t_layout[3])
    n_st = len(spike_times)
    for i in range(n_st):
        st = spike_times[i]
        ax4.scatter(
            st + t0[0],
            np.arange(1, len(st) + 1) + i * 100,
            s=5, c='k', marker='.'
        )
    ax4.set_ylim([0, 100 * n_st])
    ax4.set_xlim([15, 50])

    # horizontal separators
    for k in range(1, n_st):
        ax4.axhline(y=100 * k, color='k', linewidth=0.8)

    for i, intens in enumerate(ref['intensities']):
        ax4.text(12, (i + 1) * 100 - 80, f"{intens:.4g}%", fontsize=6)

    ax4.set_yticks([])
    ax4.spines['left'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.tick_params(direction='out')
    ax4.plot([47, 47], [1, 100], 'k', linewidth=1.5)
    ax4.text(45, 20, '100 MUs', fontsize=8, horizontalalignment='right')
    ax4.set_xlabel('Time (ms)', fontsize=10)
    ax4.set_title('MU trigger time')

    if 'figname' in ref:
        root = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(os.path.join(root, ref['figname']))

    plt.show()


# ==========================================================================
def get_iocurve(simMEP, ref):
    # reload single-trial MEP for std of IO curve
    _, _, _, _, _, y0all = load_MEP(ref['subj'], ref['intensity_idx'], [20, 50], 0)

    t0          = ref['t0']
    intensity_idx = ref['intensity_idx']
    n_int       = len(intensity_idx)
    n_trials    = y0all.shape[2]

    IO    = np.zeros((n_int, 3))   # [intensity, amplitude, std-error]
    simIO = np.zeros((n_int, 2))

    for i in range(n_int):
        tidx1 = np.where((t0 >= 24) & (t0 <= 28))[0]
        peaki1 = int(np.argmax(ref['y0'][tidx1, i]))
        peak_t  = t0[tidx1[peaki1]]
        tidx2   = np.where((t0 >= peak_t) & (t0 <= peak_t + 6))[0]

        peakv1  = ref['y0'][tidx1, i].max()
        peakv2  = ref['y0'][tidx2, i].min()
        IO[i, 0] = ref['intensities'][i]
        IO[i, 1] = peakv1 - peakv2

        tmp = np.zeros(n_trials)
        for j in range(n_trials):
            pv1 = y0all[i, tidx1, j].max()
            pv2 = y0all[i, tidx2, j].min()
            tmp[j] = pv1 - pv2
        IO[i, 2] = tmp.std() / np.sqrt(len(tmp))

        # simMEP peak
        tidx_s  = np.where((t0 >= 24) & (t0 <= 28))[0]
        peaki_s = int(np.argmax(simMEP[tidx_s, i]))
        peak_t_s = t0[tidx_s[peaki_s]]
        tidx_s2  = np.where((t0 >= peak_t_s) & (t0 <= peak_t_s + 6))[0]
        sv1 = simMEP[tidx_s,  i].max()
        sv2 = simMEP[tidx_s2, i].min()
        simIO[i, 0] = ref['intensities'][i]
        simIO[i, 1] = sv1 - sv2

    popt1 = _fit_sigmoid(IO[:, 0],    IO[:, 1])
    popt2 = _fit_sigmoid(simIO[:, 0], simIO[:, 1])

    return IO, simIO, popt1, popt2


def _fit_sigmoid(x, y):
    """
    Fit sigmoid(x, x0, r, a) = a / (1 + exp(r*(x0 - x))) to (x, y) data.

    Uses data-driven initial guesses and tries multiple candidates so that
    shallow or late-rising IO curves (e.g. subject 6) converge reliably.

    Parameters
    ----------
    x : np.ndarray  stimulus intensities
    y : np.ndarray  MEP amplitudes

    Returns
    -------
    popt : np.ndarray  [x0, r, a]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # a: plateau is at least 2x the observed max (curve may not have plateaued)
    a_est  = float(y.max()) * 2.0
    # r: approximate slope; use a small safe default
    r_est  = 0.2
    # x0: inflection point — start beyond the data range if y is still rising
    x0_est = float(x[-1]) + (float(x[-1]) - float(x[0])) * 0.5

    # p0 candidates tried in order of preference: (x0, r, a)
    candidates = [
        [x0_est,              r_est, a_est],
        [x0_est,              0.3,   a_est],
        [x0_est,              0.1,   a_est],
        [float(x[-1]),        r_est, a_est],
        [float(x[-1]) + 10,   r_est, a_est],
        [float(x[-1]) + 20,   0.3,   a_est],
        [float(x.mean()),     r_est, a_est],
    ]

    # x0 can extend well beyond the data range for late-rising curves;
    # r must stay positive; a must be positive
    x_range = float(x[-1]) - float(x[0])
    bounds = (
        [float(x[0]),                  1e-4,  0.0   ],
        [float(x[-1]) + 3 * x_range,  10.0,  np.inf],
    )

    for p0 in candidates:
        try:
            popt, _ = curve_fit(
                sigmoid, x, y,
                p0=p0, bounds=bounds,
                maxfev=20000,
            )
            return popt
        except (RuntimeError, ValueError):
            continue

    raise RuntimeError(
        f"sigmoid curve_fit failed for all initial guesses.\n"
        f"  x = {x}\n"
        f"  y = {y}\n"
        "  Consider checking the data or extending the candidate p0 list."
    )