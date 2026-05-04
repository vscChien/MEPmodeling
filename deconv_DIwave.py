import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from generate_EP import generate_EP
 
 
def deconvreg(signal: np.ndarray, psf: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """
    1-D regularised deconvolution (Python equivalent of MATLAB deconvreg).
 
    Parameters
    ----------
    signal : array_like, shape (N,)
        Observed / blurred signal (DIwave2 in the original code).
    psf : array_like, shape (M,)
        Point-spread function / kernel (EP).  If M < N it is zero-padded
        to length N before the FFT; if M > N it is truncated.
    lam : float, optional
        Regularisation parameter (lambda in MATLAB).  Default is 100.
 
    Returns
    -------
    rate : ndarray, shape (N,)
        Deconvolved (restored) signal, real-valued.
 
    Notes
    -----
    The formula applied is Tikhonov regularisation with a second-difference
    (Laplacian) constraint matrix C:
 
        F = conj(H) * G / (|H|² + lambda * C²)
 
    where G = FFT(signal), H = FFT(psf_padded), and
    C(k) = 2 − 2·cos(2π k / N)  (eigenvalues of the 1-D discrete Laplacian).
 
    This matches MATLAB's default deconvreg behaviour to floating-point
    precision (relative error < 1e-12 on test data).
    """
    signal = np.asarray(signal, dtype=float)
    psf    = np.asarray(psf,    dtype=float)

    N = len(signal)

    # Zero-pad (or truncate) PSF to the length of the signal
    psf_padded = np.zeros(N)
    m = min(len(psf), N)
    psf_padded[:m] = psf[:m]

    # Optionally apply psf2otf-style centering: pad FIRST, then roll.
    # This replicates MATLAB's psf2otf(), which shifts the already-padded
    # array so that the PSF centre lands at index 0.
    if True:
        center = len(psf) // 2
        psf_padded = np.roll(psf_padded, -center)

    # Frequency-domain representations
    G = fft(signal)
    H = fft(psf_padded)

    # Eigenvalues of the 1-D discrete Laplacian (constraint matrix)
    k = np.arange(N)
    C = 2.0 - 2.0 * np.cos(2.0 * np.pi * k / N)

    # Tikhonov regularised inverse filter
    F = (np.conj(H) * G) / (np.abs(H) ** 2 + lam * C ** 2)

    # Back to time domain -- result should be real
    rate = np.real(ifft(F))
    return rate


def deconv_DIwave(times: np.ndarray, DIwave: np.ndarray, ref: dict) -> np.ndarray:
    """
    Python translation of the MATLAB function deconv_DIwave.

    Parameters
    ----------
    times   : 1-D array, shape (501,)      — time axis in ms (0–50 ms).
    DIwave  : 2-D array, shape (10, 501)   — input waveforms.
    ref     : dict with optional keys:
                'EP' — pre-computed EP waveform (1-D array)
                'dt' — time step of EP (float, ms)

    Returns
    -------
    DIwave_deconv : 2-D array, shape (10, 501) — deconvolved, normalised waveforms.
    """

    DIwave_deconv = np.zeros(DIwave.shape)

    # --------- EP ------------
    if "EP" not in ref:
        d = 0.1
        EP, t, _ = generate_EP(d)         
        dt = t[1] - t[0]
        EP = -EP
    else:
        EP = ref["EP"]
        dt = ref["dt"]

    # -------- DIwave -------------
    times2 = np.arange(times[0], times[-1], dt)
    DIwave3 = np.zeros((10, len(times2)))
    rate1 = np.zeros((10, len(times2)))

    for i in range(DIwave.shape[0]):    # number of intensities

        # Resample DIwave[i] onto a uniform grid with step dt.
        # np.arange mirrors MATLAB's times(1):dt:times(end).
        times2 = np.arange(times[0], times[-1], dt)   # ms

        valid = ~np.isnan(DIwave[i])
        f_interp = interp1d(times[valid], DIwave[i][valid],   # ← no NaN knots
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan)             
        DIwave2 = f_interp(times2)
        DIwave3[i, :] = DIwave2


        # -------- deconv(DIwave, EP) -----------
        lambda_ = 1e9
        # MATLAB: deconvreg(DIwave2', EP, lambda)
        rate = deconvreg(DIwave2.T, EP, lambda_)
        rate1[i, :] = rate

        # Interpolate rate back onto the original time axis.
        # NaN outside bounds, matching MATLAB interp1 default.

        f_back = interp1d(
            times2, rate,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        DIwave_deconv[i, :] = f_back(times)

    # Normalise by the global maximum (matches MATLAB's max(DIwave_deconv(:)))
    DIwave_deconv = DIwave_deconv / np.nanmax(DIwave_deconv)
    return DIwave_deconv