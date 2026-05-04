import numpy as np
from scipy.interpolate import interp1d
from generate_EP import generate_EP


def deconvreg_fft(signal: np.ndarray, psf: np.ndarray, lambda_: float) -> np.ndarray:
    """
    FFT-based Tikhonov-regularized deconvolution, equivalent to MATLAB's
    deconvreg(signal, psf, lambda_) with the default identity regulariser.

    Mathematical background
    -----------------------
    The matrix-form Tikhonov problem is:

        minimise  ||H r - s||²  +  λ ||r||²

    whose closed-form solution is:

        r = (HᵀH + λI)⁻¹ Hᵀ s

    Because H is a (circular) convolution operator, in the DFT domain every
    matrix-vector product becomes a pointwise multiply:

        H  r  <-->  EP(f) · R(f)
        Hᵀ s  <-->  conj(EP(f)) · S(f)      (correlation theorem)
        HᵀH   <-->  |EP(f)|²                 (power spectrum)

    Substituting into the normal equations gives the Wiener / Tikhonov
    filter applied in the frequency domain:

        R(f) = conj(EP(f)) · S(f) / ( |EP(f)|² + λ )

    One iDFT recovers r in O(N log N) time with O(N) memory — no N×N
    matrix is ever formed.

    Zero-padding
    ------------
    Circular convolution wraps around; to approximate the *linear* convolution
    that the matrix approach computes we zero-pad both arrays to length
    N + M - 1 before the FFT and then truncate the output to length N.

    Parameters
    ----------
    signal  : 1-D array, shape (N,)  — measured signal (DIwave2).
    psf     : 1-D array, shape (M,)  — point-spread function (EP).
    lambda_ : float                  — regularisation strength.

    Returns
    -------
    rate : 1-D array, shape (N,)  — deconvolved signal.
    """

    N = len(signal)
 
    # Circular FFT at length N — matching MATLAB's fftn(x, sizeI)
    S = np.fft.rfft(signal, n=N)
    H = np.fft.rfft(psf,    n=N)
 
    # MATLAB: FILTF = PSFNORM ./ (|PSFNORM|² + λ)   — NO conj in numerator
    H_power = (np.conj(H) * H).real        # |H(f)|²  (real-valued)
    R = (H * S) / (H_power + lambda_)      # element-wise, no conjugation
 
    return np.fft.irfft(R, n=N)[:N]


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

        # -------- deconv(DIwave, EP) -----------
        lambda_ = 100
        # MATLAB: deconvreg(DIwave2', EP, lambda)
        rate = deconvreg_fft(DIwave2.T, EP, lambda_)

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