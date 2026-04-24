import numpy as np
from scipy.interpolate import interp1d
from generate_EP import generate_EP
from scipy.signal import deconvolve

def deconvreg(signal, kernel, lambd):
    # Regularized deconvolution using Wiener filter approach
    # signal and kernel are 1D numpy arrays
    # lambd is the regularization parameter
    S = np.fft.fft(signal)
    K = np.fft.fft(kernel, n=signal.size)
    K_conj = np.conj(K)
    denom = K_conj * K + lambd
    rate = np.real(np.fft.ifft(S * K_conj / denom))
    return rate

def deconv_DIwave(times, DIwave, ref):
    DIwave = np.array(DIwave)
    DIwave_deconv = np.zeros_like(DIwave)

    if not hasattr(ref, 'EP'):
        d = 0.1
        EP, t, _ = generate_EP(d)  
        dt = t[1] - t[0]
        EP = -EP
    else:
        EP = ref["EP"]
        dt = ref["dt"]

    for i in range(DIwave.shape[0]):
        times2 = np.arange(times[0], times[-1] + dt, dt)
        interp_func = interp1d(times, DIwave[i, :], kind='linear', fill_value="extrapolate")
        DIwave2 = interp_func(times2)

        lambda_ = 100
        rate = deconvreg(DIwave2.T, EP, lambda_)
        interp_rate = interp1d(times2, rate, kind='linear', fill_value="extrapolate")
        DIwave_deconv[i, :] = interp_rate(times)

    DIwave_deconv /= np.max(DIwave_deconv)
    return DIwave_deconv