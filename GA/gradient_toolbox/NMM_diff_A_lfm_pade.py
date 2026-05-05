import numpy as np


def NMM_diff_A_lfm_pade(para, houtput, myfunc, y_goal):
    """
    Estimate the Jacobian using a Padé [2/2] rational derivative approximation.

    A 5-point symmetric stencil is used to fit a [2/2] Padé approximant
    P(h)/Q(h) to the function, and the derivative at h=0 is extracted as
        f'(0) = (P'(0)·Q(0) − P(0)·Q'(0)) / Q(0)²

    Parameters
    ----------
    para    : np.ndarray  [nParams,] or [1 x nParams]  current parameter vector
    houtput : np.ndarray  [T,] or [T x 1]              function output at `para`
    myfunc  : callable    black-box function: output = myfunc(para, y_goal)
    y_goal  : target output passed through to myfunc

    Returns
    -------
    j : np.ndarray  [T x nParams]  Jacobian matrix
    """
    houtput = np.asarray(houtput)
    T = houtput.shape[0]
    para = np.atleast_1d(para).ravel(order='F')
    P_len = len(para)

    j = np.zeros((T, P_len))

    h = np.linspace(-0.02, 0.02, 5).reshape(-1, 1)   # [5 x 1] symmetric stencil

    for p in range(P_len):
        h_vec = para[p] + h.ravel()   # perturbed scalar values for parameter p
        Y = np.zeros((T, len(h_vec)))

        for k in range(len(h_vec)):
            para_perturbed = para.copy()
            para_perturbed[p] = h_vec[k]
            Y[:, k] = np.asarray(myfunc(para_perturbed, y_goal)).ravel(order='F')

        df_p = np.zeros(T)

        for t in range(T):
            y_vals = Y[t, :].reshape(-1, 1)   # [5 x 1]
            H = h                              # [5 x 1]

            # Design matrix for Padé [2/2]:
            #   [1, h, h², −y·h, −y·h²]
            A = np.hstack([
                H ** 0,
                H ** 1,
                H ** 2,
                -y_vals * H,
                -y_vals * H ** 2,
            ])   # [5 x 5]
            rhs = y_vals.ravel()   # [5,]

            lam = 1e-6   # small regularisation
            coeffs = np.linalg.solve(
                A.T @ A + lam * np.eye(A.shape[1]),
                A.T @ rhs,
            )

            a = coeffs[:3]              # a0, a1, a2
            b = np.array([1.0, coeffs[3], coeffs[4]])   # b0=1, b1, b2

            P0  = a[0]
            Pp0 = a[1]   # d/dh of P at h=0 = a1
            Q0  = b[0]
            Qp0 = b[1]   # d/dh of Q at h=0 = b1

            df_p[t] = (Pp0 * Q0 - P0 * Qp0) / Q0 ** 2

        j[:, p] = df_p

    j[np.isnan(j)] = 0.0
    j[np.isinf(j)] = 0.0

    return j