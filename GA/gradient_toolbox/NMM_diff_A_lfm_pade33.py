import numpy as np


def NMM_diff_A_lfm_pade33(para, houtput, myfunc, y_goal):
    """
    Estimate the Jacobian using a Padé [3/3] rational derivative approximation.

    A 7-point symmetric stencil is used to fit a [3/3] Padé approximant
    P(h)/Q(h), and the derivative at h=0 is extracted as
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

    h = np.linspace(-0.02, 0.02, 7).reshape(-1, 1)   # [7 x 1] stencil for [3/3]

    for p in range(P_len):
        h_vec = para[p] + h.ravel()   # perturbed scalar values for parameter p
        Y = np.zeros((T, len(h_vec)))

        for k in range(len(h_vec)):
            para_perturbed = para.copy()
            para_perturbed[p] = h_vec[k]
            Y[:, k] = np.asarray(myfunc(para_perturbed, y_goal)).ravel(order='F')

        df_p = np.zeros(T)

        for t in range(T):
            y_vals = Y[t, :].reshape(-1, 1)   # [7 x 1]
            H = h                              # [7 x 1]

            # Design matrix for Padé [3/3]:
            #   [1, h, h², h³, −y·h, −y·h², −y·h³]
            A = np.hstack([
                H ** 0,
                H ** 1,
                H ** 2,
                H ** 3,
                -y_vals * H,
                -y_vals * H ** 2,
                -y_vals * H ** 3,
            ])   # [7 x 7]
            rhs = y_vals.ravel()   # [7,]

            lam = 1e-6   # small regularisation
            coeffs = np.linalg.solve(
                A.T @ A + lam * np.eye(A.shape[1]),
                A.T @ rhs,
            )

            a = coeffs[:4]   # a0, a1, a2, a3
            b = np.concatenate([[1.0], coeffs[4:7]])   # b0=1, b1, b2, b3

            P0  = a[0]
            Q0  = b[0]

            # f'(0): first derivative of numerator / denominator polynomials at h=0
            #   P'(0) = a1 + 2·a2·0 + 3·a3·0² = a1
            #   Q'(0) = b1
            Pp0 = np.sum(np.arange(1, 4) * a[1:4])   # a1 + 2*a2 + 3*a3 (matches MATLAB)
            Qp0 = np.sum(np.arange(1, 4) * b[1:4])   # b1 + 2*b2 + 3*b3

            df_p[t] = (Pp0 * Q0 - P0 * Qp0) / Q0 ** 2

        j[:, p] = df_p

    j[np.isnan(j)] = 0.0
    j[np.isinf(j)] = 0.0

    return j