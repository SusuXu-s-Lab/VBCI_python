# /methods/tfxn.py

import numpy as np

def f(a):
    """
    Compute the function f(a) = -ln(1 + exp(a))
    
    Parameters:
    a : np.ndarray
        Input array.
    
    Returns:
    np.ndarray
        Output after applying the function.
    """
    return -np.log(1 + np.exp(a))

def df(a):
    """
    Compute the derivative of f(a), which is df(a)/da = 1 / (1 + exp(a))
    
    Parameters:
    a : np.ndarray
        Input array.
    
    Returns:
    np.ndarray
        Derivative of f with respect to a.
    """
    return 1 / (1 + np.exp(a))

def Tfxn(y, qBD, qLS, qLF, alLS, alLF, w, local, delta):
    """
    Nonlinear Transformation Function T
    
    Parameters:
    y     : np.ndarray
        Observed data.
    qBD   : np.ndarray
        Posterior probability for BD.
    qLS   : np.ndarray
        Posterior probability for LS.
    qLF   : np.ndarray
        Posterior probability for LF.
    alLS  : np.ndarray
        Alpha parameter for LS.
    alLF  : np.ndarray
        Alpha parameter for LF.
    w     : np.ndarray
        Weight vector (array of length 15).
    local : np.ndarray
        Local model classification.
    delta : float
        Tolerance parameter.
    
    Returns:
    np.ndarray
        Combined gradients for BD, LS, and LF.
    """
    # Unpack weights
    w0, weps, w0BD, w0LS, w0LF, wLSBD, wLFBD, wBDy, wLSy, wLFy, weLS, weLF, weBD, waLS, waLF = w

    y = np.log(1e-6 + y)

    # Initialize gBD, gLS, gLF
    gBD = np.zeros_like(y)
    gLS = np.zeros_like(y)
    gLF = np.zeros_like(y)

    # Compute gBD
    idx_local3 = (local == 3)
    idx_local4 = (local == 4)
    idx_local6 = (local == 6)
    idx_local3_4_6 = (local == 3) | (local == 4) | (local == 6)

    weBD_sq_div2 = (w[12] ** 2) / 2  # w[12] is weBD

    # Local == 3
    gBD[idx_local3] += (
        qLS[idx_local3] * f(- w[2] - w[5] + weBD_sq_div2)
        + (1 - qLS[idx_local3]) * f(- w[2] + weBD_sq_div2)
        - qLS[idx_local3] * f(w[2] + w[5] + weBD_sq_div2)
        - (1 - qLS[idx_local3]) * f(w[2] + weBD_sq_div2)
    )

    # Local == 4
    gBD[idx_local4] += (
        qLF[idx_local4] * f(- w[2] - w[6] + weBD_sq_div2)
        + (1 - qLF[idx_local4]) * f(- w[2] + weBD_sq_div2)
        - qLF[idx_local4] * f(w[2] + w[6] + weBD_sq_div2)
        - (1 - qLF[idx_local4]) * f(w[2] + weBD_sq_div2)
    )

    # Subtract terms involving y, wBDy, etc.
    term = ((w[7] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[7]
    gBD[idx_local3_4_6] -= term[idx_local3_4_6]

    # Additional terms for gBD
    if w[1] ** 2 != 0:
        gBD[idx_local3] -= (1 / (w[1] ** 2)) * (w[8] * w[7] * qLS[idx_local3])
        gBD[idx_local4] -= (1 / (w[1] ** 2)) * (w[9] * w[7] * qLF[idx_local4])
        gBD[idx_local6] -= (1 / (w[1] ** 2)) * (w[8] * w[9] * w[7] * qLS[idx_local6] * qLF[idx_local6])

    # Additional terms for local == 6
    if any(idx_local6):
        a1 = - w[2] - w[5] - w[6] + weBD_sq_div2
        a2 = - w[2] - w[5] + weBD_sq_div2
        a3 = - w[2] - w[6] + weBD_sq_div2
        a4 = w[2] + weBD_sq_div2
        a5 = w[2] + w[5] + w[6] + weBD_sq_div2
        a6 = w[2] + w[5] + weBD_sq_div2
        a7 = w[2] + w[6] + weBD_sq_div2

        qLS_local6 = qLS[idx_local6]
        qLF_local6 = qLF[idx_local6]

        term9 = (
            (qLS_local6 * qLF_local6 * f(a1)
             + qLS_local6 * (1 - qLF_local6) * f(a2)
             + (1 - qLS_local6) * qLF_local6 * f(a3)
             - (1 - qLS_local6) * (1 - qLF_local6) * f(a4))
            - (qLS_local6 * qLF_local6 * f(a5)
               - qLS_local6 * (1 - qLF_local6) * f(a6)
               - (1 - qLS_local6) * qLF_local6 * f(a7)
               - (1 - qLS_local6) * (1 - qLF_local6) * f(a4))
        )
        gBD[idx_local6] -= term9

    # Compute gLS
    idx_local1_3_5_6 = (local == 1) | (local == 3) | (local == 5) | (local == 6)
    weLS_sq_div2 = (w[10] ** 2) / 2

    gLS[idx_local1_3_5_6] += (
        f(- w[3] - w[13] * alLS[idx_local1_3_5_6] + weLS_sq_div2)
        - f(w[3] + w[13] * alLS[idx_local1_3_5_6] + weLS_sq_div2)
    )

    idx_local3 = (local == 3)
    idx_local5 = (local == 5)
    idx_local6 = (local == 6)

    # Additional terms for gLS
    if any(idx_local3):
        gLS[idx_local3] += (
            qBD[idx_local3] * (f(- w[2] - w[5] + weBD_sq_div2) - f(- w[2] + weBD_sq_div2))
            + (1 - qBD[idx_local3]) * (f(w[2] + w[5] + weBD_sq_div2) - f(w[2] + weBD_sq_div2))
        )

    if any(idx_local6):
        qBD_local6 = qBD[idx_local6]
        qLF_local6 = qLF[idx_local6]

        term = (
            qBD_local6 * (
                qLF_local6 * f(- w[2] - w[5] - w[6] + weBD_sq_div2)
                + (1 - qLF_local6) * f(- w[2] - w[6] + weBD_sq_div2)
                - qLF_local6 * f(- w[2] - w[5] + weBD_sq_div2)
                - (1 - qLF_local6) * f(- w[2] + weBD_sq_div2)
            )
            + (1 - qBD_local6) * (
                qLF_local6 * f(w[2] + w[5] + w[6] + weBD_sq_div2)
                + (1 - qLF_local6) * f(w[2] + w[5] + weBD_sq_div2)
                - qLF_local6 * f(w[2] + w[6] + weBD_sq_div2)
                - (1 - qLF_local6) * f(w[2] + weBD_sq_div2)
            )
        )
        gLS[idx_local6] += term

    # Subtract terms involving y, wLSy, etc.
    term = ((w[8] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[8]
    gLS[idx_local1_3_5_6] -= term[idx_local1_3_5_6]

    if w[1] ** 2 != 0:
        gLS[idx_local3] -= (1 / (w[1] ** 2)) * (w[7] * w[8] * qBD[idx_local3])
        gLS[idx_local5] -= (1 / (w[1] ** 2)) * (w[8] * w[9] * qLF[idx_local5])
        gLS[idx_local6] -= (1 / (w[1] ** 2)) * (w[7] * w[8] * w[9] * qBD[idx_local6] * qLF[idx_local6])

    # Additional terms for gLS
    idx_local5_or_6 = (local == 5) | (local == 6)
    gLS[idx_local5_or_6] -= qLF[idx_local5_or_6] / (2 * delta)

    # Compute gLF similarly
    idx_local2_4_5_6 = (local == 2) | (local == 4) | (local == 5) | (local == 6)
    weLF_sq_div2 = (w[11] ** 2) / 2

    gLF[idx_local2_4_5_6] += (
        f(- w[4] - w[14] * alLF[idx_local2_4_5_6] + weLF_sq_div2)
        - f(w[4] + w[14] * alLF[idx_local2_4_5_6] + weLF_sq_div2)
    )

    idx_local4 = (local == 4)
    idx_local5 = (local == 5)
    idx_local6 = (local == 6)

    if any(idx_local4):
        gLF[idx_local4] += (
            qBD[idx_local4] * (f(- w[2] - w[6] + weBD_sq_div2) - f(- w[2] + weBD_sq_div2))
            + (1 - qBD[idx_local4]) * (f(w[2] + w[6] + weBD_sq_div2) - f(w[2] + weBD_sq_div2))
        )

    if any(idx_local6):
        qBD_local6 = qBD[idx_local6]
        qLS_local6 = qLS[idx_local6]

        term = (
            qBD_local6 * (
                qLS_local6 * f(- w[2] - w[5] - w[6] + weBD_sq_div2)
                + (1 - qLS_local6) * f(- w[2] - w[6] + weBD_sq_div2)
                - qLS_local6 * f(- w[2] - w[5] + weBD_sq_div2)
                - (1 - qLS_local6) * f(- w[2] + weBD_sq_div2)
            )
            + (1 - qBD_local6) * (
                qLS_local6 * f(w[2] + w[5] + w[6] + weBD_sq_div2)
                + (1 - qLS_local6) * f(w[2] + w[6] + weBD_sq_div2)
                - qLS_local6 * f(w[2] + w[5] + weBD_sq_div2)
                - (1 - qLS_local6) * f(w[2] + weBD_sq_div2)
            )
        )
        gLF[idx_local6] += term

    # Subtract terms involving y, wLFy, etc.
    term = ((w[9] - 2 * y + 2 * w[0]) / (2 * (w[1] ** 2))) * w[9]
    gLF[idx_local2_4_5_6] -= term[idx_local2_4_5_6]

    if w[1] ** 2 != 0:
        gLF[idx_local4] -= (1 / (w[1] ** 2)) * (w[7] * w[9] * qBD[idx_local4])
        gLF[idx_local5] -= (1 / (w[1] ** 2)) * (w[8] * w[9] * qLS[idx_local5])
        gLF[idx_local6] -= (1 / (w[1] ** 2)) * (w[7] * w[8] * w[9] * qBD[idx_local6] * qLS[idx_local6])

    # Additional terms for gLF
    idx_local5_or_6 = (local == 5) | (local == 6)
    gLF[idx_local5_or_6] -= qLS[idx_local5_or_6] / (2 * delta)

    # Combine
    g = np.column_stack((gBD, gLS, gLF))

    return g
