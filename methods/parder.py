# methods/parder.py

import numpy as np
from .tfxn import df

def parder(y, qBD, qLS, qLF, alLS, alLF, w, local):
    """
    Compute the partial derivatives (gradients) for the optimization process.

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

    Returns:
    np.ndarray
        Gradient matrix (array of shape (len(y), 15)).
    """
    # Unpack weights
    w0, weps, w0BD, w0LS, w0LF, wLSBD, wLFBD, wBDy, wLSy, wLFy, weLS, weLF, weBD, waLS, waLF = w

    y = np.log(1e-6 + y)

    # Initialize gradient matrix
    grad = np.zeros((len(y), 15))

    # Precompute common terms
    weps_sq = weps ** 2
    weBD_sq_div2 = (weBD ** 2) / 2
    weLS_sq_div2 = (weLS ** 2) / 2
    weLF_sq_div2 = (weLF ** 2) / 2

    # Compute indicators
    idx_local1 = (local == 1)
    idx_local2 = (local == 2)
    idx_local3 = (local == 3)
    idx_local4 = (local == 4)
    idx_local5 = (local == 5)
    idx_local6 = (local == 6)

    # grad[:, 0] corresponds to w0
    grad[:, 0] = (1 / weps_sq) * (y - w0)
    grad[:, 0] -= ((idx_local3 | idx_local4 | idx_local6) * (1 / weps_sq) * (wBDy * qBD))
    grad[:, 0] -= ((idx_local1 | idx_local3 | idx_local5 | idx_local6) * (1 / weps_sq) * (wLSy * qLS))
    grad[:, 0] -= ((idx_local2 | idx_local4 | idx_local5 | idx_local6) * (1 / weps_sq) * (wLFy * qLF))

    # grad[:, 1] corresponds to weps
    term = -1 / weps - (1 / weps ** 3) * (
        - y ** 2 - w0 ** 2 + 2 * w0 * y
        - ((idx_local1 | idx_local3 | idx_local5 | idx_local6) * (wLSy ** 2 - 2 * y * wLSy + 2 * w0 * wLSy) * qLS)
        - ((idx_local2 | idx_local4 | idx_local5 | idx_local6) * (wLFy ** 2 - 2 * y * wLFy + 2 * w0 * wLFy) * qLF)
        - ((idx_local3 | idx_local4 | idx_local6) * (wBDy ** 2 - 2 * y * wBDy + 2 * w0 * wBDy) * qBD)
        - (idx_local3 * 2 * (wBDy * wLSy * qLS * qBD))
        - (idx_local4 * 2 * (wBDy * wLFy * qLF * qBD))
        - (idx_local6 * 2 * (wBDy * wLSy * wLFy * qLS * qLF * qBD))
    )
    grad[:, 1] = term

    # grad[:, 2] corresponds to w0BD
    grad[:, 2] = 0
    if np.any(idx_local3):
        grad[idx_local3, 2] += (
            qBD[idx_local3] * qLS[idx_local3] * df(w0BD + wLSBD - weBD_sq_div2)
            - (1 - qBD[idx_local3]) * qLS[idx_local3] * df(-w0BD - wLSBD - weBD_sq_div2)
            + qBD[idx_local3] * (1 - qLS[idx_local3]) * df(w0BD - weBD_sq_div2)
            - (1 - qBD[idx_local3]) * (1 - qLS[idx_local3]) * df(-w0BD - weBD_sq_div2)
        )

    if np.any(idx_local4):
        grad[idx_local4, 2] += (
            qBD[idx_local4] * qLF[idx_local4] * df(w0BD + wLFBD - weBD_sq_div2)
            - (1 - qBD[idx_local4]) * qLF[idx_local4] * df(-w0BD - wLFBD - weBD_sq_div2)
            + qBD[idx_local4] * (1 - qLF[idx_local4]) * df(w0BD - weBD_sq_div2)
            - (1 - qBD[idx_local4]) * (1 - qLF[idx_local4]) * df(-w0BD - weBD_sq_div2)
        )

    if np.any(idx_local6):
        qBD_local6 = qBD[idx_local6]
        qLS_local6 = qLS[idx_local6]
        qLF_local6 = qLF[idx_local6]

        term = (
            qBD_local6 * qLS_local6 * qLF_local6 * df(w0BD + wLSBD + wLFBD - weBD_sq_div2)
            + qBD_local6 * qLS_local6 * (1 - qLF_local6) * df(w0BD + wLSBD - weBD_sq_div2)
            + qBD_local6 * (1 - qLS_local6) * qLF_local6 * df(w0BD + wLFBD - weBD_sq_div2)
            + qBD_local6 * (1 - qLS_local6) * (1 - qLF_local6) * df(w0BD - weBD_sq_div2)
            - (1 - qBD_local6) * qLS_local6 * qLF_local6 * df(-w0BD - wLSBD - wLFBD - weBD_sq_div2)
            - (1 - qBD_local6) * qLS_local6 * (1 - qLF_local6) * df(-w0BD - wLSBD - weBD_sq_div2)
            - (1 - qBD_local6) * (1 - qLS_local6) * qLF_local6 * df(-w0BD - wLFBD - weBD_sq_div2)
            - (1 - qBD_local6) * (1 - qLS_local6) * (1 - qLF_local6) * df(-w0BD - weBD_sq_div2)
        )
        grad[idx_local6, 2] += term

    # grad[:, 3] corresponds to w0LS
    idx_LS = idx_local1 | idx_local3 | idx_local5 | idx_local6
    grad[:, 3] = 0
    grad[idx_LS, 3] += (
        qLS[idx_LS] * df(w0LS + waLS * alLS[idx_LS] - weLS_sq_div2)
        - (1 - qLS[idx_LS]) * df(-w0LS - waLS * alLS[idx_LS] - weLS_sq_div2)
    )

    # grad[:, 4] corresponds to w0LF
    idx_LF = idx_local2 | idx_local4 | idx_local5 | idx_local6
    grad[:, 4] = 0
    grad[idx_LF, 4] += (
        qLF[idx_LF] * df(w0LF + waLF * alLF[idx_LF] - weLF_sq_div2)
        - (1 - qLF[idx_LF]) * df(-w0LF - waLF * alLF[idx_LF] - weLF_sq_div2)
    )

    # grad[:, 5] corresponds to wLSBD
    grad[:, 5] = 0
    if np.any(idx_local3):
        grad[idx_local3, 5] += (
            qBD[idx_local3] * qLS[idx_local3] * df(w0BD + wLSBD - weBD_sq_div2)
            - (1 - qBD[idx_local3]) * qLS[idx_local3] * df(-w0BD - wLSBD - weBD_sq_div2)
        )

    if np.any(idx_local6):
        grad[idx_local6, 5] += (
            qBD[idx_local6] * qLS[idx_local6] * qLF[idx_local6] * df(w0BD + wLSBD + wLFBD - weBD_sq_div2)
            + qBD[idx_local6] * qLS[idx_local6] * (1 - qLF[idx_local6]) * df(w0BD + wLSBD - weBD_sq_div2)
            - (1 - qBD[idx_local6]) * qLS[idx_local6] * qLF[idx_local6] * df(-w0BD - wLSBD - wLFBD - weBD_sq_div2)
            - (1 - qBD[idx_local6]) * qLS[idx_local6] * (1 - qLF[idx_local6]) * df(-w0BD - wLSBD - weBD_sq_div2)
        )

    # grad[:, 6] corresponds to wLFBD
    grad[:, 6] = 0
    if np.any(idx_local4):
        grad[idx_local4, 6] += (
            qBD[idx_local4] * qLF[idx_local4] * df(w0BD + wLFBD - weBD_sq_div2)
            - (1 - qBD[idx_local4]) * qLF[idx_local4] * df(-w0BD - wLFBD - weBD_sq_div2)
        )

    if np.any(idx_local6):
        grad[idx_local6, 6] += (
            qBD[idx_local6] * qLF[idx_local6] * qLS[idx_local6] * df(w0BD + wLSBD + wLFBD - weBD_sq_div2)
            + qBD[idx_local6] * qLF[idx_local6] * (1 - qLS[idx_local6]) * df(w0BD + wLFBD - weBD_sq_div2)
            - (1 - qBD[idx_local6]) * qLF[idx_local6] * qLS[idx_local6] * df(-w0BD - wLSBD - wLFBD - weBD_sq_div2)
            - (1 - qBD[idx_local6]) * qLF[idx_local6] * (1 - qLS[idx_local6]) * df(-w0BD - wLFBD - weBD_sq_div2)
        )

    # grad[:, 7] corresponds to wBDy
    idx_BDy = idx_local3 | idx_local4 | idx_local6
    grad[:, 7] = 0
    grad[idx_BDy, 7] += (-1 / weps_sq) * qBD[idx_BDy] * (wBDy - y[idx_BDy] + w0)
    grad[idx_BDy, 7] -= (1 / weps_sq) * (
        (idx_local3 * wLSy * qBD * qLS)
        + (idx_local4 * wLFy * qBD * qLF)
        + (idx_local6 * wLSy * wLFy * qBD * qLS * qLF)
    )[idx_BDy]

    # grad[:, 8] corresponds to wLSy
    idx_LSy = idx_local1 | idx_local3 | idx_local5 | idx_local6
    grad[:, 8] = 0
    grad[idx_LSy, 8] += (-1 / weps_sq) * qLS[idx_LSy] * (wLSy - y[idx_LSy] + w0)
    grad[idx_LSy, 8] -= (1 / weps_sq) * (
        (idx_local3 * wBDy * qBD * qLS)
        + (idx_local6 * wBDy * wLFy * qBD * qLS * qLF)
    )[idx_LSy]

    # grad[:, 9] corresponds to wLFy
    idx_LFy = idx_local2 | idx_local4 | idx_local5 | idx_local6
    grad[:, 9] = 0
    grad[idx_LFy, 9] += (-1 / weps_sq) * qLF[idx_LFy] * (wLFy - y[idx_LFy] + w0)
    grad[idx_LFy, 9] -= (1 / weps_sq) * (
        (idx_local4 * wBDy * qBD * qLF)
        + (idx_local6 * wBDy * wLSy * qBD * qLF * qLS)
    )[idx_LFy]

    # grad[:, 10] corresponds to weLS
    grad[:, 10] = 0
    grad[idx_LS, 10] += (
        - qLS[idx_LS] * weLS * df(w0LS + waLS * alLS[idx_LS] - weLS_sq_div2)
        - (1 - qLS[idx_LS]) * weLS * df(-w0LS - waLS * alLS[idx_LS] - weLS_sq_div2)
    )

    # grad[:, 11] corresponds to weLF
    grad[:, 11] = 0
    grad[idx_LF, 11] += (
        - qLF[idx_LF] * weLF * df(w0LF + waLF * alLF[idx_LF] - weLF_sq_div2)
        - (1 - qLF[idx_LF]) * weLF * df(-w0LF - waLF * alLF[idx_LF] - weLF_sq_div2)
    )

    # grad[:, 12] corresponds to weBD
    grad[:, 12] = 0
    if np.any(idx_local3):
        grad[idx_local3, 12] += (
            - qBD[idx_local3] * qLS[idx_local3] * weBD * df(w0BD + wLSBD - weBD_sq_div2)
            - (1 - qBD[idx_local3]) * qLS[idx_local3] * weBD * df(-w0BD - wLSBD - weBD_sq_div2)
            - qBD[idx_local3] * (1 - qLS[idx_local3]) * weBD * df(w0BD - weBD_sq_div2)
            - (1 - qBD[idx_local3]) * (1 - qLS[idx_local3]) * weBD * df(-w0BD - weBD_sq_div2)
        )

    if np.any(idx_local4):
        grad[idx_local4, 12] += (
            - qBD[idx_local4] * qLF[idx_local4] * weBD * df(w0BD + wLFBD - weBD_sq_div2)
            - (1 - qBD[idx_local4]) * qLF[idx_local4] * weBD * df(-w0BD - wLFBD - weBD_sq_div2)
            - qBD[idx_local4] * (1 - qLF[idx_local4]) * weBD * df(w0BD - weBD_sq_div2)
            - (1 - qBD[idx_local4]) * (1 - qLF[idx_local4]) * weBD * df(-w0BD - weBD_sq_div2)
        )

    if np.any(idx_local6):
        grad[idx_local6, 12] += (
            - qBD[idx_local6] * qLS[idx_local6] * qLF[idx_local6] * weBD * df(w0BD + wLSBD + wLFBD - weBD_sq_div2)
            - qBD[idx_local6] * qLS[idx_local6] * (1 - qLF[idx_local6]) * weBD * df(w0BD + wLSBD - weBD_sq_div2)
            - qBD[idx_local6] * (1 - qLS[idx_local6]) * qLF[idx_local6] * weBD * df(w0BD + wLFBD - weBD_sq_div2)
            - qBD[idx_local6] * (1 - qLS[idx_local6]) * (1 - qLF[idx_local6]) * weBD * df(w0BD - weBD_sq_div2)
            - (1 - qBD[idx_local6]) * qLS[idx_local6] * qLF[idx_local6] * weBD * df(-w0BD - wLSBD - wLFBD - weBD_sq_div2)
            - (1 - qBD[idx_local6]) * qLS[idx_local6] * (1 - qLF[idx_local6]) * weBD * df(-w0BD - wLSBD - weBD_sq_div2)
            - (1 - qBD[idx_local6]) * (1 - qLS[idx_local6]) * qLF[idx_local6] * weBD * df(-w0BD - wLFBD - weBD_sq_div2)
            - (1 - qBD[idx_local6]) * (1 - qLS[idx_local6]) * (1 - qLF[idx_local6]) * weBD * df(-w0BD - weBD_sq_div2)
        )

    # grad[:, 13] corresponds to waLS
    grad[:, 13] = 0
    grad[idx_LS, 13] += (
        qLS[idx_LS] * alLS[idx_LS] * df(w0LS + waLS * alLS[idx_LS] - weLS_sq_div2)
        - (1 - qLS[idx_LS]) * alLS[idx_LS] * df(-w0LS - waLS * alLS[idx_LS] - weLS_sq_div2)
    )

    # grad[:, 14] corresponds to waLF
    grad[:, 14] = 0
    grad[idx_LF, 14] += (
        qLF[idx_LF] * alLF[idx_LF] * df(w0LF + waLF * alLF[idx_LF] - weLF_sq_div2)
        - (1 - qLF[idx_LF]) * alLF[idx_LF] * df(-w0LF - waLF * alLF[idx_LF] - weLF_sq_div2)
    )

    # Ensure no NaNs in gradient
    grad = np.nan_to_num(grad)

    return grad
