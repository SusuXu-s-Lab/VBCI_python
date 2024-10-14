# /methods/loss.py

import numpy as np
from .tfxn import f

def Loss(y, qBD, qLS, qLF, alLS, alLF, w, local, delta):
    """
    Compute the Loss function
    
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
        Total loss.
    """
    # Unpack weights
    w0, weps, w0BD, w0LS, w0LF, wLSBD, wLFBD, wBDy, wLSy, wLFy, weLS, weLF, weBD, waLS, waLF = w

    y = np.log(1e-6 + y)

    # Clip qBD, qLS, qLF to avoid log(0)
    f_qBD = np.clip(qBD, 1e-6, 1 - 1e-6)
    f_qLS = np.clip(qLS, 1e-6, 1 - 1e-6)
    f_qLF = np.clip(qLF, 1e-6, 1 - 1e-6)

    # Initialize LBD, LLS, LLF
    LBD = np.zeros_like(y)
    LLS = np.zeros_like(y)
    LLF = np.zeros_like(y)

    idx_local3 = (local == 3)
    idx_local4 = (local == 4)
    idx_local6 = (local == 6)

    weBD_sq_div2 = (weBD ** 2) / 2

    # Compute LBD
    if np.any(idx_local3):
        LBD[idx_local3] += (
            qBD[idx_local3] * (qLS[idx_local3] * f(- w0BD - wLSBD + weBD_sq_div2) + (1 - qLS[idx_local3]) * f(-w0BD + weBD_sq_div2))
            + (1 - qBD[idx_local3]) * (qLS[idx_local3] * f(w0BD + wLSBD + weBD_sq_div2) + (1 - qLS[idx_local3]) * f(w0BD + weBD_sq_div2))
            - (qBD[idx_local3] * np.log(f_qBD[idx_local3]) + (1 - qBD[idx_local3]) * np.log(1 - f_qBD[idx_local3]))
        )

    if np.any(idx_local4):
        LBD[idx_local4] += (
            qBD[idx_local4] * (qLF[idx_local4] * f(- w0BD - wLFBD + weBD_sq_div2) + (1 - qLF[idx_local4]) * f(-w0BD + weBD_sq_div2))
            + (1 - qBD[idx_local4]) * (qLF[idx_local4] * f(w0BD + wLFBD + weBD_sq_div2) + (1 - qLF[idx_local4]) * f(w0BD + weBD_sq_div2))
            - (qBD[idx_local4] * np.log(f_qBD[idx_local4]) + (1 - qBD[idx_local4]) * np.log(1 - f_qBD[idx_local4]))
        )

    if np.any(idx_local6):
        qLS_local6 = qLS[idx_local6]
        qLF_local6 = qLF[idx_local6]
        qBD_local6 = qBD[idx_local6]
        term1 = (
            qBD_local6 * (
                qLS_local6 * qLF_local6 * f(- w0BD - wLSBD - wLFBD + weBD_sq_div2)
                + qLS_local6 * (1 - qLF_local6) * f(- w0BD - wLSBD + weBD_sq_div2)
                + (1 - qLS_local6) * qLF_local6 * f(- w0BD - wLFBD + weBD_sq_div2)
                + (1 - qLS_local6) * (1 - qLF_local6) * f(- w0BD + weBD_sq_div2)
            )
            + (1 - qBD_local6) * (
                qLS_local6 * qLF_local6 * f(w0BD + wLSBD + wLFBD + weBD_sq_div2)
                + qLS_local6 * (1 - qLF_local6) * f(w0BD + wLSBD + weBD_sq_div2)
                + (1 - qLS_local6) * qLF_local6 * f(w0BD + wLFBD + weBD_sq_div2)
                + (1 - qLS_local6) * (1 - qLF_local6) * f(w0BD + weBD_sq_div2)
            )
            - (qBD_local6 * np.log(f_qBD[idx_local6]) + (1 - qBD_local6) * np.log(1 - f_qBD[idx_local6]))
        )
        LBD[idx_local6] += term1

    # Compute LLS
    idx_local1_3_5_6 = (local == 1) | (local == 3) | (local == 5) | (local == 6)
    if np.any(idx_local1_3_5_6):
        LLS[idx_local1_3_5_6] += (
            qLS[idx_local1_3_5_6] * f(- w0LS - waLS * alLS[idx_local1_3_5_6] + (weLS ** 2) / 2)
            + (1 - qLS[idx_local1_3_5_6]) * f(w0LS + waLS * alLS[idx_local1_3_5_6] + (weLS ** 2) / 2)
            - (qLS[idx_local1_3_5_6] * np.log(f_qLS[idx_local1_3_5_6]) + (1 - qLS[idx_local1_3_5_6]) * np.log(1 - f_qLS[idx_local1_3_5_6]))
        )

    # Compute LLF similarly
    idx_local2_4_5_6 = (local == 2) | (local == 4) | (local == 5) | (local == 6)
    if np.any(idx_local2_4_5_6):
        LLF[idx_local2_4_5_6] += (
            qLF[idx_local2_4_5_6] * f(- w0LF - waLF * alLF[idx_local2_4_5_6] + (weLF ** 2) / 2)
            + (1 - qLF[idx_local2_4_5_6]) * f(w0LF + waLF * alLF[idx_local2_4_5_6] + (weLF ** 2) / 2)
            - (qLF[idx_local2_4_5_6] * np.log(f_qLF[idx_local2_4_5_6]) + (1 - qLF[idx_local2_4_5_6]) * np.log(1 - f_qLF[idx_local2_4_5_6]))
        )

    # Compute Leps
    Leps = - (0.5 * np.log(weps ** 2) + (y - w0) ** 2 / (2 * (weps ** 2)))
    term = (1 / (2 * (weps ** 2))) * (
        ((local == 3) | (local == 4) | (local == 6)) * wBDy * qBD * (wBDy - 2 * y + 2 * w0)
        + ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * wLSy * qLS * (wLSy - 2 * y + 2 * w0)
        + ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * wLFy * qLF * (wLFy - 2 * y + 2 * w0)
    )
    Leps -= term

    Leps -= (1 / (weps ** 2)) * (
        ((local == 3) * wBDy * wLSy * qBD * qLS)
        + ((local == 4) * wBDy * wLFy * qBD * qLF)
        + ((local == 5) * wLSy * wLFy * qLS * qLF)
        + ((local == 6) * wBDy * wLSy * wLFy * qBD * qLS * qLF)
    )

    Leps -= y

    # L_ex
    idx_local5_6 = (local == 5) | (local == 6)
    L_ex = np.zeros_like(y)
    L_ex[idx_local5_6] = - (qLS[idx_local5_6] * qLF[idx_local5_6]) / (2 * delta)
    L_ex = np.maximum(L_ex, -1e+2)

    # Total Loss
    L = LBD + LLS + LLF + Leps + L_ex

    return L
