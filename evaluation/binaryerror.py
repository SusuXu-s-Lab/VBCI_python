# /evaluation/binaryerror.py

import numpy as np

def binaryerror(prio, post, gtruth, prio_thresh, post_thresh):
    """
    Compute binary classification error metrics for prior and posterior.

    Parameters:
    prio       : np.ndarray
        Prior probabilities.
    post       : np.ndarray
        Posterior probabilities.
    gtruth     : np.ndarray
        Ground truth labels (0 or 1).
    prio_thresh: float
        Threshold for prior classification.
    post_thresh: float
        Threshold for posterior classification.

    Returns:
    tuple
        Metrics for prior and posterior:
        (prio_TPR, prio_FPR, prio_TNR, prio_FNR, prio_PRE,
         post_TPR, post_FPR, post_TNR, post_FNR, post_PRE)
    """
    # Compute TP, TN, FN, FP for prior
    prio_TP = np.sum((gtruth == 1) & (prio > prio_thresh))
    prio_TN = np.sum((gtruth == 0) & (prio <= prio_thresh))
    prio_FN = np.sum((gtruth == 1) & (prio <= prio_thresh))
    prio_FP = np.sum((gtruth == 0) & (prio > prio_thresh))

    # Compute TP, TN, FN, FP for posterior
    post_TP = np.sum((gtruth == 1) & (post > post_thresh))
    post_TN = np.sum((gtruth == 0) & (post <= post_thresh))
    post_FN = np.sum((gtruth == 1) & (post <= post_thresh))
    post_FP = np.sum((gtruth == 0) & (post > post_thresh))

    # Compute metrics for prior
    prio_TPR = prio_TP / (prio_TP + prio_FN) if (prio_TP + prio_FN) > 0 else 0
    prio_FPR = prio_FP / (prio_FP + prio_TN) if (prio_FP + prio_TN) > 0 else 0
    prio_TNR = prio_TN / (prio_TN + prio_FP) if (prio_TN + prio_FP) > 0 else 0
    prio_FNR = prio_FN / (prio_FN + prio_TP) if (prio_FN + prio_TP) > 0 else 0
    prio_PRE = prio_TP / (prio_TP + prio_FP) if (prio_TP + prio_FP) > 0 else 0

    # Compute metrics for posterior
    post_TPR = post_TP / (post_TP + post_FN) if (post_TP + post_FN) > 0 else 0
    post_FPR = post_FP / (post_FP + post_TN) if (post_FP + post_TN) > 0 else 0
    post_TNR = post_TN / (post_TN + post_FP) if (post_TN + post_FP) > 0 else 0
    post_FNR = post_FN / (post_FN + post_TP) if (post_FN + post_TP) > 0 else 0
    post_PRE = post_TP / (post_TP + post_FP) if (post_TP + post_FP) > 0 else 0

    return prio_TPR, prio_FPR, prio_TNR, prio_FNR, prio_PRE, \
           post_TPR, post_FPR, post_TNR, post_FNR, post_PRE
