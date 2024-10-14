# /evaluation/cel.py

import numpy as np

def cel(prio, post, gtruth):
    """
    Compute the cross-entropy loss for prior and posterior.

    Parameters:
    prio   : np.ndarray
        Prior probabilities.
    post   : np.ndarray
        Posterior probabilities.
    gtruth : np.ndarray
        Ground truth labels (0 or 1).

    Returns:
    tuple
        Cross-entropy loss for prior and posterior (ploss, qloss).
    """
    tmp_prio = (prio - np.min(prio)) / (np.max(prio) - np.min(prio))
    tmp_prio = np.clip(tmp_prio, 1e-6, 1 - 1e-6)
    ploss = -np.mean(gtruth * np.log(tmp_prio) + (1 - gtruth) * np.log(1 - tmp_prio))

    tmp_post = (post - np.min(post)) / (np.max(post) - np.min(post))
    tmp_post = np.clip(tmp_post, 1e-6, 1 - 1e-6)
    qloss = -np.mean(gtruth * np.log(tmp_post) + (1 - gtruth) * np.log(1 - tmp_post))

    return ploss, qloss
