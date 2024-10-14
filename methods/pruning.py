# methods/pruning.py

import numpy as np

def pruning(BD, LS, LF, sigma, side):
    """
    Pruning function to classify the local model based on BD, LS, and LF probabilities.

    Parameters:
    BD    : np.ndarray
        Binary indicator for BD.
    LS    : np.ndarray
        Landslide probabilities.
    LF    : np.ndarray
        Liquefaction probabilities.
    sigma : float
        Threshold parameter.
    side  : str
        Type of pruning ('single' or 'double').

    Returns:
    np.ndarray
        Local model classification.
    """
    model = np.zeros(BD.shape, dtype=int)

    if side == "single":
        # 1 - LS alone
        model += ((BD == 0) & (LF <= sigma)).astype(int)

        # 2 - LF alone
        model += 2 * ((BD == 0) & (LF > sigma) & (LF > LS)).astype(int)

        # 3 - LS and BD
        model += 3 * ((BD == 1) & (LF <= sigma)).astype(int)

        # 4 - LF and BD
        model += 4 * ((BD == 1) & (LF > sigma) & (LF > LS)).astype(int)
    else:
        # 1 - LS alone
        model += ((BD == 0) & (LS > LF + sigma) & (LS > 0)).astype(int)

        # 2 - LF alone
        model += 2 * ((BD == 0) & (LF > LS + sigma) & (LF > 0)).astype(int)

        # 3 - LS and BD
        model += 3 * ((BD == 1) & (LS > LF + sigma) & (LS > 0)).astype(int)

        # 4 - LF and BD
        model += 4 * ((BD == 1) & (LF > LS + sigma) & (LF > 0)).astype(int)

        # 5 - LF and LS
        model += 5 * ((BD == 0) & (np.abs(LF - LS) <= sigma)).astype(int)

        # 6 - LF and LS and BD
        model += 6 * ((BD == 1) & (np.abs(LF - LS) <= sigma)).astype(int)

    return model
