# /evaluation/rocdetpr.py

import numpy as np
import matplotlib.pyplot as plt
from .binaryerror import binaryerror

def rocdetpr(title, prio, post, gtruth, location):
    """
    Plot ROC, DET, and PR curves for prior and posterior probabilities.

    Parameters:
    title     : str
        Title for the plots.
    prio      : np.ndarray
        Prior probabilities.
    post      : np.ndarray
        Posterior probabilities.
    gtruth    : np.ndarray
        Ground truth labels (0 or 1).
    location  : str
        Directory to save the plots.

    Returns:
    np.ndarray
        Array containing metrics for various thresholds.
    """
    acc = []
    ith_percentile = np.arange(0, 1.01, 0.01)

    for l in ith_percentile:
        # Compute thresholds
        prio_thresh = np.quantile(prio.flatten(), l)
        post_thresh = np.quantile(post.flatten(), l)

        # Compute error metrics
        prio_TPR, prio_FPR, prio_TNR, prio_FNR, prio_PRE, \
        post_TPR, post_FPR, post_TNR, post_FNR, post_PRE = binaryerror(
            prio, post, gtruth, prio_thresh, post_thresh)

        # Count the number of estimates above the threshold
        tmp_p_c = np.sum(prio > prio_thresh)
        tmp_q_c = np.sum(post > post_thresh)

        # Append
        tmp_acc = [tmp_p_c, tmp_q_c, prio_TPR, prio_FPR, prio_TNR, prio_FNR, prio_PRE,
                   post_TPR, post_FPR, post_TNR, post_FNR, post_PRE]
        acc.append(tmp_acc)

    acc = np.array(acc)

    # Plot ROC Curve
    plt.figure()
    plt.plot(acc[:, 3], acc[:, 2], linewidth=1.5, label='Prior')
    plt.plot(acc[:, 8], acc[:, 7], linewidth=1.5, label='Posterior')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title(f'{title} ROC Curve')
    plt.grid(True)
    plt.savefig(f"{location}{title}_ROC.png")
    plt.close()

    # Plot DET Curve
    plt.figure()
    plt.loglog(acc[:, 3], acc[:, 5], linewidth=1.5, label='Prior')
    plt.loglog(acc[:, 8], acc[:, 10], linewidth=1.5, label='Posterior')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.legend(loc='upper right')
    plt.title(f'{title} DET Curve')
    plt.grid(True)
    plt.savefig(f"{location}{title}_DET.png")
    plt.close()

    # Plot PR Curve
    plt.figure()
    plt.plot(acc[:, 2], acc[:, 6], linewidth=1.5, label='Prior')
    plt.plot(acc[:, 7], acc[:, 11], linewidth=1.5, label='Posterior')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right')
    plt.title(f'{title} PR Curve')
    plt.grid(True)
    plt.savefig(f"{location}{title}_PR.png")
    plt.close()

    return acc
