# methods/svi.py

import numpy as np
from .tfxn import Tfxn
from .loss import Loss
from .parder import parder
from .pruning import pruning

def SVI(Y, LS, LF, w, Nq, rho, delta, eps_0, LOCAL, lambda_, regu_type, sigma, prune_type):
    """
    Stochastic Variational Inference (SVI) function.

    Parameters:
    Y         : np.ndarray
        Observed data.
    LS        : np.ndarray
        Landslide probabilities.
    LF        : np.ndarray
        Liquefaction probabilities.
    w         : np.ndarray
        Initial weight vector.
    Nq        : int
        Number of Posterior Probability Iterations.
    rho       : float
        Step size for weight updates.
    delta     : float
        Acceptable tolerance for weight optimization.
    eps_0     : float
        Lower-bound non-negative weight.
    LOCAL     : np.ndarray
        Local model classification.
    lambda_   : float
        Regularization parameter.
    regu_type : int
        Type of regularization.
    sigma     : float
        Threshold for pruning.
    prune_type: str
        Type of pruning ('single' or 'double').

    Returns:
    tuple
        Optimized weights and posterior probabilities along with other outputs.
    """
    # Initialize variables
    QBD = 0.001 * np.random.rand(*Y.shape)
    QLS = LS.copy()
    QLF = LF.copy()
    loss = 1e+5
    loss_old = 0
    best_loss = -1e+4
    final_loss = []
    grad = np.zeros(len(w))
    epoch = 0

    # Compute the alpha that relates to prior LS and LF estimates
    t_alLS = np.log(LS / (1 - LS))
    t_alLF = np.log(LF / (1 - LF))
    t_alLS = np.clip(t_alLS, -6, 6)
    t_alLF = np.clip(t_alLF, -6, 6)
    final_w = w.copy()

    # Set stopping condition
    if np.sum(LOCAL >= 5) == 0:
        def myConditionFunc(x, y): return y > 0
    else:
        def myConditionFunc(x, y): return (x > 0) or (y > 0)

    # Main loop
    while (epoch < 30) and myConditionFunc(np.sum(LOCAL == 5), abs(loss_old - loss) - eps_0):
        # Create a sample batch
        totalnum = Y.size
        bsize = 500
        bnum = totalnum // bsize
        iD_sample = np.random.permutation(totalnum)
        iD = iD_sample[:bnum * bsize].reshape(bnum, bsize)

        # Reset loss parameters
        loss_old = loss
        loss = 0
        eloss = 0
        tmp_final_loss = []

        # Learning rate decay
        if (epoch % 10 == 0) and (epoch > 1):
            rho = max(rho * 0.1, 1e-4)

        # Iterate over batches
        for i in range(bnum):
            idx = np.unravel_index(iD[i], Y.shape)
            y = Y[idx]
            qBD = QBD[idx]
            qLS = QLS[idx]
            qLF = QLF[idx]
            local = LOCAL[idx]
            alLS = t_alLS[idx]
            alLF = t_alLF[idx]

            # Iterations
            for nq in range(Nq):
                # Apply the nonlinear T function
                q = 1 / (1 + np.exp(-Tfxn(y, qBD, qLS, qLF, alLS, alLF, w, local, delta)))
                qBD = q[:, 0]
                qLS = q[:, 1]
                qLF = q[:, 2]

                # Apply pruning
                qBD[local < 3] = 0
                qBD[local == 5] = 0
                qLS[np.isin(local, [0, 2, 4])] = 0
                qLF[np.isin(local, [0, 1, 3])] = 0

                # Remove negligible estimates
                qLS[qLS < 1e-6] = 0
                qLF[qLF < 1e-6] = 0

                # Apply sigma
                tqLS = qLS.copy()
                tqLF = qLF.copy()
                if prune_type == "single":
                    qLS[tqLS < tqLF - sigma] = 0
                    qLF[tqLF < tqLS] = 0
                else:
                    qLS[tqLS < tqLF - sigma] = 0
                    qLF[tqLF < tqLS - sigma] = 0

                # Update posterior estimates
                QBD[idx] = qBD
                QLS[idx] = qLS
                QLF[idx] = qLF

            # Classify new local model by pruning
            if np.sum(local >= 5) > 0:
                if prune_type == "single":
                    local[(tqLF < tqLS) & (local == 5)] = 1
                    local[(tqLF < tqLS) & (local == 6)] = 3
                    local[(tqLS < tqLF - sigma) & (local == 5)] = 2
                    local[(tqLS < tqLF - sigma) & (local == 6)] = 4
                else:
                    local[(tqLF < tqLS - sigma) & (local == 5)] = 1
                    local[(tqLF < tqLS - sigma) & (local == 6)] = 3
                    local[(tqLS < tqLF - sigma) & (local == 5)] = 2
                    local[(tqLS < tqLF - sigma) & (local == 6)] = 4
                LOCAL[idx] = local

            # Compute the partial derivative
            grad_D = parder(y, qBD, qLS, qLF, alLS, alLF, w, local)
            identity = np.column_stack([
                local >= 0, local >= 0, np.isin(local, [3, 4, 5]), np.isin(local, [1, 3, 5, 6]),
                np.isin(local, [2, 4, 5, 6]), np.isin(local, [3, 6]), np.isin(local, [4, 6]),
                np.isin(local, [3, 4, 6]), np.isin(local, [1, 3, 5, 6]), np.isin(local, [2, 4, 5, 6]),
                np.isin(local, [1, 3, 5, 6]), np.isin(local, [2, 4, 5, 6]), np.isin(local, [3, 4, 6]),
                np.isin(local, [1, 3, 5, 6]), np.isin(local, [2, 4, 5, 6])
            ])
            tmp_count = np.sum(identity, axis=0)
            grad = (np.sum(identity * grad_D, axis=0) / tmp_count)
            grad[tmp_count == 0] = 0

            # Compute the regularization
            if regu_type == 1:
                regu_grad = lambda_ * 100 * (1 / (1 + np.exp(-0.01 * w)) - 1 / (1 + np.exp(0.01 * w)))
                grad[[7, 8, 9]] -= regu_grad[[7, 8, 9]]
            else:
                regu_grad = lambda_ * 100 * (1 / (1 + np.exp(-0.01 * w)) - 1 / (1 + np.exp(0.01 * w)))
                grad[[13, 14]] -= regu_grad[[13, 14]]

            # Compute the new weight
            wnext = w + rho * grad
            wnext[[0, 2]] = np.minimum(wnext[[0, 2]], (-1e-6) * np.ones(2))
            wnext[[13, 14]] = np.maximum(0, wnext[[13, 14]])
            wnext[1] = np.clip(wnext[1], 1e-3, 1)

            # Compute the loss
            tmp_loss = np.mean(Loss(y, qBD, qLS, qLF, alLS, alLF, wnext, local, delta))
            loss += tmp_loss
            if i % 20 == 0:
                tmp_final_loss.append(tmp_loss)
            if i % 100 == 0:
                c_loss = np.mean(Loss(Y.flatten(), QBD.flatten(), QLS.flatten(), QLF.flatten(), t_alLS.flatten(), t_alLF.flatten(), wnext, LOCAL.flatten(), delta))
                if c_loss > best_loss:
                    final_QLS = QLS.copy()
                    final_QLF = QLF.copy()
                    final_QBD = QBD.copy()
                    final_w = wnext.copy()
                    best_loss = c_loss

            # Assign the new weight
            w = wnext.copy()

        # Show progress
        print(f"Epoch {epoch + 1}, loss: {loss / bnum}")
        epoch += 1
        final_loss.extend(tmp_final_loss)

    return final_w, final_QBD, final_QLS, final_QLF, QLS, QLF, QBD, final_loss, best_loss, LOCAL
