from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import numpy as np


def cluster_acc(y_pred, y_true, return_matched=False):
    """
    Calculate clustering accuracy. Require scipy installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    if return_matched:
        matched = np.array(list(map(lambda i: col_ind[i], y_pred)))
        return w[row_ind, col_ind].sum() / y_pred.size, matched
    else:
        return w[row_ind, col_ind].sum() / y_pred.size


def cluster_ari(y_pred, y_true):
    """
    Calculate adjusted rand index. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        ARI, in [0,1]
    """
    return adjusted_rand_score(y_true, y_pred)
