# Code adapted from https://github.com/umutsimsekli/sgd_tail_index
import math
import numpy as np

def alpha_estimator(m, X):
    # X is N by d matrix
    N = len(X)
    n = int(N / m) # must be an integer
    Y = np.sum(X.reshape(n, m, -1), axis=1)
    eps = np.spacing(1)
    Y_log_norm = np.mean(np.log(np.linalg.norm(Y, axis=1) + eps))
    X_log_norm = np.mean(np.log(np.linalg.norm(X, axis=1) + eps))
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff

