import random
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cv_score(X, y):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    clf = LogisticRegression(penalty=None)
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return np.mean(scores)


def check_both_none_or_not_none(arg1, arg2):
    return (arg1 is None and arg2 is None) or (arg1 is not None and arg2 is not None)
