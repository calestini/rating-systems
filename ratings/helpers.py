import sys
import numpy as np

"""
http://scikit-learn.org/stable/modules/model_evaluation.html
"""

def progress(count, total, status=''):
    """
    Adapted from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[{0}] {1}{2} ... {3}\r'.format(bar, percents, '%', status))
    sys.stdout.flush()


def log_loss(s, p):
    """
    Calculates the log loss given two np arrays: actual outcome and probability projection
    """
    p = np.maximum(np.minimum(p, 1 - 10**-15), 10**-15)
    arr = -(s * np.log10(p) + (1 - s) * np.log10(1 - p))
    return np.mean(arr)


def squared_error(s, p):
    """
    Calculates the mean squared error given two np arrays: actual outcome and probability projection
    """
    arr = np.power((s-p), 2)
    return np.mean(arr)