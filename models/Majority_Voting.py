import numpy as np


def inference(task_worker_class, n, m, K):
    '''
    :param task_worker_class:   Input. Each element (i, j, k) represents that worker j labeled task i as class k.
    :param n:                   The number of tasks.
    :param m:                   The number of workers. This value is not used.
    :param K:                   The number of classes.
    :return T:                  Estimates of class of each task. T[i] denotes the class of task i.
    This function estimates the true classes of tasks by using majority voting scheme.
    '''

    x = np.zeros([n, m, K])
    for i, j, k in task_worker_class:
        x[i, j, k] = 1
    g_hat_ = np.einsum('ki->ik', np.einsum('ijk->ki', x) / np.einsum('ijk->i', x))
    g_hat = np.argmax(g_hat_, axis=1)
    return g_hat
