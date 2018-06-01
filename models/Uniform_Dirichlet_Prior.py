import numpy as np
import scipy.special as sp


def convert_input(task_class_worker, n, m, K):
    x = np.zeros([n, m, K])
    for i, j, k in task_class_worker:
        x[i, j, k] = 1
    return x


def convergence_condition(elbo_new, elbo_old, epsilon):
    if elbo_new - elbo_old < 0:
        return False
    elif elbo_new - elbo_old < epsilon:
        return True
    else:
        return False


def initialize_parameter(x, delta=1e-100):
    n = x.shape[0]
    m = x.shape[1]
    K = x.shape[2]
    theta = np.einsum('ki->ik', np.einsum('ijk->ki', x) / np.einsum('ijk->i', x))

    x += delta
    alpha_comp = np.einsum('mjk->jkm', np.einsum('ik,ijm->mjk', theta, x) / np.einsum('ik,ijm->jk', theta, x))

    rho = np.einsum('ik->k', theta) / n

    alpha = solve_eq(m, K, alpha_comp)

    return theta, alpha_comp, rho, alpha


def solve_eq(m, K, alpha_comp):
    beta = np.einsum('jk->', np.einsum('jkm->jk', sp.digamma(alpha_comp))
                     - K * sp.digamma(np.einsum('jkm->jk', alpha_comp))) / (m * K * K)
    alpha0 = 0.1

    def f(x):
        return sp.digamma(x) - sp.digamma(K * x) - beta

    def f_prime(x):
        return sp.polygamma(1, x) - K * sp.polygamma(1, K * x)

    alpha = newton(f, f_prime, alpha0)
    return alpha


def newton(f, f_prime, x0, epsilon=1e-2, delta=1e-100):
    x_ = x0
    while True:
        x = x_ - f(x_) / f_prime(x_)
        if x < 0:
            x = delta
            break
        elif np.abs(x - x_) < epsilon:
            break
        else:
            x_ = x
    return x


def variational_update(x, theta, alpha_comp, rho, alpha, delta=1e-100):
    dig_alpha = sp.digamma(np.einsum('jkm->jk', alpha_comp))
    dig_alpha.shape = dig_alpha.shape + (1,)
    dig_alpha = np.broadcast_arrays(alpha_comp, dig_alpha)[1]
    theta_new = np.exp(np.einsum('ijm,jkm->ik', x, sp.digamma(alpha_comp) - dig_alpha)
                   + np.log(rho + delta))
    alpha_comp_new = np.einsum('ijm,ik->jkm', x, theta) + alpha
    return theta_new, alpha_comp_new


def hyperparameter_update(x, theta, alpha_comp):
    n = x.shape[0]
    m = x.shape[1]
    K = x.shape[2]
    alpha = solve_eq(m, K, alpha_comp)
    rho = np.einsum('ik->k', theta) / n
    return rho, alpha


def elbo(x, theta, alpha_comp, rho, alpha, delta=1e-100):
    m = x.shape[1]
    K = x.shape[2]
    dig_alpha = sp.digamma(np.einsum('jkm->jk', alpha_comp))
    dig_alpha.shape = dig_alpha.shape + (1,)
    dig_alpha = np.broadcast_arrays(alpha_comp, dig_alpha)[1]
    dig_alpha = sp.digamma(alpha_comp) - dig_alpha
    elbo = np.einsum('ijm,ik,jkm->', x, theta, dig_alpha)
    elbo += np.einsum('ik,k->', theta, np.log(rho + delta))
    elbo -= np.einsum('ik,ik->', theta, np.log(theta + delta))
    elbo += (alpha - 1) * np.einsum('jkm->', dig_alpha)
    elbo += m * K * sp.gammaln(K * alpha)
    elbo -= m * K * K * sp.gammaln(alpha)
    elbo -= np.einsum('jkm,jkm->', (alpha_comp - 1), dig_alpha)
    elbo -= np.einsum('jk->', sp.gammaln(np.einsum('jkm->jk', alpha_comp)))
    elbo += np.einsum('jkm->', sp.gammaln(alpha_comp))
    if np.isnan(elbo):
        print("theta = ", theta)
        print("alpha_comp = ", alpha_comp)
        print("rho = ", rho)
        print("alpha = ", alpha)
        raise ValueError("ELBO is NaN!")
    return elbo


def one_iteration(x, epsilon):
    theta, alpha_comp, rho, alpha = initialize_parameter(x)
    l = -1e+100
    while True:
        theta, alpha_comp = variational_update(x, theta, alpha_comp, rho, alpha)
        rho, alpha = hyperparameter_update(x, theta, alpha_comp)
        l_ = elbo(x, theta, alpha_comp, rho, alpha)
        if convergence_condition(l, l_, epsilon):
            break
        else:
            l = l_
    g = np.argmax(theta, axis=1)
    alpha_comp_ = np.einsum('jkm->mjk', alpha_comp)
    pi = np.einsum('mjk->jkm', alpha_comp_ / np.sum(alpha_comp_, axis=0))
    return g, pi, rho, alpha


def inference(task_worker_class, n, m, K, epsilon=1e-2):
    x = convert_input(task_worker_class, n, m, K)
    g, pi, rho, alpha = one_iteration(x, epsilon)
    return g, [pi, rho, alpha]