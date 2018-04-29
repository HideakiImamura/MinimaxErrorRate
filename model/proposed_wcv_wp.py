import numpy as np
from scipy import special
import pandas as pd


def is_chance_rate(theta):
    n = theta.shape[0]
    K = theta.shape[1]
    sum = np.sum(theta, axis=0)
    for k in range(K):
        if int(sum[k]) == n:
            return True
    return False


def truncate(x):
    if x < 0.0:
        return 0.0
    elif x > 1.0:
        return 1.0
    else:
        return x


def convert_input(task_class_worker, n, m, K):
    x = np.zeros([n, m, K])
    for i, j, k in task_class_worker:
        x[i, j, k] = 1
    return x


def initialize_parameter(x, K, L, random=True, delta=1e-100):
    n = x.shape[0]
    m = x.shape[1]

    if random:
        theta = np.clip(np.einsum('ki->ik', np.einsum('ijk->ki', x) / np.einsum('ijk->i', x)) + np.random.normal(scale=0.1, size=[n, K]), 0.0, 1.0)
        theta = np.einsum('ki->ik', np.einsum('ik->ki', theta) / np.sum(theta, axis=1))
    else:
        theta = np.einsum('ki->ik', np.einsum('ijk->ki', x) / np.einsum('ijk->i', x))

    x += delta
    pi = np.einsum('mjk->jkm', np.einsum('ik,ijm->mjk', theta, x) / np.einsum('ik,ijm->jk', theta, x))
    order = np.array([np.linalg.norm(pi[j], ord='nuc') for j in range(m)])
    sigma = np.array(sorted(np.c_[np.arange(m), order], key=lambda pair: pair[1], reverse=True))[:, 0].astype(dtype=int)
    J = np.array([sigma[int(m * l / L): int(m * (l + 1) / L)] for l in range(L)])
    lambda_ = np.array([(np.sum(pi[J[l]], axis=0)) * L / m for l in range(L)])

    phi = np.zeros([m, L])
    for l in range(L):
        for j in J[l]:
            phi[j, l] = 1.0

    alpha = np.ones(K, dtype=float)
    beta = np.ones(L, dtype=float)

    alpha_hat = np.sum(theta, axis=0) + 1
    beta_hat = 2 * np.ones(L, dtype=float)

    return theta, phi, alpha_hat, beta_hat, alpha, beta, lambda_


def variational_update(x, theta, phi, alpha_hat, beta_hat, alpha, beta, lambda_, delta=1e-100):
    alpha_hat_new = np.sum(theta, axis=0) + alpha
    beta_hat_new = np.sum(phi, axis=0) + beta

    log_rho = special.digamma(alpha_hat) - special.digamma(np.sum(alpha_hat))
    log_tau = special.digamma(beta_hat) - special.digamma(np.sum(beta_hat))

    theta_prime = np.exp(np.einsum('ijm,jl,lkm->ik', x, phi, np.log(lambda_ + delta)) + log_rho)
    phi_prime = np.exp(np.einsum('ijm,ik,lkm->jl', x, theta, np.log(lambda_ + delta)) + log_tau)
    theta = np.einsum('ki->ik', theta_prime.T / np.sum(theta_prime.T, axis=0))
    phi = np.einsum('lj->jl', phi_prime.T / np.sum(phi_prime.T, axis=0))

    return theta, phi, alpha_hat_new, beta_hat_new


def hyper_parameter_update(x, theta, phi, alpha_hat, beta_hat, epsilon):
    lambda_prime = np.einsum('ik,jl,ijm->mlk', theta, phi, x)
    lambda_ = np.einsum('mlk->lkm', lambda_prime / np.sum(lambda_prime, axis=0))
    '''
    log_rho = special.digamma(alpha_hat) - special.digamma(np.sum(alpha_hat))
    log_tau = special.digamma(beta_hat) - special.digamma(np.sum(beta_hat))

    alpha = solve_digamma_equation(log_rho, epsilon)
    beta = solve_digamma_equation(log_tau, epsilon)
    '''
    alpha = alpha_hat
    beta = beta_hat
    return alpha, beta, lambda_


def solve_digamma_equation(log, epsilon):
    K = log.shape[0]
    alpha = np.ones(K)
    while True:
        f = open("log.txt", 'a')
        f.write("       alp = " + str(alpha) + "\n")
        f.close()

        log_alpha = log + special.digamma(np.sum(alpha))

        alpha_ = np.array([digamma_inv(log_alpha[k], epsilon) for k in range(K)])
        if convergence_condition_vector(alpha_, alpha, epsilon):
            break
        else:
            alpha = alpha_
    return alpha


def digamma_inv(y, epsilon):
    if y >= -2.22:
        x = np.exp(y) + 0.5
    else:
        x = - 1.0 / (1.0 - special.digamma(1))
    while True:
        dx = (y - special.digamma(x)) / special.polygamma(1, x)
        x += dx
        f = open("log.txt", 'a')
        f.write("               " + str(dx) + "\n")
        f.close()
        if np.abs(dx) < epsilon:
            break
    return x


def convergence_condition_vector(x_, x, epsilon):
    value = True
    for i in range(x.shape[0]):
        if np.abs(x_[i] - x[i]) / x_[i] >= epsilon:
            value = False
            break
    return value


def digamma_sum(alpha):
    return special.digamma(alpha) - special.digamma(np.sum(alpha))


def log_gamma(alpha):
    return special.gammaln(np.sum(alpha)) - np.sum(special.gammaln(alpha))


def elbo(x, theta, phi, alpha_hat, beta_hat, alpha, beta, lambda_, delta=1e-100):
    l = np.einsum('ik,jl,ijm,lkm->', theta, phi, x, np.log(lambda_ + delta)) + \
        np.einsum('ik,k->', theta, digamma_sum(alpha_hat)) + \
        np.einsum('jl,l->', phi, digamma_sum(beta_hat)) + \
        np.sum((alpha - alpha_hat) * digamma_sum(alpha_hat)) + \
        np.sum((beta - beta_hat) * digamma_sum(beta_hat)) - \
        np.einsum('ik,ik->', theta, np.log(theta + delta)) - \
        np.einsum('jl,jl->', phi, np.log(phi + delta)) + \
        log_gamma(alpha) - log_gamma(alpha_hat) + log_gamma(beta) - log_gamma(beta_hat)
    if np.isnan(l):
        return 1
        print("theta = ", theta)
        print("phi = ", phi)
        print("alpha_hat = ", alpha_hat)
        print("beta_hat = ", beta_hat)
        print("alpha = ", alpha)
        print("beta = ", beta)
        print("Lambda = ", lambda_)
        raise ValueError("ELBO is Nan!")
    return l


def convergence_condition(elbo_new, elbo_old, epsilon):
    if elbo_new - elbo_old < 0:
        # f = open("log.txt", "a")
        # f.write("ELBO is decrease!!!\n")
        # f.close()
        return False
    elif elbo_new - elbo_old < epsilon:
        return True
    else:
        return False


def EVI(task_worker_class, n, m, K, L, epsilon=1e-2):
    # f = open("log.txt", 'w')
    # f.close()
    acc_param = 1e-2
    x = convert_input(task_worker_class, n, m, K)

    theta, phi, alpha_hat, beta_hat, alpha, beta, lambda_ = initialize_parameter(x, K, L, random=False)
    l = -1e+100
    c = 1
    while True:
        theta, phi, alpha_hat, beta_hat = variational_update(x,
                                                                theta,
                                                                phi,
                                                                alpha_hat,
                                                                beta_hat,
                                                                alpha,
                                                                beta,
                                                                lambda_
                                                                )
        alpha, beta, lambda_ = hyper_parameter_update(x,
                                                        theta,
                                                        phi,
                                                        alpha_hat,
                                                        beta_hat,
                                                        acc_param)
        l_ = elbo(x, theta, phi, alpha_hat, beta_hat, alpha, beta, lambda_)
        if convergence_condition(l_, l, epsilon):
            break
        else:
            l = l_

    while is_chance_rate(theta):
        c += 1
        theta, phi, alpha_hat, beta_hat, alpha, beta, lambda_ = initialize_parameter(x, K, L)
        l = -1e+100
        while True:
            theta, phi, alpha_hat, beta_hat = variational_update(x,
                                                                theta,
                                                                phi,
                                                                alpha_hat,
                                                                beta_hat,
                                                                alpha,
                                                                beta,
                                                                lambda_
                                                                )
            alpha, beta, lambda_ = hyper_parameter_update(x,
                                                        theta,
                                                        phi,
                                                        alpha_hat,
                                                        beta_hat,
                                                        acc_param)
            l_ = elbo(x, theta, phi, alpha_hat, beta_hat, alpha, beta, lambda_)
            if convergence_condition(l_, l, epsilon):
                break
            else:
                l = l_
        if c >= 100:
            break
    g_hat = np.argmax(theta, axis=1)
    pi_hat = lambda_[np.argmax(phi, axis=1)]
    return g_hat, pi_hat, c
