import numpy as np


def is_chance_rate(theta):
    n = theta.shape[0]
    K = theta.shape[1]
    sum = np.sum(theta, axis=0)
    for k in range(K):
        if int(sum[k]) == n:
            return True
    return False


def convert_input(task_class_worker, n, m, K):
    x = np.zeros([n, m, K])
    for i, j, k in task_class_worker:
        x[i, j, k] = 1
    return x


def initialize_parameter(x, K, L, random=True, delta=1e-100):
    n = x.shape[0]
    m = x.shape[1]

    if random:
        theta = np.clip(
            np.einsum('ki->ik', np.einsum('ijk->ki', x) / np.einsum('ijk->i', x)) + np.random.normal(scale=0.1,
                                                                                                     size=[n, K])
            , 0.0, 1.0)
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

    rho = np.einsum('ik->k', theta) / n
    tau = np.einsum('jl->l', phi) / m
    return theta, phi, rho, tau, lambda_


def variational_update(x, theta, phi, rho, tau, lambda_, delta=1e-100):
    theta_prime = np.exp(np.einsum('ijm,jl,lkm->ik', x, phi, np.log(lambda_ + delta)) + np.log(rho + delta))
    phi_prime = np.exp(np.einsum('ijm,ik,lkm->jl', x, theta, np.log(lambda_ + delta)) + np.log(tau + delta))
    theta = np.einsum('ki->ik', theta_prime.T / np.sum(theta_prime.T, axis=0))
    phi = np.einsum('lj->jl', phi_prime.T / np.sum(phi_prime.T, axis=0))
    return theta, phi


def hyper_parameter_update(x, theta, phi):
    n = x.shape[0]
    m = x.shape[1]

    lambda_prime = np.einsum('ik,jl,ijm->mlk', theta, phi, x)
    lambda_ = np.einsum('mlk->lkm', lambda_prime / np.sum(lambda_prime, axis=0))

    rho = np.einsum('ik->k', theta) / n
    tau = np.einsum('jl->l', phi) / m

    return rho, tau, lambda_


def elbo(x, theta, phi, rho, tau, lambda_, delta=1e-100):
    l = np.einsum('ik,jl,ijm,lkm->', theta, phi, x, np.log(lambda_ + delta)) + \
        np.einsum('ik,k->', theta, np.log(rho + delta)) + \
        np.einsum('jl,l->', phi, np.log(tau + delta)) - \
        np.einsum('ik,ik->', theta, np.log(theta + delta)) - \
        np.einsum('jl,jl->', phi, np.log(phi + delta))
    if np.isnan(l):
        print("theta = ", theta)
        print("phi = ", phi)
        print("rho = ", rho)
        print("tau = ", tau)
        print("Lambda = ", lambda_)
        raise ValueError("ELBO is Nan!")
    return l


def convergence_condition(elbo_new, elbo_old, epsilon):
    if elbo_new - elbo_old < 0:
        return False
    elif elbo_new - elbo_old < epsilon:
        return True
    else:
        return False


def one_iteration(x, K, L, epsilon=1e-2, random=False):
    theta, phi, rho, tau, lambda_ = initialize_parameter(x, K, L, random=random)
    l = -1e+100
    while True:
        theta, phi = variational_update(x,
                                        theta,
                                        phi,
                                        rho,
                                        tau,
                                        lambda_
                                        )
        rho, tau, lambda_ = hyper_parameter_update(x,
                                                   theta,
                                                   phi)
        l_ = elbo(x, theta, phi, rho, tau, lambda_)
        if convergence_condition(l_, l, epsilon):
            break
        else:
            l = l_
    return theta, phi, lambda_, rho


def inference(task_worker_class, n, m, K, L, epsilon=1e-2):
    '''
    Worker Clustering Version
    :param task_worker_class:
    :param n:
    :param m:
    :param K:
    :param L:
    :param epsilon:
    :return:
    '''
    x = convert_input(task_worker_class, n, m, K)

    c = 1
    theta, phi, lambda_, rho = one_iteration(x, K, L, epsilon=epsilon, random=False)

    while is_chance_rate(theta):
        c += 1
        theta, phi, lambda_, rho = one_iteration(x, K, L, epsilon=epsilon, random=True)
        print("Drop into " + str(c) + "th chance rate!!")
        if c >= 100:
            break

    g_hat = np.argmax(theta, axis=1)
    pi_hat = lambda_[np.argmax(phi, axis=1)]
    return g_hat, [pi_hat, rho], c
