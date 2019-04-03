# -*- coding: utf-8 -*-
"""
@author: Aghiles Salah
"""
import numpy as np
from ...utils import common


# PMF (Gaussian non-linear model version using sigmoid function)  SGD_RMSProp optimizer
def pmf_seperable(trainset, n_X, d_X, k, fixedParameter=None, n_epochs=100, lamda=0.001, learning_rate=0.001, gamma=0.9, init_params=None):
    # some useful variables
    loss = np.full(n_epochs, 0.0)
    n = n_X
    d = d_X
    cache_u = np.zeros((n, k))
    cache_v = np.zeros((d, k))
    grad_u = np.zeros((n, k))
    grad_v = np.zeros((d, k))
    eps = 1e-8

    # Parameter initialization
    # User factors
    if init_params['U'] is None:
        U = np.random.normal(loc=0.0, scale=0.001, size=n * k).reshape(n, k)
    else:
        U = np.zeros([n, k])
        initialU = init_params['U']
        for oldindex, newindex in trainset._uid_map.items():
            U[newindex, :]=initialU[int(oldindex),:]

    # Item factors
    if init_params['V'] is None:
        V = np.random.normal(loc=0.0, scale=0.001, size=d * k).reshape(d, k)
    else:
        V = np.zeros([d, k])
        initialV = init_params['V']
        for oldindex, newindex in trainset._iid_map.items():
            V[newindex, :] = initialV[int(oldindex), :]

    # Optimization

    if fixedParameter =='V':
        print("fixed V, just update U")
        for epoch in range(n_epochs):
            for u_, i_, val in trainset.uir_iter(batch_size=1, shuffle=False):
                u_, i_ = int(u_), int(i_)

                sg = common.sigmoid(np.dot(U[u_, :], V[i_, :].T))
                e = (val - sg)  # Error for the obseved rating u, i, val
                we = e * sg * (1. - sg)  # Weighted error for the obseved rating u, i, val
                grad_u[u_, :] = we * V[i_, :] - lamda * U[u_, :]
                cache_u[u_, :] = gamma * cache_u[u_, :] + (1 - gamma) * (grad_u[u_, :] * grad_u[u_, :])
                U[u_, :] += learning_rate * (grad_u[u_, :] / (np.sqrt(cache_u[u_,:]) + eps))
                # Update the user factor, better to reweight the L2 regularization terms acoording the number of ratings per-user
                loss[epoch] += e * e + lamda * (np.dot(U[u_, :].T, U[u_, :]) + np.dot(V[i_, :].T, V[i_, :]))
            # print('epoch %i, loss: %f' % (epoch, loss[epoch]))
        res = {'U': U, 'V': V, 'loss': loss}
        return res
    else:
        for epoch in range(n_epochs):
            for u_, i_, val in trainset.uir_iter(batch_size=1, shuffle=False):
                u_, i_ = int(u_), int(i_)
                sg = common.sigmoid(np.dot(U[u_, :], V[i_, :].T))
                e = (val - sg)  # Error for the obseved rating u, i, val
                we = e * sg * (1. - sg)  # Weighted error for the obseved rating u, i, val
                grad_u[u_, :] = we * V[i_, :] - lamda * U[u_, :]
                cache_u[u_, :] = gamma * cache_u[u_, :] + (1 - gamma) * (grad_u[u_, :] * grad_u[u_, :])
                U[u_, :] += learning_rate * (grad_u[u_, :] / (np.sqrt(cache_u[u_,
                                                                      :]) + eps))  # Update the user factor, better to reweight the L2 regularization terms acoording the number of ratings per-user

                # update item factors
                grad_v[i_, :] = we * U[u_, :] - lamda * V[i_, :]
                cache_v[i_, :] = gamma * cache_v[i_, :] + (1 - gamma) * (grad_v[i_, :] * grad_v[i_, :])
                V[i_, :] += learning_rate * (
                        grad_v[i_, :] / (np.sqrt(cache_v[i_, :]) + eps))
                loss[epoch] += e * e + lamda * (np.dot(U[u_, :].T, U[u_, :]) + np.dot(V[i_, :].T, V[i_, :]))
            # print('epoch %i, loss: %f' % (epoch, loss[epoch]))
        res = {'U': U, 'V': V, 'loss': loss}
        return res







