# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao <jyguo@smu.edu.sg>
"""

import numpy as np
import torch
import math
import scipy
import torch.nn as nn
from cornac.utils import data_utils
import pdb

def sorec(train_set, l, n_epochs=100, learning_rate=0.001, lamda_C=10, lamda=0.01, init_params=None):

    X = train_set.matrix
    n = train_set.num_users
    d = train_set.num_items

    graph = train_set.user_graph
    user_n = np.amax(graph.map_data)+1
    trust_raw = scipy.sparse.csc_matrix((graph.map_data[:, 2], (graph.map_data[:, 0], graph.map_data[:, 1])), shape=(user_n, user_n))
    outdegree = np.array(trust_raw.sum(axis=0)).flatten()
    indegree = np.array(trust_raw.sum(axis=1)).flatten()
    weighted_trust = []
    for ui, uk, cik in graph.map_data:
        i_out = outdegree[int(ui)]
        k_in = indegree[int(uk)]
        cik_weighted = math.sqrt(k_in/(k_in+i_out))*cik
        weighted_trust.append(cik_weighted)

    trust_matrix =scipy.sparse.csc_matrix((weighted_trust, (graph.map_data[:, 0], graph.map_data[:, 1])), shape=(user_n, user_n))

    if init_params['U'] is None:
        U = torch.randn(l, n, requires_grad=True)
    else:
        U = init_params['U']
        newU = np.zeros((l, n))
        for oldidx, newidx in train_set._uid_map.items():
            newU[:, newidx] = U[:, int(oldidx)]
        U = torch.tensor(newU, dtype=torch.float32, requires_grad=True)

    if init_params['V'] is None:
        V = torch.randn(l, d, requires_grad=True)
    else:
        V = init_params['V']
        newV = np.zeros((l, d))
        for oldidx, newidx in train_set._iid_map.items():
            newV[:, newidx]=V[:, int(oldidx)]
        V = torch.tensor(newV, dtype=torch.float32, requires_grad=True)

    if init_params['Z'] is None:
        Z = torch.randn(l, n, requires_grad=True)
    else:
        Z = init_params['Z']
        newZ = np.zeros((l, n))
        for oldidx, newidx in train_set._uid_map.items():
            newZ[:, newidx]=Z[:, int(oldidx)]
        Z = torch.tensor(newZ, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([U, V, Z], lr=learning_rate)

    for epoch in range(n_epochs):

        for uid, iid, val in train_set.uij_iter(1, shuffle=True):
            uid = uid[0]
            iid = iid[0]
            val = val[0]
            loss = (torch.add(torch.sigmoid(U[:, uid].dot(V[:,iid])), -val)).pow(2)
            print('epoch:', epoch, 'loss:', loss)
            print("cik: ", trust_matrix.getrow(uid))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch:', epoch, 'loss:', loss)

    U = U.data.numpy()
    V = V.data.numpy()
    Z = Z.data.numpy()

    res = {'U': U, 'V': V, 'Z': Z}
    return res

