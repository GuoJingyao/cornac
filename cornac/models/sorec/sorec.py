# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao <jyguo@smu.edu.sg>
"""

import numpy as np
import torch
import math
import scipy
from ...utils import common
import torch.nn as nn
from cornac.utils import data_utils
import pdb


# fit the recommender model to the traning data
# def fit(self, train_set):
#     """Fit the model to observations.
#
#     Parameters
#     ----------
#     train_set: object of type TrainSet, required
#         An object contraining the user-item preference in csr scipy sparse format,\
#         as well as some useful attributes such as mappings to the original user/item ids.\
#         Please refer to the class TrainSet in the "data" module for details.
#     """
#
#     Recommender.fit(self, train_set)
#
#     X = self.train_set.matrix
#
#     if self.trainable:
#         # converting data to the triplet format (needed for cython function pmf)
#         (rid, cid, val) = sp.find(X)
#         val = np.array(val, dtype='float32')
#
#         if [self.train_set.min_rating, self.train_set.max_rating] != [0, 1]:
#             if self.train_set.min_rating == self.train_set.max_rating:
#                 val = scale(val, 0., 1., 0., self.train_set.max_rating)
#             else:
#                 val = (val - 1) / (max(val) - 1)
#
#         self.train_set.uir_tuple = tuple([self.train_set.uir_tuple[0], self.train_set.uir_tuple[1], val])
#
#         if self.verbose:
#             print('Learning...')
#
#         res = sorec(train_set, l=self.l, n_epochs=self.max_iter, learning_rate=self.learning_rate,
#                     lamda_C=self.lamda_C, lamda=self.lamda, init_params=self.init_params)
#
#         self.U = np.asarray(res['U'])
#         self.V = np.asarray(res['V'])
#
#         if self.verbose:
#             print('Learning completed')
#     elif self.verbose:
#         print('%s is trained already (trainable = False)' % (self.name))


def sorec(train_set, l, n_epochs=100, learning_rate=0.001, lamda_C=10, lamda=0.01, init_params=None):

    X = train_set.matrix
    n = train_set.num_users
    d = train_set.num_items

    # generate global weighted trust matrix
    graph = train_set.user_graph
    user_n = int(np.amax(graph.map_data))+1
    trust_raw = scipy.sparse.csc_matrix((graph.map_data[:, 2], (graph.map_data[:, 0], graph.map_data[:, 1])), shape=(user_n, user_n))
    outdegree = np.array(trust_raw.sum(axis=0)).flatten()
    indegree = np.array(trust_raw.sum(axis=1)).flatten()
    weighted_trust = []
    for ui, uk, cik in graph.map_data:
        i_out = outdegree[int(ui)]
        k_in = indegree[int(uk)]
        cik_weighted = math.sqrt(k_in/(k_in+i_out))*cik
        weighted_trust.append(cik_weighted)

    # index = (graph.map_data[:, 0] < train_set.num_users) & (graph.map_data[:, 1] < train_set.num_users)
    # w_trust = np.array(weighted_trust)[index]
    # rowind = np.array(graph.map_data[:, 0])[index]
    # colind = np.array(graph.map_data[:, 1])[index]

    trust_matrix =scipy.sparse.csr_matrix((weighted_trust, (graph.map_data[:, 0], graph.map_data[:, 1])), shape=(user_n, user_n))
    # t_matrix = scipy.sparse.csc_matrix((w_trust, (rowind, colind)), shape=(n, n)).todense()
    # t_matrix = torch.tensor(t_matrix,dtype=torch.float32)
    # mask = scipy.sparse.csc_matrix((np.ones(len(w_trust)), (rowind, colind)), shape=(n, n)).todense()
    # mask = torch.tensor(mask,dtype=torch.float32)

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
        Z = torch.randn(l, user_n, requires_grad=True)
    else:
        Z = init_params['Z']
        newZ = np.zeros((l, user_n))
        for oldidx, newidx in train_set._uid_map.items():
            newZ[:, newidx]=Z[:, int(oldidx)]
        Z = torch.tensor(newZ, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([U, V, Z], lr=learning_rate)

    for epoch in range(n_epochs):

        for uid, iid, val in train_set.uir_iter(1000, shuffle=True):

            trust_tensor = torch.tensor(trust_matrix[uid].A, dtype=torch.float32)
            mask = (trust_tensor > 0).float() * 1

            l = (mask * (torch.sigmoid(torch.t(U[:, uid]).mm(Z)) - trust_tensor)).pow(2).sum()

            # torch.sparse.mm(trust_tensor, mat2)
            # for u in uid:
            #     ptr1, ptr2 = trust_matrix.indptr[u], trust_matrix.indptr[u + 1]
            #     utrust = trust_matrix.indices[ptr1:ptr2]
            #     trustvalue = trust_matrix.data[ptr1:ptr2]
            #     # _, utrust, trustvalue = scipy.sparse.find(trust_matrix[u])
            #     # u_nid.extend(utrust.tolist())
            #     if len(utrust)>0:
            #         u_nid.extend(utrust.tolist())
            #         l += (torch.sigmoid(U[:, u].matmul(Z[:, utrust.astype(int)]))
            #               - torch.tensor(trustvalue, dtype=torch.float32)).pow(2).sum()

            loss = (torch.sigmoid(U[:, uid].mul(V[:, iid]).sum(0)) - torch.tensor(val, dtype=torch.float32)).pow(2).sum() + \
                   l * lamda_C + \
                   lamda * (V[:, iid].norm().pow(2) + U[:, uid].norm().pow(2))# + Z[:, u_nid].norm().pow(2))

            print('loss:', loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch:', epoch, 'loss:', loss)

    U = U.data.numpy()
    V = V.data.numpy()
    Z = Z.data.numpy()

    res = {'U': U, 'V': V, 'Z': Z}
    return res

