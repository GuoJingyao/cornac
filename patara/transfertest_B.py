import cornac
from cornac.datasets import movielens
from cornac.models import BPR_seperable
from cornac.eval_methods import BaseMethod
import random
from cornac.data import Reader
import numpy
import scipy.io
import pickle
from sklearn.model_selection import train_test_split
from cornac.utils.data_utils import *
from collections import Counter

# Load the MovieLens dataset
rawdata = cornac.datasets.netflix.load_data_small(reader=Reader(bin_threshold=1.0))
data = numpy.unique(rawdata, axis=0)

c = Counter(data[:, 0]).most_common(100)
users = list(numpy.asarray(c)[:, 0])
index = numpy.isin(data[:, 0], users)
data = data[index, :]

# users = list(numpy.unique(data[:, 0]))
print(len(users))
items = list(numpy.unique(data[:, 1]))

# build source and target dataset
experimentSet = random.sample(users, round(len(users) * 0.2))  # save 20% users as experiment set
indexS = numpy.isin(data[:, 0], experimentSet)
indexB = numpy.isin(data[:, 0], experimentSet, invert=True)
SourceData = data[indexB, :]
TargetData = data[indexS, :]
print("Soucedata size", SourceData.shape[0])
print("TargetData size", TargetData.shape[0])

initialU = numpy.random.normal(loc=0.0, scale=1.0, size=len(users) * 10).reshape(len(users), 10)
initialV = numpy.random.normal(loc=0.0, scale=1.0, size=len(items) * 10).reshape(len(items), 10)

# global dictionary contains sorce data and target data
# key: raw data index, value: index for initial feature vector
globalD_users = dict(zip(users, list(range(len(users)))))
globalD_items = dict(zip(items, list(range(len(items)))))
#
scipy.io.savemat("initailSetting", {"SourceData": SourceData, "TargetData": TargetData, "initialU": initialU,
                                     "initialV": initialV})

# # load initial setting
# parameters = scipy.io.loadmat("initailSetting.mat")
# initialU = parameters['initialU']
# initialV = parameters['initialV']
# SourceData = parameters['SourceData']
# TargetData = parameters['TargetData']

mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=100)
pre_20 = cornac.metrics.Precision(k=100)
auc = cornac.metrics.AUC()

res_MAE = numpy.empty((3, 0))
res_RMSE = numpy.empty((3, 0))
res_RECALL20 = numpy.empty((3, 0))
res_PRECISION20 = numpy.empty((3, 0))
res_AUC = numpy.empty((3, 0))
maxiter = 200

Strain_Ttest = BaseMethod.from_splits(train_data=SourceData, test_data=TargetData, exclude_unknowns=False,
                                      verbose=True)
# Instantiate a baseline PMF recommender model and then run an experiment.
pretrain = BPR_seperable(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01,
                              init_params={'V': numpy.copy(initialV), 'U': numpy.copy(initialU), 'globalD_users': globalD_users, 'globalD_items': globalD_items})
exp_pretrainV = cornac.Experiment(eval_method=Strain_Ttest, models=[pretrain],
                                  metrics=[mae, rmse, rec_20, pre_20], user_based=True)
exp_pretrainV.run()

trainedV = numpy.copy(initialV)
for oldindex, newindex in Strain_Ttest.train_set.iid_map.items():
    trainedV[globalD_items.get(oldindex), :] = pretrain.i_factors[newindex, :]

scipy.io.savemat("trainedV", {"trainedV": trainedV})

# # trainedV=loadmat("trainedV.mat")['trainedV']

baseline = BPR_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lambda_reg=0.001,
                        init_params={'V': numpy.copy(initialV), 'U': numpy.copy(initialU), 'globalD_users': globalD_users, 'globalD_items': globalD_items}, verbose=True)
fixV = BPR_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lambda_reg=0.001, fixedParameter='V',
                    init_params={'V': numpy.copy(trainedV), 'U': numpy.copy(initialU), 'globalD_users': globalD_users, 'globalD_items': globalD_items}, verbose=True)
transferV = BPR_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lambda_reg=0.001,
                        init_params={'V': numpy.copy(trainedV), 'U': numpy.copy(initialU), 'globalD_users': globalD_users, 'globalD_items': globalD_items}, verbose=True)

for Tration in range(5, 100, 5):

    # sparse target training data
    Tusers = list(numpy.unique(TargetData[:, 0]))
    sparse_target_train = numpy.empty((0, 3))
    target_test = numpy.empty((0, 3))
    for u in Tusers:
        index_Udata = numpy.isin(TargetData[:, 0], u)
        Udata = TargetData[index_Udata, :]
        Utest, Utrain = train_test_split(Udata, test_size=(Tration * 0.01))
        # sampled_Udata = Udata[numpy.random.choice(Udata.shape[0], round(Udata.shape[0]*0.25), replace=False), :]
        sparse_target_train = numpy.vstack((sparse_target_train, Utrain))
        target_test = numpy.vstack((target_test, Utest))

    print("sparse_target_train size", sparse_target_train.shape[0])
    print("target_test size", target_test.shape[0])

    Tsparsetrain_Ttest = BaseMethod.from_splits(train_data=sparse_target_train, test_data=target_test,
                                                exclude_unknowns=False, verbose=True)

    exp_Baseline = cornac.Experiment(eval_method=Tsparsetrain_Ttest, models=[baseline],
                                     metrics=[mae, rmse, rec_20, pre_20, auc], user_based=True)
    exp_Baseline.run()

    exp_fixV = cornac.Experiment(eval_method=Tsparsetrain_Ttest, models=[fixV],
                                 metrics=[mae, rmse, rec_20, pre_20, auc], user_based=True)
    exp_fixV.run()

    exp_transferV = cornac.Experiment(eval_method=Tsparsetrain_Ttest, models=[transferV],
                                      metrics=[mae, rmse, rec_20, pre_20, auc], user_based=True)
    exp_transferV.run()

    # res_MAE = numpy.hstack((res_MAE, np.array(
    #     [[exp_Baseline.result[0].metric_avg_results.get("MAE")], [exp_fixV.result[0].metric_avg_results.get("MAE")],
    #      [exp_transferV.result[0].metric_avg_results.get("MAE")]])))
    #
    # res_RMSE = numpy.hstack((res_RMSE, np.array(
    #     [[exp_Baseline.result[0].metric_avg_results.get("RMSE")], [exp_fixV.result[0].metric_avg_results.get("RMSE")],
    #      [exp_transferV.result[0].metric_avg_results.get("RMSE")]])))

    res_RECALL20 = numpy.hstack((res_RECALL20, np.array(
        [[exp_Baseline.result[0].metric_avg_results.get("Recall@100")],
         [exp_fixV.result[0].metric_avg_results.get("Recall@100")],
         [exp_transferV.result[0].metric_avg_results.get("Recall@100")]])))

    res_PRECISION20 = numpy.hstack((res_PRECISION20, np.array(
        [[exp_Baseline.result[0].metric_avg_results.get("Precision@100")],
         [exp_fixV.result[0].metric_avg_results.get("Precision@100")],
         [exp_transferV.result[0].metric_avg_results.get("Precision@100")]])))

    res_AUC = numpy.hstack((res_AUC, np.array(
        [[exp_Baseline.result[0].metric_avg_results.get("AUC")], [exp_fixV.result[0].metric_avg_results.get("AUC")],
         [exp_transferV.result[0].metric_avg_results.get("AUC")]])))

# print("res_MAE, Baseline", res_MAE[0, :])
# print("res_MAE, fixV", res_MAE[1, :])
# print("res_MAE, transferV", res_MAE[2, :])

import pandas as pd

# table_MAE = pd.DataFrame(res_MAE)
# filepath = 'MAE.xlsx'
# table_MAE.to_excel(filepath, index=False)
#
# table_RMSE = pd.DataFrame(res_RMSE)
# filepath = 'RMSE.xlsx'
# table_RMSE.to_excel(filepath, index=False)

table_RECALL20 = pd.DataFrame(res_RECALL20)
filepath = 'RECALL100.xlsx'
table_RECALL20.to_excel(filepath, index=False)

table_PRECISION20 = pd.DataFrame(res_PRECISION20)
filepath = 'PRECISION100.xlsx'
table_PRECISION20.to_excel(filepath, index=False)

table_AUC = pd.DataFrame(res_AUC)
filepath = 'AUC.xlsx'
table_AUC.to_excel(filepath, index=False)