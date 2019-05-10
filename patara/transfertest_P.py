import cornac
from cornac.datasets import movielens
from cornac.models import PMF_seperable
from cornac.eval_methods import BaseMethod
import random
import numpy
from sklearn.model_selection import train_test_split
from cornac.utils.data_utils import *

# Load the MovieLens dataset
rawdata = movielens.load_1m()
data = numpy.unique(rawdata, axis=0)
users = list(numpy.unique(data[:, 0]))

valid_data, validUsers, validItems = Dataset(data).index_trans()
# build source and target dataset
experimentSet = random.sample(users, round(len(users) * 0.2))  # save 20% users as experiment set
indexS = numpy.isin(valid_data[:, 0], experimentSet)
indexB = numpy.isin(valid_data[:, 0], experimentSet, invert=True)
SourceData = valid_data[indexB, :]
TargetData = valid_data[indexS, :]
print("Soucedata size", SourceData.size)
print("TargetData size", TargetData.size)

u_max = max(valid_data[:, 0].astype(int)) + 1
i_max = max(valid_data[:, 1].astype(int)) + 1

# the user and item index corresponding to the valid_data set index
initialU = numpy.random.normal(loc=0.0, scale=1.0, size=u_max * 10).reshape(u_max, 10)
initialV = numpy.random.normal(loc=0.0, scale=1.0, size=i_max * 10).reshape(i_max, 10)

import scipy.io

scipy.io.savemat("initailP", {"initialU": initialU, "initailV": initialV})
# load initial parameters
# parameters = loadmat("init_parameters.mat")
# initialU = parameters['U']
# initialV = parameters['V']

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

# Instantiate a baseline PMF recommender model and then run an experiment.
pmf_pretrain = PMF_seperable(k=10, max_iter=200, learning_rate=0.001, lamda=0.001,
                             init_params={'V': numpy.copy(initialV), 'U': numpy.copy(initialU)})

# pretrain V
Strain_Ttest = BaseMethod.from_splits(train_data=SourceData, test_data=TargetData, exclude_unknowns=False,
                                      verbose=True)
exp_pretrainV = cornac.Experiment(eval_method=Strain_Ttest, models=[pmf_pretrain],
                                  metrics=[mae, rmse, rec_20, pre_20], user_based=True)
exp_pretrainV.run()
trainedV = numpy.copy(initialV)
for oldindex, newindex in Strain_Ttest.train_set.iid_map.items():
    trainedV[int(oldindex), :] = pmf_pretrain.V[newindex, :]

scipy.io.savemat("trainedV", {"trainedV": trainedV, "SourceData": SourceData, "TargetData": TargetData})

# trainedV=loadmat("trainedV.mat")['trainedV']

pmf_baseline = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001,
                             init_params={'V': numpy.copy(initialV), 'U': numpy.copy(initialU)}, verbose=True)
pmf_fixV = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001, fixedParameter='V',
                         init_params={'V': numpy.copy(trainedV), 'U': numpy.copy(initialU)}, verbose=True)
pmf_transferV = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001,
                              init_params={'V': numpy.copy(trainedV), 'U': numpy.copy(initialU)}, verbose=True)

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

    print("sparse_target_train size", sparse_target_train.size)
    print("target_test size", target_test.size)

    Tsparsetrain_Ttest = BaseMethod.from_splits(train_data=sparse_target_train, test_data=target_test,
                                                exclude_unknowns=False, verbose=True)

    exp_Baseline = cornac.Experiment(eval_method=Tsparsetrain_Ttest, models=[pmf_baseline],
                                     metrics=[mae, rmse, rec_20, pre_20, auc], user_based=True)
    exp_Baseline.run()

    exp_fixV = cornac.Experiment(eval_method=Tsparsetrain_Ttest, models=[pmf_fixV],
                                 metrics=[mae, rmse, rec_20, pre_20, auc], user_based=True)
    exp_fixV.run()

    exp_transferV = cornac.Experiment(eval_method=Tsparsetrain_Ttest, models=[pmf_transferV],
                                      metrics=[mae, rmse, rec_20, pre_20, auc], user_based=True)
    exp_transferV.run()

    res_MAE = numpy.hstack((res_MAE, np.array(
        [[exp_Baseline.result[0].metric_avg_results.get("MAE")], [exp_fixV.result[0].metric_avg_results.get("MAE")],
         [exp_transferV.result[0].metric_avg_results.get("MAE")]])))

    res_RMSE = numpy.hstack((res_RMSE, np.array(
        [[exp_Baseline.result[0].metric_avg_results.get("RMSE")], [exp_fixV.result[0].metric_avg_results.get("RMSE")],
         [exp_transferV.result[0].metric_avg_results.get("RMSE")]])))

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

table_MAE = pd.DataFrame(res_MAE)
filepath = '100k_MAE.xlsx'
table_MAE.to_excel(filepath, index=False)

table_RMSE = pd.DataFrame(res_RMSE)
filepath = '100k_RMSE.xlsx'
table_RMSE.to_excel(filepath, index=False)

table_RECALL20 = pd.DataFrame(res_RECALL20)
filepath = '100k_RECALL100.xlsx'
table_RECALL20.to_excel(filepath, index=False)

table_PRECISION20 = pd.DataFrame(res_PRECISION20)
filepath = '100k_PRECISION100.xlsx'
table_PRECISION20.to_excel(filepath, index=False)

table_AUC = pd.DataFrame(res_AUC)
filepath = '100k_AUC.xlsx'
table_AUC.to_excel(filepath, index=False)