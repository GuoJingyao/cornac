import cornac
from cornac.models import PMF_seperable
from cornac.eval_methods import BaseMethod
from cornac.data import Reader
import random
import numpy

# Load the MovieLens dataset
rawdata = cornac.datasets.netflix.load_data_small()
data = numpy.unique(rawdata, axis=0)
users = list(numpy.unique(data[:, 0]))
items = list(numpy.unique(data[:, 1]))

print("total user number:", len(users))
print("total item number:", len(items))

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

import scipy.io

# global dictionary contains sorce data and target data
# key: raw data index, value: index for initial feature vector
globalD_users = dict(zip(users, list(range(len(users)))))
globalD_items = dict(zip(items, list(range(len(items)))))
#
scipy.io.savemat("initailSetting", {"SourceData": SourceData, "TargetData": TargetData, "initialU": initialU,
                                     "initialV": initialV})
# load initial parameters
# parameters = loadmat("init_parameters.mat")
# initialU = parameters['U']
# initialV = parameters['V']

mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec = cornac.metrics.Recall(k=100)
pre = cornac.metrics.Precision(k=100)
auc = cornac.metrics.AUC()
maxiter = 200

# Instantiate a baseline PMF recommender model and then run an experiment.
pmf_pretrain = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001,
                             init_params={'V': numpy.copy(initialV), 'U': numpy.copy(initialU), 'globalD_users': globalD_users, 'globalD_items': globalD_items})

# pretrain V
Strain_Ttest = BaseMethod.from_splits(train_data=SourceData, test_data=TargetData, exclude_unknowns=False,
                                      verbose=True)
exp_pretrainV = cornac.Experiment(eval_method=Strain_Ttest, models=[pmf_pretrain],
                                  metrics=[mae, rmse, rec, pre], user_based=True)
exp_pretrainV.run()

trainedV = numpy.copy(initialV)
for oldindex, newindex in Strain_Ttest.train_set.iid_map.items():
    trainedV[globalD_items.get(oldindex), :] = pmf_pretrain.V[newindex, :]

scipy.io.savemat("trainedV", {"trainedV": trainedV, "SourceData": SourceData, "TargetData": TargetData})

pmf_baseline = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001,
                             init_params={'V': numpy.copy(initialV), 'U': numpy.copy(initialU), 'globalD_users': globalD_users, 'globalD_items': globalD_items}, verbose=False)
pmf_fixV = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001, fixedParameter='V',
                         init_params={'V': numpy.copy(trainedV), 'U': numpy.copy(initialU), 'globalD_users': globalD_users, 'globalD_items': globalD_items}, verbose=False)
pmf_transferV = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001,
                              init_params={'V': numpy.copy(trainedV), 'U': numpy.copy(initialU), 'globalD_users': globalD_users, 'globalD_items': globalD_items}, verbose=False)


Tusers = list(numpy.unique(TargetData[:, 0]))
Blocks =[]
splitK = 10
MAE = numpy.empty((3, 10, 9))
RMSE = numpy.empty((3, 10, 9))
REC = numpy.empty((3, 10, 9))
PRE = numpy.empty((3, 10, 9))
AUC = numpy.empty((3, 10, 9))

for u in Tusers:
    index_Udata = numpy.isin(TargetData[:, 0], u)
    Udata = TargetData[index_Udata, :]
    ublocks = numpy.array_split(Udata, splitK)
    blocks = random.sample(ublocks, splitK)

    if Blocks == []:
        Blocks = blocks
    else:
        for i in range(splitK):
            Blocks[i] = numpy.vstack((Blocks[i], blocks[i]))

for TestId in range(splitK):
    target_test = Blocks[TestId]
    for TrainSize in range(1, splitK):
        target_train = numpy.concatenate(random.sample((Blocks[:TestId] + Blocks[TestId+1:]), TrainSize), axis=0)
        print("target_train size", target_train.shape[0])
        print("target_test size", target_test.shape[0])

        Ttrain_Ttest = BaseMethod.from_splits(train_data=target_train, test_data=target_test,
                                                    exclude_unknowns=False, verbose=True)

        exp_Baseline = cornac.Experiment(eval_method=Ttrain_Ttest, models=[pmf_baseline],
                                     metrics=[mae, rmse, rec, pre, auc], user_based=True)
        exp_Baseline.run()

        exp_fixV = cornac.Experiment(eval_method=Ttrain_Ttest, models=[pmf_fixV],
                                 metrics=[mae, rmse, rec, pre, auc], user_based=True)
        exp_fixV.run()

        exp_transferV = cornac.Experiment(eval_method=Ttrain_Ttest, models=[pmf_transferV],
                                      metrics=[mae, rmse, rec, pre, auc], user_based=True)
        exp_transferV.run()

        MAE[0][TestId][TrainSize-1] = exp_Baseline.result[0].metric_avg_results.get("MAE")
        MAE[1][TestId][TrainSize-1] = exp_fixV.result[0].metric_avg_results.get("MAE")
        MAE[2][TestId][TrainSize-1] = exp_transferV.result[0].metric_avg_results.get("MAE")

        RMSE[0][TestId][TrainSize-1] = exp_Baseline.result[0].metric_avg_results.get("RMSE")
        RMSE[1][TestId][TrainSize-1] = exp_fixV.result[0].metric_avg_results.get("RMSE")
        RMSE[2][TestId][TrainSize-1] = exp_transferV.result[0].metric_avg_results.get("RMSE")

        REC[0][TestId][TrainSize-1] = exp_Baseline.result[0].metric_avg_results.get("Recall@100")
        REC[1][TestId][TrainSize-1] = exp_fixV.result[0].metric_avg_results.get("Recall@100")
        REC[2][TestId][TrainSize-1] = exp_transferV.result[0].metric_avg_results.get("Recall@100")

        PRE[0][TestId][TrainSize-1] = exp_Baseline.result[0].metric_avg_results.get("Precision@100")
        PRE[1][TestId][TrainSize-1] = exp_fixV.result[0].metric_avg_results.get("Precision@100")
        PRE[2][TestId][TrainSize-1] = exp_transferV.result[0].metric_avg_results.get("Precision@100")

        AUC[0][TestId][TrainSize-1] = exp_Baseline.result[0].metric_avg_results.get("AUC")
        AUC[1][TestId][TrainSize-1] = exp_fixV.result[0].metric_avg_results.get("AUC")
        AUC[2][TestId][TrainSize-1] = exp_transferV.result[0].metric_avg_results.get("AUC")

# import pandas as pd
# table_RECALL20 = pd.DataFrame(REC[0])
# filepath = 'C:/Users\jyguo\Desktop/result/PMF/netflix_small/exp2/REC_baseline.xlsx'
# table_RECALL20.to_excel(filepath, index=False)
# table_RECALL20 = pd.DataFrame(REC[1])
# filepath = 'C:/Users\jyguo\Desktop/result/PMF/netflix_small/exp2/REC_fixV.xlsx'
# table_RECALL20.to_excel(filepath, index=False)
# table_RECALL20 = pd.DataFrame(REC[2])
# filepath = 'C:/Users\jyguo\Desktop/result/PMF/netflix_small/exp2/REC__transV.xlsx'
# table_RECALL20.to_excel(filepath, index=False)

# import pandas as pd
#
# table_MAE = pd.DataFrame(MAE)
# filepath = 'C:/Users\jyguo\Desktop\movielens1m/MAE.xlsx'
# table_MAE.to_excel(filepath, index=False)
#
# table_RMSE = pd.DataFrame(RMSE)
# filepath = 'C:/Users\jyguo\Desktop\movielens1m/RMSE.xlsx'
# table_RMSE.to_excel(filepath, index=False)
#
# table_RECALL20 = pd.DataFrame(REC)
# filepath = '100k_RECALL100.xlsx'
# table_RECALL20.to_excel(filepath, index=False)
#
# table_PRECISION20 = pd.DataFrame(PRE)
# filepath = '100k_PRECISION100.xlsx'
# table_PRECISION20.to_excel(filepath, index=False)
#
# table_AUC = pd.DataFrame(AUC)
# filepath = '100k_AUC.xlsx'
# table_AUC.to_excel(filepath, index=False)