import cornac
from cornac.datasets import movielens
from cornac.models import PMF_seperable
from cornac.eval_methods import BaseMethod
from cornac.data import trainset
import random
import numpy
from sklearn.model_selection import train_test_split
from cornac.utils.data_utils import *
from scipy.io import loadmat

# Load the MovieLens dataset
rawdata =movielens.load_100k()
data = numpy.unique(rawdata, axis=0)
users = list(numpy.unique(data[:, 0]))

valid_data, validUsers, validItems = Dataset(data).index_trans()
# build source and target dataset
experimentSet = random.sample(users, round(len(users)*0.2))  # save 20% users as experiment set
indexS = numpy.isin(valid_data[:, 0], experimentSet)
indexB = numpy.isin(valid_data[:, 0], experimentSet, invert=True)
SourceData = valid_data[indexB, :]
TargetData = valid_data[indexS, :]
print("Soucedata size", SourceData.size)
print("TargetData size", TargetData.size)

# u_max = max(valid_data[:, 0].astype(int))+1
# i_max = max(valid_data[:, 1].astype(int))+1
#
# # the user and item index corresponding to the valid_data set index
# initialU = numpy.random.normal(loc=0.0, scale=1.0, size=u_max*10).reshape(u_max, 10).astype(numpy.float32)
# initialV = numpy.random.normal(loc=0.0, scale=1.0, size=i_max*10).reshape(i_max, 10).astype(numpy.float32)
# load initial parameters
parameters= loadmat("init_parameters.mat")
initialU = parameters['U']
initialV = parameters['V']

mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

res_MAE = numpy.empty((3, 0))
res_RMSE = numpy.empty((3, 0))
res_RECALL20 = numpy.empty((3, 0))
res_PRECISION20 = numpy.empty((3, 0))

maxiter = 100

# Instantiate a baseline PMF recommender model and then run an experiment.
pmf_baseline = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001,
                             init_params={'V': numpy.copy(initialV), 'U': numpy.copy(initialU)})


# pretrain V
Strain_Ttest = BaseMethod.from_splits(train_data=SourceData, test_data=TargetData, exclude_unknowns=False,
                                      verbose=True)
exp_pretrainV = cornac.Experiment(eval_method=Strain_Ttest, models=[pmf_baseline],
                                  metrics=[mae, rmse, rec_20, pre_20], user_based=True)
exp_pretrainV.run()
trainedV = numpy.copy(initialV)
for oldindex, newindex in Strain_Ttest.global_uid_map.items():
    trainedV[int(oldindex), :] = pmf_baseline.V[newindex, :]


pmf_fixV = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001, fixedParameter='V',
                             init_params={'V': trainedV, 'U': numpy.copy(initialU)})
pmf_transferV = PMF_seperable(k=10, max_iter=maxiter, learning_rate=0.001, lamda=0.001,
                                  init_params={'V': numpy.copy(trainedV), 'U': numpy.copy(initialU)})


for Tration in range(5, 100, 5):

    # sparse target training data
    Tusers = list(numpy.unique(TargetData[:, 0]))
    sparse_target_train = numpy.empty((0, 3))
    target_test = numpy.empty((0, 3))
    for u in Tusers:
        index_Udata = numpy.isin(TargetData[:, 0], u)
        Udata = TargetData[index_Udata, :]
        Utest, Utrain = train_test_split(Udata, test_size=(Tration/100))
        # sampled_Udata = Udata[numpy.random.choice(Udata.shape[0], round(Udata.shape[0]*0.25), replace=False), :]
        sparse_target_train = numpy.vstack((sparse_target_train, Utrain))
        target_test = numpy.vstack((target_test, Utest))

    print("sparse_target_train size", sparse_target_train.size)
    print("target_test size", target_test.size)
    # Instantiate an evaluation method.
    # Strain_Ttest = BaseMethod.from_splits(train_data=SourceData, test_data=target_test, exclude_unknowns=False,
    #                                       verbose=True)
    Tsparsetrain_Ttest = BaseMethod.from_splits(train_data=sparse_target_train, test_data=target_test,
                                                exclude_unknowns=False, verbose=True)

    exp_Baseline = cornac.Experiment(eval_method=Tsparsetrain_Ttest, models=[pmf_baseline],
                                     metrics=[mae, rmse, rec_20, pre_20], user_based=True)
    exp_Baseline.run()

    exp_fixV = cornac.Experiment(eval_method=Tsparsetrain_Ttest, models=[pmf_fixV], metrics=[mae, rmse, rec_20, pre_20],
                                 user_based=True)
    exp_fixV.run()

    exp_transferV = cornac.Experiment(eval_method=Tsparsetrain_Ttest, models=[pmf_transferV],
                                      metrics=[mae, rmse, rec_20, pre_20], user_based=True)
    exp_transferV.run()

    res_MAE = numpy.hstack((res_MAE, np.array(
        [[exp_Baseline.result[0].metric_avg_results.get("MAE")], [exp_fixV.result[0].metric_avg_results.get("MAE")],
         [exp_transferV.result[0].metric_avg_results.get("MAE")]])))

    res_RMSE = numpy.hstack((res_RMSE, np.array(
        [[exp_Baseline.result[0].metric_avg_results.get("RMSE")], [exp_fixV.result[0].metric_avg_results.get("RMSE")],
         [exp_transferV.result[0].metric_avg_results.get("RMSE")]])))

    res_RECALL20 = numpy.hstack((res_RECALL20, np.array(
        [[exp_Baseline.result[0].metric_avg_results.get("Recall@20")], [exp_fixV.result[0].metric_avg_results.get("Recall@20")],
         [exp_transferV.result[0].metric_avg_results.get("Recall@20")]])))

    res_PRECISION20 = numpy.hstack((res_PRECISION20, np.array(
        [[exp_Baseline.result[0].metric_avg_results.get("Precision@20")], [exp_fixV.result[0].metric_avg_results.get("Precision@20")],
         [exp_transferV.result[0].metric_avg_results.get("Precision@20")]])))

print("res_MAE", res_MAE)
print("res_RMSE", res_RMSE)
print("res_RECALL20", res_RECALL20)
print("res_PRECISION20", res_PRECISION20)