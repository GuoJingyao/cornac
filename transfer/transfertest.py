import cornac
from cornac.datasets import movielens
from cornac.models import PMF_seperable
from cornac.eval_methods import BaseMethod
from cornac.data import trainset
import random
import numpy
from sklearn.model_selection import train_test_split
from cornac.utils.data_utils import *

# Load the MovieLens dataset
rawdata =movielens.load_100k()
data = numpy.unique(rawdata, axis=0)
users = list(numpy.unique(data[:, 0]))

valid_data = numpy.empty((0,3))
for u in users:
    index_Udata = numpy.isin(data[:, 0], u)
    Udata = data[index_Udata, :]
    sampled_Udata = Udata[numpy.random.choice(Udata.shape[0], round(Udata.shape[0]*0.25), replace=False), :]
    valid_data = numpy.vstack((valid_data, sampled_Udata))

valid_data, validUsers, validItems = Dataset(valid_data).index_trans()
# build source and target dataset
experimentSet = random.sample(users, round(len(users)*0.2))  # save 20% users as experiment set
indexS = numpy.isin(valid_data[:, 0], experimentSet)
indexB = numpy.isin(valid_data[:, 0], experimentSet, invert=True)
SourceData = valid_data[indexB, :]
TargetData = valid_data[indexS, :]

target_train, target_test = train_test_split(TargetData, test_size=0.2)

u_max = max(valid_data[:, 0].astype(int))+1
i_max = max(valid_data[:, 1].astype(int))+1

# the user and item index corresponding to the valid_data set index
initialU = numpy.random.normal(loc=0.0, scale=1.0, size=u_max*10).reshape(u_max, 10).astype(numpy.float32)
initialV = numpy.random.normal(loc=0.0, scale=1.0, size=i_max*10).reshape(i_max, 10).astype(numpy.float32)

# Instantiate an evaluation method.
Strain_Ttest = BaseMethod.from_splits(train_data=SourceData, test_data=target_test, exclude_unknowns=False, verbose=True)
Ttrain_Ttest = BaseMethod.from_splits(train_data=target_train, test_data=target_test, exclude_unknowns=False, verbose=True)

# Instantiate a PMF recommender model.
pmf = PMF_seperable(k=10, max_iter=100, learning_rate=0.001, lamda=0.001, init_params={'V':initialV, 'U':initialU})

# Instantiate evaluation metrics.
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
exp_Baseline = cornac.Experiment(eval_method=Ttrain_Ttest, models=[pmf], metrics=[mae, rmse], user_based=True)
exp_Baseline.run()