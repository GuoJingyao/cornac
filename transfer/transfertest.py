import cornac
from cornac.datasets import movielens
from cornac.models import PMF
from cornac.eval_methods import BaseMethod
from cornac.data import trainset
import random
import numpy
from sklearn.model_selection import train_test_split

# Load the MovieLens dataset
rawdata = movielens.load_100k()
valid_data, valid_users, valid_items = rawdata.index_trans()

experimentSet = random.sample(valid_users, round(len(valid_users)*0.2))  # save 20% users as experiment set
indexS = numpy.isin(valid_data[:, 0], experimentSet)
indexB = numpy.isin(valid_data[:, 0], experimentSet, invert=True)
SourceData = valid_data[indexB, :]
TargetData = valid_data[indexS, :]

target_train, target_test = train_test_split(TargetData, test_size=0.2)

# Instantiate an evaluation method.
eval_method = BaseMethod.from_splits(train_data=SourceData, test_data=target_test,
                                     exclude_unknowns=False, verbose=True)

# Instantiate a PMF recommender model.
pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)

# Instantiate evaluation metrics.
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
exp = cornac.Experiment(eval_method=eval_method,
                        models=[pmf],
                        metrics=[mae, rmse, rec_20, pre_20],
                        user_based=True)
exp.run()